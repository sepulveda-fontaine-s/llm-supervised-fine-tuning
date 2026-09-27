from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time

from llm_fine_tuning.preprocessing import build_squad_messages


MODEL_PATH = Path(
    "results/checkpoints/squad_v2_intermediate_v3/best_model"
)

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("llm_fine_tuning.api")

class QARequest(BaseModel):
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)


class QAResponse(BaseModel):
    answer: str
    input_tokens: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to serve the fine-tuned model."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    )

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    model.config.use_cache = True

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device

    yield

    del app.state.model
    torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen2.5 SQuAD Fine-Tuning API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "device": "cuda",
    }


@app.post("/answer", response_model=QAResponse)
def answer(request: QARequest):
    start_time = time.perf_counter()
    tokenizer = app.state.tokenizer
    model = app.state.model
    device = app.state.device

    example = {
        "context": request.context.strip(),
        "question": request.question.strip(),
        "answers": {"text": []},
    }

    messages = build_squad_messages(example)[:-1]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = inputs["input_ids"].shape[1]

    if input_tokens > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Input contains {input_tokens} tokens; "
                f"maximum supported is {MAX_INPUT_TOKENS}."
            ),
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    eos_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[:, input_tokens:]

    prediction = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "inference_complete | input_tokens=%s | latency_ms=%.2f",
        input_tokens,
        elapsed_ms,
    )
    return QAResponse(
        answer=prediction,
        input_tokens=input_tokens,
    )
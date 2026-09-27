import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/no_robots_qwen2_5_0_5b"
BATCH_SIZE = 4


def gib(bytes_value: int) -> float:
    return bytes_value / (1024**3)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_from_disk(DATASET_PATH)
    train = dataset["train"]

    longest_indices = sorted(
        range(len(train)),
        key=lambda index: len(train[index]["input_ids"]),
        reverse=True,
    )[:BATCH_SIZE]

    features = [train[index] for index in longest_indices]

    collator = SFTDataCollator(tokenizer=tokenizer)
    batch = collator(features)

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    print("=== BATCH ===")
    print("micro-batch size:", BATCH_SIZE)
    print("shape:", batch["input_ids"].shape)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.train()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    torch.cuda.reset_peak_memory_stats()

    outputs = model(**batch)
    loss = outputs.loss

    print("loss:", loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    print(
        "peak VRAM allocated:",
        f"{gib(torch.cuda.max_memory_allocated()):.2f} GiB",
    )
    print(
        "peak VRAM reserved:",
        f"{gib(torch.cuda.max_memory_reserved()):.2f} GiB",
    )

    print("=== BATCH MEMORY TEST PASSED ===")


if __name__ == "__main__":
    main()
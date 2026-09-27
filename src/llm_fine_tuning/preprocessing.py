from transformers import PreTrainedTokenizerBase

SYSTEM_MESSAGE = (
    "Answer the question using only the provided context. "
    "Return only the answer supported by the context. "
    "If the context does not contain the answer, respond exactly: "
    "I don't know."
)


def build_squad_messages(example):
    user_message = (
        f"Context:\n{example['context']}\n\n"
        f"Question:\n{example['question']}"
    )

    if len(example["answers"]["text"]) == 0:
        answer = "I don't know."
    else:
        answer = example["answers"]["text"][0]

    return [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_sft_example(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[int]]:
    """
    Tokenize one conversation for supervised fine-tuning.

    Tokens belonging to system/user messages and structural assistant
    prefixes receive label -100, so they are ignored by the language-model
    loss. Only assistant response tokens are supervised.
    """

    input_ids: list[int] = []
    labels: list[int] = []

    # Qwen inserts this system message automatically when none is supplied.
    if messages[0]["role"] != "system":
        system_text = (
            f"{IM_START}system\n"
            "You are a helpful assistant."
            f"{IM_END}\n"
        )

        system_ids = tokenizer(
            system_text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids.extend(system_ids)
        labels.extend([-100] * len(system_ids))

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            prefix = f"{IM_START}assistant\n"
            response = f"{content}{IM_END}\n"

            prefix_ids = tokenizer(
                prefix,
                add_special_tokens=False,
            )["input_ids"]

            response_ids = tokenizer(
                response,
                add_special_tokens=False,
            )["input_ids"]

            input_ids.extend(prefix_ids)
            labels.extend([-100] * len(prefix_ids))

            input_ids.extend(response_ids)
            labels.extend(response_ids)

        else:
            text = f"{IM_START}{role}\n{content}{IM_END}\n"

            ids = tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]

            input_ids.extend(ids)
            labels.extend([-100] * len(ids))

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

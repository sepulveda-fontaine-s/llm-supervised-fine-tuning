from collections import Counter

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "rajpurkar/squad_v2"
MAX_LENGTH = 1024

SYSTEM_MESSAGE = (
    "Answer the question using only the provided context. "
    "Return only the answer supported by the context. "
    "If the context does not contain the answer, respond exactly: "
    "I don't know."
)


def build_messages(example):
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


def audit_split(
    dataset,
    split_name,
    tokenizer,
):
    split = dataset[split_name]

    counts = Counter()
    lengths = []

    for example in split:
        if len(example["answers"]["text"]) == 0:
            counts["unanswerable"] += 1
        else:
            counts["answerable"] += 1

        messages = build_messages(example)

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        token_ids = tokenizer(
            formatted_text,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(token_ids))

    lengths = np.asarray(lengths)

    print(f"\n=== {split_name.upper()} ===")

    print("examples:", len(split))
    print(
        "answerable:",
        counts["answerable"],
    )
    print(
        "unanswerable:",
        counts["unanswerable"],
    )
    print(
        "unanswerable ratio:",
        counts["unanswerable"] / len(split),
    )

    print(
        "token length min:",
        int(lengths.min()),
    )
    print(
        "token length p50:",
        int(np.percentile(lengths, 50)),
    )
    print(
        "token length p90:",
        int(np.percentile(lengths, 90)),
    )
    print(
        "token length p95:",
        int(np.percentile(lengths, 95)),
    )
    print(
        "token length p99:",
        int(np.percentile(lengths, 99)),
    )
    print(
        "token length max:",
        int(lengths.max()),
    )

    overlength = int(
        (lengths > MAX_LENGTH).sum()
    )

    print(
        f"over {MAX_LENGTH} tokens:",
        overlength,
    )
    print(
        "overlength ratio:",
        overlength / len(split),
    )


def main():
    print("=== LOADING SQUAD V2 ===")

    dataset = load_dataset(
        DATASET_NAME
    )

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    audit_split(
        dataset,
        "train",
        tokenizer,
    )

    audit_split(
        dataset,
        "validation",
        tokenizer,
    )

    print("\n=== ANSWERABLE EXAMPLE ===")

    for example in dataset["train"]:
        if len(example["answers"]["text"]) > 0:
            print(build_messages(example))
            break

    print("\n=== UNANSWERABLE EXAMPLE ===")

    for example in dataset["train"]:
        if len(example["answers"]["text"]) == 0:
            print(build_messages(example))
            break


if __name__ == "__main__":
    main()
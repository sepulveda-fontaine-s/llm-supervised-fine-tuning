from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/processed/no_robots"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_from_disk(DATASET_PATH)

    print("=== TOKENIZER ===")
    print("model:", MODEL_NAME)
    print("vocab size:", len(tokenizer))
    print("bos token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("pad token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("unk token:", tokenizer.unk_token, tokenizer.unk_token_id)

    print("\n=== CHAT TEMPLATE ===")
    if tokenizer.chat_template:
        print(tokenizer.chat_template)
    else:
        print("NONE")

    example = dataset["train"][0]

    print("\n=== ORIGINAL MESSAGES ===")
    print(example["messages"])

    if tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        print("\n=== FORMATTED EXAMPLE ===")
        print(formatted)

        tokens = tokenizer(
            formatted,
            add_special_tokens=False,
        )

        print("\n=== TOKEN COUNT ===")
        print(len(tokens["input_ids"]))


if __name__ == "__main__":
    main()
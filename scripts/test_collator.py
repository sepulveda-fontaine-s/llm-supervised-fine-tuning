from datasets import load_from_disk
from transformers import AutoTokenizer

from llm_fine_tuning.collator import SFTDataCollator


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/tokenized/no_robots_qwen2_5_0_5b"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_from_disk(DATASET_PATH)
    train = dataset["train"]

    collator = SFTDataCollator(tokenizer=tokenizer)

    features = [
        train[0],
        train[1],
        train[2],
    ]

    batch = collator(features)

    print("=== BATCH SHAPES ===")
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    print("\n=== ORIGINAL LENGTHS ===")
    for index, feature in enumerate(features):
        print(index, len(feature["input_ids"]))

    print("\n=== PADDED LENGTH ===")
    print(batch["input_ids"].shape[1])

    print("\n=== LABEL PADDING CHECK ===")
    for index, feature in enumerate(features):
        original_length = len(feature["labels"])
        padded_labels = batch["labels"][index]

        padding = padded_labels[original_length:]

        print(
            f"example {index}:",
            "all padding labels == -100:",
            bool((padding == -100).all()) if len(padding) > 0 else True,
        )


if __name__ == "__main__":
    main()
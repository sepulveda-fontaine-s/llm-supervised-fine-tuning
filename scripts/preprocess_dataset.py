from pathlib import Path

import yaml
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from llm_fine_tuning.preprocessing import (
    build_sft_example,
    build_squad_messages,
)


CONFIG_PATH = Path("configs/training.yaml")
OUTPUT_PATH = Path(
    "data/tokenized/squad_v2_qwen2_5_0_5b"
)


def main() -> None:
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]
    dataset_path = config["data"]["path"]
    max_length = config["data"]["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    dataset = load_from_disk(
        dataset_path
    )

    prepared = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split = dataset[split_name]

        split = split.map(
            lambda example: build_sft_example(
                build_squad_messages(example),
                tokenizer,
            )
        )

        before = len(split)

        split = split.filter(
            lambda example:
            len(example["input_ids"])
            <= max_length
        )

        after = len(split)

        split = split.select_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        )

        prepared[split_name] = split

        print(
            f"=== {split_name.upper()} ==="
        )
        print("before:", before)
        print("after:", after)
        print("removed:", before - after)

    tokenized_dataset = DatasetDict(
        prepared
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenized_dataset.save_to_disk(
        OUTPUT_PATH
    )

    print("\n=== SAVED ===")
    print("path:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
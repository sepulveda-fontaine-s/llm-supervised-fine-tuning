from pathlib import Path

import yaml
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from llm_fine_tuning.preprocessing import (
    build_sft_example,
    build_squad_messages,
)


CONFIG_PATH = Path("configs/training.yaml")

DATASET_PATH = Path(
    "data/processed/squad_v2_intermediate_v3"
)

OUTPUT_PATH = Path(
    "data/tokenized/squad_v2_intermediate_v3_qwen2_5_0_5b"
)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. Refusing to overwrite it."
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]
    max_length = config["data"]["max_length"]

    print("model:", model_name)
    print("max_length:", max_length)
    print("input:", DATASET_PATH)
    print("output:", OUTPUT_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    dataset = load_from_disk(
        str(DATASET_PATH)
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
            len(example["input_ids"]) <= max_length
        )

        after = len(split)

        zero_supervised = sum(
            all(label == -100 for label in labels)
            for labels in split["labels"]
        )

        max_tokens = max(
            len(ids)
            for ids in split["input_ids"]
        )

        split = split.select_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        )

        prepared[split_name] = split

        print(
            f"\n=== {split_name.upper()} ==="
        )
        print("before:", before)
        print("after:", after)
        print("removed:", before - after)
        print("max tokens:", max_tokens)
        print(
            "zero-supervised examples:",
            zero_supervised,
        )

        assert zero_supervised == 0

    tokenized_dataset = DatasetDict(
        prepared
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenized_dataset.save_to_disk(
        str(OUTPUT_PATH)
    )

    print("\n=== SAVED ===")
    print("path:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

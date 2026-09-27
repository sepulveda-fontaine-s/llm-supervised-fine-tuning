import json
import random
from pathlib import Path

from datasets import load_from_disk


PREDICTIONS_PATH = Path("results/test_final_v3.json")
DATASET_PATH = Path("data/processed/squad_v2")
OUTPUT_PATH = Path("results/test_final_v3_qualitative_enriched.json")

SEED = 42
EXAMPLES_PER_CATEGORY = 5


def sample_examples(items, n, rng):
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def main():
    with PREDICTIONS_PATH.open("r", encoding="utf-8") as file:
        predictions_data = json.load(file)

    dataset = load_from_disk(str(DATASET_PATH))
    test = dataset["test"]

    metadata_by_id = {
        example["id"]: {
            "question": example["question"],
            "context": example["context"],
            "gold_answers": example["answers"]["text"],
        }
        for example in test
    }

    results = predictions_data["final_v3"]["predictions"]

    categories = {
        "answerable_exact_match": [],
        "answerable_partial_match": [],
        "answerable_false_abstention": [],
        "answerable_wrong_answer": [],
        "unanswerable_correct_abstention": [],
        "unanswerable_false_answer": [],
    }

    for item in results:
        if item["answerable"]:
            if item["exact_match"] == 1.0:
                category = "answerable_exact_match"
            elif item["abstained"]:
                category = "answerable_false_abstention"
            elif item["f1"] > 0.0:
                category = "answerable_partial_match"
            else:
                category = "answerable_wrong_answer"
        else:
            if item["abstained"]:
                category = "unanswerable_correct_abstention"
            else:
                category = "unanswerable_false_answer"

        metadata = metadata_by_id[item["id"]]

        categories[category].append(
            {
                "id": item["id"],
                "question": metadata["question"],
                "context": metadata["context"],
                "prediction": item["prediction"],
                "gold_answers": metadata["gold_answers"],
                "exact_match": item["exact_match"],
                "f1": item["f1"],
                "abstained": item["abstained"],
            }
        )

    rng = random.Random(SEED)

    output = {
        "source_predictions": str(PREDICTIONS_PATH),
        "source_dataset": str(DATASET_PATH),
        "seed": SEED,
        "examples_per_category": EXAMPLES_PER_CATEGORY,
        "counts": {
            key: len(items)
            for key, items in categories.items()
        },
        "examples": {
            key: sample_examples(items, EXAMPLES_PER_CATEGORY, rng)
            for key, items in categories.items()
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=== ENRICHED QUALITATIVE ANALYSIS ===")
    for key, items in categories.items():
        print(f"{key}: {len(items)}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

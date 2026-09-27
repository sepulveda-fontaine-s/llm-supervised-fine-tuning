import json
import random
from pathlib import Path


INPUT_PATH = Path("results/test_final_v3.json")
OUTPUT_PATH = Path("results/test_final_v3_qualitative_analysis.json")

SEED = 42
EXAMPLES_PER_CATEGORY = 5


def sample_examples(items, n, rng):
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def compact_example(item):
    return {
        "id": item["id"],
        "prediction": item["prediction"],
        "gold_answers": item["gold_answers"],
        "exact_match": item["exact_match"],
        "f1": item["f1"],
        "abstained": item["abstained"],
    }


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    results = data["final_v3"]["predictions"]

    answerable_correct = []
    answerable_partial = []
    answerable_false_abstention = []
    answerable_wrong_answer = []
    unanswerable_correct_abstention = []
    unanswerable_false_answer = []

    for item in results:
        if item["answerable"]:
            if item["exact_match"] == 1.0:
                answerable_correct.append(item)
            elif item["abstained"]:
                answerable_false_abstention.append(item)
            elif item["f1"] > 0.0:
                answerable_partial.append(item)
            else:
                answerable_wrong_answer.append(item)
        else:
            if item["abstained"]:
                unanswerable_correct_abstention.append(item)
            else:
                unanswerable_false_answer.append(item)

    categories = {
        "answerable_exact_match": answerable_correct,
        "answerable_partial_match": answerable_partial,
        "answerable_false_abstention": answerable_false_abstention,
        "answerable_wrong_answer": answerable_wrong_answer,
        "unanswerable_correct_abstention": unanswerable_correct_abstention,
        "unanswerable_false_answer": unanswerable_false_answer,
    }

    rng = random.Random(SEED)

    output = {
        "source": str(INPUT_PATH),
        "seed": SEED,
        "examples_per_category": EXAMPLES_PER_CATEGORY,
        "counts": {
            key: len(items)
            for key, items in categories.items()
        },
        "examples": {},
    }

    for key, items in categories.items():
        sampled = sample_examples(
            items,
            EXAMPLES_PER_CATEGORY,
            rng,
        )

        output["examples"][key] = [
            compact_example(item)
            for item in sampled
        ]

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

    print("=== QUALITATIVE ANALYSIS COUNTS ===")

    for key, items in categories.items():
        print(f"{key}: {len(items)}")

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

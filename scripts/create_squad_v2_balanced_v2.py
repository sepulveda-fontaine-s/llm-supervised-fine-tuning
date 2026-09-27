from pathlib import Path
import json

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


SEED = 42

V1_PATH = Path("data/processed/squad_v2")
V2_PATH = Path("data/processed/squad_v2_balanced_v2")
AUDIT_PATH = Path("results/squad_v2_balanced_v2_audit.json")


def is_answerable(answers):
    texts = answers["text"]
    return len(texts) > 0 and any(text.strip() for text in texts)


def main():
    if V2_PATH.exists():
        raise FileExistsError(
            f"{V2_PATH} already exists. Refusing to overwrite it."
        )

    print("Loading V1 processed dataset...")
    v1 = load_from_disk(str(V1_PATH))

    train_v1 = v1["train"]

    answerable_idx = [
        i for i, answers in enumerate(train_v1["answers"])
        if is_answerable(answers)
    ]

    unanswerable_idx = [
        i for i, answers in enumerate(train_v1["answers"])
        if not is_answerable(answers)
    ]

    print(f"V1 train total: {len(train_v1)}")
    print(f"V1 answerable: {len(answerable_idx)}")
    print(f"V1 unanswerable: {len(unanswerable_idx)}")

    assert len(answerable_idx) == 20_000
    assert len(unanswerable_idx) == 10_000

    # Keep all 10k V1 unanswerable examples.
    v1_unanswerable = train_v1.select(unanswerable_idx)

    # Keep 15k of the original V1 answerable examples.
    v1_answerable = (
        train_v1
        .select(answerable_idx)
        .shuffle(seed=SEED)
        .select(range(15_000))
    )

    print("Loading original SQuAD v2 train...")
    raw_train = load_dataset(
        "rajpurkar/squad_v2",
        split="train",
    )

    # Avoid reusing examples already present in V1.
    existing_ids = set(train_v1["id"])

    candidate_extra_unanswerable_idx = [
        i
        for i, (example_id, answers)
        in enumerate(zip(raw_train["id"], raw_train["answers"]))
        if example_id not in existing_ids
        and not is_answerable(answers)
    ]

    print(
        "Available new unanswerable examples:",
        len(candidate_extra_unanswerable_idx),
    )

    extra_unanswerable = (
        raw_train
        .select(candidate_extra_unanswerable_idx)
        .shuffle(seed=SEED)
        .select(range(5_000))
    )

    # Final train: 15k answerable + 15k unanswerable.
    train_v2 = concatenate_datasets(
        [
            v1_answerable,
            v1_unanswerable,
            extra_unanswerable,
        ]
    ).shuffle(seed=SEED)

    answerable_v2 = sum(
        is_answerable(answers)
        for answers in train_v2["answers"]
    )
    unanswerable_v2 = len(train_v2) - answerable_v2

    print("\n=== V2 AUDIT ===")
    print(f"train total: {len(train_v2)}")
    print(f"answerable: {answerable_v2}")
    print(f"unanswerable: {unanswerable_v2}")

    assert len(train_v2) == 30_000
    assert answerable_v2 == 15_000
    assert unanswerable_v2 == 15_000

    # Validation and test are copied untouched.
    v2 = DatasetDict(
        {
            "train": train_v2,
            "validation": v1["validation"],
            "test": v1["test"],
        }
    )

    assert v2["validation"]["id"] == v1["validation"]["id"]
    assert v2["test"]["id"] == v1["test"]["id"]

    print(f"validation unchanged: {len(v2['validation'])}")
    print(f"test unchanged: {len(v2['test'])}")

    V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    v2.save_to_disk(str(V2_PATH))

    audit = {
        "seed": SEED,
        "train_total": len(train_v2),
        "train_answerable": answerable_v2,
        "train_unanswerable": unanswerable_v2,
        "validation_total": len(v2["validation"]),
        "test_total": len(v2["test"]),
        "validation_unchanged": True,
        "test_unchanged": True,
        "v1_unanswerable_retained": 10_000,
        "v1_answerable_retained": 15_000,
        "new_unanswerable_added": 5_000,
    }

    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"\nSaved dataset to: {V2_PATH}")
    print(f"Saved audit to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
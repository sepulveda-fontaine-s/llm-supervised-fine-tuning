from pathlib import Path
import json

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


SEED = 42

V1_PATH = Path("data/processed/squad_v2")
V3_PATH = Path("data/processed/squad_v2_intermediate_v3")
AUDIT_PATH = Path("results/squad_v2_intermediate_v3_audit.json")


def is_answerable(answers):
    texts = answers["text"]
    return len(texts) > 0 and any(text.strip() for text in texts)


def main():
    if V3_PATH.exists():
        raise FileExistsError(
            f"{V3_PATH} already exists. Refusing to overwrite it."
        )

    print("Loading V1 processed dataset...")
    v1 = load_from_disk(str(V1_PATH))
    train_v1 = v1["train"]

    answerable_idx = [
        i
        for i, answers in enumerate(train_v1["answers"])
        if is_answerable(answers)
    ]

    unanswerable_idx = [
        i
        for i, answers in enumerate(train_v1["answers"])
        if not is_answerable(answers)
    ]

    print(f"V1 train total: {len(train_v1)}")
    print(f"V1 answerable: {len(answerable_idx)}")
    print(f"V1 unanswerable: {len(unanswerable_idx)}")

    assert len(answerable_idx) == 20_000
    assert len(unanswerable_idx) == 10_000

    # Keep all 10k V1 unanswerable examples.
    v1_unanswerable = train_v1.select(unanswerable_idx)

    # Use the same deterministic shuffle as V2, but retain 18k
    # instead of 15k. Therefore V2's 15k retained answerables
    # are nested inside this V3 selection.
    v1_answerable = (
        train_v1
        .select(answerable_idx)
        .shuffle(seed=SEED)
        .select(range(18_000))
    )

    print("Loading original SQuAD v2 train...")
    raw_train = load_dataset(
        "rajpurkar/squad_v2",
        split="train",
    )

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

    # Same deterministic candidate shuffle as V2, but take only
    # the first 2k instead of 5k. Thus V3's new unanswerables are
    # a subset of the V2 additions.
    extra_unanswerable = (
        raw_train
        .select(candidate_extra_unanswerable_idx)
        .shuffle(seed=SEED)
        .select(range(2_000))
    )

    # Final train: 18k answerable + 12k unanswerable.
    train_v3 = concatenate_datasets(
        [
            v1_answerable,
            v1_unanswerable,
            extra_unanswerable,
        ]
    ).shuffle(seed=SEED)

    answerable_v3 = sum(
        is_answerable(answers)
        for answers in train_v3["answers"]
    )
    unanswerable_v3 = len(train_v3) - answerable_v3

    print("\n=== V3 AUDIT ===")
    print(f"train total: {len(train_v3)}")
    print(f"answerable: {answerable_v3}")
    print(f"unanswerable: {unanswerable_v3}")

    assert len(train_v3) == 30_000
    assert answerable_v3 == 18_000
    assert unanswerable_v3 == 12_000

    v3 = DatasetDict(
        {
            "train": train_v3,
            "validation": v1["validation"],
            "test": v1["test"],
        }
    )

    assert v3["validation"]["id"] == v1["validation"]["id"]
    assert v3["test"]["id"] == v1["test"]["id"]

    print(f"validation unchanged: {len(v3['validation'])}")
    print(f"test unchanged: {len(v3['test'])}")

    V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    v3.save_to_disk(str(V3_PATH))

    audit = {
        "seed": SEED,
        "train_total": len(train_v3),
        "train_answerable": answerable_v3,
        "train_unanswerable": unanswerable_v3,
        "unanswerable_ratio": unanswerable_v3 / len(train_v3),
        "validation_total": len(v3["validation"]),
        "test_total": len(v3["test"]),
        "validation_unchanged": True,
        "test_unchanged": True,
        "v1_unanswerable_retained": 10_000,
        "v1_answerable_retained": 18_000,
        "new_unanswerable_added": 2_000,
        "selection_note": (
            "Uses the same seed and deterministic selection procedure as V2. "
            "V3 retained answerables contain the V2 retained answerables, and "
            "V3 added unanswerables are a subset of the V2 additions."
        ),
    }

    with AUDIT_PATH.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    print(f"\nSaved dataset to: {V3_PATH}")
    print(f"Saved audit to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()

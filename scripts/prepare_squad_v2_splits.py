import random
from collections import defaultdict

from datasets import DatasetDict, load_dataset


DATASET_NAME = "rajpurkar/squad_v2"
OUTPUT_PATH = "data/processed/squad_v2"

SEED = 42

TRAIN_ANSWERABLE = 20_000
TRAIN_UNANSWERABLE = 10_000


def is_answerable(example):
    return len(example["answers"]["text"]) > 0


def count_answerability(dataset):
    answerable = sum(
        is_answerable(example)
        for example in dataset
    )

    unanswerable = len(dataset) - answerable

    return answerable, unanswerable


def build_train_subset(train_dataset):
    answerable_indices = []
    unanswerable_indices = []

    for index, example in enumerate(
        train_dataset
    ):
        if is_answerable(example):
            answerable_indices.append(index)
        else:
            unanswerable_indices.append(index)

    rng = random.Random(SEED)

    rng.shuffle(answerable_indices)
    rng.shuffle(unanswerable_indices)

    selected_indices = (
        answerable_indices[
            :TRAIN_ANSWERABLE
        ]
        + unanswerable_indices[
            :TRAIN_UNANSWERABLE
        ]
    )

    rng.shuffle(selected_indices)

    return train_dataset.select(
        selected_indices
    )


def split_validation_by_context(
    validation_dataset,
):
    context_groups = defaultdict(list)

    for index, example in enumerate(
        validation_dataset
    ):
        context_groups[
            example["context"]
        ].append(index)

    contexts = list(
        context_groups.keys()
    )

    rng = random.Random(SEED)
    rng.shuffle(contexts)

    validation_indices = []
    test_indices = []

    for context in contexts:
        indices = context_groups[context]

        if (
            len(validation_indices)
            <= len(test_indices)
        ):
            validation_indices.extend(
                indices
            )
        else:
            test_indices.extend(
                indices
            )

    validation_split = (
        validation_dataset.select(
            validation_indices
        )
    )

    test_split = (
        validation_dataset.select(
            test_indices
        )
    )

    return validation_split, test_split


def main():
    print("=== LOADING SQUAD V2 ===")

    dataset = load_dataset(
        DATASET_NAME
    )

    print("\n=== BUILDING TRAIN SUBSET ===")

    train = build_train_subset(
        dataset["train"]
    )

    print(
        "\n=== SPLITTING OFFICIAL VALIDATION "
        "BY CONTEXT ==="
    )

    validation, test = (
        split_validation_by_context(
            dataset["validation"]
        )
    )

    train_answerable, train_unanswerable = (
        count_answerability(train)
    )

    val_answerable, val_unanswerable = (
        count_answerability(validation)
    )

    test_answerable, test_unanswerable = (
        count_answerability(test)
    )

    print("\n=== FINAL SPLITS ===")

    print(
        "train:",
        len(train),
        "answerable:",
        train_answerable,
        "unanswerable:",
        train_unanswerable,
    )

    print(
        "validation:",
        len(validation),
        "answerable:",
        val_answerable,
        "unanswerable:",
        val_unanswerable,
    )

    print(
        "test:",
        len(test),
        "answerable:",
        test_answerable,
        "unanswerable:",
        test_unanswerable,
    )

    validation_contexts = set(
        validation["context"]
    )

    test_contexts = set(
        test["context"]
    )

    overlap = (
        validation_contexts
        & test_contexts
    )

    print(
        "\nvalidation/test context overlap:",
        len(overlap),
    )

    assert len(train) == 30_000
    assert train_answerable == 20_000
    assert train_unanswerable == 10_000
    assert len(overlap) == 0

    final_dataset = DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": test,
        }
    )

    final_dataset.save_to_disk(
        OUTPUT_PATH
    )

    print(
        "\nsaved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
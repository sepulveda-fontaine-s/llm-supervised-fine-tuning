import csv
from pathlib import Path

import optuna
import yaml


CONFIG_PATH = "configs/training.yaml"
OUTPUT_PATH = Path(
    "results/optuna_squad_v1_trials.csv"
)


def main():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    tuning = config["tuning"]

    study = optuna.load_study(
        study_name=tuning["study_name"],
        storage=tuning["storage"],
    )

    rows = []

    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "learning_rate":
                trial.params.get("learning_rate"),
            "weight_decay":
                trial.params.get("weight_decay"),
            "warmup_ratio":
                trial.params.get("warmup_ratio"),
            "scheduler":
                trial.params.get("scheduler"),
            "pruned_at_step":
                trial.user_attrs.get(
                    "pruned_at_step"
                ),
            "training_time_seconds":
                trial.user_attrs.get(
                    "training_time_seconds"
                ),
            "peak_vram_gib":
                trial.user_attrs.get(
                    "peak_vram_gib"
                ),
        }

        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["value"] is None,
            (
                row["value"]
                if row["value"] is not None
                else float("inf")
            ),
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=== OPTUNA TRIALS ===")

    for row in rows:
        print(
            f"trial={row['trial']} "
            f"state={row['state']} "
            f"value={row['value']} "
            f"lr={row['learning_rate']} "
            f"wd={row['weight_decay']} "
            f"warmup={row['warmup_ratio']} "
            f"scheduler={row['scheduler']} "
            f"pruned_at={row['pruned_at_step']}"
        )

    print()
    print("saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
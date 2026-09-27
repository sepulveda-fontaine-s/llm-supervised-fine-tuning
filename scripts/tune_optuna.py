import gc
import time
import argparse
import optuna
import torch
import yaml
import math
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/squad_v2_qwen2_5_0_5b"


def evaluate(model, dataloader, device):
    """Mean cross-entropy over supervised assistant tokens."""

    model.eval()

    total_loss = 0.0
    total_supervised_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(**batch)

            supervised_tokens = (
                batch["labels"] != -100
            ).sum().item()

            total_loss += (
                outputs.loss.item()
                * supervised_tokens
            )

            total_supervised_tokens += supervised_tokens

    model.train()

    return total_loss / total_supervised_tokens


def optimizer_update(
    model,
    optimizer,
    scheduler,
    accumulated_tokens,
    max_grad_norm,
):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_tokens)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return float(grad_norm)


def create_scheduler(
    name,
    optimizer,
    warmup_steps,
    total_steps,
):
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    raise ValueError(
        f"Unsupported scheduler: {name}"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=None,
        help="Maximum number of new trials to run in this invocation.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    training = config["training"]
    tuning = config["tuning"]

    seed = training["seed"]
    batch_size = training["per_device_batch_size"]
    accumulation_steps = (
        training["gradient_accumulation_steps"]
    )

    max_grad_norm = config["pilot"]["max_grad_norm"]

    max_optimizer_steps = tuning["max_optimizer_steps"]
    evaluation_interval = tuning["evaluation_interval"]
    pruning_start_step = tuning["pruning_start_step"]

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    dataset = load_from_disk(DATASET_PATH)

    validation_dataset = (
        dataset["validation"]
        .shuffle(seed=seed)
        .select(
            range(tuning["validation_examples"])
        )
    )

    collator = SFTDataCollator(
        tokenizer=tokenizer
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    micro_batches_per_epoch = math.ceil(
        len(dataset["train"]) / batch_size
    )

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / accumulation_steps
    )

    full_training_steps = (
        optimizer_steps_per_epoch
        * training["epochs"]
    )

    print("full training optimizer steps:", full_training_steps)

    def objective(trial):
        # Every trial begins from the same RNG state so that
        # hyperparameters are compared under the same conditions.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        learning_rate = trial.suggest_float(
            "learning_rate",
            tuning["learning_rate_min"],
            tuning["learning_rate_max"],
            log=True,
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            tuning["weight_decay_min"],
            tuning["weight_decay_max"],
        )

        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            tuning["warmup_ratio_min"],
            tuning["warmup_ratio_max"],
        )

        scheduler_name = trial.suggest_categorical(
            "scheduler",
            tuning["schedulers"],
        )

        warmup_steps = int(
            full_training_steps * warmup_ratio
        )

        # Explicit generator gives every trial the same
        # shuffled training order.
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"]
        )

        model.to(device)
        model.train()
        model.config.use_cache = False

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = create_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=full_training_steps,
        )

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats()

        optimizer_step = 0
        micro_steps_since_update = 0
        accumulated_tokens = 0

        last_validation_loss = None

        trial_start = time.perf_counter()

        try:
            for batch_index, batch in enumerate(
                train_loader,
                start=1,
            ):
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss: {loss.item()}"
                    )

                supervised_tokens = (
                    batch["labels"] != -100
                ).sum().item()

                if supervised_tokens == 0:
                    raise RuntimeError(
                        "Batch contains zero supervised tokens."
                    )

                (
                    loss * supervised_tokens
                ).backward()

                accumulated_tokens += supervised_tokens
                micro_steps_since_update += 1

                is_accumulation_boundary = (
                    micro_steps_since_update
                    == accumulation_steps
                )

                is_last_batch = (
                    batch_index == len(train_loader)
                )

                if (
                    is_accumulation_boundary
                    or is_last_batch
                ):
                    optimizer_update(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        accumulated_tokens=accumulated_tokens,
                        max_grad_norm=max_grad_norm,
                    )

                    optimizer_step += 1
                    accumulated_tokens = 0
                    micro_steps_since_update = 0

                    if (
                        optimizer_step
                        % evaluation_interval
                        == 0
                    ):
                        validation_loss = evaluate(
                            model,
                            validation_loader,
                            device,
                        )

                        last_validation_loss = (
                            validation_loss
                        )

                        trial.report(
                            validation_loss,
                            step=optimizer_step,
                        )

                        print(
                            f"trial={trial.number} "
                            f"step={optimizer_step} "
                            f"val_loss={validation_loss:.6f}"
                        )

                        if (
                            optimizer_step
                            >= pruning_start_step
                            and trial.should_prune()
                        ):
                            trial.set_user_attr(
                                "pruned_at_step",
                                optimizer_step,
                            )

                            raise optuna.TrialPruned()

                    if (
                        optimizer_step
                        >= max_optimizer_steps
                    ):
                        break

   

            elapsed = (
                time.perf_counter()
                - trial_start
            )

            peak_vram_gib = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            trial.set_user_attr(
                "training_time_seconds",
                elapsed,
            )

            trial.set_user_attr(
                "peak_vram_gib",
                peak_vram_gib,
            )

            if last_validation_loss is None:
                raise RuntimeError(
                    "Trial ended without validation."
                )

            return last_validation_loss

        finally:
            del model
            del optimizer
            del scheduler
            del train_loader

            gc.collect()
            torch.cuda.empty_cache()

    study_summaries = optuna.study.get_all_study_summaries(
        storage=tuning["storage"]
    )

    existing_trial_count = next(
        (
            summary.n_trials
            for summary in study_summaries
            if summary.study_name == tuning["study_name"]
        ),
        0,
    )

    effective_sampler_seed = (
        tuning["sampler_seed"]
        + existing_trial_count
    )

    print(
        "effective sampler seed:",
        effective_sampler_seed,
    )

    sampler = optuna.samplers.TPESampler(
        seed=effective_sampler_seed
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tuning["startup_trials"],
        n_warmup_steps=pruning_start_step,
        interval_steps=evaluation_interval,
    )

    study = optuna.create_study(
        study_name=tuning["study_name"],
        storage=tuning["storage"],
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # n_trials in the YAML means TARGET total number of
    # completed/pruned trials, rather than "add another 16"
    # every time this script is resumed.
    finished_trials = [
        trial
        for trial in study.trials
        if trial.state
        in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        )
    ]

    remaining_trials = max(
        0,
        tuning["n_trials"]
        - len(finished_trials),
    )

    if args.max_new_trials is not None:
        remaining_trials = min(
            remaining_trials,
            args.max_new_trials,
        )
    print("=== OPTUNA STUDY ===")
    print("study name:", tuning["study_name"])
    print("finished trials:", len(finished_trials))
    print("target trials:", tuning["n_trials"])
    print("remaining trials:", remaining_trials)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=True,
        )

    print("\n=== OPTUNA RESULT ===")
    print("best trial:", study.best_trial.number)
    print("best validation loss:", study.best_value)
    print("best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
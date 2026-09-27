import csv
import gc
import math
import time
from pathlib import Path

import torch
import yaml
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
DATASET_PATH = "data/tokenized/no_robots_qwen2_5_0_5b"
OUTPUT_PATH = Path("results/hparam_confirmation.csv")


def evaluate(model, dataloader, device):
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


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    training = config["training"]
    confirmation = config["confirmation"]

    seed = training["seed"]
    batch_size = training["per_device_batch_size"]
    accumulation_steps = (
        training["gradient_accumulation_steps"]
    )

    max_grad_norm = config["pilot"]["max_grad_norm"]

    confirmation_steps = confirmation["optimizer_steps"]

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    dataset = load_from_disk(DATASET_PATH)

    validation_dataset = dataset["validation"]

    if confirmation["validation_examples"] is not None:
        validation_dataset = validation_dataset.select(
            range(confirmation["validation_examples"])
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

    print("=== CONFIRMATION PLAN ===")
    print(
        "validation examples:",
        len(validation_dataset),
    )
    print(
        "confirmation optimizer steps:",
        confirmation_steps,
    )
    print(
        "full training optimizer steps:",
        full_training_steps,
    )
    print(
        "candidates:",
        len(confirmation["candidates"]),
    )

    results = []

    for candidate in confirmation["candidates"]:
        print()
        print(
            f"=== CANDIDATE {candidate['name']} ==="
        )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
            lr=candidate["learning_rate"],
            weight_decay=candidate["weight_decay"],
        )

        warmup_steps = int(
            full_training_steps
            * candidate["warmup_ratio"]
        )

        scheduler = create_scheduler(
            name=candidate["scheduler"],
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=full_training_steps,
        )

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        optimizer_step = 0
        micro_steps_since_update = 0
        accumulated_tokens = 0

        start_time = time.perf_counter()

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
                    >= confirmation_steps
                ):
                    break

        validation_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        peak_vram_gib = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        result = {
            "name": candidate["name"],
            "validation_loss": validation_loss,
            "learning_rate":
                candidate["learning_rate"],
            "weight_decay":
                candidate["weight_decay"],
            "warmup_ratio":
                candidate["warmup_ratio"],
            "scheduler":
                candidate["scheduler"],
            "warmup_steps": warmup_steps,
            "optimizer_steps": optimizer_step,
            "training_time_seconds": elapsed,
            "peak_vram_gib": peak_vram_gib,
        }

        results.append(result)

        print(
            "validation loss:",
            validation_loss,
        )
        print(
            "optimizer steps:",
            optimizer_step,
        )
        print(
            "warmup steps:",
            warmup_steps,
        )
        print(
            "training time seconds:",
            elapsed,
        )
        print(
            "peak VRAM GiB:",
            peak_vram_gib,
        )

        del model
        del optimizer
        del scheduler
        del train_loader

        gc.collect()
        torch.cuda.empty_cache()

    results.sort(
        key=lambda item: item["validation_loss"]
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
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=== CONFIRMATION RANKING ===")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. {result['name']} "
            f"validation_loss="
            f"{result['validation_loss']}"
        )

    print()
    print(
        "CSV saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
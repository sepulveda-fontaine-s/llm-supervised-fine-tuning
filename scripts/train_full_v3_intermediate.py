import math
import shutil
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/squad_v2_intermediate_v3_qwen2_5_0_5b"

CHECKPOINT_DIR = Path(
    "results/checkpoints/squad_v2_intermediate_v3"
)
MLFLOW_DIR = Path("mlruns")


def evaluate(model, dataloader, device):
    """Return mean cross-entropy per supervised assistant token."""
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

            if supervised_tokens == 0:
                raise RuntimeError(
                    "Batch contains zero supervised tokens."
                )

            total_loss += (
                outputs.loss.item() * supervised_tokens
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
    """
    Normalize accumulated gradients by the number of supervised tokens,
    then perform one optimizer update.
    """

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
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    # Record the actual V3 tokenized dataset used by this run
    # in saved checkpoint metadata.
    config["data"]["path"] = DATASET_PATH

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    seed = config["training"]["seed"]
    batch_size = config["training"]["per_device_batch_size"]
    accumulation_steps = (
        config["training"]["gradient_accumulation_steps"]
    )
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    epochs = config["training"]["epochs"]
    warmup_ratio = config["training"]["warmup_ratio"]
    scheduler_name = config["training"]["scheduler"]
    max_grad_norm = config["pilot"]["max_grad_norm"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    dataset = load_from_disk(
        DATASET_PATH
    )

    collator = SFTDataCollator(
        tokenizer=tokenizer
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    micro_batches_per_epoch = len(
        train_loader
    )

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        * warmup_ratio
    )

    print("=== TRAINING PLAN ===")
    print(
        "train examples:",
        len(dataset["train"]),
    )
    print(
        "validation examples:",
        len(dataset["validation"]),
    )
    print(
        "micro-batches per epoch:",
        micro_batches_per_epoch,
    )
    print(
        "optimizer steps per epoch:",
        optimizer_steps_per_epoch,
    )
    print("epochs:", epochs)
    print(
        "total optimizer steps:",
        total_optimizer_steps,
    )
    print(
        "warmup steps:",
        warmup_steps,
    )
    print(
        "learning rate:",
        learning_rate,
    )
    print(
        "weight decay:",
        weight_decay,
    )
    print(
        "scheduler:",
        scheduler_name,
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
        total_steps=total_optimizer_steps,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MLFLOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        MLFLOW_DIR.resolve().as_uri()
    )

    mlflow.set_experiment(
        "qwen_squad_v2_full_sft_training_v3_intermediate"
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.reset_peak_memory_stats()

    global_optimizer_step = 0
    best_validation_loss = float("inf")
    best_epoch = None

    start_time = time.perf_counter()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": config["model"]["name"],
                "dataset_variant": "intermediate_60_40",
                "dataset_path": DATASET_PATH,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "gradient_accumulation_steps":
                    accumulation_steps,
                "effective_batch_size":
                    batch_size
                    * accumulation_steps,
                "epochs": epochs,
                "scheduler": scheduler_name,
                "precision":
                    "bf16_autocast",
                "max_length":
                    config["data"]["max_length"],
                "warmup_ratio":
                    warmup_ratio,
                "warmup_steps":
                    warmup_steps,
                "total_optimizer_steps":
                    total_optimizer_steps,
                "seed": seed,
            }
        )

        for epoch in range(
            1,
            epochs + 1,
        ):
            epoch_loss_sum = 0.0
            epoch_supervised_tokens = 0

            accumulated_tokens = 0
            micro_steps_since_update = 0

            for batch_index, batch in enumerate(
                train_loader,
                start=1,
            ):
                batch = {
                    key: value.to(device)
                    for key, value
                    in batch.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite loss: "
                        f"{loss.item()}"
                    )

                supervised_tokens = (
                    batch["labels"] != -100
                ).sum().item()

                if supervised_tokens == 0:
                    raise RuntimeError(
                        "Batch contains zero "
                        "supervised tokens."
                    )

                (
                    loss
                    * supervised_tokens
                ).backward()

                epoch_loss_sum += (
                    loss.item()
                    * supervised_tokens
                )

                epoch_supervised_tokens += (
                    supervised_tokens
                )

                accumulated_tokens += (
                    supervised_tokens
                )

                micro_steps_since_update += 1

                is_accumulation_boundary = (
                    micro_steps_since_update
                    == accumulation_steps
                )

                is_last_batch = (
                    batch_index
                    == micro_batches_per_epoch
                )

                if (
                    is_accumulation_boundary
                    or is_last_batch
                ):
                    grad_norm = optimizer_update(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        accumulated_tokens=(
                            accumulated_tokens
                        ),
                        max_grad_norm=(
                            max_grad_norm
                        ),
                    )

                    global_optimizer_step += 1

                    mlflow.log_metrics(
                        {
                            "learning_rate":
                                scheduler
                                .get_last_lr()[0],
                            "gradient_norm":
                                grad_norm,
                        },
                        step=global_optimizer_step,
                    )

                    accumulated_tokens = 0
                    micro_steps_since_update = 0

            train_loss = (
                epoch_loss_sum
                / epoch_supervised_tokens
            )

            validation_loss = evaluate(
                model,
                validation_loader,
                device,
            )

            mlflow.log_metrics(
                {
                    "epoch_train_loss":
                        train_loss,
                    "epoch_validation_loss":
                        validation_loss,
                },
                step=epoch,
            )

            print(
                f"\n=== EPOCH {epoch} ==="
            )
            print(
                "train loss:",
                train_loss,
            )
            print(
                "validation loss:",
                validation_loss,
            )

            if (
                validation_loss
                < best_validation_loss
            ):
                best_validation_loss = (
                    validation_loss
                )
                best_epoch = epoch

                best_dir = (
                    CHECKPOINT_DIR
                    / "best_model"
                )

                model.save_pretrained(
                    best_dir,
                    safe_serialization=True,
                )

                tokenizer.save_pretrained(
                    best_dir
                )

                with open(
                    best_dir
                    / "training_config.yaml",
                    "w",
                    encoding="utf-8",
                ) as file:
                    yaml.safe_dump(
                        config,
                        file,
                        sort_keys=False,
                    )

                print(
                    "new best checkpoint:",
                    best_dir,
                )

            last_dir = (
                CHECKPOINT_DIR
                / "last_checkpoint"
            )

            if last_dir.exists():
                shutil.rmtree(
                    last_dir
                )

            last_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                last_dir,
                safe_serialization=True,
            )

            tokenizer.save_pretrained(
                last_dir
            )

            torch.save(
                {
                    "epoch": epoch,
                    "global_optimizer_step":
                        global_optimizer_step,
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "scheduler_state_dict":
                        scheduler.state_dict(),
                    "best_validation_loss":
                        best_validation_loss,
                    "best_epoch":
                        best_epoch,
                    "train_generator_state":
                        train_generator
                        .get_state(),
                    "torch_rng_state":
                        torch.get_rng_state(),
                    "cuda_rng_state_all":
                        torch.cuda
                        .get_rng_state_all(),
                },
                last_dir
                / "training_state.pt",
            )

            with open(
                last_dir
                / "training_config.yaml",
                "w",
                encoding="utf-8",
            ) as file:
                yaml.safe_dump(
                    config,
                    file,
                    sort_keys=False,
                )

            print(
                "latest recovery checkpoint:",
                last_dir,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        peak_vram_gib = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        mlflow.log_metrics(
            {
                "best_validation_loss":
                    best_validation_loss,
                "best_epoch":
                    best_epoch,
                "training_time_seconds":
                    elapsed,
                "peak_vram_gib":
                    peak_vram_gib,
            }
        )

        print(
            "\n=== TRAINING COMPLETE ==="
        )
        print(
            "best epoch:",
            best_epoch,
        )
        print(
            "best validation loss:",
            best_validation_loss,
        )
        print(
            "training time seconds:",
            elapsed,
        )
        print(
            "peak VRAM GiB:",
            peak_vram_gib,
        )


if __name__ == "__main__":
    main()

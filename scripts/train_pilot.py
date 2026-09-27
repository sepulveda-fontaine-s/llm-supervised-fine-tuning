from pathlib import Path
import argparse
import time

import mlflow
import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/squad_v2_qwen2_5_0_5b"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        required=True,
    )

    return parser.parse_args()

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
                outputs.loss.item() * supervised_tokens
            )
            total_supervised_tokens += supervised_tokens

    model.train()

    return total_loss / total_supervised_tokens



def main():
    args = parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["training"]["seed"]
    batch_size = config["training"]["per_device_batch_size"]
    accumulation_steps = config["training"]["gradient_accumulation_steps"]

    pilot = config["pilot"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    dataset = load_from_disk(DATASET_PATH)

    collator = SFTDataCollator(tokenizer=tokenizer)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=train_generator,
    )

    validation_subset = (
        dataset["validation"]
        .shuffle(seed=seed)
        .select(
            range(pilot["validation_examples"])
        )
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Keep master model weights in FP32.
    # BF16 is used during forward computation via autocast.
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"]
    )

    model.to(device)
    model.train()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=pilot["warmup_steps"],
        num_training_steps=pilot["max_optimizer_steps"],
    )

    tracking_dir = Path("mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())

    mlflow.set_experiment("qwen_squad_v2_full_sft_lr_sweep")

    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    micro_step = 0
    running_loss_sum = 0.0
    running_supervised_tokens = 0

    with mlflow.start_run():
        mlflow.log_params(
            {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "precision": "bf16_autocast",
                "max_length": config["data"]["max_length"],
                "seed": seed,
            }
        )

        while optimizer_step < pilot["max_optimizer_steps"]:
            for batch in train_loader:
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

                supervised_tokens = (batch["labels"] != -100).sum().item()

                # Convert the mean token loss back into a summed token loss.
                (loss * supervised_tokens).backward()

                running_loss_sum += loss.item() * supervised_tokens
                running_supervised_tokens += supervised_tokens

                micro_step += 1

                if micro_step % accumulation_steps == 0:

                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(running_supervised_tokens)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        pilot["max_grad_norm"],
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1

                    mean_train_loss = (running_loss_sum / running_supervised_tokens)

                    running_loss_sum = 0.0
                    running_supervised_tokens = 0

                    mlflow.log_metrics(
                        {
                            "train_loss": mean_train_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "gradient_norm": float(grad_norm),
                        },
                        step=optimizer_step,
                    )

                    print(
                        f"step={optimizer_step} "
                        f"loss={mean_train_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                    if optimizer_step >= pilot["max_optimizer_steps"]:
                        break

        validation_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        elapsed = time.perf_counter() - start_time

        peak_vram_gib = (
            torch.cuda.max_memory_allocated()
            / (1024**3)
        )

        mlflow.log_metrics(
            {
                "validation_loss": validation_loss,
                "training_time_seconds": elapsed,
                "peak_vram_gib": peak_vram_gib,
            }
        )

        print("\n=== PILOT RESULT ===")
        print("learning rate:", args.learning_rate)
        print("validation loss:", validation_loss)
        print("training time seconds:", elapsed)
        print("peak VRAM GiB:", peak_vram_gib)


if __name__ == "__main__":
    main()
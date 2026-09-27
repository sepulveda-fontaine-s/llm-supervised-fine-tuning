import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/squad_v2_qwen2_5_0_5b"

def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    dataset = load_from_disk(DATASET_PATH)

    seed = config["training"]["seed"]

    validation_subset = (
        dataset["validation"]
        .shuffle(seed=seed)
        .select(
            range(config["pilot"]["validation_examples"])
        )
    )

    collator = SFTDataCollator(tokenizer=tokenizer)

    validation_loader = DataLoader(
        validation_subset,
        batch_size=config["training"]["per_device_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"]
    )

    model.to(device)
    model.eval()
    model.config.use_cache = False

    total_loss = 0.0
    total_supervised_tokens = 0

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in validation_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(**batch)

            supervised_tokens = (batch["labels"] != -100).sum().item()

            total_loss += outputs.loss.item() * supervised_tokens
            total_supervised_tokens += supervised_tokens

    validation_loss = total_loss / total_supervised_tokens

    print("=== BASE MODEL BASELINE ===")
    print("validation examples:", len(validation_subset))
    print("validation loss:", validation_loss)
    print(
        "peak VRAM GiB:",
        torch.cuda.max_memory_allocated() / (1024**3),
    )


if __name__ == "__main__":
    main()
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tuning.collator import SFTDataCollator


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/tokenized/no_robots_qwen2_5_0_5b"


def gib(bytes_value: int) -> float:
    """Convert bytes to GiB."""
    return bytes_value / (1024**3)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not report BF16 support.")

    device = torch.device("cuda")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_from_disk(DATASET_PATH)
    train = dataset["train"]

    # Use one of the longest available examples to make the smoke test
    # representative of the upper end of our training sequence lengths.
    longest_index = max(
        range(len(train)),
        key=lambda index: len(train[index]["input_ids"]),
    )

    feature = train[longest_index]

    collator = SFTDataCollator(tokenizer=tokenizer)
    batch = collator([feature])

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    print("=== DEVICE ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("sequence length:", batch["input_ids"].shape[1])

    print("\n=== LOAD MODEL ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.train()
    model.config.use_cache = False

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("total parameters:", total_parameters)
    print("trainable parameters:", trainable_parameters)
    print(
        "all parameters trainable:",
        total_parameters == trainable_parameters,
    )

    print(
        "VRAM after model load:",
        f"{gib(torch.cuda.memory_allocated()):.2f} GiB",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    torch.cuda.reset_peak_memory_stats()

    print("\n=== ONE TRAINING STEP ===")

    outputs = model(**batch)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    print("loss:", loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    print(
        "peak VRAM allocated:",
        f"{gib(torch.cuda.max_memory_allocated()):.2f} GiB",
    )

    print(
        "peak VRAM reserved:",
        f"{gib(torch.cuda.max_memory_reserved()):.2f} GiB",
    )

    print("\n=== FULL FINE-TUNING SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
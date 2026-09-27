from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class SFTDataCollator:
    """Dynamically pad SFT examples within each batch."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = []

        for feature in features:
            labels = feature["labels"]
            padding_length = max_length - len(labels)

            if self.tokenizer.padding_side == "right":
                labels = labels + [-100] * padding_length
            else:
                labels = [-100] * padding_length + labels

            padded_labels.append(labels)

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch
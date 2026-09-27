import unittest

import torch
from fastapi import HTTPException

from backend.app import (
    MAX_INPUT_TOKENS,
    QARequest,
    answer,
    app,
    health,
)


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, overlength=False):
        self.overlength = overlength

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        return "fake prompt"

    def __call__(
        self,
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        length = (
            MAX_INPUT_TOKENS + 1
            if self.overlength
            else 10
        )

        return {
            "input_ids": torch.ones(
                (1, length),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (1, length),
                dtype=torch.long,
            ),
        }

    def convert_tokens_to_ids(self, token):
        return 2

    def batch_decode(
        self,
        generated,
        skip_special_tokens=True,
    ):
        return ["Tony Hawk"]


class FakeModel:
    def generate(self, **inputs):
        input_ids = inputs["input_ids"]

        generated_token = torch.tensor(
            [[99]],
            dtype=torch.long,
            device=input_ids.device,
        )

        return torch.cat(
            [input_ids, generated_token],
            dim=1,
        )


class BackendTests(unittest.TestCase):
    def setUp(self):
        app.state.tokenizer = FakeTokenizer()
        app.state.model = FakeModel()
        app.state.device = torch.device("cpu")

    def test_health(self):
        response = health()

        self.assertEqual(
            response["status"],
            "ok",
        )
        self.assertEqual(
            response["device"],
            "cuda",
        )
        self.assertIn(
            "squad_v2_intermediate_v3",
            response["model"],
        )

    def test_answer(self):
        request = QARequest(
            context=(
                "Professional skateboarder Tony Hawk "
                "lives in southern California."
            ),
            question=(
                "What is the name of the "
                "professional skateboarder?"
            ),
        )

        response = answer(request)

        self.assertEqual(
            response.answer,
            "Tony Hawk",
        )
        self.assertEqual(
            response.input_tokens,
            10,
        )

    def test_overlength_input(self):
        app.state.tokenizer = FakeTokenizer(
            overlength=True
        )

        request = QARequest(
            context="context",
            question="question",
        )

        with self.assertRaises(HTTPException) as raised:
            answer(request)

        self.assertEqual(
            raised.exception.status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
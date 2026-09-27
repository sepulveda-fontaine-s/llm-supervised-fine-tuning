import argparse
import gc
import json
import re
import string
from collections import Counter
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_fine_tuning.preprocessing import build_squad_messages


CONFIG_PATH = "configs/training.yaml"
DATASET_PATH = "data/processed/squad_v2"

V1_MODEL_PATH = "results/checkpoints/squad_v2/best_model"
V2_MODEL_PATH = "results/checkpoints/squad_v2_balanced_v2/best_model"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        return "".join(
            char for char in value if char not in string.punctuation
        )

    def normalize_whitespace(value):
        return " ".join(value.split())

    return normalize_whitespace(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        )
    )


def exact_match_score(prediction, ground_truth):
    return float(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    return 2 * precision * recall / (precision + recall)


def best_answerable_scores(prediction, gold_answers):
    exact_match = max(
        exact_match_score(prediction, answer)
        for answer in gold_answers
    )

    f1 = max(
        f1_score(prediction, answer)
        for answer in gold_answers
    )

    return exact_match, f1


def is_abstention(prediction):
    return (
        normalize_answer(prediction)
        == normalize_answer("I don't know.")
    )


def generate_predictions(
    model_path,
    model_name,
    examples,
    tokenizer,
    generation_config,
    device,
    batch_size=16,
):
    print(f"\n=== {model_name} ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()
    model.config.use_cache = True

    eos_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    results = []

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]

        prompts = []

        for example in batch_examples:
            messages = build_squad_messages(example)[:-1]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"],
                repetition_penalty=generation_config["repetition_penalty"],
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[:, input_length:]

        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        for example, prediction in zip(batch_examples, predictions):
            prediction = prediction.strip()
            gold_answers = example["answers"]["text"]
            answerable = len(gold_answers) > 0

            if answerable:
                exact_match, f1 = best_answerable_scores(
                    prediction,
                    gold_answers,
                )
                abstained = is_abstention(prediction)
            else:
                abstained = is_abstention(prediction)
                exact_match = float(abstained)
                f1 = float(abstained)

            results.append(
                {
                    "id": example["id"],
                    "answerable": answerable,
                    "prediction": prediction,
                    "gold_answers": gold_answers,
                    "exact_match": exact_match,
                    "f1": f1,
                    "abstained": abstained,
                }
            )

        print(
            f"processed "
            f"{min(start + batch_size, len(examples))}"
            f"/{len(examples)}"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def summarize(results):
    answerable = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable = [
        result
        for result in results
        if not result["answerable"]
    ]

    answerable_em = sum(
        result["exact_match"]
        for result in answerable
    ) / len(answerable)

    answerable_f1 = sum(
        result["f1"]
        for result in answerable
    ) / len(answerable)

    answerable_abstentions = sum(
        result["abstained"]
        for result in answerable
    )

    correct_abstentions = sum(
        result["abstained"]
        for result in unanswerable
    )

    abstention_accuracy = (
        correct_abstentions
        / len(unanswerable)
    )

    false_answer_rate = 1.0 - abstention_accuracy

    task_exact_match = sum(
        result["exact_match"]
        for result in results
    ) / len(results)

    task_f1 = sum(
        result["f1"]
        for result in results
    ) / len(results)

    return {
        "examples": len(results),
        "answerable_examples": len(answerable),
        "unanswerable_examples": len(unanswerable),
        "answerable_exact_match": answerable_em,
        "answerable_f1": answerable_f1,
        "answerable_false_abstention_rate":
            answerable_abstentions / len(answerable),
        "unanswerable_abstention_accuracy":
            abstention_accuracy,
        "unanswerable_false_answer_rate":
            false_answer_rate,
        "task_exact_match": task_exact_match,
        "task_f1": task_f1,
    }


def main():
    args = parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    tokenizer.padding_side = "left"

    dataset = load_from_disk(DATASET_PATH)

    validation = dataset["validation"].shuffle(
        seed=config["training"]["seed"]
    )

    if args.limit is not None:
        validation = validation.select(range(args.limit))

    examples = [
        validation[index]
        for index in range(len(validation))
    ]

    print("validation examples:", len(examples))

    generation_config = config["generation"]

    print("generation config:", generation_config)

    v1_results = generate_predictions(
        model_path=V1_MODEL_PATH,
        model_name="V1 - 33% UNANSWERABLE",
        examples=examples,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device,
    )

    v2_results = generate_predictions(
        model_path=V2_MODEL_PATH,
        model_name="V2 - 50% UNANSWERABLE",
        examples=examples,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device,
    )

    v1_summary = summarize(v1_results)
    v2_summary = summarize(v2_results)

    output = {
        "split": "validation",
        "limit": args.limit,
        "generation_config": generation_config,
        "v1": {
            "summary": v1_summary,
            "predictions": v1_results,
        },
        "v2_balanced": {
            "summary": v2_summary,
            "predictions": v2_results,
        },
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== V1 SUMMARY ===")

    for key, value in v1_summary.items():
        print(f"{key}: {value}")

    print("\n=== V2 BALANCED SUMMARY ===")

    for key, value in v2_summary.items():
        print(f"{key}: {value}")

    print("\nsaved to:", output_path)


if __name__ == "__main__":
    main()

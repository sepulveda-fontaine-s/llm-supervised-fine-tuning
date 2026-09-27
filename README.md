# LLM Full Fine-Tuning - Qwen2.5 Context-Grounded QA

Full supervised fine-tuning (SFT) of `Qwen/Qwen2.5-0.5B` for context-grounded question answering with explicit abstention on unanswerable inputs. The project covers dataset construction, tokenization, hyperparameter search, full-parameter training, checkpoint selection, validation/test evaluation, qualitative error analysis, FastAPI serving, a browser frontend, tests, logging, and GPU execution under SLURM.

This repository is a hands-on portfolio project designed to demonstrate an inspectable, production-oriented ML/LLM engineering workflow. It is not presented as long-term commercial production experience.

## 1. Technical scope

The implemented system includes:

- SQuAD v2-based answerable/unanswerable question answering
- deterministic dataset variants for training-distribution experiments
- Qwen chat-template preprocessing and assistant-only loss masking
- maximum sequence length of 1024 tokens
- full-parameter supervised fine-tuning of `Qwen2.5-0.5B`
- Optuna-based hyperparameter search followed by candidate confirmation
- epoch-level validation-loss checkpoint selection
- MLflow experiment tracking
- validation comparison across v1, v2 balanced, and v3 intermediate variants
- frozen final-test evaluation against the unfine-tuned base model
- qualitative error categorization
- FastAPI backend with `/health` and `/answer`
- static browser frontend
- CUDA inference on an NVIDIA A30
- backend unit tests and request/inference logging
- SLURM batch and interactive GPU workflows

## 2. System flow

```text
Processed SQuAD v2
        |
        v
Training-variant construction
(v1 / v2 balanced / v3 intermediate)
        |
        v
Qwen chat-template tokenization
(max_length = 1024)
        |
        v
Full SFT of Qwen2.5-0.5B
        |
        v
Validation-loss checkpoint selection
        |
        v
Validation comparison across variants
        |
        v
Final v3 best_model
        |
        +--------------------------+
        |                          |
        v                          v
Frozen test evaluation       FastAPI /answer
(base vs fine-tuned)              |
                                   v
                              Browser frontend
```

The final model is served with the same chat-template construction and deterministic generation policy used during evaluation.

## 3. Dataset and final training variant

The final v3 training variant uses a 60/40 answerable-unanswerable mixture:

| Item | Value |
|---|---:|
| Processed train examples | 30,000 |
| Answerable | 18,000 |
| Unanswerable | 12,000 |
| Unanswerable ratio | 40% |
| Validation examples | 5,933 |
| Test examples | 5,940 |

Validation and test are unchanged across the compared variants. During tokenization, three training sequences exceeded 1024 tokens and were removed, leaving 29,997 tokenized training examples.

## 4. Repository structure

```text
LLM_fine_tuning/
├── backend/
│   └── app.py
├── configs/
│   └── training.yaml
├── data/
│   ├── processed/
│   └── tokenized/
├── docs/
│   └── screenshots/
├── frontend/
│   └── index.html
├── monitoring/
├── results/
│   ├── checkpoints/
│   └── *.json / *.csv
├── scripts/
├── slurm/
├── src/
│   └── llm_fine_tuning/
│       ├── collator.py
│       └── preprocessing.py
├── tests/
│   └── test_backend.py
├── pyproject.toml
├── requirements-lock.txt
├── README.md
└── technical_report.pdf
```

Large model/checkpoint artifacts are intentionally not appropriate for normal Git history and should be restored or regenerated outside the repository clone.

## 5. Validated runtime

Canonical cluster runtime:

- Python `3.11.11`
- NVIDIA A30 24 GB
- PyTorch `2.11.0+cu128`
- Transformers `5.17.0`
- Datasets `5.0.1`
- MLflow `3.16.0`
- FastAPI `0.141.1`
- Uvicorn `0.53.0`
- SLURM GPU node `g-0`

The model is loaded in `bfloat16` for evaluation/serving and requires CUDA in the canonical deployment path.

## 6. Installation

### 6.1 Create a Python 3.11 environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 6.2 Install the CUDA PyTorch build

The validated cluster environment used the CUDA 12.8 wheel:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install the CUDA build appropriate to the target GPU/driver environment if reproducing elsewhere.

### 6.3 Install the project

```bash
python -m pip install -e .
python -m pip check
```

For the exact validated environment snapshot, see:

```text
requirements-lock.txt
```

## 7. Required model artifact

The serving backend expects the selected checkpoint at:

```text
results/checkpoints/squad_v2_intermediate_v3/best_model/
```

The normal `best_model` artifact is approximately 954 MB in the validated project state. Recovery checkpoints are substantially larger because they also retain training state such as optimizer/scheduler/RNG information.

Weights should not be committed to ordinary Git history. A fresh clone therefore requires restoring the final checkpoint or reproducing the training pipeline before serving.

## 8. Preprocessing

The system instruction used for SFT and inference is intentionally restrictive:

```text
Answer the question using only the provided context.
Return only the answer supported by the context.
If the context does not contain the answer, respond exactly: I don't know.
```

User messages are formatted as:

```text
Context:
<source context>

Question:
<question>
```

Only assistant response tokens are supervised. System/user tokens and structural assistant prefixes are masked with label `-100`.

The v3 preprocessing script removes examples whose tokenized sequence exceeds `max_length = 1024` and asserts that all retained samples contain supervised assistant tokens.

## 9. Full fine-tuning configuration

Final configured training parameters:

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Epochs | 3 |
| Per-device batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Precision | `bf16` |
| Learning rate | `2.5789950779174585e-05` |
| Weight decay | `0.02433937943676302` |
| Scheduler | linear |
| Warmup ratio | `0.018118910790805944` |
| Seed | 42 |

This is full-parameter training: no LoRA or QLoRA adapters are used.

The training loop normalizes accumulated gradients by supervised-token count, clips gradient norm, evaluates validation loss after each epoch, and saves a new `best_model` when validation loss improves. `last_checkpoint` additionally stores optimizer, scheduler, and RNG state for recovery.

## 10. Hyperparameter search

The tracked configuration defines a 12-trial Optuna search with a 600-optimizer-step surrogate horizon. Search dimensions include:

- learning rate: `2e-5` to `8e-5`
- weight decay: `0.0` to `0.08`
- warmup ratio: `0.005` to `0.03`
- scheduler: linear or cosine

Three candidates were then configured for confirmation. The final training configuration uses the candidate labelled `trial_3`.

## 11. Training telemetry

A dedicated rerun of the final v3 configuration was executed to recover operational training telemetry without overwriting the original checkpoint.

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.34499 | **0.21490** |
| 2 | 0.08130 | 0.23041 |
| 3 | 0.03463 | 0.27515 |

Validated rerun telemetry:

- best epoch: `1`
- best validation loss: `0.2149047066`
- training time: `2548.86 s` (~42 min 29 s)
- peak allocated VRAM: `14.998 GiB`

The divergence between continually falling training loss and worsening validation loss after epoch 1 makes the validation-loss checkpoint selection important.

The rerun used the same saved configuration but did not produce bitwise-identical weights to the original model. This project therefore claims reproducibility of the pipeline/configuration, not deterministic CUDA bitwise identity.

## 12. Validation comparison

All compared variants use the same 5,933-example validation set.

| Variant | Task EM | Task F1 | Answerable F1 | Unanswerable abstention accuracy | Answerable false-abstention rate |
|---|---:|---:|---:|---:|---:|
| v1 | 0.7367 | 0.7762 | 0.7861 | 0.7664 | 0.1172 |
| v2 balanced | **0.7484** | **0.7827** | 0.7128 | **0.8520** | 0.2052 |
| v3 intermediate | 0.7479 | 0.7825 | 0.7354 | 0.8292 | 0.1727 |

v2 and v3 are nearly tied on aggregate EM/F1 but expose a different answer/abstention trade-off. The final artifact is v3: relative to v2 it reduces false abstention on answerable examples while retaining essentially the same aggregate validation performance.

## 13. Frozen final-test evaluation

The final test contains 5,940 examples and is evaluated only after model selection.

| Metric | Base model | Fine-tuned v3 |
|---|---:|---:|
| Task EM | 0.0042 | **0.7271** |
| Task F1 | 0.1070 | **0.7580** |
| Answerable F1 | 0.2136 | **0.7140** |
| Unanswerable abstention accuracy | 0.0000 | **0.8020** |
| Unanswerable false-answer rate | 1.0000 | **0.1980** |

The base model effectively never abstains on unanswerable test examples. Full SFT changes both answer extraction and abstention behavior substantially.

## 14. Qualitative error analysis

Final v3 test predictions are categorized as:

| Category | Count |
|---|---:|
| Answerable exact match | 1,941 |
| Answerable partial match | 294 |
| Answerable false abstention | 547 |
| Answerable wrong answer | 193 |
| Unanswerable correct abstention | 2,378 |
| Unanswerable false answer | 587 |

The enriched analysis stores the question, context, prediction, gold answers, EM/F1, and abstention state for sampled examples. Observed failure modes include:

- exact-span boundary differences (`British Gas` vs `British Gas plc`)
- semantic near-matches not rewarded by token-overlap scoring (`stationary` vs `stagnant`)
- false abstention when the answer is present but requires selecting a non-salient phrase
- answering a related fact from context when the SQuAD question is deliberately unanswerable
- choosing a plausible nearby entity rather than the annotated answer

## 15. Generation policy

Serving and evaluation use deterministic generation:

```text
do_sample = false
repetition_penalty = 1.0
max_new_tokens = 64
```

The FastAPI backend reconstructs the same Qwen chat prompt used during evaluation and enforces the same 1024-token input limit.

## 16. Run the backend

On the GPU node, with the environment active:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8040
```

Health endpoint:

```text
GET http://127.0.0.1:8040/health
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8040/docs
```

Inference endpoint:

```text
POST /answer
```

Request body:

```json
{
  "context": "Southern California is home to a large surf and skateboard culture. Professional skateboarder Tony Hawk lives in southern California.",
  "question": "What is the name of the professional skateboarder that lives in southern California?"
}
```

Example response:

```json
{
  "answer": "Tony Hawk",
  "input_tokens": 89
}
```

## 17. Run the frontend

The frontend is a static HTML application. On the GPU node:

```bash
.venv/bin/python -m http.server 8050 --bind 127.0.0.1 --directory frontend
```

Local address after tunneling/forwarding:

```text
http://127.0.0.1:8050
```

The UI reports API availability, sends context/question pairs to FastAPI, and displays the generated answer and input-token count.

## 18. Remote DANTZIG/UMH workflow

Validated interactive GPU allocation:

```bash
sinteractive --partition GPU --qos gpu -w g-0 --cpus-per-task=24 --mem=72G --gres=gpu:1 --time=08:00:00
```

In the validated VS Code Remote SSH workflow, the browser remains on the local Windows machine while backend/frontend processes run on `g-0`.

Two tunnels were maintained from `dantzig` to `g-0`:

```bash
ssh -N -L 8040:127.0.0.1:8040 g-0
ssh -N -L 8050:127.0.0.1:8050 g-0
```

VS Code then forwards ports `8040` and `8050` to the local browser.

## 19. Tests

Lightweight backend tests:

```bash
python -m unittest tests/test_backend.py -v
```

Validated result:

```text
test_answer ... ok
test_health ... ok
test_overlength_input ... ok

Ran 3 tests
OK
```

The tests use lightweight fake tokenizer/model objects, so they validate API logic without loading the full checkpoint or requiring a GPU.

## 20. Logging

The backend logs inference completion with token count and latency. Example shape:

```text
inference_complete | input_tokens=81 | latency_ms=416.71
```

Standard Uvicorn access logs remain available alongside the application log.

## 21. Reproducibility

Reproducibility is supported by:

- tracked `training.yaml`
- preprocessing and training scripts
- deterministic dataset-selection seed (`42`)
- saved `training_config.yaml` inside checkpoints
- `requirements-lock.txt` with the exact validated Python environment
- SLURM wrappers for cluster execution
- validation/test result JSONs
- MLflow file-store experiment artifacts

Important boundary: fixed random seeds and identical configuration do not imply bitwise-identical GPU weights. A controlled rerun produced different SHA-256 hashes for the original and rerun `model.safetensors`, so the repository does not claim bitwise deterministic CUDA training.

## 22. Known limitations

- The final checkpoint is large and is not suitable for normal Git history.
- The canonical serving path requires CUDA.
- The browser demo is not maintained as a 24/7 public service.
- Backend tests currently focus on API logic rather than full GPU integration.
- No load/concurrency benchmark has been performed.
- No Docker image or CI/CD pipeline is currently claimed.
- Exact bitwise GPU-training reproducibility is not claimed.
- SQuAD-style lexical EM/F1 can penalize semantically plausible aliases or paraphrases.
- The final model still produces false answers on a subset of unanswerable examples and false abstentions on a subset of answerable examples.

## 23. Technical report

A separate illustrated report documents the methodology, training evidence, validation/test metrics, qualitative errors, API/frontend evidence, runtime workflow, reproducibility boundary, limitations, and production-scale extensions:

```text
technical_report.pdf
```

The README intentionally stays focused on implementation and reproducibility. Visual evidence and narrative analysis belong in the technical report.

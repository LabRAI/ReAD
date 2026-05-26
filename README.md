# ReAD

This repository contains an anonymous implementation of ReAD, a reinforcement-guided framework for budgeted capability distillation of large language models.

ReAD treats distillation as an allocation problem over interacting capabilities. Given a fixed token budget, it estimates which capabilities matter for a downstream task, generates capability-targeted teacher supervision, trains the student on the generated data, and updates future capability allocations with an uncertainty-aware contextual bandit.

## Method Overview

ReAD has four main components:

1. Task requirement identification: each task card defines a requirement vector `r_tau` over measurable capabilities. A learned requirement identifier is also supported through offline intervention data.
2. Capability-targeted generation: prompt templates are sampled by capability and difficulty. Teacher generations become distillation examples, with optional eval-prompt hash filtering to reduce leakage.
3. Token-budgeted distillation: the student is trained with a masked token-level cross-entropy objective while tracking prompt, label, and generation tokens.
4. Adaptive allocation: a bootstrap ensemble reward model selects the next capability allocation with an upper-confidence rule. The reward favors task-aligned capability gains and penalizes harmful spillover on required capabilities.

The implementation follows the paper story while keeping the public artifact small: no benchmark datasets, local checkpoints, generated outputs, or baseline contender implementations are included.

## Repository Organization

```text
read/                         Core ReAD package
  allocation.py               Candidate actions and spillover penalty
  bandit.py                   UCB ensemble contextual bandit
  generation.py               vLLM teacher generation wrapper
  probe_suite.py              Capability-profile probe evaluation
  requirements.py             Task requirement vectors / learned identifier
  template_library.py         Prompt templates, difficulty, and hash filtering
  training_utils.py           Token-counted student distillation

scripts/
  read_train.py               End-to-end ReAD run
  read_generate_batch.py      Generate one capability-weighted data batch
  read_build_probes.py        Build probe data from the teacher
  read_collect_gphi.py        Collect intervention data for requirement learning
  train_requirement_identifier.py
  build_eval_hashes.py        Build prompt hashes for eval-overlap filtering

configs/                      Default capabilities, task cards, and budgets
data/distill_prompts/         Small synthetic prompt-template pools
examples/toy_read_demo.py     Lightweight runnable sanity check
tests/                        Unit tests for core public code
```

## Installation

For the lightweight package, toy demo, and unit tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For full teacher generation and benchmark utilities:

```bash
pip install -e ".[generation,eval]"
```

Large-model runs require GPUs and access to the selected teacher/student model checkpoints. The default configs use Llama-family model names as placeholders; change them to checkpoints you can access.

## Quick Sanity Check

Run the synthetic ReAD demo:

```bash
python examples/toy_read_demo.py --seed 1 --steps 40 --output runs/toy_read/seed1.json
```

This demo does not use external datasets or LLM checkpoints. It simulates cross-capability transfer, diminishing returns, and spillover, then verifies that the ReAD allocation loop can adapt budget toward the task-relevant capabilities. It is intended as a fast check that the core bandit/allocation logic runs.

Run unit tests:

```bash
pytest -q
```

## Full ReAD Run

1. Prepare datasets locally. Datasets are not vendored in this repository. Task cards in `configs/task_cards.yaml` show the expected paths and schemas. Local generic JSONL tasks use:

```json
{"prompt": "...", "answer": "..."}
```

2. Optionally build evaluation prompt hashes to reduce train/eval prompt overlap:

```bash
python scripts/build_eval_hashes.py \
  --eval_config configs/eval_config.json \
  --output data/eval_prompt_hashes.json \
  --allow_missing_bfcl
```

3. Run ReAD with a small budget first:

```bash
python scripts/read_train.py \
  --config configs/read.yml \
  --task_name math_gsm8k \
  --teacher_model <teacher-checkpoint-or-hf-id> \
  --student_model <student-checkpoint-or-hf-id> \
  --budget_total_tokens 200000 \
  --budget_gen_tokens 80000 \
  --budget_train_tokens 120000 \
  --seed 1
```

Outputs are written under:

```text
checkpoints/read/<student>/<task>/budget_<tokens>/seed_<seed>/
```

Each run records generated examples, interval manifests, bandit history, dev checkpoints, final student weights, and a top-level run manifest.

## Training a Requirement Identifier

ReAD can use manually specified task requirement vectors from the task cards or a learned identifier. To train the identifier:

```bash
python scripts/read_collect_gphi.py \
  --config configs/read.yml \
  --tasks math_gsm8k,code_humaneval \
  --num_interventions 5 \
  --budget_gen_tokens 2000 \
  --budget_train_tokens 3000 \
  --output_jsonl checkpoints/read_gphi/interventions.jsonl

python scripts/train_requirement_identifier.py \
  --train_jsonl checkpoints/read_gphi/interventions.jsonl \
  --num_capabilities 8 \
  --output checkpoints/read_gphi/gphi.pt
```

Then set `requirement_model: checkpoints/read_gphi/gphi.pt` in the corresponding task card.

## Double-Blind Artifact Hygiene

The Git ignore rules are intentionally strict. The repository excludes:

- the paper PDF and PDF metadata
- notebooks, scratch notes, backup logs, and generated results
- benchmark datasets and local copies of public datasets
- private cluster scripts, absolute paths, usernames, emails, and affiliations
- model checkpoints and generated distillation/evaluation outputs

Before release, audit the tracked files with `git grep`, a secret scanner, and a manual review of generated manifests.

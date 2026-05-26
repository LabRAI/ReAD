#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from read.eval_utils import evaluate_gsm8k, evaluate_logprob, load_examples
from read.generation import GeneratorConfigs, VllmBatchGenerator
from read.probe_suite import ProbeSuite
from read.task_cards import load_task_cards
from read.training_utils import TokenCountingRules, train_for_tokens
from read.utils import load_yaml, set_seed


def sample_allocation(capabilities: List[str], support_size: int, alpha: float, rng: random.Random) -> List[float]:
    idx = rng.sample(range(len(capabilities)), k=support_size)
    weights = [0.0] * len(capabilities)
    # Dirichlet sampling via gamma
    draws = [rng.gammavariate(alpha, 1.0) for _ in idx]
    total = sum(draws) if draws else 1.0
    for i, d in zip(idx, draws):
        weights[i] = d / total
    return weights


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def evaluate_dev(
    task_card,
    model,
    tokenizer,
    examples,
    device,
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
) -> float:
    if task_card.eval_type == "gsm8k_exact":
        return evaluate_gsm8k(model, tokenizer, examples, max_new_tokens, device, batch_size=batch_size)

    if task_card.eval_type == "logprob":
        return evaluate_logprob(model, tokenizer, examples, max_length, device, batch_size=batch_size)

    if task_card.eval_type == "humaneval_pass@1":
        samples_path = Path("/tmp") / f"gphi_humaneval_samples_{os.getpid()}.jsonl"
        with samples_path.open("w", encoding="utf-8") as f:
            batch_size = max(1, batch_size)
            with torch.inference_mode():
                for i in range(0, len(examples), batch_size):
                    batch = examples[i : i + batch_size]
                    prompts = [ex.prompt for ex in batch]
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                    gen = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    for ex, out, in_len in zip(batch, gen, input_lens):
                        completion = tokenizer.decode(out[int(in_len) :], skip_special_tokens=True)
                        f.write(json.dumps({"task_id": ex.task_id, "completion": completion}) + "\n")

        problem_file = Path("data/humaneval/HumanEval.jsonl")
        cmd = [
            "evaluate_functional_correctness",
            str(samples_path),
            "--problem_file",
            str(problem_file),
        ]
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = p.stdout.strip().splitlines()
        result_line = ""
        for line in reversed(stdout):
            if line.strip().startswith("{") and line.strip().endswith("}"):
                result_line = line.strip()
                break
        if not result_line:
            raise RuntimeError("evaluate_functional_correctness produced no JSON result")
        import ast
        res = ast.literal_eval(result_line)
        return float(res.get("pass@1", 0.0))

    raise ValueError(f"Unsupported eval_type: {task_card.eval_type}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/read.yml")
    ap.add_argument("--task_cards", default="configs/task_cards.yaml")
    ap.add_argument("--tasks", default="", help="Comma-separated task names; default all in task_cards")
    ap.add_argument("--output_jsonl", default="checkpoints/read_gphi/interventions.jsonl")
    ap.add_argument("--num_interventions", type=int, default=5)
    ap.add_argument("--support_size", type=int, default=2)
    ap.add_argument("--dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--budget_gen_tokens", type=int, default=2000)
    ap.add_argument("--budget_train_tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher_model", default="")
    ap.add_argument("--student_model", default="")
    ap.add_argument("--probe_suite", default="")
    ap.add_argument("--teacher_visible_devices", default="")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(args.seed)
    rng = random.Random(args.seed)

    teacher_model = args.teacher_model or cfg["teacher"]["model"]
    student_model = args.student_model or cfg["student"]["model"]
    capabilities = cfg["capabilities"]

    # Prepare probe suite
    if args.probe_suite:
        probe_path = Path(args.probe_suite)
    else:
        probe_path = Path("checkpoints/read_gphi/probes.jsonl")

    if not probe_path.exists():
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if args.teacher_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.teacher_visible_devices
        cmd = [
            "python",
            "scripts/read_build_probes.py",
            "--capabilities",
            ",".join(capabilities),
            "--num_probes",
            str(cfg["probe"]["num_per_capability"]),
            "--teacher_model",
            teacher_model,
            "--output_jsonl",
            str(probe_path),
            "--exp_config",
            cfg["exp_config"],
            "--prompt_source_config",
            cfg["prompt_source_config"],
            "--eval_hashes",
            cfg["eval_hashes"],
            "--temperature",
            str(cfg["probe"]["temperature"]),
            "--top_p",
            str(cfg["probe"]["top_p"]),
            "--max_new_tokens",
            str(cfg["probe"]["max_new_tokens"]),
            "--dtype",
            cfg["teacher"]["dtype"],
            "--tp",
            str(cfg["teacher"]["tp"]),
            "--max_model_len",
            str(cfg["teacher"]["max_model_len"]),
        ]
        if cfg["teacher"].get("trust_remote_code", False):
            cmd.append("--trust_remote_code")
        print("[RUN]", " ".join(cmd))
        probe_start = time.perf_counter()
        subprocess.check_call(cmd, env=env)
        print(f"[TIME] build probes: {format_duration(time.perf_counter() - probe_start)}")

    overall_start = time.perf_counter()
    probes = ProbeSuite.load(probe_path)

    all_task_cards = load_task_cards(Path(args.task_cards))
    tasks = [t for t in args.tasks.split(",") if t] if args.tasks else list(all_task_cards.keys())
    task_cards = {name: all_task_cards[name] for name in tasks}

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for task_name in tasks:
        task_card = task_cards[task_name]
        task_text = task_card.to_text()
        for m in range(args.num_interventions):
            weights = sample_allocation(capabilities, args.support_size, args.dirichlet_alpha, rng)
            run_dir = out_path.parent / f"{task_name}_m{m:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            runs.append(
                {
                    "task_name": task_name,
                    "task_text": task_text,
                    "m": m,
                    "weights": weights,
                    "gen_jsonl": run_dir / "train.jsonl",
                    "gen_manifest": run_dir / "manifest.json",
                }
            )

    if args.teacher_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.teacher_visible_devices

    gen_configs = GeneratorConfigs(
        exp_config=Path(cfg["exp_config"]),
        prompt_source_config=Path(cfg["prompt_source_config"]),
        eval_hashes=Path(cfg["eval_hashes"]),
        difficulty_percentiles=(0.33, 0.66),
        difficulty_sample_size=256,
    )
    gen = VllmBatchGenerator(
        capabilities=capabilities,
        teacher_model=teacher_model,
        seed=args.seed,
        dtype=cfg["teacher"]["dtype"],
        tp=cfg["teacher"]["tp"],
        max_model_len=cfg["teacher"]["max_model_len"],
        trust_remote_code=cfg["teacher"].get("trust_remote_code", False),
        configs=gen_configs,
    )

    gen_total_start = time.perf_counter()
    for run in runs:
        gen_start = time.perf_counter()
        manifest = gen.generate(
            weights=run["weights"],
            budget_gen_tokens=args.budget_gen_tokens,
            output_jsonl=run["gen_jsonl"],
            output_manifest=run["gen_manifest"],
            progress=0.0,
            temperature=cfg["generation"]["temperature"],
            top_p=cfg["generation"]["top_p"],
            top_k=cfg["generation"]["top_k"],
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            batch_size=cfg["generation"]["batch_size"],
        )
        run["actual_gen"] = int(manifest.get("gen_tokens", args.budget_gen_tokens))
        run["gen_time_s"] = time.perf_counter() - gen_start
        print(
            f"[TIME] gen {run['task_name']} m{run['m']:03d}: {format_duration(run['gen_time_s'])}"
        )

    gen_total = time.perf_counter() - gen_total_start
    gen.shutdown()
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[TIME] generation total: {format_duration(gen_total)}")

    tokenizer = AutoTokenizer.from_pretrained(
        student_model, trust_remote_code=cfg["student"].get("trust_remote_code", False)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = getattr(torch, cfg["student"]["dtype"], torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=cfg["student"].get("trust_remote_code", False),
    )
    device = next(model.parameters()).device
    base_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    s0_start = time.perf_counter()
    probe_batch = int(cfg.get("probe", {}).get("batch_size", 8))
    eval_batch = int(cfg.get("eval", {}).get("batch_size", 8))
    s0_dict = probes.compute_profile(
        model,
        tokenizer,
        capabilities,
        cfg["student"]["max_length"],
        device,
        batch_size=probe_batch,
    )
    s0 = [s0_dict.get(c, 0.0) for c in capabilities]
    print(f"[TIME] base probe profile: {format_duration(time.perf_counter() - s0_start)}")

    rules = TokenCountingRules(
        count_prompt_tokens=cfg["token_counting"]["count_prompt_tokens"],
        count_label_tokens=cfg["token_counting"]["count_label_tokens"],
        count_padding_tokens=cfg["token_counting"]["count_padding_tokens"],
    )

    dev_examples_by_task = {name: load_examples(task_cards[name], "dev") for name in tasks}

    total_train = 0.0
    total_probe = 0.0
    total_eval = 0.0

    with out_path.open("w", encoding="utf-8") as f:
        for run in runs:
            task_name = run["task_name"]
            task_card = task_cards[task_name]
            dev_examples = dev_examples_by_task[task_name]

            model.load_state_dict(base_state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["student"]["learning_rate"],
                weight_decay=cfg["student"]["weight_decay"],
            )

            train_start = time.perf_counter()
            train_tokens, prompt_tokens, label_tokens, steps = train_for_tokens(
                model,
                tokenizer,
                run["gen_jsonl"],
                optimizer,
                rules,
                args.budget_train_tokens,
                cfg["student"]["batch_size"],
                cfg["student"]["gradient_accumulation_steps"],
                cfg["student"]["max_length"],
                device,
                cfg["student"]["log_every_steps"],
            )
            train_time = time.perf_counter() - train_start
            total_train += train_time

            probe_start = time.perf_counter()
            s1_dict = probes.compute_profile(
                model,
                tokenizer,
                capabilities,
                cfg["student"]["max_length"],
                device,
                batch_size=probe_batch,
            )
            probe_time = time.perf_counter() - probe_start
            total_probe += probe_time
            s1 = [s1_dict.get(c, 0.0) for c in capabilities]
            delta_s = [s1[i] - s0[i] for i in range(len(s0))]

            eval_start = time.perf_counter()
            delta_u = evaluate_dev(
                task_card,
                model,
                tokenizer,
                dev_examples,
                device,
                cfg["eval"]["max_new_tokens"],
                cfg["student"]["max_length"],
                eval_batch,
            )
            eval_time = time.perf_counter() - eval_start
            total_eval += eval_time

            row = {
                "task_name": task_name,
                "task_text": run["task_text"],
                "delta_s": delta_s,
                "delta_u": delta_u,
                "allocation": run["weights"],
                "budget_gen_tokens": run["actual_gen"],
                "budget_train_tokens": train_tokens,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            print(
                f"[TIME] {task_name} m{run['m']:03d} "
                f"train {format_duration(train_time)} "
                f"probe {format_duration(probe_time)} "
                f"eval {format_duration(eval_time)} "
                f"gen {format_duration(run.get('gen_time_s', 0.0))}"
            )

    overall_elapsed = time.perf_counter() - overall_start
    print(f"[TIME] train total: {format_duration(total_train)}")
    print(f"[TIME] probe total: {format_duration(total_probe)}")
    print(f"[TIME] eval total: {format_duration(total_eval)}")
    print(f"[TIME] overall total: {format_duration(overall_elapsed)}")
    print(f"[OK] Wrote interventions -> {out_path}")


if __name__ == "__main__":
    main()

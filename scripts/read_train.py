#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from read.allocation import build_candidate_actions, compute_spillover
from read.bandit import BanditConfig, UCBEnsembleBandit
from read.eval_utils import evaluate_gsm8k, evaluate_logprob, load_examples
from read.probe_suite import ProbeSuite
from read.requirements import get_requirement_vector
from read.task_cards import load_task_card
from read.training_utils import TokenCountingRules, train_for_tokens
from read.utils import load_yaml, safe_model_name, set_seed


def run_subprocess(cmd: List[str], env: Dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def build_probes_if_missing(
    probe_path: Path,
    capabilities: List[str],
    teacher_model: str,
    cfg: Dict[str, Any],
    env: Dict[str, str],
) -> None:
    if probe_path.exists():
        return
    probe_path.parent.mkdir(parents=True, exist_ok=True)
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
    run_subprocess(cmd, env)


def generate_interval_data(
    out_jsonl: Path,
    out_manifest: Path,
    capabilities: List[str],
    weights: List[float],
    budget_gen_tokens: int,
    teacher_model: str,
    cfg: Dict[str, Any],
    progress: float,
    env: Dict[str, str],
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cap_weights = ",".join(f"{c}:{w:.6f}" for c, w in zip(capabilities, weights))
    cmd = [
        "python",
        "scripts/read_generate_batch.py",
        "--capability_weights",
        cap_weights,
        "--capabilities",
        ",".join(capabilities),
        "--budget_gen_tokens",
        str(budget_gen_tokens),
        "--teacher_model",
        teacher_model,
        "--seed",
        str(cfg["seed"]),
        "--output_jsonl",
        str(out_jsonl),
        "--output_manifest",
        str(out_manifest),
        "--exp_config",
        cfg["exp_config"],
        "--prompt_source_config",
        cfg["prompt_source_config"],
        "--eval_hashes",
        cfg["eval_hashes"],
        "--progress",
        str(progress),
        "--temperature",
        str(cfg["generation"]["temperature"]),
        "--top_p",
        str(cfg["generation"]["top_p"]),
        "--top_k",
        str(cfg["generation"]["top_k"]),
        "--max_new_tokens",
        str(cfg["generation"]["max_new_tokens"]),
        "--dtype",
        cfg["teacher"]["dtype"],
        "--tp",
        str(cfg["teacher"]["tp"]),
        "--max_model_len",
        str(cfg["teacher"]["max_model_len"]),
        "--batch_size",
        str(cfg["generation"]["batch_size"]),
    ]
    if cfg["teacher"].get("trust_remote_code", False):
        cmd.append("--trust_remote_code")
    run_subprocess(cmd, env)


def evaluate_dev(
    task_card,
    model,
    tokenizer,
    examples,
    device,
    cfg: Dict[str, Any],
    outdir: Path,
    batch_size: int,
) -> float:
    if task_card.eval_type == "gsm8k_exact":
        return evaluate_gsm8k(
            model,
            tokenizer,
            examples,
            cfg["eval"]["max_new_tokens"],
            device,
            batch_size=batch_size,
        )

    if task_card.eval_type == "logprob":
        return evaluate_logprob(
            model,
            tokenizer,
            examples,
            cfg["student"]["max_length"],
            device,
            batch_size=batch_size,
        )

    if task_card.eval_type == "humaneval_pass@1":
        # Generate completions with current model and call evaluator
        outdir.mkdir(parents=True, exist_ok=True)
        samples_path = outdir / "samples.jsonl"
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
                        max_new_tokens=cfg["eval"]["max_new_tokens"],
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
        (outdir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return float(res.get("pass@1", 0.0))

    raise ValueError(f"Unsupported eval_type: {task_card.eval_type}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/read.yml")
    ap.add_argument("--task_name", default="")
    ap.add_argument("--task_cards", default="configs/task_cards.yaml")
    ap.add_argument("--output_root", default="checkpoints/read")
    ap.add_argument("--budget_total_tokens", type=int, default=0)
    ap.add_argument("--budget_gen_tokens", type=int, default=0)
    ap.add_argument("--budget_train_tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher_model", default="")
    ap.add_argument("--student_model", default="")
    ap.add_argument("--teacher_visible_devices", default="")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.task_name:
        cfg["task_name"] = args.task_name
    if args.task_cards:
        cfg["task_cards"] = args.task_cards

    cfg["seed"] = args.seed or cfg.get("seed", 1)
    set_seed(cfg["seed"])

    teacher_model = args.teacher_model or cfg["teacher"]["model"]
    student_model = args.student_model or cfg["student"]["model"]
    capabilities = cfg["capabilities"]

    task_card = load_task_card(Path(cfg["task_cards"]), cfg["task_name"])

    total_budget = args.budget_total_tokens or cfg["budget"]["total_tokens"]
    gen_budget = args.budget_gen_tokens or cfg["budget"]["gen_tokens"]
    train_budget = args.budget_train_tokens or cfg["budget"]["train_tokens"]
    if gen_budget + train_budget != total_budget:
        raise ValueError("budget_total_tokens must equal gen_tokens + train_tokens")

    run_dir = (
        Path(args.output_root)
        / safe_model_name(student_model)
        / cfg["task_name"]
        / f"budget_{total_budget}"
        / f"seed_{cfg['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_env = os.environ.copy()
    if args.teacher_visible_devices:
        teacher_env["CUDA_VISIBLE_DEVICES"] = args.teacher_visible_devices

    probe_path = run_dir / "probes.jsonl"
    build_probes_if_missing(probe_path, capabilities, teacher_model, cfg, teacher_env)
    probes = ProbeSuite.load(probe_path)

    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=cfg["student"].get("trust_remote_code", False))
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

    probe_batch = int(cfg.get("probe", {}).get("batch_size", 8))
    eval_batch = int(cfg.get("eval", {}).get("batch_size", 8))
    r_tau = get_requirement_vector(task_card, capabilities, None)
    s0_dict = probes.compute_profile(
        model,
        tokenizer,
        capabilities,
        cfg["student"]["max_length"],
        device,
        batch_size=probe_batch,
    )
    s_t = [s0_dict.get(c, 0.0) for c in capabilities]

    dev_examples = load_examples(task_card, "dev")

    bandit_cfg = BanditConfig(
        num_models=cfg["bandit"]["num_models"],
        hidden_dim=cfg["bandit"]["hidden_dim"],
        lr=cfg["bandit"]["lr"],
        train_steps=cfg["bandit"]["train_steps"],
        batch_size=cfg["bandit"]["batch_size"],
        kappa=cfg["bandit"]["kappa"],
        warmup_steps=cfg["bandit"]["warmup_steps"],
    )
    bandit = UCBEnsembleBandit(input_dim=len(capabilities) * 2 + 1 + len(capabilities), cfg=bandit_cfg, seed=cfg["seed"])

    actions = build_candidate_actions(
        capabilities,
        r_tau,
        max_support=cfg["allocation"]["max_support"],
        weight_grid=cfg["allocation"]["weight_grid"],
        top_k=cfg["allocation"]["top_k"],
        max_candidates=cfg["allocation"]["max_candidates"],
        include_one_hot=cfg["allocation"].get("include_one_hot", True),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["student"]["learning_rate"], weight_decay=cfg["student"]["weight_decay"])
    rules = TokenCountingRules(
        count_prompt_tokens=cfg["token_counting"]["count_prompt_tokens"],
        count_label_tokens=cfg["token_counting"]["count_label_tokens"],
        count_padding_tokens=cfg["token_counting"]["count_padding_tokens"],
    )

    history_path = run_dir / "history.jsonl"
    used_gen = 0
    used_train = 0
    interval = 0
    last_utility = None
    kappa = cfg["bandit"]["kappa"]
    proxy_cum = 0.0
    proxy_points: List[float] = []
    utility_points: List[float] = []
    reward_scale = 1.0
    checkpoint_fraction = float(cfg["eval"].get("checkpoint_fraction", 0.0) or 0.0)
    next_checkpoint = checkpoint_fraction * total_budget if checkpoint_fraction > 0 else None

    while used_gen < gen_budget and used_train < train_budget:
        interval += 1
        remaining_gen = gen_budget - used_gen
        remaining_train = train_budget - used_train
        interval_train = min(cfg["interval"]["train_tokens"], remaining_train)
        if cfg["interval"]["gen_tokens"] > 0:
            interval_gen = min(cfg["interval"]["gen_tokens"], remaining_gen)
        else:
            if remaining_train <= 0 or remaining_gen <= 0:
                interval_gen = 0
            else:
                import math
                intervals_left = max(1, math.ceil(remaining_train / max(1, interval_train)))
                interval_gen = min(remaining_gen, int(math.ceil(remaining_gen / intervals_left)))
        progress = (used_gen + used_train) / max(1, total_budget)

        context = r_tau + s_t + [float(total_budget - used_gen - used_train)]
        w_t = bandit.select_action(context, actions, kappa)

        interval_dir = run_dir / f"interval_{interval:04d}"
        gen_jsonl = interval_dir / "train.jsonl"
        gen_manifest = interval_dir / "manifest.json"
        generate_interval_data(
            gen_jsonl,
            gen_manifest,
            capabilities,
            w_t,
            interval_gen,
            teacher_model,
            cfg,
            progress,
            teacher_env,
        )
        manifest = json.loads(gen_manifest.read_text(encoding="utf-8"))
        actual_gen = int(manifest.get("gen_tokens", interval_gen))
        used_gen += actual_gen

        optimizer.zero_grad()
        train_tokens, prompt_tokens, label_tokens, steps = train_for_tokens(
            model,
            tokenizer,
            gen_jsonl,
            optimizer,
            rules,
            interval_train,
            cfg["student"]["batch_size"],
            cfg["student"]["gradient_accumulation_steps"],
            cfg["student"]["max_length"],
            device,
            cfg["student"]["log_every_steps"],
        )
        used_train += train_tokens

        s1_dict = probes.compute_profile(
            model,
            tokenizer,
            capabilities,
            cfg["student"]["max_length"],
            device,
            batch_size=probe_batch,
        )
        s1 = [s1_dict.get(c, 0.0) for c in capabilities]
        delta_s = [s1[i] - s_t[i] for i in range(len(s_t))]
        spill = compute_spillover(r_tau, delta_s, cfg["reward"]["spillover_top_k"])
        cost = float(actual_gen + train_tokens)
        reward_raw = sum(r_tau[i] * delta_s[i] for i in range(len(r_tau))) - cfg["reward"]["beta"] * spill - cfg["reward"]["lambda_cost"] * cost
        reward = reward_raw * reward_scale
        proxy_cum += reward_raw

        bandit.add(context + w_t, reward)
        bandit.fit()
        s_t = s1

        record = {
            "interval": interval,
            "w_t": w_t,
            "gen_tokens": actual_gen,
            "train_tokens": train_tokens,
            "reward_raw": reward_raw,
            "reward_scaled": reward,
            "spillover": spill,
            "cost": cost,
            "delta_s": delta_s,
            "s_t": s_t,
            "reward_scale": reward_scale,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        do_eval = False
        if next_checkpoint is not None:
            if (used_gen + used_train) >= next_checkpoint:
                do_eval = True
                next_checkpoint += checkpoint_fraction * total_budget
        elif cfg["eval"].get("every_intervals", 0) and interval % cfg["eval"]["every_intervals"] == 0:
            do_eval = True

        if do_eval:
            util = evaluate_dev(
                task_card,
                model,
                tokenizer,
                dev_examples,
                device,
                cfg,
                run_dir / "dev_eval",
                eval_batch,
            )
            proxy_points.append(proxy_cum)
            utility_points.append(util)
            if len(proxy_points) >= 2:
                x = torch.tensor(proxy_points, dtype=torch.float32)
                y = torch.tensor(utility_points, dtype=torch.float32)
                x_mean = x.mean()
                y_mean = y.mean()
                denom = ((x - x_mean) ** 2).sum()
                if float(denom.item()) > 1e-8:
                    a = float(((x - x_mean) * (y - y_mean)).sum().item() / denom.item())
                    if a > 0:
                        reward_scale = a
            if last_utility is not None and util <= last_utility:
                kappa = max(0.0, kappa * cfg["bandit"]["kappa_decay"])
            last_utility = util
            (run_dir / "dev_eval" / f"interval_{interval:04d}.json").write_text(
                json.dumps({"utility": util, "interval": interval}, indent=2), encoding="utf-8"
            )

        if used_gen >= gen_budget or used_train >= train_budget:
            break

    model.save_pretrained(run_dir / "final", safe_serialization=True)
    tokenizer.save_pretrained(run_dir / "final")

    manifest = {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "task_name": cfg["task_name"],
        "capabilities": capabilities,
        "r_tau": r_tau,
        "budget_total_tokens": total_budget,
        "budget_gen_tokens": gen_budget,
        "budget_train_tokens": train_budget,
        "used_gen_tokens": used_gen,
        "used_train_tokens": used_train,
        "seed": cfg["seed"],
        "config": cfg,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Saved ReAD run to {run_dir}")


if __name__ == "__main__":
    main()

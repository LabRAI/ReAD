#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch

from read.allocation import build_candidate_actions, compute_spillover
from read.bandit import BanditConfig, UCBEnsembleBandit


CAPABILITIES = ["general_knowledge", "reasoning", "math", "code"]

# Rows are training targets and columns are measured capability changes.
# The matrix encodes the paper story: on-target gains are strongest, related
# capabilities can transfer positively, and unrelated capabilities can degrade.
TRANSFER = [
    [0.045, 0.004, 0.000, -0.002],
    [0.006, 0.065, 0.025, -0.010],
    [0.003, 0.030, 0.070, -0.006],
    [-0.002, 0.010, -0.004, 0.060],
]


def utility(profile: List[float], requirement: List[float]) -> float:
    return sum(p * r for p, r in zip(profile, requirement))


def transition(profile: List[float], action: List[float], rng: random.Random) -> List[float]:
    delta = [0.0 for _ in profile]
    for target, weight in enumerate(action):
        for cap_idx, raw_effect in enumerate(TRANSFER[target]):
            saturation = max(0.05, 1.0 - profile[cap_idx])
            delta[cap_idx] += weight * raw_effect * saturation

    next_profile = []
    for score, change in zip(profile, delta):
        noisy = score + change + rng.gauss(0.0, 0.0015)
        next_profile.append(min(1.0, max(0.0, noisy)))
    return next_profile


def run_fixed_policy(
    action: List[float],
    requirement: List[float],
    initial_profile: List[float],
    steps: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    profile = initial_profile[:]
    history = []
    for step in range(steps):
        old = profile
        profile = transition(profile, action, rng)
        history.append(
            {
                "step": step + 1,
                "action": action,
                "delta_s": [profile[i] - old[i] for i in range(len(profile))],
                "utility": utility(profile, requirement),
            }
        )
    return {"profile": profile, "utility": utility(profile, requirement), "history": history}


def run_read_demo(seed: int = 1, steps: int = 40) -> Dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)

    requirement = [0.05, 0.40, 0.45, 0.10]
    initial_profile = [0.25, 0.24, 0.20, 0.26]
    actions = build_candidate_actions(
        CAPABILITIES,
        requirement,
        max_support=2,
        weight_grid=[0.25, 0.50, 0.75],
        top_k=2,
        max_candidates=32,
        include_one_hot=True,
    )

    bandit_cfg = BanditConfig(
        num_models=3,
        hidden_dim=16,
        lr=5e-3,
        train_steps=8,
        batch_size=16,
        kappa=0.8,
        warmup_steps=5,
    )
    input_dim = len(CAPABILITIES) * 2 + 1 + len(CAPABILITIES)
    bandit = UCBEnsembleBandit(input_dim=input_dim, cfg=bandit_cfg, seed=seed)

    rng = random.Random(seed)
    profile = initial_profile[:]
    allocation_mass = [0.0 for _ in CAPABILITIES]
    history = []
    beta = 0.6
    lambda_cost = 0.001

    for step in range(steps):
        remaining = float(steps - step)
        context = requirement + profile + [remaining]
        action = bandit.select_action(context, actions, bandit_cfg.kappa)
        old_profile = profile
        profile = transition(profile, action, rng)
        delta_s = [profile[i] - old_profile[i] for i in range(len(profile))]
        reward = (
            sum(requirement[i] * delta_s[i] for i in range(len(requirement)))
            - beta * compute_spillover(requirement, delta_s, top_k=3)
            - lambda_cost
        )
        bandit.add(context + action, reward)
        bandit.fit()
        allocation_mass = [allocation_mass[i] + action[i] for i in range(len(action))]
        history.append(
            {
                "step": step + 1,
                "action": dict(zip(CAPABILITIES, action)),
                "reward": reward,
                "profile": dict(zip(CAPABILITIES, profile)),
                "utility": utility(profile, requirement),
            }
        )

    uniform_action = [1.0 / len(CAPABILITIES) for _ in CAPABILITIES]
    top_cap_action = [0.0 for _ in CAPABILITIES]
    top_cap_action[requirement.index(max(requirement))] = 1.0

    uniform = run_fixed_policy(uniform_action, requirement, initial_profile, steps, seed)
    single = run_fixed_policy(top_cap_action, requirement, initial_profile, steps, seed)

    allocation_share = [mass / max(1, steps) for mass in allocation_mass]
    return {
        "capabilities": CAPABILITIES,
        "requirement_vector": dict(zip(CAPABILITIES, requirement)),
        "initial_profile": dict(zip(CAPABILITIES, initial_profile)),
        "read_final_profile": dict(zip(CAPABILITIES, profile)),
        "read_allocation_share": dict(zip(CAPABILITIES, allocation_share)),
        "initial_utility": utility(initial_profile, requirement),
        "read_final_utility": utility(profile, requirement),
        "uniform_final_utility": uniform["utility"],
        "single_capability_final_utility": single["utility"],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight ReAD synthetic sanity check.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="runs/toy_read/seed1.json")
    args = parser.parse_args()

    result = run_read_demo(seed=args.seed, steps=args.steps)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("ReAD toy demo")
    print(f"initial utility: {result['initial_utility']:.4f}")
    print(f"uniform utility: {result['uniform_final_utility']:.4f}")
    print(f"single-capability utility: {result['single_capability_final_utility']:.4f}")
    print(f"ReAD utility: {result['read_final_utility']:.4f}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

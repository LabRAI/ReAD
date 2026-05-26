#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from read.generation import GeneratorConfigs, VllmBatchGenerator, parse_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability_weights", required=True)
    ap.add_argument("--capabilities", required=True, help="Comma-separated list")
    ap.add_argument("--budget_gen_tokens", type=int, required=True)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--output_manifest", required=True)
    ap.add_argument("--exp_config", default="configs/exp.yml")
    ap.add_argument("--prompt_source_config", default="configs/distill_prompt_sources.yaml")
    ap.add_argument("--eval_hashes", default="data/eval_prompt_hashes.json")
    ap.add_argument("--difficulty_calibration", default="")
    ap.add_argument("--difficulty_percentiles", default="0.33,0.66")
    ap.add_argument("--difficulty_sample_size", type=int, default=256)
    ap.add_argument("--difficulty_schedule", default="")
    ap.add_argument("--progress", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--usage_window", type=int, default=128)
    ap.add_argument("--usage_max_share", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    weights = parse_weights(args.capability_weights, capabilities)

    percentiles = tuple(float(x) for x in args.difficulty_percentiles.split(","))
    configs = GeneratorConfigs(
        exp_config=Path(args.exp_config),
        prompt_source_config=Path(args.prompt_source_config),
        eval_hashes=Path(args.eval_hashes),
        difficulty_percentiles=percentiles,
        difficulty_sample_size=args.difficulty_sample_size,
    )

    schedule = None
    if args.difficulty_schedule:
        schedule = json.loads(args.difficulty_schedule)

    generator = VllmBatchGenerator(
        capabilities=capabilities,
        teacher_model=args.teacher_model,
        seed=args.seed,
        dtype=args.dtype,
        tp=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        configs=configs,
    )
    generator.generate(
        weights=weights,
        budget_gen_tokens=args.budget_gen_tokens,
        output_jsonl=Path(args.output_jsonl),
        output_manifest=Path(args.output_manifest),
        progress=args.progress,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        usage_window=args.usage_window,
        usage_max_share=args.usage_max_share,
        difficulty_schedule=schedule,
    )


if __name__ == "__main__":
    main()

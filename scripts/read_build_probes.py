#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from read.template_library import DifficultyThresholds, HashPolicy, TemplateLibrary, UsageTracker
from read.utils import load_yaml, write_jsonl


def load_eval_hashes(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[WARN] Eval hash file not found: {path}. Proceeding without overlap filtering.")
        return {"hashes": set(), "normalize": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"hashes": set(payload), "normalize": {}}
    if isinstance(payload, dict):
        hashes = payload.get("hashes", payload.get("hash_list", []))
        return {"hashes": set(hashes), "normalize": payload.get("normalize", {})}
    return {"hashes": set(), "normalize": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capabilities", required=True)
    ap.add_argument("--num_probes", type=int, default=16)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--exp_config", default="configs/exp.yml")
    ap.add_argument("--prompt_source_config", default="configs/distill_prompt_sources.yaml")
    ap.add_argument("--eval_hashes", default="data/eval_prompt_hashes.json")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]

    exp_cfg = load_yaml(Path(args.exp_config))
    prompt_cfg = load_yaml(Path(args.prompt_source_config))

    eval_hash_payload = load_eval_hashes(Path(args.eval_hashes))
    hash_policy = HashPolicy(
        eval_hashes=eval_hash_payload["hashes"],
        normalize_rules=prompt_cfg.get("hash_policy", {}).get("normalize", {}),
    )

    templates_by_cap: Dict[str, List[Any]] = {}
    thresholds: Dict[str, DifficultyThresholds] = {}
    for cap in capabilities:
        cap_cfg = prompt_cfg.get(cap)
        if not cap_cfg:
            raise ValueError(f"Capability missing in prompt config: {cap}")
        template_path = Path(cap_cfg["template_path"])
        templates = TemplateLibrary.from_jsonl(cap, template_path)
        templates_by_cap[cap] = templates
        thresholds[cap] = TemplateLibrary.calibrate_thresholds(templates, args.seed, 128, (0.33, 0.66))

    usage_tracker = UsageTracker(window_size=0, max_share=1.0)
    library = TemplateLibrary(templates_by_cap, thresholds, hash_policy, usage_tracker)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.teacher_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    rows: List[Dict[str, Any]] = []
    for cap in capabilities:
        for _ in range(args.num_probes):
            tmpl, prompt, slot_values, diff_label, prompt_hash = library.sample_prompt(
                cap, "medium", rng.randint(1, 1_000_000)
            )
            params = SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else -1,
                seed=args.seed,
            )
            outputs = llm.generate([prompt], params)
            response = outputs[0].outputs[0].text
            rows.append(
                {
                    "capability": cap,
                    "prompt": prompt,
                    "response": response,
                    "template_id": tmpl.template_id,
                    "difficulty": diff_label,
                    "prompt_hash": prompt_hash,
                }
            )

    write_jsonl(Path(args.output_jsonl), rows)


if __name__ == "__main__":
    main()

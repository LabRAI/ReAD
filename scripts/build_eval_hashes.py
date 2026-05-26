#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("pyyaml is required: pip install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_text(text: str, rules: Dict[str, Any]) -> str:
    out = text
    if rules.get("strip", True):
        out = out.strip()
    if rules.get("collapse_whitespace", True):
        out = re.sub(r"\s+", " ", out)
    if rules.get("lowercase", False):
        out = out.lower()
    return out


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_lm_eval_prompts(task_names: List[str]) -> Dict[str, List[str]]:
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        task_dict = tm.get_task_dict(task_names)
    except Exception:
        try:
            from lm_eval.tasks import get_task_dict
            task_dict = get_task_dict(task_names)
        except Exception as exc:
            raise RuntimeError("Failed to import lm_eval task loader") from exc

    prompts: Dict[str, List[str]] = {}
    for name, task in task_dict.items():
        docs = None
        if hasattr(task, "validation_docs"):
            docs = task.validation_docs()
        elif hasattr(task, "test_docs"):
            docs = task.test_docs()
        elif hasattr(task, "training_docs"):
            docs = task.training_docs()
        if docs is None:
            continue

        items: List[str] = []
        for doc in docs:
            try:
                text = task.doc_to_text(doc)
            except Exception:
                continue
            if hasattr(task, "doc_to_choice"):
                try:
                    choices = task.doc_to_choice(doc)
                    if choices:
                        text = text + "\n" + "\n".join([str(c) for c in choices])
                except Exception:
                    pass
            items.append(text)
        prompts[name] = items
    return prompts


def collect_humaneval_prompts(path: Path) -> List[str]:
    prompts = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if prompt:
                prompts.append(prompt)
    return prompts


def extract_text_fields(obj: Any, keys: List[str]) -> List[str]:
    values = []
    if isinstance(obj, dict):
        for key in keys:
            val = obj.get(key)
            if isinstance(val, str):
                values.append(val)
            elif isinstance(val, list):
                values.extend([v for v in val if isinstance(v, str)])
    return values


def collect_prompts_from_json(path: Path) -> List[str]:
    prompts: List[str] = []
    if path.suffix.lower() in {".jsonl"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.extend(extract_text_fields(obj, ["prompt", "input", "instruction", "query", "question"]))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                prompts.extend(extract_text_fields(obj, ["prompt", "input", "instruction", "query", "question"]))
        elif isinstance(data, dict):
            for _, obj in data.items():
                if isinstance(obj, (dict, list)):
                    prompts.extend(extract_text_fields(obj, ["prompt", "input", "instruction", "query", "question"]))
    return prompts


def collect_bfcl_prompts(extra_paths: List[Path]) -> List[str]:
    prompts: List[str] = []
    for p in extra_paths:
        if p.is_dir():
            for file in p.rglob("*.json*"):
                prompts.extend(collect_prompts_from_json(file))
        elif p.is_file():
            prompts.extend(collect_prompts_from_json(p))
    return prompts


def find_bfcl_data_dir() -> Optional[Path]:
    try:
        import bfcl_eval
    except Exception:
        return None
    base = Path(bfcl_eval.__file__).resolve().parent
    data_dir = base / "data"
    if data_dir.exists():
        return data_dir
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_config", default="configs/eval_config.json")
    ap.add_argument("--distill_prompt_config", default="configs/distill_prompt_sources.yaml")
    ap.add_argument("--output", default="data/eval_prompt_hashes.json")
    ap.add_argument("--humaneval_path", default="data/humaneval/HumanEval.jsonl")
    ap.add_argument("--bfcl_prompt_paths", nargs="*", default=[])
    ap.add_argument("--allow_missing_bfcl", action="store_true")
    args = ap.parse_args()

    eval_cfg = json.loads(Path(args.eval_config).read_text(encoding="utf-8"))
    distill_cfg = load_yaml(Path(args.distill_prompt_config))
    norm_rules = distill_cfg.get("hash_policy", {}).get("normalize", {})

    task_names = []
    for group in eval_cfg["lm_eval"]["tasks"].values():
        task_names.extend(group)

    lm_prompts = collect_lm_eval_prompts(task_names)
    missing_tasks = [t for t in task_names if not lm_prompts.get(t)]
    if missing_tasks:
        raise RuntimeError(f"Missing lm_eval prompts for tasks: {missing_tasks}")

    humaneval_prompts = collect_humaneval_prompts(Path(args.humaneval_path))
    if not humaneval_prompts:
        raise RuntimeError("HumanEval prompts not found. Check humaneval_path.")

    bfcl_paths = [Path(p) for p in args.bfcl_prompt_paths]
    if not bfcl_paths:
        auto = find_bfcl_data_dir()
        if auto:
            bfcl_paths = [auto]
    bfcl_prompts = collect_bfcl_prompts(bfcl_paths)

    if not bfcl_prompts and not args.allow_missing_bfcl:
        raise RuntimeError(
            "BFCL prompts not found. Provide --bfcl_prompt_paths or install bfcl_eval data."
        )

    hashes = set()
    counts = {}

    for task, prompts in lm_prompts.items():
        counts[task] = len(prompts)
        for text in prompts:
            hashes.add(sha256_hex(normalize_text(text, norm_rules)))

    counts["humaneval"] = len(humaneval_prompts)
    for text in humaneval_prompts:
        hashes.add(sha256_hex(normalize_text(text, norm_rules)))

    counts["bfcl"] = len(bfcl_prompts)
    for text in bfcl_prompts:
        hashes.add(sha256_hex(normalize_text(text, norm_rules)))

    payload = {
        "hash_method": "sha256",
        "normalize": norm_rules,
        "counts": counts,
        "tasks": task_names,
        "bfcl_prompt_paths": [str(p) for p in bfcl_paths],
        "hashes": sorted(hashes),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_path} with {len(hashes)} hashes")


if __name__ == "__main__":
    main()

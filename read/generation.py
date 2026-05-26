from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .template_library import DifficultyThresholds, HashPolicy, TemplateLibrary, UsageTracker
from .utils import load_yaml, write_jsonl


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


def parse_weights(raw: str, capabilities: List[str]) -> List[float]:
    if ":" in raw:
        weights = {k: 0.0 for k in capabilities}
        for part in raw.split(","):
            if not part.strip():
                continue
            name, val = part.split(":", 1)
            weights[name.strip()] = float(val)
        vec = [weights.get(c, 0.0) for c in capabilities]
    else:
        vec = [float(x) for x in raw.split(",") if x.strip()]
    if len(vec) != len(capabilities):
        raise ValueError("Capability weights length mismatch")
    total = sum(vec)
    if total <= 0:
        raise ValueError("Capability weights must sum to > 0")
    return [v / total for v in vec]


def choose_difficulty(progress: float, schedule: List[Dict[str, Any]], rng: random.Random) -> str:
    for entry in schedule:
        if progress <= float(entry.get("until", 1.0)):
            probs = entry.get("probs", {})
            labels = list(probs.keys())
            weights = [float(probs[l]) for l in labels]
            total = sum(weights)
            if total <= 0:
                return "medium"
            pick = rng.random() * total
            cum = 0.0
            for label, w in zip(labels, weights):
                cum += w
                if pick <= cum:
                    return label
    return "medium"


DEFAULT_DIFFICULTY_SCHEDULE = [
    {"until": 0.3, "probs": {"easy": 0.1, "medium": 0.8, "hard": 0.1}},
    {"until": 0.7, "probs": {"easy": 0.1, "medium": 0.5, "hard": 0.4}},
    {"until": 1.0, "probs": {"easy": 0.05, "medium": 0.2, "hard": 0.75}},
]


@dataclass
class GeneratorConfigs:
    exp_config: Path
    prompt_source_config: Path
    eval_hashes: Path
    difficulty_percentiles: Tuple[float, float] = (0.33, 0.66)
    difficulty_sample_size: int = 256


class VllmBatchGenerator:
    def __init__(
        self,
        capabilities: List[str],
        teacher_model: str,
        seed: int,
        dtype: str,
        tp: int,
        max_model_len: int,
        trust_remote_code: bool,
        configs: GeneratorConfigs,
    ) -> None:
        self.capabilities = capabilities
        self.teacher_model = teacher_model
        self.seed = seed
        self.prompt_source_config_path = configs.prompt_source_config
        self.exp_cfg = load_yaml(configs.exp_config)
        self.prompt_cfg = load_yaml(configs.prompt_source_config)

        eval_hash_payload = load_eval_hashes(configs.eval_hashes)
        self.hash_policy = HashPolicy(
            eval_hashes=eval_hash_payload["hashes"],
            normalize_rules=self.prompt_cfg.get("hash_policy", {}).get("normalize", {}),
        )

        self.templates_by_cap: Dict[str, List[Any]] = {}
        self.thresholds: Dict[str, DifficultyThresholds] = {}
        for cap in capabilities:
            cap_cfg = self.prompt_cfg.get(cap)
            if not cap_cfg:
                raise ValueError(f"Capability missing in prompt config: {cap}")
            template_path = Path(cap_cfg["template_path"])
            templates = TemplateLibrary.from_jsonl(cap, template_path)
            self.templates_by_cap[cap] = templates
            self.thresholds[cap] = TemplateLibrary.calibrate_thresholds(
                templates,
                seed,
                configs.difficulty_sample_size,
                configs.difficulty_percentiles,
            )

        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=teacher_model,
            dtype=dtype,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model, trust_remote_code=trust_remote_code
        )

    def shutdown(self) -> None:
        if hasattr(self.llm, "shutdown"):
            self.llm.shutdown()

    def generate(
        self,
        weights: List[float],
        budget_gen_tokens: int,
        output_jsonl: Path,
        output_manifest: Path,
        progress: float,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        batch_size: int,
        usage_window: int = 128,
        usage_max_share: float = 0.25,
        difficulty_schedule: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        schedule = difficulty_schedule or DEFAULT_DIFFICULTY_SCHEDULE

        if max_new_tokens <= 0:
            max_map = self.exp_cfg["token_counting"]["gen_tokens_teacher"]["max_new_tokens_by_capability"]
            max_new_tokens = int(max_map.get(self.capabilities[0], 256))

        rng = random.Random(self.seed)
        usage_tracker = UsageTracker(window_size=usage_window, max_share=usage_max_share)
        library = TemplateLibrary(
            self.templates_by_cap, self.thresholds, self.hash_policy, usage_tracker
        )

        rows: List[Dict[str, Any]] = []
        gen_tokens = 0
        total_items = 0

        while gen_tokens < budget_gen_tokens:
            prompts: List[str] = []
            metas: List[Dict[str, Any]] = []
            remaining = budget_gen_tokens - gen_tokens
            for _ in range(batch_size):
                if remaining <= 0:
                    break
                cap = rng.choices(self.capabilities, weights=weights, k=1)[0]
                difficulty = choose_difficulty(progress, schedule, rng)
                tmpl, prompt, slot_values, diff_label, prompt_hash = library.sample_prompt(
                    cap, difficulty, rng.randint(1, 1_000_000)
                )
                prompts.append(prompt)
                metas.append(
                    {
                        "prompt": prompt,
                        "capability_target": cap,
                        "template_id": tmpl.template_id,
                        "difficulty": diff_label,
                        "slot_values": slot_values,
                        "prompt_hash": prompt_hash,
                    }
                )
                remaining -= 1
            if not prompts:
                break

            params = self.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else -1,
                seed=self.seed,
            )
            outputs = self.llm.generate(prompts, params)
            for meta, out in zip(metas, outputs):
                response = out.outputs[0].text
                resp_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
                if gen_tokens + resp_tokens > budget_gen_tokens:
                    continue
                row = {
                    "prompt": meta.pop("prompt"),
                    "response": response,
                    **meta,
                }
                rows.append(row)
                gen_tokens += resp_tokens
                total_items += 1

        write_jsonl(Path(output_jsonl), rows)
        manifest = {
            "teacher_model": self.teacher_model,
            "capabilities": self.capabilities,
            "capability_weights": weights,
            "budget_gen_tokens": budget_gen_tokens,
            "gen_tokens": gen_tokens,
            "items": total_items,
            "difficulty_schedule": schedule,
            "prompt_source_config": str(self.prompt_source_config_path),
        }
        Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(output_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

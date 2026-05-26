from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import normalize_text, read_jsonl, sha256_hex


@dataclass
class SlotSpec:
    name: str
    type: str = "str"
    values: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    fmt: Optional[str] = None


@dataclass
class Template:
    template_id: str
    capability: str
    prompt_template: str
    slots: List[SlotSpec] = field(default_factory=list)
    format_constraints: List[str] = field(default_factory=list)
    difficulty_fn: Optional[str] = None
    difficulty_features: Dict[str, Any] = field(default_factory=dict)
    seed_mode: str = "template"
    meta: Dict[str, Any] = field(default_factory=dict)

    def fill(self, seed: int) -> Tuple[str, Dict[str, Any]]:
        rng = random.Random(seed)
        slot_values: Dict[str, Any] = {}
        for slot in self.slots:
            if slot.type == "int":
                lo = int(slot.min or 0)
                hi = int(slot.max or lo)
                val = rng.randint(lo, hi)
            elif slot.type == "float":
                lo = float(slot.min or 0.0)
                hi = float(slot.max or lo)
                val = rng.random() * (hi - lo) + lo
                if slot.fmt:
                    val = format(val, slot.fmt)
            elif slot.type == "choice":
                if not slot.values:
                    val = ""
                else:
                    val = rng.choice(slot.values)
            else:
                if slot.values:
                    val = rng.choice(slot.values)
                else:
                    val = f"{slot.name}_{seed}"
            if slot.fmt and slot.type != "float":
                try:
                    val = format(val, slot.fmt)
                except Exception:
                    pass
            slot_values[slot.name] = val
        try:
            prompt = self.prompt_template.format(**slot_values)
        except Exception:
            prompt = self.prompt_template
        return prompt, slot_values

    def difficulty_score(self, prompt: str, slot_values: Dict[str, Any]) -> float:
        fn = (self.difficulty_fn or "").lower().strip()
        if fn == "num_slots":
            return float(len(self.slots))
        if fn == "length":
            return float(len(prompt))
        if fn == "num_lines":
            return float(prompt.count("\n") + 1)
        if fn == "constraints":
            return float(len(self.format_constraints))

        score = float(len(prompt))
        score += 10.0 * float(len(self.slots))
        score += 5.0 * float(len(self.format_constraints))
        if isinstance(self.difficulty_features, dict):
            for val in self.difficulty_features.values():
                try:
                    score += float(val)
                except Exception:
                    continue
        return score


@dataclass
class DifficultyThresholds:
    easy: float
    hard: float

    def label(self, score: float) -> str:
        if score <= self.easy:
            return "easy"
        if score >= self.hard:
            return "hard"
        return "medium"


@dataclass
class HashPolicy:
    eval_hashes: Optional[set] = None
    normalize_rules: Dict[str, Any] = field(default_factory=dict)

    def hash_prompt(self, prompt: str) -> str:
        normalized = normalize_text(prompt, self.normalize_rules)
        return sha256_hex(normalized)

    def is_blocked(self, prompt: str) -> bool:
        if not self.eval_hashes:
            return False
        return self.hash_prompt(prompt) in self.eval_hashes


@dataclass
class UsageTracker:
    window_size: int
    max_share: float
    history: Dict[str, List[str]] = field(default_factory=dict)

    def allow(self, capability: str, template_id: str) -> bool:
        if self.window_size <= 0:
            return True
        recent = self.history.get(capability, [])
        if len(recent) < max(1, self.window_size):
            return True
        count = sum(1 for t in recent if t == template_id)
        return count / max(1, len(recent)) < self.max_share

    def record(self, capability: str, template_id: str) -> None:
        if self.window_size <= 0:
            return
        recent = self.history.setdefault(capability, [])
        recent.append(template_id)
        if len(recent) > self.window_size:
            del recent[: len(recent) - self.window_size]


class TemplateLibrary:
    def __init__(
        self,
        templates_by_capability: Dict[str, List[Template]],
        difficulty_thresholds: Dict[str, DifficultyThresholds],
        hash_policy: HashPolicy,
        usage_tracker: UsageTracker,
    ) -> None:
        self.templates_by_capability = templates_by_capability
        self.difficulty_thresholds = difficulty_thresholds
        self.hash_policy = hash_policy
        self.usage_tracker = usage_tracker

    @staticmethod
    def from_jsonl(
        capability: str,
        path: Path,
    ) -> List[Template]:
        templates: List[Template] = []
        bad_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    if bad_lines <= 3:
                        print(f"[WARN] Skipping invalid JSONL line {line_no} in {path}")
                    continue
                template_id = row.get("template_id") or row.get("id") or ""
                if not template_id:
                    template_id = sha256_hex(json.dumps(row, sort_keys=True))[:12]
                cap = row.get("capability") or row.get("meta", {}).get("capability") or capability
                prompt_template = row.get("template") or row.get("prompt") or ""
                slots_raw = row.get("slots") or []
                slots: List[SlotSpec] = []
                for s in slots_raw:
                    slots.append(
                        SlotSpec(
                            name=str(s.get("name")),
                            type=str(s.get("type", "str")),
                            values=s.get("values"),
                            min=s.get("min"),
                            max=s.get("max"),
                            fmt=s.get("format"),
                        )
                    )
                templates.append(
                    Template(
                        template_id=template_id,
                        capability=cap,
                        prompt_template=prompt_template,
                        slots=slots,
                        format_constraints=row.get("format_constraints") or [],
                        difficulty_fn=row.get("difficulty_fn") or row.get("meta", {}).get("difficulty_fn"),
                        difficulty_features=row.get("difficulty_features") or row.get("meta", {}).get("difficulty_features") or {},
                        seed_mode=row.get("seed_mode") or "template",
                        meta=row.get("meta") or {},
                    )
                )
        if bad_lines > 0:
            print(f"[WARN] Skipped {bad_lines} invalid JSONL lines in {path}")
        if not templates:
            raise ValueError(f"No templates found in {path}")
        return templates

    @staticmethod
    def calibrate_thresholds(
        templates: List[Template],
        rng_seed: int,
        sample_size: int,
        percentiles: Tuple[float, float],
    ) -> DifficultyThresholds:
        rng = random.Random(rng_seed)
        scores: List[float] = []
        for i in range(sample_size):
            tmpl = rng.choice(templates)
            seed = rng.randint(1, 1_000_000)
            prompt, slot_values = tmpl.fill(seed)
            scores.append(tmpl.difficulty_score(prompt, slot_values))
        if not scores:
            return DifficultyThresholds(easy=0.0, hard=0.0)
        scores.sort()
        lo = scores[int(percentiles[0] * (len(scores) - 1))]
        hi = scores[int(percentiles[1] * (len(scores) - 1))]
        return DifficultyThresholds(easy=lo, hard=hi)

    def sample_prompt(
        self,
        capability: str,
        difficulty: Optional[str],
        seed: int,
        max_attempts: int = 50,
    ) -> Tuple[Template, str, Dict[str, Any], str, str]:
        templates = self.templates_by_capability.get(capability, [])
        if not templates:
            raise ValueError(f"No templates found for capability: {capability}")
        rng = random.Random(seed)

        for _ in range(max_attempts):
            tmpl = rng.choice(templates)
            if tmpl.seed_mode == "template":
                local_seed = int(sha256_hex(f"{tmpl.template_id}:{seed}")[:16], 16) % 1_000_000_007
            else:
                local_seed = rng.randint(1, 1_000_000_000)
            prompt, slot_values = tmpl.fill(local_seed)
            if self.hash_policy.is_blocked(prompt):
                continue
            score = tmpl.difficulty_score(prompt, slot_values)
            thresholds = self.difficulty_thresholds.get(
                capability, DifficultyThresholds(easy=0.0, hard=0.0)
            )
            label = thresholds.label(score)
            if difficulty and label != difficulty:
                continue
            if not self.usage_tracker.allow(capability, tmpl.template_id):
                continue
            self.usage_tracker.record(capability, tmpl.template_id)
            prompt_hash = self.hash_policy.hash_prompt(prompt)
            return tmpl, prompt, slot_values, label, prompt_hash

        tmpl = templates[0]
        prompt, slot_values = tmpl.fill(seed)
        prompt_hash = self.hash_policy.hash_prompt(prompt)
        return tmpl, prompt, slot_values, "medium", prompt_hash

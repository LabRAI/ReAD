from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import load_yaml


@dataclass
class TaskSplit:
    source: str
    path: Optional[str] = None
    dataset: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    format: Optional[str] = None
    limit: Optional[int] = None
    seed: int = 1234


@dataclass
class TaskCard:
    name: str
    capability: str
    description: str
    input_format: str
    output_format: str
    exemplars: List[Dict[str, Any]] = field(default_factory=list)
    dev: Optional[TaskSplit] = None
    test: Optional[TaskSplit] = None
    eval_type: str = "exact_match"
    requirement_vector: Optional[Dict[str, float]] = None
    requirement_model: Optional[str] = None

    def to_text(self) -> str:
        parts = [f"Task: {self.name}", f"Capability: {self.capability}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.input_format:
            parts.append(f"Input: {self.input_format}")
        if self.output_format:
            parts.append(f"Output: {self.output_format}")
        if self.exemplars:
            for ex in self.exemplars[:5]:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                parts.append(f"Example input: {inp}")
                parts.append(f"Example output: {out}")
        return "\n".join(parts)


def _parse_split(raw: Optional[Dict[str, Any]]) -> Optional[TaskSplit]:
    if not raw:
        return None
    return TaskSplit(
        source=str(raw.get("source", "local")),
        path=raw.get("path"),
        dataset=raw.get("dataset"),
        subset=raw.get("subset"),
        split=raw.get("split"),
        format=raw.get("format"),
        limit=raw.get("limit"),
        seed=int(raw.get("seed", 1234)),
    )


def load_task_cards(path: Path) -> Dict[str, TaskCard]:
    cfg = load_yaml(path)
    tasks: Dict[str, TaskCard] = {}
    raw_tasks = cfg.get("tasks", {})
    for name, raw in raw_tasks.items():
        card = TaskCard(
            name=name,
            capability=str(raw.get("capability", "")),
            description=str(raw.get("description", "")),
            input_format=str(raw.get("input_format", "")),
            output_format=str(raw.get("output_format", "")),
            exemplars=raw.get("exemplars", []) or [],
            dev=_parse_split(raw.get("dev")),
            test=_parse_split(raw.get("test")),
            eval_type=str(raw.get("eval_type", "exact_match")),
            requirement_vector=raw.get("requirement_vector"),
            requirement_model=raw.get("requirement_model"),
        )
        tasks[name] = card
    return tasks


def load_task_card(path: Path, name: str) -> TaskCard:
    cards = load_task_cards(path)
    if name not in cards:
        raise KeyError(f"Task card '{name}' not found in {path}")
    return cards[name]

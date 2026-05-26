from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .task_cards import TaskCard


@dataclass
class RequirementModelConfig:
    encoder_model: str
    hidden_dim: int
    num_capabilities: int


class RequirementIdentifier(nn.Module):
    def __init__(self, encoder_model: str, hidden_dim: int, num_capabilities: int) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder_name = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_capabilities),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.proj[0].weight.device) for k, v in enc.items()}
        outputs = self.encoder(**enc)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.proj(pooled)
        return torch.softmax(logits, dim=-1)


def load_requirement_model(path: Path) -> RequirementIdentifier:
    payload = torch.load(path, map_location="cpu")
    cfg = payload["config"]
    model = RequirementIdentifier(
        encoder_model=cfg["encoder_model"],
        hidden_dim=int(cfg["hidden_dim"]),
        num_capabilities=int(cfg["num_capabilities"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model


def save_requirement_model(path: Path, model: RequirementIdentifier, config: RequirementModelConfig) -> None:
    payload = {
        "config": {
            "encoder_model": config.encoder_model,
            "hidden_dim": config.hidden_dim,
            "num_capabilities": config.num_capabilities,
        },
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def get_requirement_vector(
    task_card: TaskCard,
    capabilities: List[str],
    requirement_model_path: Optional[Path] = None,
) -> List[float]:
    if task_card.requirement_vector:
        vec = [float(task_card.requirement_vector.get(c, 0.0)) for c in capabilities]
        total = sum(vec)
        if total <= 0:
            raise ValueError("Requirement vector must have positive mass.")
        return [v / total for v in vec]

    model_path = requirement_model_path or (
        Path(task_card.requirement_model) if task_card.requirement_model else None
    )
    if model_path and model_path.exists():
        model = load_requirement_model(model_path)
        model.eval()
        with torch.no_grad():
            probs = model([task_card.to_text()])[0].tolist()
        total = sum(probs)
        if total <= 0:
            raise ValueError("Requirement model produced non-positive weights.")
        return [p / total for p in probs]

    # Fallback: one-hot on declared capability with small smoothing.
    eps = 1e-3
    vec = [eps for _ in capabilities]
    if task_card.capability in capabilities:
        idx = capabilities.index(task_card.capability)
        vec[idx] = 1.0
    total = sum(vec)
    return [v / total for v in vec]

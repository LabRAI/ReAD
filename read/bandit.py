from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


class RewardRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class BanditConfig:
    num_models: int = 5
    hidden_dim: int = 64
    lr: float = 1e-3
    train_steps: int = 200
    batch_size: int = 64
    kappa: float = 1.0
    warmup_steps: int = 5


class UCBEnsembleBandit:
    def __init__(self, input_dim: int, cfg: BanditConfig, seed: int = 1) -> None:
        self.cfg = cfg
        self.models = [RewardRegressor(input_dim, cfg.hidden_dim) for _ in range(cfg.num_models)]
        self.opts = [torch.optim.Adam(m.parameters(), lr=cfg.lr) for m in self.models]
        self.history_x: List[List[float]] = []
        self.history_y: List[float] = []
        self.rng = random.Random(seed)

    def add(self, x: List[float], reward: float) -> None:
        self.history_x.append(x)
        self.history_y.append(float(reward))

    def _sample_indices(self, n: int) -> List[int]:
        return [self.rng.randrange(n) for _ in range(n)]

    def fit(self) -> None:
        if not self.history_x:
            return
        x_all = torch.tensor(self.history_x, dtype=torch.float32)
        y_all = torch.tensor(self.history_y, dtype=torch.float32)
        n = x_all.shape[0]
        for model, opt in zip(self.models, self.opts):
            model.train()
            idx = self._sample_indices(n)
            x = x_all[idx]
            y = y_all[idx]
            for _ in range(self.cfg.train_steps):
                batch_idx = torch.randint(0, n, (min(self.cfg.batch_size, n),))
                xb = x[batch_idx]
                yb = y[batch_idx]
                pred = model(xb)
                loss = torch.mean((pred - yb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, x: List[float]) -> Tuple[float, float]:
        if not self.history_x:
            return 0.0, 0.0
        x_t = torch.tensor([x], dtype=torch.float32)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(float(model(x_t).item()))
        mean = sum(preds) / len(preds)
        if len(preds) <= 1:
            return mean, 0.0
        var = sum((p - mean) ** 2 for p in preds) / (len(preds) - 1)
        return mean, var ** 0.5

    def select_action(self, context: List[float], actions: List[List[float]], kappa: float) -> List[float]:
        if len(self.history_x) < self.cfg.warmup_steps:
            return self.rng.choice(actions)
        best = actions[0]
        best_score = -1e9
        for a in actions:
            x = context + a
            mean, std = self.predict(x)
            score = mean + kappa * std
            if score > best_score:
                best_score = score
                best = a
        return best

from __future__ import annotations

from typing import List


def build_candidate_actions(
    capabilities: List[str],
    r_tau: List[float],
    max_support: int,
    weight_grid: List[float],
    top_k: int,
    max_candidates: int,
    include_one_hot: bool,
) -> List[List[float]]:
    ranked = sorted(range(len(capabilities)), key=lambda i: r_tau[i], reverse=True)
    pool = ranked[: max(1, top_k)]
    actions: List[List[float]] = []

    if include_one_hot:
        for idx in pool:
            action = [0.0] * len(capabilities)
            action[idx] = 1.0
            actions.append(action)

    if max_support >= 2:
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                for weight in weight_grid:
                    if weight <= 0.0 or weight >= 1.0:
                        continue
                    action = [0.0] * len(capabilities)
                    action[pool[i]] = weight
                    action[pool[j]] = 1.0 - weight
                    actions.append(action)

    unique: List[List[float]] = []
    seen = set()
    for action in actions:
        total = sum(action)
        if total <= 0.0:
            continue
        action = [value / total for value in action]
        key = tuple(round(value, 6) for value in action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
        if len(unique) >= max_candidates:
            break

    if not unique:
        unique.append([1.0 / len(capabilities) for _ in capabilities])
    return unique


def compute_spillover(r_tau: List[float], delta_s: List[float], top_k: int) -> float:
    required = sorted(range(len(r_tau)), key=lambda i: r_tau[i], reverse=True)[: max(1, top_k)]
    spillover = 0.0
    for idx in required:
        if delta_s[idx] < 0.0:
            spillover += r_tau[idx] * (-delta_s[idx])
    return spillover

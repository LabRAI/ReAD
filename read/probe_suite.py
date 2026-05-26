from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .eval_utils import batched_logprob_from_ids, encode_prompt_response
from .utils import read_jsonl, write_jsonl


@dataclass
class ProbeExample:
    prompt: str
    response: str
    prompt_hash: Optional[str] = None


@dataclass
class ProbeSuite:
    probes: Dict[str, List[ProbeExample]]
    _token_cache: Dict[Tuple[str, int], Dict[str, List[Tuple[List[int], List[int]]]]] = field(
        default_factory=dict, init=False
    )

    @staticmethod
    def load(path: Path) -> "ProbeSuite":
        probes: Dict[str, List[ProbeExample]] = {}
        for row in read_jsonl(path):
            cap = row.get("capability")
            if not cap:
                continue
            probes.setdefault(cap, []).append(
                ProbeExample(
                    prompt=row.get("prompt", ""),
                    response=row.get("response", ""),
                    prompt_hash=row.get("prompt_hash"),
                )
            )
        return ProbeSuite(probes=probes)

    def save(self, path: Path) -> None:
        rows = []
        for cap, items in self.probes.items():
            for ex in items:
                rows.append(
                    {
                        "capability": cap,
                        "prompt": ex.prompt,
                        "response": ex.response,
                        "prompt_hash": ex.prompt_hash,
                    }
                )
        write_jsonl(path, rows)

    def compute_profile(
        self,
        model,
        tokenizer,
        capabilities: List[str],
        max_length: int,
        device: Optional[torch.device] = None,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        model.eval()
        scores: Dict[str, float] = {}
        device = device or next(model.parameters()).device
        cache_key = (getattr(tokenizer, "name_or_path", "tokenizer"), max_length)
        cached = self._token_cache.get(cache_key)
        if cached is None:
            cached = {}
            for cap, items in self.probes.items():
                pairs: List[Tuple[List[int], List[int]]] = []
                for ex in items:
                    encoded = encode_prompt_response(ex.prompt, ex.response, tokenizer, max_length)
                    if encoded is None:
                        continue
                    pairs.append(encoded)
                cached[cap] = pairs
            self._token_cache[cache_key] = cached

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for cap in capabilities:
            pairs = cached.get(cap, [])
            if not pairs:
                scores[cap] = 0.0
                continue
            scores_list = batched_logprob_from_ids(
                model,
                pairs,
                device=device,
                batch_size=batch_size,
                pad_token_id=pad_id,
            )
            if not scores_list:
                scores[cap] = 0.0
            else:
                scores[cap] = sum(scores_list) / len(scores_list)
        return scores

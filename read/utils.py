import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("pyyaml is required: pip install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_model_name(model: str) -> str:
    import re
    m = model.strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", m)
    return s[:120]


def normalize_text(text: str, rules: Dict[str, Any]) -> str:
    out = text
    if rules.get("strip", True):
        out = out.strip()
    if rules.get("collapse_whitespace", True):
        import re
        out = re.sub(r"\s+", " ", out)
    if rules.get("lowercase", False):
        out = out.lower()
    return out


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

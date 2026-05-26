#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from read.requirements import RequirementIdentifier, RequirementModelConfig, save_requirement_model
from read.utils import read_jsonl, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="Each row: {task_text, delta_s, delta_u}")
    ap.add_argument("--encoder_model", default="distilroberta-base")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_capabilities", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--encode_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(Path(args.train_jsonl))
    if not rows:
        raise RuntimeError("No training rows found.")

    model = RequirementIdentifier(args.encoder_model, args.hidden_dim, args.num_capabilities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.encoder.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    texts = [r["task_text"] for r in rows]
    deltas = torch.tensor([r["delta_s"] for r in rows], dtype=torch.float32, device=device)
    utilities = torch.tensor([r["delta_u"] for r in rows], dtype=torch.float32, device=device)

    with torch.inference_mode():
        embeds = []
        for i in range(0, len(texts), max(1, args.encode_batch_size)):
            batch = texts[i : i + max(1, args.encode_batch_size)]
            enc = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model.encoder(**enc)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeds.append(pooled)
        feats = torch.cat(embeds, dim=0)

    for epoch in range(args.epochs):
        model.proj.train()
        preds = torch.softmax(model.proj(feats), dim=-1)
        proj = (preds * deltas).sum(dim=-1)
        mse = ((utilities - proj) ** 2).mean()
        entropy = -(preds * (preds.clamp(min=1e-8).log())).sum(dim=-1).mean()
        loss = mse + args.entropy_weight * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[EPOCH {epoch+1}] loss={loss.item():.4f} mse={mse.item():.4f} ent={entropy.item():.4f}")

    cfg = RequirementModelConfig(
        encoder_model=args.encoder_model,
        hidden_dim=args.hidden_dim,
        num_capabilities=args.num_capabilities,
    )
    save_requirement_model(Path(args.output), model, cfg)
    print(f"[OK] Saved requirement model to {args.output}")


if __name__ == "__main__":
    main()

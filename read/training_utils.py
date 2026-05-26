from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .utils import read_jsonl


@dataclass
class TokenCountingRules:
    count_prompt_tokens: bool = True
    count_label_tokens: bool = True
    count_padding_tokens: bool = False


class JsonlDataset:
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        mask_prompt: bool,
        include_rejected: bool,
    ) -> None:
        self.examples = []
        skipped = 0
        for row in rows:
            if not include_rejected and row.get("accepted") is False:
                continue
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if not prompt or not response:
                skipped += 1
                continue
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            total_ids = prompt_ids + response_ids
            if len(total_ids) > max_length:
                max_resp = max_length - len(prompt_ids)
                if max_resp <= 0:
                    skipped += 1
                    continue
                response_ids = response_ids[:max_resp]
                total_ids = prompt_ids + response_ids
            labels = total_ids[:]
            if mask_prompt:
                labels = [-100] * len(prompt_ids) + response_ids
            self.examples.append(
                {
                    "input_ids": total_ids,
                    "labels": labels,
                    "prompt_tokens": len(prompt_ids),
                    "label_tokens": len(response_ids),
                }
            )
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    prompt_tokens = []
    label_tokens = []

    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        prompt_tokens.append(item["prompt_tokens"])
        label_tokens.append(item["label_tokens"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
        "label_tokens": torch.tensor(label_tokens, dtype=torch.long),
    }


def train_for_tokens(
    model,
    tokenizer,
    train_jsonl: Path,
    optimizer,
    rules: TokenCountingRules,
    budget_tokens: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    device: torch.device,
    log_every_steps: int = 50,
) -> Tuple[int, int, int, int]:
    rows = read_jsonl(train_jsonl)
    dataset = JsonlDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_prompt=True,
        include_rejected=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training examples after filtering.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id))

    train_tokens = 0
    train_prompt_tokens = 0
    train_label_tokens = 0
    step = 0

    model.train()
    while train_tokens < budget_tokens:
        for batch in dataloader:
            prompt_sum = int(batch["prompt_tokens"].sum().item())
            label_sum = int(batch["label_tokens"].sum().item())
            tokens_in_batch = 0
            if rules.count_prompt_tokens:
                tokens_in_batch += prompt_sum
            if rules.count_label_tokens:
                tokens_in_batch += label_sum
            if rules.count_padding_tokens:
                tokens_in_batch = int(batch["attention_mask"].numel())

            if train_tokens + tokens_in_batch > budget_tokens:
                if step > 0 and step % gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                return train_tokens, train_prompt_tokens, train_label_tokens, step

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            train_tokens += tokens_in_batch
            train_prompt_tokens += prompt_sum
            train_label_tokens += label_sum

            if step % log_every_steps == 0:
                print(f"[TRAIN] step={step} loss={loss.item():.4f} tokens={train_tokens}/{budget_tokens}")

            if train_tokens >= budget_tokens:
                if step % gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                return train_tokens, train_prompt_tokens, train_label_tokens, step

    if step > 0 and step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return train_tokens, train_prompt_tokens, train_label_tokens, step

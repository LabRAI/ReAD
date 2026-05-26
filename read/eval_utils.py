from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .task_cards import TaskCard, TaskSplit
from .utils import read_jsonl


@dataclass
class EvalExample:
    task_id: str
    prompt: str
    answer: Optional[str] = None
    test: Optional[str] = None
    entry_point: Optional[str] = None


def _load_local(split: TaskSplit) -> List[EvalExample]:
    path = Path(split.path or "")
    if not path.exists():
        raise FileNotFoundError(f"Task split path not found: {path}")
    rows = read_jsonl(path)
    examples: List[EvalExample] = []
    fmt = split.format or ""
    if fmt == "humaneval_jsonl":
        for row in rows:
            examples.append(
                EvalExample(
                    task_id=row.get("task_id", ""),
                    prompt=row.get("prompt", ""),
                    test=row.get("test"),
                    entry_point=row.get("entry_point"),
                )
            )
    elif fmt == "gsm8k_jsonl":
        for i, row in enumerate(rows):
            examples.append(
                EvalExample(
                    task_id=row.get("id", f"gsm8k_{i}"),
                    prompt=row.get("question", ""),
                    answer=row.get("answer", ""),
                )
            )
    else:
        for i, row in enumerate(rows):
            prompt = row.get("prompt") or row.get("question") or ""
            answer = row.get("answer") or row.get("output")
            examples.append(
                EvalExample(
                    task_id=row.get("id", f"ex_{i}"),
                    prompt=prompt,
                    answer=answer,
                )
            )
    if split.limit and split.limit > 0 and len(examples) > split.limit:
        rng = torch.Generator().manual_seed(split.seed)
        idx = torch.randperm(len(examples), generator=rng)[: split.limit].tolist()
        examples = [examples[i] for i in idx]
    return examples


def load_examples(task_card: TaskCard, split: str) -> List[EvalExample]:
    spec: Optional[TaskSplit] = task_card.dev if split == "dev" else task_card.test
    if not spec:
        raise ValueError(f"Task card missing split '{split}'")
    if spec.source == "local":
        return _load_local(spec)
    if spec.source == "hf":
        try:
            import datasets  # type: ignore
        except Exception as exc:
            raise RuntimeError("datasets is required for hf sources") from exc
        ds = datasets.load_dataset(spec.dataset, spec.subset, split=spec.split)
        rows = [ds[i] for i in range(len(ds))]
        examples: List[EvalExample] = []
        for i, row in enumerate(rows):
            prompt = row.get("question") or row.get("prompt") or ""
            answer = row.get("answer") or row.get("output")
            examples.append(
                EvalExample(task_id=row.get("id", f"hf_{i}"), prompt=prompt, answer=answer)
            )
        if spec.limit and spec.limit > 0 and len(examples) > spec.limit:
            rng = torch.Generator().manual_seed(spec.seed)
            idx = torch.randperm(len(examples), generator=rng)[: spec.limit].tolist()
            examples = [examples[i] for i in idx]
        return examples
    raise ValueError(f"Unknown split source: {spec.source}")


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # GSM8K answers are often after "####"
    if "####" in text:
        return text.split("####")[-1].strip()
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return None
    return nums[-1]


def encode_prompt_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int,
) -> Optional[Tuple[List[int], List[int]]]:
    if not prompt or not response:
        return None
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    total_ids = prompt_ids + response_ids
    if len(total_ids) > max_length:
        max_resp = max_length - len(prompt_ids)
        if max_resp <= 0:
            return None
        response_ids = response_ids[:max_resp]
        total_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return total_ids, labels


def _iter_batches(items: List[Tuple[List[int], List[int]]], batch_size: int) -> Iterable[List[Tuple[List[int], List[int]]]]:
    for i in range(0, len(items), max(1, batch_size)):
        yield items[i : i + max(1, batch_size)]


def batched_logprob_from_ids(
    model,
    pairs: List[Tuple[List[int], List[int]]],
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
) -> List[float]:
    if not pairs:
        return []
    model.eval()
    scores: List[float] = []
    with torch.inference_mode():
        for batch in _iter_batches(pairs, batch_size):
            max_len = max(len(item[0]) for item in batch)
            input_ids = []
            labels = []
            attention_mask = []
            for ids, labs in batch:
                pad_len = max_len - len(ids)
                input_ids.append(ids + [pad_token_id] * pad_len)
                labels.append(labs + [-100] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)

            outputs = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                labels=labels_t,
                use_cache=False,
            )
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_t[:, 1:].contiguous()
            vocab = shift_logits.shape[-1]
            token_losses = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.shape)
            mask = shift_labels != -100
            denom = mask.sum(dim=1).clamp(min=1)
            per_example = (token_losses * mask).sum(dim=1) / denom
            scores.extend((-per_example).detach().cpu().tolist())
    return scores


def evaluate_gsm8k(
    model,
    tokenizer,
    examples: List[EvalExample],
    max_new_tokens: int,
    device: torch.device,
    batch_size: int = 1,
) -> float:
    model.eval()
    correct = 0
    total = 0
    batch_size = max(1, batch_size)
    with torch.inference_mode():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            prompts = [ex.prompt for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for ex, out, in_len in zip(batch, gen, input_lens):
                completion = tokenizer.decode(out[int(in_len) :], skip_special_tokens=True)
                pred = _extract_gsm8k_answer(completion)
                gold = _extract_gsm8k_answer(ex.answer or "")
                if pred is not None and gold is not None and pred.strip() == gold.strip():
                    correct += 1
                total += 1
    return correct / total if total else 0.0


def evaluate_logprob(
    model,
    tokenizer,
    examples: List[EvalExample],
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
) -> float:
    pairs: List[Tuple[List[int], List[int]]] = []
    for ex in examples:
        if not ex.answer:
            continue
        encoded = encode_prompt_response(ex.prompt, ex.answer, tokenizer, max_length)
        if encoded is None:
            continue
        pairs.append(encoded)
    if not pairs:
        return 0.0
    scores = batched_logprob_from_ids(
        model,
        pairs,
        device=device,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

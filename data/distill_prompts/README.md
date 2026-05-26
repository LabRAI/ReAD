# Distill Prompt Pools (No Eval Leakage)

These prompt pools are used for distillation only. They must not include any
Table-1 benchmark prompts. The generator should:
- Load prompts from the JSONL files below.
- Hash and reject any prompt that overlaps with eval prompt hashes in
  `data/eval_prompt_hashes.json`.
- Log removed counts and hash policy in the run manifest.

Format: each line is JSON with fields:
- id: unique string
- prompt: the user prompt text
- meta: optional metadata (capability, template tags, etc.)

Prompt pools:
- general_knowledge: `data/distill_prompts/general_knowledge/templates.jsonl`
- steerability: `data/distill_prompts/steerability/templates.jsonl`
- reasoning: `data/distill_prompts/reasoning/templates.jsonl`
- math: `data/distill_prompts/math/templates.jsonl`
- code: `data/distill_prompts/code/templates.jsonl`
- tool_use: `data/distill_prompts/tool_use/templates.jsonl`
- long_context: `data/distill_prompts/long_context/templates.jsonl`
- multilingual: `data/distill_prompts/multilingual/templates.jsonl`

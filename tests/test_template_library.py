import json

from read.template_library import HashPolicy, TemplateLibrary, UsageTracker


def test_template_library_loads_every_jsonl_row(tmp_path):
    path = tmp_path / "templates.jsonl"
    rows = [
        {"id": "a", "prompt": "Prompt A {x}", "slots": [{"name": "x", "type": "int", "min": 1, "max": 2}]},
        {"id": "b", "prompt": "Prompt B"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    templates = TemplateLibrary.from_jsonl("math", path)

    assert [template.template_id for template in templates] == ["a", "b"]


def test_template_sampling_is_deterministic_for_template_seed(tmp_path):
    path = tmp_path / "templates.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "slot_template",
                "prompt": "Add {a} and {b}.",
                "slots": [
                    {"name": "a", "type": "int", "min": 1, "max": 9},
                    {"name": "b", "type": "int", "min": 1, "max": 9},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    templates = TemplateLibrary.from_jsonl("math", path)
    thresholds = {"math": TemplateLibrary.calibrate_thresholds(templates, 1, 4, (0.33, 0.66))}
    library = TemplateLibrary(
        {"math": templates},
        thresholds,
        HashPolicy(eval_hashes=set(), normalize_rules={}),
        UsageTracker(window_size=0, max_share=1.0),
    )

    first = library.sample_prompt("math", None, seed=123)
    second = library.sample_prompt("math", None, seed=123)

    assert first[1] == second[1]

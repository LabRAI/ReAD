from examples.toy_read_demo import run_read_demo


def test_toy_demo_runs_and_improves_over_initial_profile():
    result = run_read_demo(seed=1, steps=12)

    assert result["read_final_utility"] > result["initial_utility"]
    assert set(result["read_allocation_share"]) == set(result["capabilities"])

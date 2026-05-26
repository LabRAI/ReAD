from read.allocation import build_candidate_actions, compute_spillover


def test_build_candidate_actions_normalizes_and_deduplicates():
    caps = ["general", "reasoning", "math"]
    r_tau = [0.1, 0.4, 0.5]
    actions = build_candidate_actions(
        caps,
        r_tau,
        max_support=2,
        weight_grid=[0.25, 0.5, 0.75],
        top_k=2,
        max_candidates=10,
        include_one_hot=True,
    )

    assert actions
    assert len({tuple(round(v, 6) for v in action) for action in actions}) == len(actions)
    for action in actions:
        assert abs(sum(action) - 1.0) < 1e-8
        assert all(value >= 0.0 for value in action)


def test_compute_spillover_penalizes_required_drops_only():
    r_tau = [0.7, 0.2, 0.1]
    delta_s = [-0.5, 0.3, -0.4]

    assert abs(compute_spillover(r_tau, delta_s, top_k=1) - 0.35) < 1e-8
    assert abs(compute_spillover(r_tau, delta_s, top_k=3) - 0.39) < 1e-8

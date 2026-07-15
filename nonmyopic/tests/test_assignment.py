"""Sequential-greedy multi-robot assignment over the per-robot minimax tree."""
import numpy as np

from nonmyopic import MinimaxConfig, paper_noise, greedy_assignment


def _R_fn():
    return lambda r, t: paper_noise(r, t, 0.5, 1.0, 50.0, 5.0)


def test_two_robots_two_targets_assigns_distinctly_and_steers():
    """Two robots, each near a different target -> each is assigned to its nearer target and
    its planned first move reduces the robot-target distance."""
    cfg = MinimaxConfig(horizon=1, n_directions=8, include_stay=True, n_meas=5, v_max=5.0, q=0.05)
    robots = np.array([[0.0, 0.0], [40.0, 40.0]])
    means = np.array([[30.0, 0.0, 0.0, 0.0], [35.0, 40.0, 0.0, 0.0]])
    covs = np.stack([np.eye(4) * 6.0, np.eye(4) * 6.0])
    R_fn = _R_fn()
    new, off, asg, vals = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, dt=1.0)

    assert asg == [0, 1]                                   # each robot -> its nearer target
    for r in range(2):
        before = np.linalg.norm(robots[r] - means[asg[r]][:2])
        after = np.linalg.norm(new[r] - means[asg[r]][:2])
        assert after <= before                            # moves toward (or stays at) target
    assert all(np.isfinite(v) for v in vals)


def test_two_robots_one_target_can_share():
    """With a single target both robots may be assigned to it (one target, multiple robots);
    the second robot conditions on the first's commitment (tighter prior -> smaller value)."""
    cfg = MinimaxConfig(horizon=1, n_directions=8, include_stay=True, n_meas=5, v_max=5.0, q=0.05)
    robots = np.array([[-20.0, 0.0], [20.0, 0.0]])
    means = np.array([[0.0, 0.0, 0.0, 0.0]])
    covs = np.stack([np.eye(4) * 8.0])
    R_fn = _R_fn()
    new, off, asg, vals = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, dt=1.0)

    assert asg == [0, 0]                                   # both serve the only target
    # the second robot plans against the first's tightened posterior -> not larger value
    assert vals[1] <= vals[0] + 1e-9


def test_no_targets_returns_robots_unmoved():
    cfg = MinimaxConfig(horizon=1, n_directions=4, n_meas=5, v_max=5.0)
    robots = np.array([[1.0, 2.0], [3.0, 4.0]])
    R_fn = _R_fn()
    new, off, asg, vals = greedy_assignment(robots, np.zeros((0, 4)), np.zeros((0, 4, 4)),
                                            [R_fn, R_fn], cfg, dt=1.0)
    assert np.allclose(new, robots)
    assert asg == [-1, -1]


def test_weight_by_phd_biases_assignment_to_high_weight_target():
    """weight_by_phd: a single robot whose UNWEIGHTED marginal reward (baseline - minimax
    value) is larger for target 0 is reassigned to target 1 once target 1 carries enough PHD
    weight, because the assignment maximizes w_m*(baseline_m - value_m) (the marginal gain to
    the multi-target leaf sum_j w_j tr(Sigma_j)). Default (weights=None) is unchanged; the
    weight_by_phd flag gates the behaviour."""
    cfg = MinimaxConfig(horizon=2, n_directions=8, include_stay=True, n_meas=5, v_max=8.0,
                        q=0.0, weight_by_phd=True)
    robot = np.array([[0.0, 0.0]])
    # target 0 gives the larger unweighted reward (~4.75); target 1 a smaller one (~1.69).
    means = np.array([[12.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]])
    covs = np.stack([np.eye(4) * 8.0, np.eye(4) * 9.0])
    R_fn = _R_fn()

    # unweighted (weights=None): picks target 0 (larger raw reward)
    _n0, _o0, asg0, _v0 = greedy_assignment(robot, means, covs, [R_fn], cfg, dt=1.0)
    assert asg0 == [0]

    # weight 3x on target 1 -> 3*1.69 > 4.75 -> assignment flips to target 1
    _n1, _o1, asg1, _v1 = greedy_assignment(robot, means, covs, [R_fn], cfg, dt=1.0,
                                            weights=[1.0, 3.0])
    assert asg1 == [1]
    # the reported value stays the UNWEIGHTED per-robot minimax trace (finite, positive)
    assert np.isfinite(_v1[0])

    # flag off -> weights ignored, back to the unweighted choice
    cfg_off = MinimaxConfig(horizon=2, n_directions=8, include_stay=True, n_meas=5, v_max=8.0,
                            q=0.0, weight_by_phd=False)
    _n2, _o2, asg2, _v2 = greedy_assignment(robot, means, covs, [R_fn], cfg_off, dt=1.0,
                                            weights=[1.0, 3.0])
    assert asg2 == [0]


def test_jax_rparams_assignment_matches_numpy():
    """Routing the per-robot value through the genuine-JAX jitted vectorized minimax (via
    r_params) gives the SAME assignment/moves/values as the numpy planner under matching
    paper Eqs 4-5 noise."""
    cfg = MinimaxConfig(horizon=2, n_directions=4, n_meas=5, v_max=5.0, q=0.05)
    robots = np.array([[0.0, 0.0], [50.0, 10.0]])
    means = np.array([[20.0, 5.0, 0.0, 0.0], [45.0, 12.0, 0.0, 0.0]])
    covs = np.stack([np.eye(4) * 6.0, np.eye(4) * 6.0])
    rp = (0.5, 1.0, 50.0, 5.0)
    R_fn = lambda r, t: paper_noise(r, t, *rp)
    a = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, 1.0, use_pruning=False)
    b = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, 1.0, r_params=[rp, rp])
    assert a[2] == b[2]
    assert np.allclose(a[0], b[0], atol=1e-6)
    assert np.allclose(a[3], b[3], atol=1e-6)


def test_pruned_and_exact_assignment_agree():
    """Greedy assignment with the pruned per-robot planner gives the same assignment/moves as
    the exact tree (pruning preserves each robot's per-target optimum)."""
    cfg = MinimaxConfig(horizon=2, n_directions=4, n_meas=5, v_max=5.0, q=0.05)
    robots = np.array([[0.0, 0.0], [50.0, 10.0]])
    means = np.array([[20.0, 5.0, 0.0, 0.0], [45.0, 12.0, 0.0, 0.0]])
    covs = np.stack([np.eye(4) * 6.0, np.eye(4) * 6.0])
    R_fn = _R_fn()
    a = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, 1.0, use_pruning=True)
    b = greedy_assignment(robots, means, covs, [R_fn, R_fn], cfg, 1.0, use_pruning=False)
    assert a[2] == b[2]                                    # same assignment
    assert np.allclose(a[0], b[0])                        # same planned positions
    assert np.allclose(a[3], b[3], atol=1e-9)            # same minimax values

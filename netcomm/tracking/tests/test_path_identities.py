"""Identity labels from the path generators (the respawn-teleport contract).

random_walk respawns an exited object IN PLACE at the same array row — a new
physical object. return_ids must (a) not perturb the walk itself, (b) allocate a
fresh id exactly when a teleport happens, so evaluation can split tracks there.
"""
import numpy as np

from netcomm.tracking.paths import preset_trajectories, random_walk_trajectories


AREA = (0.0, 100.0, 0.0, 100.0)


def test_ids_do_not_perturb_the_walk():
    tr_plain = random_walk_trajectories(5, 120, 0.2, AREA, 8.0, seed=1)
    tr, ids = random_walk_trajectories(5, 120, 0.2, AREA, 8.0, seed=1, return_ids=True)
    assert np.array_equal(tr, tr_plain)
    assert ids.shape == (120, 5)


def test_id_changes_exactly_at_teleports():
    # small area + fast walk => plenty of respawns
    tr, ids = random_walk_trajectories(5, 120, 0.2, AREA, 8.0, seed=1, return_ids=True)
    step = 8.0 * 0.2                                   # max honest per-step displacement
    jumps = np.linalg.norm(np.diff(tr, axis=0), axis=2)      # (T-1, N)
    changed = ids[1:] != ids[:-1]
    assert changed.any(), "scenario has no respawns; test is vacuous"
    # every non-change is a physically-possible step; every teleport-sized jump changed id
    assert np.all(jumps[~changed] <= step + 1e-9)
    assert np.all(changed[jumps > step + 1e-9])
    # ids never repeat once retired
    for i in range(5):
        col = ids[:, i]
        seen_after_change = col[np.concatenate([[False], col[1:] != col[:-1]])]
        assert len(set(seen_after_change)) == len(seen_after_change)


def test_pattern_ids_are_stable():
    tr, ids = preset_trajectories("circle", 4, 50, 0.1, AREA, 5.0, return_ids=True)
    assert np.array_equal(ids, np.tile(np.arange(4), (50, 1)))


def test_preset_dispatch_matches_plain_call():
    tr_plain = preset_trajectories("random_walk", 4, 80, 0.2, AREA, 8.0, seed=7)
    tr, ids = preset_trajectories("random_walk", 4, 80, 0.2, AREA, 8.0, seed=7,
                                  return_ids=True)
    assert np.array_equal(tr, tr_plain)
    assert ids.shape == (80, 4)

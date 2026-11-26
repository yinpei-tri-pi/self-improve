"""The S1 span rules (selfimprove/data/s1_select.py) on a synthetic episode."""

from types import SimpleNamespace

import numpy as np

from selfimprove.data import s1_select as S
from selfimprove.data.episode import Segment

STATS = {p: {m: {"p99": 0.5, "p99.9": 0.6} for m in S.METRICS} for p in ("reach", "grasp", "go", "carry", "lift")}


def seg(i, start, end, by, subgoal, judge, stop=None):
    sg = Segment(episode="T/0", turn=i + 1, index=i, start=start, end=end, by=by, subgoal=subgoal, judge=judge)
    sg.s1_stop = stop
    return sg


def episode(segs, n, success=True, act=None, level=None):
    a = np.zeros((n, 11)) if act is None else act
    lev = np.full(n, -1.0) if level is None else level
    return SimpleNamespace(success=success, segments=segs, traj=SimpleNamespace(lean_act=a, level=lev))


def test_floor_length():
    assert [S.floor_length(n) for n in (5, 60, 112, 140, 260, 2000)] == [50, 50, 100, 125, 250, 1300]


def test_rules():
    n = 1300
    act = np.zeros((n, 11))
    act[200:700, 0] = 0.3                 # the harness drive: base only, as the demos drive
    act[1100:1140, 4] = 0.9 * (-1.0) ** np.arange(40)  # a shaking harness reach (jerk far over the p99.9 of 0.6)
    segs = [
        seg(0, 0, 59, "s1", "reach to the bowl", "subgoal_complete"),
        seg(1, 60, 99, "s1", "grasp the bowl", "subgoal_incomplete"),
        seg(2, 100, 139, "s1", "continue to grasp the bowl", "subgoal_complete"),
        seg(3, 140, 199, "s1", "lift the bowl", "subgoal_complete", stop="budget"),
        seg(4, 200, 449, "agent", "go to the stool", "subgoal_incomplete"),
        seg(5, 450, 699, "agent", "continue to go to the stool", "subgoal_complete"),
        seg(6, 700, 703, "agent", "lower the bowl", "subgoal_complete"),
        seg(7, 704, 799, "agent", "carry the bowl", "subgoal_failed"),
        seg(8, 800, 899, "agent", "carry the bowl again", "subgoal_complete"),
        seg(9, 900, 1099, "agent", "reach to the knob", "subgoal_complete"),
        seg(10, 1100, 1139, "agent", "reach to the mug", "subgoal_complete"),
        seg(11, 1140, 1179, "agent", "reach to the plate", "subgoal_incomplete"),
        seg(12, 1180, 1299, "s1", "lower the bowl onto the plate", "subgoal_complete"),
    ]
    level = np.full(n, -1.0)
    level[900:1000] = 0.2                 # a partial grip (no gripper command)
    spans = S.select(episode(segs, n, act=act, level=level), STATS)
    # System1's partway grasp finished by its own "continue to": one recovery span over both, all frames trained
    assert [(sp.start, sp.end, sp.subgoal, sp.train_from) for sp in spans if sp.kind == "recovery_within"] == [
        (60, 139, "grasp the bowl", None)]
    spans = [sp for sp in spans if sp.kind != "recovery_within"]
    got = [(sp.start, sp.end, sp.subgoal, sp.kept, sp.reason.split(" (")[0], sp.est_length) for sp in spans]
    assert got[0] == (0, 59, "reach to the bowl", True, "S1 step completed", 50)
    assert got[1][3] is False and got[2][2] == "continue to grasp the bowl" and got[2][3]
    assert got[3][4] == "S1 timeout"
    # the two harness drive turns merged (500 steps): a navigation, kept whole
    assert got[4] == (200, 699, "go to the stool", True, "harness step completed, not jerky", 500)
    # the harness streak 200-1179 has 6 manipulation steps and misses (the failed carry, the plate reach left
    # partway): a struggle - its manipulation steps up to the last miss are dropped; the drive (navigation) stays
    assert [got[k][4] for k in (5, 7, 8, 10)] == ["before the last miss of a long harness struggle"] * 4
    assert got[6][4] == "judged subgoal_failed" and got[7][2] == "carry the bowl again" and got[9][4] == "very jerky"


def test_long_navigation_is_not_cut():
    n = 1100
    act = np.zeros((n, 11))
    act[0:500, 0] = 0.3                   # a 500-step harness drive (base only)
    act[500:1000, 4] = 0.2                # a 500-step harness carry (arm only)
    segs = [seg(0, 0, 249, "agent", "go to the stool", "subgoal_incomplete"),
            seg(1, 250, 499, "agent", "continue to go to the stool", "subgoal_complete"),
            seg(2, 500, 999, "agent", "carry the bowl to the stool", "subgoal_complete"),
            seg(3, 1000, 1099, "s1", "lower the bowl onto the stool", "subgoal_complete")]
    got = [(sp.start, sp.end, sp.subgoal, sp.est_length) for sp in S.select(episode(segs, n, act=act), STATS) if sp.kept]
    assert got[0] == (0, 499, "go to the stool", 500)
    assert got[1:3] == [(500, 899, "carry the bowl to the stool", 400), (900, 999, "continue to carry the bowl to the stool", 100)]
    halts = [sp.halt for sp in S.select(episode(segs, n, act=act), STATS) if sp.kept]
    assert halts == [True, False, True, True]  # the cut carry: only its last piece halts
    assert S.is_navigation("move", act[0:500]) and not S.is_navigation("move", act[500:1000])


def test_failed_episode_keeps_nothing():
    segs = [seg(0, 0, 99, "s1", "reach to the bowl", "subgoal_complete")]
    assert [sp.reason for sp in S.select(episode(segs, 100, success=False), STATS)] == ["failed episode"]


def test_trainer_args_seed_and_objective():
    from selfimprove.config import Config
    from selfimprove.train import launch

    cfg = Config.load(None, ["exp_id=e"])
    a0, a1 = launch.trainer_args(cfg, 0), launch.trainer_args(cfg, 1)
    assert "--si.enabled" not in a0 and "--seed=" in a0 and a0 != a1
    assert cfg["train"]["new_data_fraction"] == 0.7 and launch.repeat_factor(2_000_000, 34_200_000, 0.7) == 40
    si = Config.load(None, ["exp_id=e", "train.objective=si"])
    assert "--si.enabled" in launch.trainer_args(si, 0)


def test_relabel_grip():
    lean, raw = np.zeros((6, 11)), np.zeros((6, 12))
    lean[1, 10] = raw[1, 11] = 1.0                        # a step that moves the gripper keeps its sign
    level = np.array([-1.0, -1.0, -0.6, -0.6, 0.2, 0.2])  # held partly open, then partly closed
    l2, r2 = S.relabel_grip(lean, raw, level, 1, 4)
    assert l2[:, 10].tolist() == [0.0, 1.0, -1.0, -1.0, 1.0, 0.0] and r2[:, 11].tolist() == l2[:, 10].tolist()
    assert lean[2, 10] == 0.0  # the original is untouched


def test_partial_grip_relabeled_and_partway_kept():
    n = 400
    level = np.full(n, -1.0)
    level[100:200] = -0.6                 # a reach with the fingers 80% open (grip=.8)
    segs = [seg(0, 0, 99, "s1", "reach to the bowl", "subgoal_complete"),
            seg(1, 100, 199, "agent", "reach to the mug", "subgoal_complete"),
            seg(2, 200, 249, "agent", "reach to the plate", "subgoal_incomplete"),
            seg(3, 250, 399, "s1", "lower the bowl onto the plate", "subgoal_complete")]
    spans = S.select(episode(segs, n, level=level), STATS)
    assert spans[1].kept and spans[1].grip_relabel and "relabeled" in spans[1].reason
    # a harness step left partway, not continued in the same words: kept, without a halt frame
    assert spans[2].kept and not spans[2].halt and spans[0].halt


def test_clean_long_streak_kept():
    segs = [seg(0, 0, 59, "s1", "reach to the knob", "subgoal_complete")] + [
        seg(k, 60 + 40 * (k - 1), 99 + 40 * (k - 1), "agent", w, "subgoal_complete")
        for k, w in enumerate(["grasp the knob", "turn the knob clockwise", "release the knob", "retract the arm"], 1)]
    spans = S.select(episode(segs, 220), STATS)
    # nothing dropped: each step kept, the short ones (40 steps) as merged fragments
    assert all(sp.kept or sp.reason.startswith("merged into") for sp in spans)
    assert [(sp.start, sp.end, sp.subgoal) for sp in spans if sp.kind == "harness_merged"] == [
        (60, 139, "turn the knob clockwise"), (140, 219, "retract the arm")]


def test_harness_fragments_merged():
    n = 400
    act = np.zeros((n, 11))
    act[100:140, 0] = 0.3                 # a drive: navigation
    segs = [seg(0, 0, 19, "agent", "reach to the mug", "subgoal_complete"),
            seg(1, 20, 34, "agent", "grasp the mug", "subgoal_complete"),
            seg(2, 35, 64, "agent", "lift the mug", "subgoal_complete"),
            seg(3, 65, 99, "agent", "carry the mug to the plate", "subgoal_complete"),
            seg(4, 100, 139, "agent", "go to the plate", "subgoal_complete"),
            seg(5, 140, 399, "s1", "lower the mug onto the plate", "subgoal_complete")]
    spans = S.select(episode(segs, n, act=act), STATS)
    merged = [(sp.start, sp.end, sp.subgoal, sp.est_length) for sp in spans if sp.kind == "harness_merged"]
    # 20 + 15 < 50, + 30 = 65: merged, worded as the last; the carry (35) is left alone, the drive not joined to it
    assert merged == [(0, 64, "lift the mug", 50)]
    kept = {(sp.start, sp.kind) for sp in spans if sp.kept}
    assert (65, "harness") in kept and (100, "harness") in kept and (140, "s1") in kept
    assert [sp.reason for sp in spans if sp.start == 20][0].startswith("merged into the harness span 0-64")


def test_recovery_within_the_step():
    segs = [seg(0, 0, 59, "s1", "grasp the mug", "subgoal_failed"),
            seg(1, 60, 99, "agent", "grasp the mug again", "subgoal_complete"),
            seg(2, 100, 159, "s1", "lift the mug", "subgoal_incomplete"),
            seg(3, 160, 199, "agent", "continue to lift the mug", "subgoal_complete"),
            seg(4, 200, 299, "s1", "carry the mug to the plate", "subgoal_complete")]
    spans = S.select(episode(segs, 300), STATS)
    rec = [sp for sp in spans if sp.by == "recovery"]
    assert [(sp.start, sp.end, sp.subgoal, sp.train_from) for sp in rec] == [
        (0, 99, "grasp the mug", 60),      # after a failure: System1's frames are context, the fix's trained
        (100, 199, "lift the mug", None)]  # after an incomplete step: all frames trained
    assert rec[0].trained_steps == 40 and rec[0].est_length == 100 and rec[0].kept
    # the fixes alone stay spans too (recovery when System2 asks for it)
    assert [(sp.start, sp.subgoal) for sp in spans if sp.by == "harness" and sp.kept] == [
        (60, "grasp the mug again"), (160, "continue to lift the mug")]
    assert {sp.kind for sp in spans if sp.start in (60, 160) and sp.by == "harness"} == {"recovery_request"}
    assert {sp.kind for sp in rec} == {"recovery_within"}
    # a miss + fix of 125 steps or more gives no recovery span
    long = [seg(0, 0, 59, "s1", "grasp the mug", "subgoal_failed"),
            seg(1, 60, 259, "agent", "grasp the mug again", "subgoal_complete")]
    assert not [sp for sp in S.select(episode(long, 260), STATS) if sp.by == "recovery"]


def test_navigation_recovery_has_no_length_bound():
    """NavigateKitchen l88 (check run): System1's search failed (64 steps), the harness's search again (72) found it:
    one span 'search for the microwave' over both, though longer than RECOVERY_TOTAL."""
    n = 200
    act = np.zeros((n, 11))
    act[0:136, 2] = 0.4                   # the base turning: navigation
    segs = [seg(0, 0, 63, "s1", "search for the microwave", "subgoal_failed"),
            seg(1, 64, 135, "agent", "search for the microwave again", "subgoal_complete"),
            seg(2, 136, 199, "s1", "go to the microwave", "subgoal_complete")]
    rec = [sp for sp in S.select(episode(segs, n, act=act), STATS) if sp.kind == "recovery_within"]
    assert [(sp.start, sp.end, sp.subgoal, sp.train_from) for sp in rec] == [(0, 135, "search for the microwave", 64)]


def test_system1_own_retry_is_a_recovery():
    """WashLettuce l81 (check run): System1's grasp failed; the next turn opened the gripper (harness) and System1
    grasped again: one step 'grasp the colander again' (prep included), and both recovery examples."""
    segs = [seg(0, 0, 58, "s1", "grasp the colander", "subgoal_failed"),
            seg(1, 59, 72, "agent", "grasp the colander again", "subgoal_complete"),
            seg(2, 73, 129, "s1", "grasp the colander again", "subgoal_complete"),
            seg(3, 130, 199, "s1", "lift the colander", "subgoal_complete")]
    segs[2].turn = segs[1].turn  # one turn, two movers
    spans = S.select(episode(segs, 200), STATS)
    got = {(sp.kind, sp.start, sp.end, sp.subgoal, sp.train_from) for sp in spans if sp.kept}
    assert ("recovery_within", 0, 129, "grasp the colander", 59) in got
    assert ("recovery_request", 59, 129, "grasp the colander again", None) in got
    assert not any(sp.start == 59 and sp.end == 72 for sp in spans)  # the opening alone is no step of its own

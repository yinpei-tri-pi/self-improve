from selfimprove import tasks as T
from selfimprove.config import Config


def test_task_filter_and_seeds():
    tasks = T.select_tasks(0.8)
    assert len(tasks) == 287 and "PlaceVegetablesEvenly" not in tasks and "CloseFridge" in tasks
    a = T.draw_seeds("e", 0, "CloseFridge", 10)
    assert a == T.draw_seeds("e", 0, "CloseFridge", 10) and len(set(a)) == 10
    b = T.draw_seeds("e", 1, "CloseFridge", 10, used=set(a))
    assert not set(a) & set(b) and all(100000 <= s <= 999999 for s in a + b)


def test_plan_round_pilot_then_full():
    cfg = Config.load(None, ["exp_id=e", "rollout.tasks=pretrain300", "rollout.split=pretrain",
                             "rollout.seeds_per_task=10", "rollout.pilot_seeds_per_task=2"])
    assert len(T.plan_round(cfg, 0)) == 287 * 2
    assert len(T.plan_round(cfg, 1)) == 287 * 10
    order = T.interleave(T.plan_round(cfg, 0))
    assert order[0].kind == "composite" and order[0].horizon >= order[-1].horizon


def test_plan_round_selected_shared_seeds():
    from selfimprove.config import Config

    a = Config.load(None, ["exp_id=ab_with", "rollout.tasks=selected", "rollout.seed_key=ab108",
                           "rollout.pilot_seeds_per_task=5", "rollout.split=pretrain"])
    b = Config.load(None, ["exp_id=ab_without", "rollout.tasks=selected", "rollout.seed_key=ab108",
                           "rollout.pilot_seeds_per_task=5", "rollout.split=pretrain"])
    pa, pb = T.plan_round(a, 0), T.plan_round(b, 0)
    assert len(pa) == len(T.selected_tasks()) * 5 and pa == pb
    assert not set(e.task for e in pa) & set(T.target_tasks())
    c = Config.load(None, ["exp_id=ab_with", "rollout.tasks=selected", "rollout.pilot_seeds_per_task=5",
                           "rollout.split=pretrain"])
    assert T.plan_round(c, 0) != pa  # without a seed_key the draw is the experiment's own


def test_plan_round_development_tasks():
    import json
    import pytest

    cfg = Config.load(None, ["exp_id=dev"])  # rollout.tasks: rollout is the default
    plan = T.plan_round(cfg, 0)
    tasks = {e.task for e in plan}
    target = T.target_tasks()
    seen = {t for t, v in target.items() if v["task_set"] in ("atomic_seen", "composite_seen")}
    assert len(tasks) == 120 and len(plan) == 120 * 30
    assert seen <= tasks and not tasks & T.held_out_tasks() and len(T.held_out_tasks()) == 16
    dropped = {r["task"] for r in json.loads((T.ASSETS / "rollout_tasks.json").read_text())["dropped"]}
    assert len(dropped) == 22 and not tasks & dropped
    assert len([t for t in tasks if t not in target]) == 44 + 42
    for e in plan:  # target tasks keep their official horizons
        if e.task in target:
            assert e.horizon == int(target[e.task]["horizon"])
    # a check round may replay the target tasks, unseen ones included; a training round may not
    check = Config.load(None, ["exp_id=c", "rollout.tasks=target50", "rollout.fixed_seeds=[0]"])
    assert {e.task for e in T.plan_round(check, 0)} >= T.held_out_tasks()
    bad = Config.load(None, ["exp_id=b", "rollout.tasks=rollout", "rollout.include_tasks=[ArrangeTea]"])
    with pytest.raises(ValueError):
        T.plan_round(bad, 0)


def test_held_out_tasks_never_reach_training_data(tmp_path):
    import pytest

    from selfimprove.data import s1_shards

    with pytest.raises(ValueError, match="held-out"):
        s1_shards.build([(str(tmp_path), "WashFruitColander", 1)], tmp_path / "out", exp_id="x", iteration=0, fcfg={})


def test_plan_round_demo_split():
    """split demo: 30 of each task's own recorded episodes per round, never one an earlier round played."""
    import collections

    cfg = Config.load(None, ["exp_id=dev"])
    recorded = T.demo_episodes()
    r0 = T.plan_round(cfg, 0)
    used = collections.defaultdict(set)
    for e in r0:
        used[e.task].add(e.seed)
        assert e.demo == recorded[e.task][e.seed]
    r1 = T.plan_round(cfg, 1, used)
    assert len(r1) == len(r0) == 120 * 30 and r0 == T.plan_round(cfg, 0)  # deterministic
    assert not {e.key for e in r0} & {e.key for e in r1}
    # a task with few recordings left gets what is left, no repeats
    ids = recorded["CloseDrawer"]
    left = T.draw_demos("k", 5, "CloseDrawer", len(ids), 30, used=set(range(len(ids) - 7)))
    assert left == list(range(len(ids) - 7, len(ids)))

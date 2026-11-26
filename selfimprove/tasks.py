"""Which (task, seed) episodes a rollout round plays.

Tasks: RoboCasa's ``pretrain300`` (assets/pretrain300.json), keeping those whose recorded demos replay
to success at least ``min_replay_success`` of the time (assets/replay_success.json; a task whose
success check rejects most of its own demonstrations cannot be learnt from rollouts either).

Seeds: per (experiment, iteration, task) a deterministic draw from ``seed_range`` that never repeats
a seed an earlier iteration of the experiment used for that task.  The rollouts run on RoboCasa's
``pretrain`` split (kitchens with layouts and styles 11-60 and the pretraining objects), which is
disjoint from the target split (kitchens 1-10 and the target objects) whatever the seed is; the seed
range also stays clear of HexaShell's own development seeds (0-49).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"


@dataclass(frozen=True)
class Episode:
    task: str
    seed: int
    kind: str  # atomic | composite
    horizon: int
    #: split demo: the recorded episode the seed starts (its id in the task's index; the seed is its position)
    demo: str | None = None
    #: the start perturbation's salt (HexaShell start_salt), None = the recorded start as it is
    start_salt: int | None = None

    @property
    def key(self) -> str:
        return f"{self.task}/{self.seed}"


def pretrain_tasks() -> dict[str, dict]:
    return json.loads((ASSETS / "pretrain300.json").read_text())["tasks"]


def replay_rates(split: str = "pretrain") -> dict[str, float]:
    table = json.loads((ASSETS / "replay_success.json").read_text())[split]
    return {task: float(v["rate"]) for task, v in table.items()}


def select_tasks(min_replay_success: float = 0.8, include=(), exclude=()) -> list[str]:
    """The round's tasks, sorted: pretrain300 minus low-replay tasks, restricted / reduced as asked."""
    rates = replay_rates()
    tasks = sorted(t for t in pretrain_tasks() if rates.get(t, 0.0) >= float(min_replay_success))
    if include:
        unknown = sorted(set(include) - set(pretrain_tasks()))
        if unknown:
            raise ValueError(f"include_tasks not in pretrain300: {unknown}")
        tasks = [t for t in tasks if t in set(include)]
    return [t for t in tasks if t not in set(exclude)]


def _rng(exp_id: str, iteration: int, task: str) -> random.Random:
    digest = hashlib.sha256(f"{exp_id}|{iteration}|{task}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def draw_seeds(exp_id: str, iteration: int, task: str, n: int, seed_range=(100000, 999999),
               used: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """``n`` distinct seeds for this (experiment, iteration, task), none of them in ``used``."""
    lo, hi = int(seed_range[0]), int(seed_range[1])
    if hi - lo + 1 < n + len(used):
        raise ValueError("seed_range is too small")
    rng = _rng(exp_id, iteration, task)
    out: list[int] = []
    while len(out) < n:
        s = rng.randint(lo, hi)
        if s not in used and s not in out:
            out.append(s)
    return out


def selected_tasks() -> list[str]:
    """The rollout tasks of rule A (assets/selected_tasks.json, scripts/select_tasks.py)."""
    return [r["task"] for r in json.loads((ASSETS / "selected_tasks.json").read_text())["tasks"]]


def rollout_tasks() -> list[dict]:
    """The development tasks of the self-improvement rollouts (assets/rollout_tasks.json, scripts/rollout_tasks.py):
    the atomic_seen and composite_seen target tasks plus the selected pretrain tasks that stayed after the prompt A/B,
    each with its kind and horizon."""
    return json.loads((ASSETS / "rollout_tasks.json").read_text())["tasks"]


def demo_episodes() -> dict[str, list[str]]:
    """task -> its recorded pretrain-split episodes, in index order (assets/demo_episodes.json, from the sim states
    scripts/extract_sim_states.py extracted): seed n of HexaShell's split demo starts ids[n]."""
    return json.loads((ASSETS / "demo_episodes.json").read_text())["tasks"]


def demo_starts(key: str, iteration: int, task: str, n_recorded: int, n: int, mode: str = "sequential",
                augment_from: int = 1) -> list[tuple[int, int | None]]:
    """``n`` (recording position, start salt) of a task for iteration ``iteration`` - always ``n``, every round.

    sequential: the task's recordings in one fixed shuffled order (per experiment and task), ``n`` per round, wrapping
    to the start when they run out; random: ``n`` distinct recordings drawn afresh each round. The start is perturbed
    (salt = the round + 1) from round ``augment_from`` on, and always when sequential wraps (a recording played
    again), so no two starts of the experiment are the same; earlier rounds play the recorded start (salt None)."""
    n_recorded, n = int(n_recorded), int(n)
    if n > n_recorded:
        raise ValueError(f"{task}: {n} episodes per round but only {n_recorded} recordings (a round would repeat one)")
    if mode == "sequential":
        order = list(range(n_recorded))
        _rng(key, -1, task).shuffle(order)  # one order for the whole experiment
        picks = [(k, (int(iteration) * n + j) // n_recorded) for j, k in enumerate(
            order[(int(iteration) * n + j) % n_recorded] for j in range(n))]
    elif mode == "random":
        picks = [(k, 0) for k in _rng(key, iteration, task).sample(range(n_recorded), n)]
    else:
        raise ValueError(f"rollout.start_draw: sequential or random, not {mode!r}")
    out = []
    for pos, passes in picks:
        perturb = int(iteration) >= int(augment_from) or passes > 0
        out.append((pos, int(iteration) + 1 if perturb else None))
    return sorted(out)


def draw_demos(key: str, iteration: int, task: str, n_recorded: int, n: int,
               used: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """``n`` of a task's recorded episodes (their positions 0..n_recorded-1) for this (experiment, iteration), none
    played in an earlier iteration; fewer when fewer are left."""
    free = [i for i in range(int(n_recorded)) if i not in used]
    _rng(key, iteration, task).shuffle(free)
    return sorted(free[:int(n)])


def held_out_tasks() -> frozenset[str]:
    """The composite_unseen target tasks: never rolled out for training and never in any training data (they measure
    generalization). Only a check round (``rollout.tasks: target50``) may play them."""
    return frozenset(t for t, v in target_tasks().items() if v["task_set"] == "composite_unseen")


def target_tasks() -> dict[str, dict]:
    """The 50 target tasks with their official horizons (assets/target50.json, from HexaShell's protocol)."""
    return json.loads((ASSETS / "target50.json").read_text())["tasks"]


def plan_round(cfg, iteration: int, used_seeds: dict[str, set[int]] | None = None) -> list[Episode]:
    """Every episode of iteration ``iteration``: the selected tasks x seeds_per_task fresh seeds.

    ``rollout.tasks: target50`` with ``rollout.fixed_seeds: [0, 1, 2]`` instead replays a fixed set (e.g. a local
    HexaShell run such as full2_s1first, to compare the cluster with this machine episode by episode).
    ``rollout.tasks: rollout`` (the default) plays the development tasks of assets/rollout_tasks.json;
    ``rollout.tasks: selected`` plays the tasks of assets/selected_tasks.json (with include/exclude as usual), and
    ``rollout.seed_key`` (default: the experiment's id) names the draw, so that two experiments (the arms of an A/B)
    can play the same seeds."""
    r = cfg["rollout"]
    if r.get("tasks") == "target50":
        table = target_tasks()
        names = [t for t in sorted(table) if not r.get("include_tasks") or t in set(r["include_tasks"])]
        seeds = [int(s) for s in (r.get("fixed_seeds") or [])]
        if not seeds:
            raise ValueError("rollout.tasks=target50 needs rollout.fixed_seeds")
        return [Episode(t, s, table[t]["kind"], int(table[t]["horizon"])) for t in names for s in seeds]
    if r.get("tasks") == "rollout":
        rows = {row["task"]: row for row in rollout_tasks()}
        names = [t for t in sorted(rows) if (not r.get("include_tasks") or t in set(r["include_tasks"]))
                 and t not in set(r.get("exclude_tasks") or ())]
        unknown = sorted(set(r.get("include_tasks") or ()) - set(rows))
        if unknown:
            raise ValueError(f"include_tasks not in assets/rollout_tasks.json: {unknown}")
        tables = {t: (rows[t]["kind"], int(rows[t]["horizon"])) for t in names}
    else:
        tables = None
    table = pretrain_tasks()
    include = list(r.get("include_tasks") or ())
    if r.get("tasks") == "selected":
        chosen = selected_tasks()
        include = [t for t in chosen if not include or t in set(include)]
    elif r.get("tasks") not in (None, "pretrain300", "rollout"):
        raise ValueError(f"rollout.tasks: rollout, pretrain300, selected or target50, not {r.get('tasks')!r}")
    if tables is not None:
        tasks = list(tables)
    else:
        tasks = select_tasks(r.get("min_replay_success", 0.8), include, r.get("exclude_tasks") or ())
    if r.get("tasks") == "selected" and not tasks:
        raise ValueError("rollout.tasks=selected: no task left")
    n = cfg.seeds_per_task(iteration)
    key = str(r.get("seed_key") or cfg.exp_id)
    used_seeds = used_seeds or {}
    out = []
    if r.get("split") == "demo":  # the tasks' own recorded episodes, never one an earlier iteration played
        recorded = demo_episodes()
        missing = sorted(t for t in tasks if t not in recorded)
        if missing:
            raise ValueError(f"split demo: no recorded pretrain episodes of {missing}")
        mode = str(r.get("start_draw") or "fresh")
        for task in tasks:
            kind, horizon = tables[task] if tables is not None else (table[task]["kind"], int(table[task]["horizon"]))
            ids = recorded[task]
            if mode == "fresh":  # legacy: never a recording an earlier round played (rounds shrink once they run out)
                salt = int(iteration) + 1 if int(iteration) >= int(r.get("augment_from_round", 10**6)) else None
                for i in draw_demos(key, iteration, task, len(ids), n, used_seeds.get(task, set())):
                    out.append(Episode(task, i, kind, horizon, demo=ids[i], start_salt=salt))
                continue
            aug = r.get("augment_from_round", 1)  # null: never by round, only a recording played again is perturbed
            for i, salt in demo_starts(key, iteration, task, len(ids), n, mode, 10**9 if aug is None else int(aug)):
                out.append(Episode(task, i, kind, horizon, demo=ids[i], start_salt=salt))
        tasks = []
    for task in tasks:
        for seed in draw_seeds(key, iteration, task, n, r.get("seed_range", (100000, 999999)),
                               used_seeds.get(task, set())):
            kind, horizon = tables[task] if tables is not None else (table[task]["kind"], int(table[task]["horizon"]))
            out.append(Episode(task, seed, kind, horizon))
    held = sorted({e.task for e in out} & held_out_tasks())
    if held:
        raise ValueError(f"composite_unseen tasks are held out of every training rollout: {held}")
    return out


def interleave(episodes: list[Episode]) -> list[Episode]:
    """The order to play them in: long (composite) episodes first so the tail of the round is short, and
    the tasks spread out (a slot keeps a task's environment warm, but many slots on one task would make
    the round's early statistics lopsided)."""
    by_task: dict[str, list[Episode]] = {}
    for e in sorted(episodes, key=lambda e: (-e.horizon, e.task, e.seed)):
        by_task.setdefault(e.task, []).append(e)
    order = sorted(by_task, key=lambda t: -by_task[t][0].horizon)
    out: list[Episode] = []
    while any(by_task.values()):
        for t in order:
            if by_task[t]:
                out.append(by_task[t].pop(0))
    return out

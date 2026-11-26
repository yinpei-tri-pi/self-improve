"""Freeze the self-improvement rollout tasks: the target tasks of ``--target-sets``, plus the selected tasks (rule A,
assets/selected_tasks.json) without the ``--drop-worst`` weakest of them (of kind ``--drop-kind``) in the prompt A/B.

    .venv/bin/python scripts/rollout_tasks.py [--drop-worst 22] [--drop-kind composite]
        [--target-sets atomic_seen composite_seen]

The target tasks run in RoboCasa's pretrain kitchens with the pretrain objects (the loop's split), never in the target
split's; composite_unseen is left out by default so that it stays unseen. The selected tasks are ranked by their success with the current task notes. That is the
A/B's ``with`` arm (ab108_with, 5 seeds), or for the tasks whose notes were revised after it, the re-check on fresh
seeds (ab108r2_with, 6 seeds). Ties are broken by the ``without`` arm (ab108_without), then by name; this is the
ranking of docs/ab108_worst20.md. Writes assets/rollout_tasks.json (every task with its kind, horizon and source;
the dropped ones with their numbers) and docs/rollout_tasks.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402
from selfimprove.config import Config  # noqa: E402

#: the tasks whose notes were revised after the A/B (docs/ab108_findings.md): ranked on the re-check
REVISED = ("BeverageOrganization", "CloseOven", "MoveToFreezerDrawer", "ScalePortioning", "SlideOvenRack")


def results(exp: str) -> dict[str, list[bool]]:
    """task -> successes of the experiment's finished episodes (status ok)."""
    cfg = Config.load(None, [f"exp_id={exp}"])
    plan = s3.read_json(f"{cfg.rollout_uri(0)}/_plan.json")
    keys = [(e["task"], int(e["seed"])) for e in plan["episodes"]]
    with ThreadPoolExecutor(32) as pool:
        rows = pool.map(lambda k: s3.read_json(f"{cfg.episode_uri(0, *k)}/rollout.json", default=None), keys)
    out: dict[str, list[bool]] = {}
    for (task, _), r in zip(keys, rows):
        if r and r.get("status") == "ok":
            out.setdefault(task, []).append(bool(r.get("success")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-worst", type=int, default=22)
    ap.add_argument("--drop-kind", choices=("composite", "atomic", "all"), default="composite")
    ap.add_argument("--target-sets", nargs="*", default=["atomic_seen", "composite_seen"])
    args = ap.parse_args()
    cfg = Config.load()
    s3.configure(cfg.get("aws_profile"), cfg.region)
    selected = {r["task"]: r for r in json.loads((ROOT / "assets/selected_tasks.json").read_text())["tasks"]}
    target = {t: v for t, v in json.loads((ROOT / "assets/target50.json").read_text())["tasks"].items()
              if v["task_set"] in args.target_sets}
    with_, without, r2 = results("ab108_with"), results("ab108_without"), results("ab108r2_with")
    rank = []
    for task in selected:
        cur = r2[task] if task in REVISED else with_[task]
        rank.append({"task": task, "current": f"{sum(cur)}/{len(cur)}", "rate": sum(cur) / len(cur),
                     "ab_with": f"{sum(with_[task])}/{len(with_[task])}",
                     "ab_without": f"{sum(without[task])}/{len(without[task])}",
                     "without_rate": sum(without[task]) / len(without[task]), "revised": task in REVISED})
    rank.sort(key=lambda r: (r["rate"], r["without_rate"], r["task"]))
    pool = [r for r in rank if args.drop_kind == "all" or selected[r["task"]]["kind"] == args.drop_kind]
    dropped = pool[:args.drop_worst]
    drop = {r["task"] for r in dropped}
    tasks = [{"task": t, "kind": v["kind"], "task_set": v["task_set"], "horizon": int(v["horizon"]), "source": "target"}
             for t, v in sorted(target.items())]
    tasks += [{"task": t, "kind": v["kind"], "task_set": f"pretrain_{v['kind']}", "horizon": int(v["horizon"]),
               "source": "selected", "family": v.get("family")} for t, v in sorted(selected.items()) if t not in drop]
    out = {"rule": f"target sets {'+'.join(args.target_sets)} + the selected tasks (rule A) without the "
                   f"{args.drop_worst} weakest {args.drop_kind} ones in the prompt A/B (current notes; ties by the "
                   "without arm, then name)",
           "total": len(tasks), "target": len(target), "selected": len(tasks) - len(target),
           "tasks": tasks, "dropped": dropped, "ranking": rank}
    (ROOT / "assets/rollout_tasks.json").write_text(json.dumps(out, indent=1) + "\n")
    kinds = {}
    for t in tasks:
        kinds[t["task_set"]] = kinds.get(t["task_set"], 0) + 1
    lines = [f"# Rollout tasks ({len(tasks)})", "",
             f"The {len(target)} target tasks of {', '.join(args.target_sets)}, run in RoboCasa's pretrain kitchens with the",
             "pretrain objects (never the target split's), plus "
             f"{len(tasks) - len(target)} of the 108 selected tasks (`docs/task_selection.md`): every selected task except",
             f"the {args.drop_worst} weakest {args.drop_kind} ones in the prompt A/B (`docs/ab108_findings.md`). Written by `scripts/rollout_tasks.py`;"
             "`rollout.tasks: rollout` plays them.", "",
             "| task set | tasks |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(kinds.items())]
    lines += ["", f"## Dropped ({len(dropped)})", "", "| # | task | current notes | A/B with | A/B without |",
              "|---|---|---|---|---|"]
    lines += [f"| {i} | {r['task']}{' (revised notes)' if r['revised'] else ''} | {r['current']} | {r['ab_with']} | "
              f"{r['ab_without']} |" for i, r in enumerate(dropped, 1)]
    lines += ["", "## All rollout tasks", "", "| task | source | task set | horizon |", "|---|---|---|---|"]
    lines += [f"| {t['task']} | {t['source']} | {t['task_set']} | {t['horizon']} |" for t in tasks]
    (ROOT / "docs/rollout_tasks.md").write_text("\n".join(lines) + "\n")
    print(f"{len(tasks)} rollout tasks: {kinds}; dropped {len(dropped)}: {', '.join(r['task'] for r in dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

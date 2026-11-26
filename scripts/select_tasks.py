"""Select the rollout tasks (rule A): every pretrain atomic task, and every pretrain composite task of a RoboCasa
task family (``robocasa/environments/kitchen/composite/<family>/``) that holds one of the 16 composite_unseen target
tasks.  Tasks whose recorded demos replay to success less often than ``--min-replay`` are dropped (atomic or
composite); the 50 target tasks themselves are not candidates (their results come from the target-task runs).

    .venv/bin/python scripts/select_tasks.py [--min-replay 0.8]

Reads assets/{pretrain300,target50,replay_success,task_families,task_signatures}.json; writes
assets/selected_tasks.json and docs/task_selection.md.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "assets" / name).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-replay", type=float, default=0.8)
    args = ap.parse_args()
    pre = load("pretrain300.json")["tasks"]
    target = load("target50.json")["tasks"]
    replay = {t: v["rate"] for t, v in load("replay_success.json")["pretrain"].items()}
    family = load("task_families.json")["task_family"]
    sig = load("task_signatures.json")
    unseen = sorted(t for t, v in target.items() if v["task_set"] == "composite_unseen")
    fam_of_unseen = {family[u]: u for u in unseen}
    if len(fam_of_unseen) != len(unseen):
        raise SystemExit("two composite_unseen tasks share a family: the table below keeps only one per family")
    rows, dropped = [], {}
    for t in sorted(pre):
        if t in target:
            continue
        kind = pre[t]["kind"]
        fam = family.get(t)
        if kind == "composite" and fam not in fam_of_unseen:
            continue
        if replay.get(t, 0.0) < args.min_replay:
            dropped[t] = {"kind": kind, "family": fam, "replay_success": replay.get(t)}
            continue
        s = sig.get(t, {})
        rows.append({"task": t, "kind": kind, "family": fam if kind == "composite" else None,
                     "unseen_in_family": fam_of_unseen.get(fam) if kind == "composite" else None,
                     "horizon": pre[t]["horizon"], "replay_success": replay.get(t),
                     "pilot": f"{s.get('success', 0)}/{s.get('episodes', 0)}", "goal": (s.get("goals") or [""])[0]})
    by_fam = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "composite":
            by_fam[r["family"]].append(r["task"])
    atomic = [r for r in rows if r["kind"] == "atomic"]
    composite = [r for r in rows if r["kind"] == "composite"]
    out = {"rule": "A: all pretrain atomic + all pretrain composite members of the composite_unseen tasks' families",
           "total": len(rows), "atomic": len(atomic), "composite": len(composite), "min_replay_success": args.min_replay,
           "excluded_target": sorted(target), "dropped_low_replay": dropped,
           "families": {f: {"unseen": u, "members": by_fam.get(f, [])} for f, u in sorted(fam_of_unseen.items())},
           "tasks": rows}
    (ROOT / "assets/selected_tasks.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [f"# Selected rollout tasks ({len(rows)}: {len(atomic)} atomic + {len(composite)} composite)", "",
             "Rule A (`scripts/select_tasks.py`): every pretrain atomic task, plus every pretrain composite task of a",
             "RoboCasa task family that holds one of the 16 composite_unseen target tasks. Tasks whose demos replay to",
             f"success less than {args.min_replay:.0%} of the time are dropped; the 50 target tasks are not candidates.", "",
             "## Families of the composite_unseen tasks", "", "| family | composite_unseen task | selected members |",
             "|---|---|---|"]
    for f, u in sorted(fam_of_unseen.items()):
        lines.append(f"| {f} | {u} | {', '.join(by_fam.get(f, [])) or '-'} |")
    lines += ["", "## Dropped for demo replay", ""]
    lines += [f"- {t} ({d['kind']}{', ' + d['family'] if d['family'] and d['kind'] == 'composite' else ''}): "
              f"{(d['replay_success'] or 0):.0%}" for t, d in sorted(dropped.items())]
    lines += ["", "## All selected tasks", "", "| task | kind | family | horizon | replay | pilot | goal |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['kind']} | {r['family'] or '-'} | {r['horizon']} | {r['replay_success']:.0%} | "
                     f"{r['pilot']} | {r['goal'][:90]} |")
    (ROOT / "docs/task_selection.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} tasks: {len(atomic)} atomic + {len(composite)} composite; dropped {len(dropped)} for replay: "
          + ", ".join(sorted(dropped)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

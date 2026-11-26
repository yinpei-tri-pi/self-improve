"""Check the S1 spans and S2 rows of every episode of one or more rollouts against the preprocessing rules.

    .venv/bin/python scripts/check_training_data.py si_sub50 sub50_check_20261125 [--jobs 16]

Reads the downloaded records (<work_dir>/<exp>/iter_00/rollout, or the legacy <work_dir>/<exp>/rollout) and prints, per
rule, how many episodes break it with a few examples. Run it after every change of the rules
(docs/data_preprocessing.md): a fix is only done when it holds on all the data.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_episode(folder: str) -> list[tuple[str, str]]:
    sys.path.insert(0, str(ROOT))
    from selfimprove.data import s1_select as S
    from selfimprove.data.episode import base_step, load_episode
    from selfimprove.data.trajectory import _LEN_GRID
    from selfimprove.s2 import rows_v2

    rec = load_episode(folder)
    name = f"{rec.task}_{rec.seed}"
    bad: list[tuple[str, str]] = []
    spans = S.select(rec)
    segs = S._segments(rec)
    lean = rec.traj.lean_act
    milestone = {int(e["turn"]): str(e.get("milestone") or "").split(":")[0].strip()
                 for e in rec.events if e.get("kind") in ("judge", "planner") and e.get("turn")}
    grid = set(_LEN_GRID) | {50}
    for sp in spans:
        if not (0 <= sp.start <= sp.end < len(rec.traj)):
            bad.append(("S1 span outside the trajectory", f"{name} {sp.start}-{sp.end}"))
        if not sp.kept:
            continue
        if not rec.success:
            bad.append(("S1 span kept in a failed episode", f"{name} {sp.start}-{sp.end}"))
        if sp.trained_steps < S.MIN_STEPS:
            bad.append(("S1 kept span under MIN_STEPS trained steps", f"{name} {sp.start}-{sp.end}"))
        if sp.est_length not in grid:
            bad.append(("S1 Estimated Length off the grid", f"{name} {sp.start}-{sp.end} {sp.est_length}"))
        if sp.train_from is not None and not (sp.start < sp.train_from <= sp.end):
            bad.append(("S1 train_from outside its span", f"{name} {sp.start}-{sp.end} from {sp.train_from}"))
        if sp.kind == "harness_merged":
            ms = {milestone.get(t) for t in sp.turns}
            if len(ms) > 1:
                bad.append(("S1 merge across milestones", f"{name} turns {sp.turns}"))
        if sp.kind == "recovery_within":
            first = next((g for g in segs if g.start == sp.start), None)
            if first is None or first.by != "s1" or first.judge not in ("subgoal_failed", "subgoal_incomplete") and (
                    getattr(first, "s1_stop", None) or "") != "budget":
                bad.append(("S1 recovery not starting at a System1 miss", f"{name} {sp.start}-{sp.end}"))
            if base_step(sp.subgoal) != base_step(first.subgoal if first else ""):
                bad.append(("S1 recovery worded unlike the miss", f"{name} {sp.start}-{sp.end}"))
            context = first is not None and (first.judge == "subgoal_failed" or (getattr(first, "s1_stop", None) or "") == "budget")
            if (sp.train_from is not None) != context:
                bad.append(("S1 recovery context rule", f"{name} {sp.start}-{sp.end}"))
    # two-part turns: no harness-only span of a turn that also called System1
    by_turn = collections.defaultdict(set)
    for g in rec.segments:
        by_turn[g.turn].add(g.by)
    for sp in spans:
        if sp.kept and sp.kind in ("harness", "harness_merged") and any("s1" in by_turn[t] for t in sp.turns):
            bad.append(("S1 harness span inside a two-part turn", f"{name} turns {sp.turns}"))
    # merges: no navigation with manipulation
    for sp in spans:
        if sp.kept and sp.kind == "harness_merged":
            parts = [p for p in spans if p.reason.startswith(f"merged into the harness span {sp.start}-{sp.end}")]
            nav = {S.is_navigation(p.primitive, lean[p.start:p.end + 1]) for p in parts}
            if len(nav) > 1:
                bad.append(("S1 merge of a navigation with a manipulation", f"{name} turns {sp.turns}"))
    # System2 rows
    rows = rows_v2.episode_rows(folder, Path("/tmp/_unused_s2"), make_media=False)
    moved = sorted({g.turn for g in rec.segments})
    covered = set()
    for r in rows:
        if r["meta"]["kind"] == "closing":
            continue
        covered.update(r["meta"].get("merged_turns") or [r["meta"]["turn"]])
    two_mover = {t for t, b in by_turn.items() if len(b) > 1}
    missing = [t for t in moved if t not in covered]
    if missing:
        bad.append(("S2 moved turn without a row", f"{name} turns {missing}"))
    closing = [r for r in rows if r["meta"]["kind"] == "closing"]
    if len(closing) != 1 or closing[0]["meta"]["turn"] != max(moved) + 1:
        bad.append(("S2 closing row missing or misnumbered", name))
    if rec.success:
        s1_groups = sorted(sp.turns for sp in spans if sp.kind == "harness_merged" and sp.kept)
        s2_groups = sorted(r["meta"]["merged_turns"] for r in rows if r["meta"].get("merged_turns"))
        if [g for g in s2_groups if g not in s1_groups]:
            bad.append(("S2 merge unlike S1's", f"{name} S2 {s2_groups} S1 {s1_groups}"))
    text = " ".join(m["content"] for r in rows for m in r["messages"][1:])
    if re.search(r"\b(harness|System1)\b", text):
        bad.append(("S2 text says harness / System1", name))
    for r in rows:
        u = r["messages"][1]["content"]
        if "Milestones:\n" in u and len(u.split("Milestones:\n")[1].split("\n\nThe last step")[0].splitlines()) > 30:
            bad.append(("S2 milestone view over 30 lines", f"{name} turn {r['meta']['turn']}"))
    # the milestone view of each row = the latest list emitted in an earlier row's target (S2 rebuilds it at inference)
    emitted: list = []
    for r in rows:
        u, a = r["messages"][1]["content"], r["messages"][2]["content"]
        if "Milestones:\n" in u and emitted:
            view = u.split("Milestones:\n")[1].split("\n\nThe last step")[0]
            listed = [ln.strip().rsplit(" (", 1)[0] for ln in view.splitlines() if re.match(r"\s+M\d+", ln)]
            want = [txt for _, txt in emitted]
            if listed[:len(want)] != want:
                bad.append(("S2 milestone view unlike the emitted list", f"{name} turn {r['meta']['turn']}"))
        m = re.search(r"<milestones>\n(.*?)\n</milestones>", a, re.S)
        if m:
            emitted = rows_v2.parse_milestones(m.group(1))
    for r in rows_v2.trained(rows):
        if r["meta"].get("clip_frames") is not None and r["meta"]["clip_frames"] < rows_v2.MIN_FRAMES:
            bad.append(("S2 trained row with a video under MIN_FRAMES", f"{name} turn {r['meta']['turn']}"))
    del two_mover
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    work = Path.home() / "selfimprove_work"
    folders = []
    for exp in args.exps:
        for root in (work / exp / "iter_00" / "rollout", work / exp / "rollout"):
            if root.exists():
                folders += [str(p.parent) for p in sorted(root.glob("*/*/result.json"))]
                break
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(check_episode, folders))
    by_rule = collections.defaultdict(list)
    for bad in results:
        for rule, where in bad:
            by_rule[rule].append(where)
    print(f"{len(folders)} episodes checked ({', '.join(args.exps)})")
    if not by_rule:
        print("all rules hold")
    for rule, wheres in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(wheres):5}  {rule}: {'; '.join(wheres[:4])}")
    return 1 if by_rule else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare two rollout experiments that played the same (task, seed) plan (an A/B of HexaShell settings).

    .venv/bin/python scripts/ab_compare.py ab108_with ab108_without [--iteration 0] [--out docs/ab108.md]

Reads rollout/<task>/<seed>/rollout.json of both from S3 (episodes with status 'ok' only; an episode one arm has
not finished is left out of both, so the comparison is paired). Prints and writes: success per arm overall, per
kind and per family (assets/selected_tasks.json), per task, the paired difference, steps used, turns, System1's
step share, Astra's prompt tokens; and the tasks to keep for the self-improvement rollouts (`--keep-min`: the
better arm's success at least this; the arm with the prompts must not be worse by more than one episode).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402
from selfimprove.config import Config  # noqa: E402


def load(exp: str, k: int) -> dict[tuple[str, int], dict]:
    from concurrent.futures import ThreadPoolExecutor

    cfg = Config.load(None, [f"exp_id={exp}"])
    plan = s3.read_json(f"{cfg.rollout_uri(k)}/_plan.json", default=None) or {"episodes": []}
    keys = [(e["task"], int(e["seed"])) for e in plan["episodes"]]
    with ThreadPoolExecutor(32) as pool:
        rows = pool.map(lambda key: s3.read_json(f"{cfg.episode_uri(k, *key)}/rollout.json", default=None), keys)
    return {key: r for key, r in zip(keys, rows) if r and r.get("status") == "ok"}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-min", type=float, default=0.6)
    args = ap.parse_args()
    cfg = Config.load()
    s3.configure(cfg.get("aws_profile"), cfg.region)
    A, B = load(args.a, args.iteration), load(args.b, args.iteration)
    both = sorted(set(A) & set(B))
    sel = {r["task"]: r for r in json.loads((ROOT / "assets/selected_tasks.json").read_text())["tasks"]}
    by_task = collections.defaultdict(list)
    for key in both:
        by_task[key[0]].append(key)

    def row(keys):
        return {arm: {"n": len(keys), "success": mean(bool(R[k].get("success")) for k in keys),
                      "steps": mean(R[k].get("step") or 0 for k in keys if R[k].get("success")),
                      "turns": mean(R[k].get("turns") or 0 for k in keys),
                      "s1_share": mean(float(R[k].get("s1_step_share") or 0) for k in keys),
                      "tokens_k": mean((R[k].get("tokens_prompt") or 0) / 1000 for k in keys)}
                for arm, R in (("a", A), ("b", B))}

    groups = {"all": both}
    for kind in ("atomic", "composite"):
        groups[kind] = [k for k in both if sel.get(k[0], {}).get("kind") == kind]
    fams = collections.defaultdict(list)
    for k in both:
        if sel.get(k[0], {}).get("family"):
            fams[sel[k[0]]["family"]].append(k)
    lines = [f"# A/B: {args.a} (a) vs {args.b} (b), iteration {args.iteration}", "",
             f"{len(both)} paired episodes on {len(by_task)} tasks ({len(A)} ok in a, {len(B)} ok in b).", "",
             "| group | n | success a | success b | a - b | steps a | steps b | turns a | turns b | S1 share a | S1 share b"
             " | tokens a (k) | tokens b (k) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(name, keys):
        r = row(keys)
        a, b = r["a"], r["b"]
        return (f"| {name} | {a['n']} | {a['success']:.1%} | {b['success']:.1%} | {a['success'] - b['success']:+.1%} | "
                f"{a['steps']:.0f} | {b['steps']:.0f} | {a['turns']:.1f} | {b['turns']:.1f} | {a['s1_share']:.2f} | "
                f"{b['s1_share']:.2f} | {a['tokens_k']:.0f} | {b['tokens_k']:.0f} |")

    for name, keys in groups.items():
        lines.append(fmt(f"**{name}**", keys))
    for fam, keys in sorted(fams.items()):
        lines.append(fmt(fam, keys))
    lines += ["", "## Per task", "", "| task | kind | family | n | a | b | a - b | keep |", "|---|---|---|---|---|---|---|---|"]
    keep, table = [], []
    for task in sorted(by_task, key=lambda t: (sel.get(t, {}).get("kind", ""), sel.get(t, {}).get("family") or "", t)):
        keys = by_task[task]
        sa = sum(bool(A[k].get("success")) for k in keys)
        sb = sum(bool(B[k].get("success")) for k in keys)
        n = len(keys)
        ok = max(sa, sb) / n >= args.keep_min
        if ok:
            keep.append(task)
        table.append({"task": task, "n": n, "a": sa, "b": sb, "keep": ok, "kind": sel.get(task, {}).get("kind"),
                      "family": sel.get(task, {}).get("family")})
        lines.append(f"| {task} | {sel.get(task, {}).get('kind')} | {sel.get(task, {}).get('family') or '-'} | {n} | "
                     f"{sa}/{n} | {sb}/{n} | {sa - sb:+d} | {'yes' if ok else 'no'} |")
    worse = [t["task"] for t in table if t["a"] < t["b"] - 1]
    lines += ["", f"Keep (better arm >= {args.keep_min:.0%}): {len(keep)} of {len(table)} tasks.",
              f"Prompts worse by 2+ episodes (revise those notes): {', '.join(worse) or 'none'}."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        args.out.with_suffix(".json").write_text(json.dumps({"a": args.a, "b": args.b, "tasks": table, "keep": keep,
                                                              "prompts_worse": worse}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

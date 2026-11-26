"""Per-span checks of System1 training shards (used by preprocessing on all shards and by scripts/build_shard_index.py)."""

from __future__ import annotations

import collections
import json
import tarfile


def s1_scan(path: str) -> list[dict]:
    """Every sample of one S1 shard in tar order: key + the meta fields the checks need."""
    out = []
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.name.endswith(".meta.json"):
                meta = json.loads(t.extractfile(m).read())
                out.append({"key": m.name[:-len(".meta.json")], "span": meta.get("span"), "f": meta.get("frame_index"),
                            "kind": meta.get("si_kind"), "src": meta.get("si_source"), "turns": meta.get("si_turns"),
                            "sub": meta.get("subgoal"), "est": meta.get("est_length"), "p": meta.get("subgoal_progress_frac"),
                            "fk": meta.get("frame_kind"), "grip": meta.get("gripper_flag"), "task": meta.get("task"),
                            "seed": meta.get("seed"), "q": meta.get("quality")})
    return out


def s1_spans(samples: list[dict]) -> list[dict]:
    by = collections.defaultdict(list)
    for r in samples:
        by[(r["task"], r["seed"], tuple(r["span"] or ()), r["kind"])].append(r)
    spans = []
    for (task, seed, span, kind), rs in by.items():
        rs.sort(key=lambda r: r["f"])
        s, e = span
        real = [r for r in rs if r["fk"] != "halt" and r["f"] <= e]
        halt = [r for r in rs if r["fk"] == "halt" or r["f"] > e]
        fs = [r["f"] for r in real]
        issues = []
        if fs and fs != list(range(fs[0], fs[-1] + 1)):
            issues.append("frames not contiguous")
        if fs and fs[-1] != e:
            issues.append(f"last frame {fs[-1]} != span end {e}")
        if fs and fs[0] != s and kind != "recovery_within":
            issues.append(f"first frame {fs[0]} != span start {s}")
        ps = [r["p"] for r in real if r["p"] is not None]
        if ps and any(b < a - 1e-6 for a, b in zip(ps, ps[1:])):
            issues.append("progress not monotone")
        if ps and not (0 <= ps[0] <= 1 and 0 <= ps[-1] <= 1):
            issues.append("progress outside [0, 1]")
        if len({r["sub"] for r in rs}) > 1:
            issues.append("several subgoals")
        if len(halt) > 1:
            issues.append(f"{len(halt)} halt frames")
        spans.append({"task": task, "seed": seed, "s": s, "e": e, "kind": kind, "src": rs[0]["src"], "turns": rs[0]["turns"],
                      "sub": rs[0]["sub"], "est": rs[0]["est"], "n": len(rs), "real": len(real), "halt": len(halt),
                      "first": real[0]["i"] if real else rs[0]["i"], "halt_i": halt[0]["i"] if halt else None,
                      "from": fs[0] if fs else None, "p0": ps[0] if ps else None, "p1": ps[-1] if ps else None,
                      "grip": rs[0]["grip"], "issues": issues})
    spans.sort(key=lambda x: (x["task"], int(x["seed"]), x["s"]))
    return spans

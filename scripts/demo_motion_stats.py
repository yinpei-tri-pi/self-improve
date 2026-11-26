"""Per-primitive motion statistics of RoboCasa's demonstrations (the S1 base set), for the smoothness filter on harness
spans: per control step |eef dpos|, |eef drot|, |base vel| and the second differences (jerk) of the eef deltas, from
the 20-step action labels of the samples of ``--shards`` globally shuffled base shards.

    .venv/bin/python scripts/demo_motion_stats.py [--shards 20] [--out assets/demo_motion_stats.json]

Primitive = the subgoal's verb ("continue to X" / "X again" count as X's). Writes p50/p95/p99/p99.9 per primitive and
metric (lean action units: the controller's normalized deltas, as in the rollouts' trajectories).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402

BASE = "s3://tri-ml-datasets-uw2/yinpeidai/robocasa_final_training_shard/system1_fullset_0717/shards/"
METRICS = ("pos", "rot", "base", "pos_jerk", "rot_jerk")


def primitive(subgoal: str) -> str:
    s = re.sub(r"^(continue to|finish|keep)\s+", "", str(subgoal or "").strip().lower())
    s = re.sub(r"\s+again\.?$", "", s)
    return s.split()[0] if s else "-"


def motion(a: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step metrics of a [T, 11] lean action sequence (base 0:3, mode 3, eef dpos 4:7, eef drot 7:10, grip 10)."""
    out = {"pos": np.linalg.norm(a[:, 4:7], axis=1), "rot": np.linalg.norm(a[:, 7:10], axis=1),
           "base": np.linalg.norm(a[:, 0:3], axis=1)}
    out["pos_jerk"] = np.linalg.norm(np.diff(a[:, 4:7], n=2, axis=0), axis=1) if len(a) > 2 else np.zeros(0)
    out["rot_jerk"] = np.linalg.norm(np.diff(a[:, 7:10], n=2, axis=0), axis=1) if len(a) > 2 else np.zeros(0)
    return out


def one_shard(uri: str) -> dict[str, dict[str, list]]:
    bucket, key = s3.split(uri)
    body = s3.client().get_object(Bucket=bucket, Key=key)["Body"]
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cur, arr, meta = None, None, None
    with tarfile.open(fileobj=body, mode="r|") as tf:
        for m in tf:
            k, ext = m.name.split(".", 1)
            if ext not in ("arrays.npz", "meta.json"):
                continue
            data = tf.extractfile(m).read()
            if k != cur:
                cur, arr, meta = k, None, None
            if ext == "arrays.npz":
                arr = np.load(io.BytesIO(data))
            else:
                meta = json.loads(data)
            if arr is not None and meta is not None:
                a = np.asarray(arr["lean_action_subgoal_pad"], float)
                mask = np.asarray(arr["action_pad_mask_subgoal"], bool)
                a = a[mask] if mask.any() else a[:0]
                if len(a) >= 3:
                    prim = primitive(meta.get("subgoal"))
                    for name, v in motion(a).items():
                        acc[prim][name].extend(np.round(v, 5).tolist())
                arr, meta = None, None
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "demo_motion_stats.json")
    args = ap.parse_args()
    s3.configure("sagemaker", "us-west-2")
    keys = sorted(u for u, _ in s3.iter_keys(BASE) if u.endswith(".tar"))
    pick = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, args.shards)]
    total: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(8) as pool:
        for acc in pool.map(one_shard, pick):
            for prim, d in acc.items():
                for name, v in d.items():
                    total[prim][name].extend(v)
    out = {"source": f"{args.shards} base shards of system1_fullset_0717 ({BASE})", "units": "lean action units per control step",
           "primitives": {}}
    for prim, d in sorted(total.items(), key=lambda kv: -len(kv[1]["pos"])):
        n = len(d["pos"])
        if n < 200:
            continue
        out["primitives"][prim] = {"steps": n, **{name: {q: round(float(np.percentile(d[name], float(q[1:]))), 5)
                                                         for q in ("p50", "p95", "p99", "p99.9")} for name in METRICS}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for prim, d in list(out["primitives"].items())[:20]:
        print(f"{prim:10} steps {d['steps']:7}  " + "  ".join(f"{m} p99 {d[m]['p99']:.3f}" for m in METRICS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

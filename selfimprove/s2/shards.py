"""Rollout episodes -> System2 training shards (the ``system2-shards`` layout of sys2_train_eval's plugin).

    .venv/bin/python -m selfimprove.s2.shards sub50_check_20261125 [more exps] --iterations 0 \\
        --out ~/selfimprove_work/sub50_check_20261125/s2_training_shard [--workers 8] [--min-shards 64]

Every episode of the given rounds (failures too; composite_unseen tasks refused by rows_v2) -> its rows
(``rows_v2.episode_rows``), the trained ones (``rows_v2.trained``: no video under MIN_FRAMES frames) written as
``<key>.json`` ({messages, meta, labels}) + ``<key>.videobin`` (the last step's clip) or ``<key>.imagebin0`` (the
opening scene), shuffled across at least ``--min-shards`` shards: the trainer gives every rank whole shards, so a small
set still needs many. ``manifest.jsonl`` (num_samples per shard), ``meta.json`` (num_samples, per_task, per kind /
judge / decision), README.md. The old System2 set (convert_full0804) is a separate shard root that the trainer
mixes in at load time (SYS2_BASE_SHARD_ROOT, SYS2_NEW_FRACTION).
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WORK = Path.home() / "selfimprove_work"


def _key(exp: str, k: int, task: str, seed: int, turn: int, kind: str) -> str:
    return f"si__{exp}__i{k:02d}__{task}__s{seed}__t{turn:03d}" + ("__closing" if kind == "closing" else "")


def episode_records(job: tuple[str, str, int, str]) -> list[dict]:
    """One episode -> [{key, json, media_name, media_path}] (media copied into the build's media dir)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from selfimprove.s2 import rows_v2

    folder, exp, k, media_dir = job
    with tempfile.TemporaryDirectory() as tmp:
        rows = rows_v2.episode_rows(folder, Path(tmp), make_media=True)
        out = []
        for r in rows_v2.trained(rows):
            m = r["meta"]
            key = _key(exp, k, m["task"], int(m["seed"]), int(m["turn"]), m["kind"])
            m.update(source="rollout", exp=exp, iteration=k)
            if m.get("clip"):
                src, member = Path(tmp) / m["clip"], "videobin"
            else:
                src, member = Path(tmp) / m["image"], "imagebin0"
            dst = Path(media_dir) / f"{key}.{member}"
            shutil.copyfile(src, dst)
            out.append({"key": key, "json": json.dumps(r), "member": member, "media": str(dst),
                        "task": m["task"], "kind": m["kind"], "judge": r["labels"]["judge"],
                        "decision": m.get("decision"), "success": bool(m.get("episode_success"))})
        return out


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0
    tar.addfile(info, io.BytesIO(data))


def build(exps: list[str], iterations: list[int], out: Path, *, workers: int = 8, min_shards: int = 64,
          per_shard: int = 2048, seed: int = 0) -> dict:
    t0 = time.time()
    jobs = []
    if out.exists():
        shutil.rmtree(out)
    (out / "shards").mkdir(parents=True)
    media = out / "_media"
    media.mkdir()
    for exp in exps:
        for k in iterations:
            for root in (WORK / exp / f"iter_{k:02d}" / "rollout", WORK / exp / "rollout"):
                if root.exists():
                    jobs += [(str(p.parent), exp, k, str(media)) for p in sorted(root.glob("*/*/result.json"))]
                    break
    with ProcessPoolExecutor(workers) as pool:
        recs = [r for rs in pool.map(episode_records, jobs) for r in rs]
    keys = [r["key"] for r in recs]
    assert len(keys) == len(set(keys)), "duplicate S2 keys"
    random.Random(seed).shuffle(recs)
    n_shards = max(min_shards, -(-len(recs) // per_shard))
    n_shards = min(n_shards, len(recs)) or 1
    manifest = []
    for i in range(n_shards):
        part = recs[i::n_shards]
        name = f"shards/shard_{i:05d}.tar"
        with tarfile.open(out / name, "w") as tar:
            for r in part:
                _add(tar, f"{r['key']}.json", r["json"].encode())
                _add(tar, f"{r['key']}.{r['member']}", Path(r["media"]).read_bytes())
        manifest.append({"shard": name, "num_samples": len(part)})
    shutil.rmtree(media)
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    C = lambda f: dict(collections.Counter(f(r) for r in recs).most_common())  # noqa: E731
    meta = {"dataset": f"selfimprove/{'+'.join(exps)}/s2_training_shard",
            "format": "system2-shards (sys2_train_eval): <key>.json {messages, meta, labels} + <key>.videobin | "
                      "<key>.imagebin0; rows_v2 single-mode S2 rows",
            "experiments": exps, "iterations": iterations, "episodes": len(jobs), "num_shards": n_shards,
            "num_samples": len(recs), "per_kind": C(lambda r: r["kind"]), "per_judge": C(lambda r: r["judge"]),
            "per_decision": C(lambda r: r["decision"]), "from_successful_episodes": sum(r["success"] for r in recs),
            "per_task": C(lambda r: r["task"]), "seed": seed,
            "clip": {"fps": 4.0, "member": "videobin", "max_frames": 32, "min_frames": 4, "size": "768x256"},
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "README.md").write_text(
        f"# System2 rollout rows: {', '.join(exps)} (rounds {iterations})\n\n"
        f"{meta['num_samples']} rows from {meta['episodes']} episodes in {n_shards} shards; built by "
        f"selfimprove/s2/shards.py from selfimprove/s2/rows_v2.py rows (docs/data_preprocessing.md).\n\n"
        f"- kinds: {meta['per_kind']}\n- judges: {meta['per_judge']}\n- decisions: {meta['per_decision']}\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--iterations", default="0", help="comma-separated rounds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-shards", type=int, default=64)
    args = ap.parse_args()
    meta = build(args.exps, [int(x) for x in args.iterations.split(",")], args.out.expanduser(),
                 workers=args.workers, min_shards=args.min_shards)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_task"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

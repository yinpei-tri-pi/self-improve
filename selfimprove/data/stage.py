"""The preprocessing stage of a round: download the rollouts, score and select segments, write the S1 shards.

``iter_k/rollout/<task>/<seed>/`` (only what the shards need: result.json, events.jsonl, rollout.json,
chunk_scores.jsonl, trajectory/; not the pictures the model saw) -> ``<work_dir>/iter_k/rollout`` ->
``<work_dir>/iter_k/s1_training_shard`` -> ``iter_k/s1_training_shard/`` on S3, with ``_DONE``.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import s3
from .. import tasks as T
from ..config import Config
from . import s1_shards

NEEDED = ("result.json", "events.jsonl", "rollout.json", "chunk_scores.jsonl", "trajectory/*")


def planned_episodes(cfg: Config, k: int) -> list[dict]:
    plan = s3.read_json(f"{cfg.rollout_uri(k)}/_plan.json", default=None)
    if plan is None:
        raise RuntimeError(f"iteration {k} has no rollout/_plan.json")
    return plan["episodes"]


def download_round(cfg: Config, k: int, dest: Path, workers: int = 16, log=print) -> list[tuple[str, str, int]]:
    """(local folder, task, seed) of every planned episode that was played (has rollout.json); never one of a
    held-out (composite_unseen) task."""
    eps = planned_episodes(cfg, k)
    held = T.held_out_tasks()
    skipped = sorted({e["task"] for e in eps if e["task"] in held})
    if skipped:
        log(f"iteration {k}: held-out composite_unseen tasks left out of the training data: {', '.join(skipped)}")
        eps = [e for e in eps if e["task"] not in held]

    def one(e):
        uri = cfg.episode_uri(k, e["task"], e["seed"])
        local = dest / e["task"] / str(e["seed"])
        if (local / "rollout.json").exists() and (local / "trajectory" / "trajectory.json").exists():
            return str(local), e["task"], int(e["seed"])
        if not s3.exists(f"{uri}/rollout.json"):
            return None
        s3.sync_down(uri, local, excludes=("*",), includes=NEEDED)
        return str(local), e["task"], int(e["seed"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = [r for r in ex.map(one, eps) if r is not None]
    log(f"iteration {k}: {len(out)}/{len(eps)} planned episodes downloaded to {dest}")
    return out


def prune_rollouts(cfg: Config, k: int, log=print) -> None:
    """Raw rollout copies of at most filter.keep_rollout_rounds rounds on this machine (the rest are on S3): before
    round k's download, the copies of rounds <= k - keep are removed, each only if its rollout _DONE is on S3."""
    keep = int(cfg["filter"].get("keep_rollout_rounds", 3))
    for j in range(0, int(k) - keep + 1):
        d = cfg.work_dir / f"iter_{j:02d}" / "rollout"
        if d.exists() and s3.read_json(f"{cfg.rollout_uri(j)}/_DONE", default=None) is not None:
            shutil.rmtree(d)
            log(f"iteration {k}: removed round {j}'s local rollout copy (on S3: {cfg.rollout_uri(j)})")


def prune_shards(out: Path, uri: str, keep: int, log=print) -> None:
    """After a confirmed upload (uri/_DONE): only the first `keep` shards stay on this machine, for checking."""
    if s3.read_json(f"{uri}/_DONE", default=None) is None:
        log(f"{out}: no {uri}/_DONE, local shards kept")
        return
    shutil.rmtree(out / "_tmp", ignore_errors=True)
    tars = sorted((out / "shards").glob("*.tar"))
    for t in tars[keep:]:
        t.unlink()
    log(f"{out}: kept {min(keep, len(tars))} of {len(tars)} shards locally (all on {uri})")


def run(cfg: Config, k: int, *, upload: bool = True, keep_local: bool = True, log=print) -> dict:
    work = cfg.work_dir / f"iter_{int(k):02d}"
    if upload:
        prune_rollouts(cfg, k, log=log)
    episodes = download_round(cfg, k, work / "rollout", log=log)
    out = work / "s1_training_shard"
    if out.exists():
        shutil.rmtree(out)
    f = cfg["filter"]
    meta = s1_shards.build(episodes, out, exp_id=cfg.exp_id, iteration=k, fcfg=f, workers=int(f.get("workers", 16)),
                           log=log)
    if upload:
        s3.sync_up(out, cfg.s1_shard_uri(k), delete=True)
        s3.write_json(f"{cfg.s1_shard_uri(k)}/_DONE", meta)
    if not keep_local:
        shutil.rmtree(out / "shards", ignore_errors=True)
    elif upload:
        prune_shards(out, cfg.s1_shard_uri(k), int(f.get("keep_local_s1_shards", 8)), log=log)
    meta["s2"] = s2_rows(cfg, k, work, upload=upload, log=log)
    if upload and meta["s2"].get("status") == "done":
        prune_shards(work / "s2_training_shard", s2_uri(cfg, k), int(f.get("keep_local_s2_shards", 4)), log=log)
    return meta


def s2_uri(cfg: Config, k: int) -> str:
    return f"{cfg.exp_uri}/iter_{int(k):02d}/s2_training_shard"


def s2_rows(cfg: Config, k: int, work: Path, *, upload: bool = True, log=print) -> dict:
    """The round's System2 rows (selfimprove/s2/shards.py, every episode, failures too) -> iter_k/s2_training_shard/ on
    S3 with _DONE. S2 is optional (only the evaluation needs it): a failure here is recorded, it does not stop S1."""
    from ..s2 import shards as s2_shards

    out = work / "s2_training_shard"
    try:
        meta = s2_shards.build([cfg.exp_id], [int(k)], out, workers=int(cfg["filter"].get("workers", 4)))
    except Exception as exc:  # noqa: BLE001
        log(f"iteration {k}: the S2 rows failed: {type(exc).__name__}: {exc}")
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}
    if upload:
        s3.sync_up(out, s2_uri(cfg, k), delete=True)
        s3.write_json(f"{s2_uri(cfg, k)}/_DONE", meta)
    log(f"iteration {k}: {meta['num_samples']} S2 training samples in {meta['num_shards']} shards")
    return {"status": "done", "uri": s2_uri(cfg, k), **{a: meta[a] for a in ("num_samples", "num_shards", "per_kind",
                                                                              "per_judge", "per_decision", "built", "seconds")}}


def summary_metrics(meta: dict) -> dict:
    """The data/* points of the loop's W&B run."""
    out = {f"data/{k}": meta[k] for k in ("episodes", "episodes_used", "segments", "samples", "positive_samples",
                                          "negative_samples", "shards") if k in meta}
    for k, v in (meta.get("segment_use") or {}).items():
        out[f"data/use/{k.replace('None', 'dropped')}"] = v
    for k, v in (meta.get("segment_scores") or {}).items():
        out[f"data/score/{k}"] = v
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    s3.configure(cfg.get("aws_profile"), cfg.region)
    meta = run(cfg, args.iteration, upload=not args.no_upload)
    print(json.dumps({k: v for k, v in meta.items() if k != "filter"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

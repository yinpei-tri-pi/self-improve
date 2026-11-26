"""Extract every recorded RoboCasa episode's initial simulator state from the downloaded LeRobot datasets.

    .venv/bin/python scripts/extract_sim_states.py [--data ~/data/robocasa_dataset/v1.0]
        [--out ~/selfimprove_work/sim_states/v1.0] [--upload]

Each LeRobot dataset (``<split>/<atomic|composite>/<Task>/<date>/lerobot/``) keeps, per episode, ``extras/episode_N/``
with ``states.npz`` (the flattened MuJoCo state of every recorded step), ``model.xml.gz`` (the scene: kitchen layout and
style, fixtures, objects, robot) and ``ep_meta.json`` (layout/style ids, fixture refs, object configs, the instruction).
An episode starts exactly as recorded with RoboCasa's own ``reset_to(env, {"states": states[0], "model": xml,
"ep_meta": json})`` (robocasa/scripts/dataset_scripts/playback_dataset.py, as openpi's combined_eval.py does for the
target protocol). So starting episodes from these states uses only the dataset's own kitchens and scenes, never a
layout RoboCasa would draw for a fresh seed.

Writes ``<out>/<split>/<Task>/<date>_episode_NNNNNN.npz`` (``state``: states[0]; ``model_xml_gz``: the scene as
recorded; ``ep_meta``: its JSON; ``n_frames``: the demo's length), ``<out>/<split>/<Task>/index.json`` (one row per
episode: id, layout/style ids, instruction, frames, source) and ``<out>/manifest.json``. ``--upload`` syncs ``<out>``
to ``s3://.../self_improve/sim_states/v1.0/``. The target split is extracted to ``target/`` for the evaluation only:
the development rollouts read ``pretrain/`` (assets/rollout_tasks.json lists the tasks they use).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

S3_PREFIX = "s3://tri-ml-datasets-uw2/yinpeidai/self_improve/sim_states/v1.0"


def episode(args: tuple[str, str, str]) -> dict:
    """One episode -> its npz; returns its index row."""
    ep_dir, out_path, ep_id = args
    ep_dir = Path(ep_dir)
    with np.load(ep_dir / "states.npz", allow_pickle=False) as z:
        states = z["states"]
    meta_text = (ep_dir / "ep_meta.json").read_text()
    meta = json.loads(meta_text)
    xml_gz = (ep_dir / "model.xml.gz").read_bytes()
    gzip.decompress(xml_gz)  # a truncated download fails here, not at reset time
    np.savez_compressed(out_path, state=states[0].astype(np.float64), model_xml_gz=np.frombuffer(xml_gz, np.uint8),
                        ep_meta=np.frombuffer(meta_text.encode(), np.uint8), n_frames=np.int64(len(states)))
    return {"id": ep_id, "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"),
            "lang": meta.get("lang"), "n_frames": int(len(states)), "state_dim": int(states.shape[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("~/data/robocasa_dataset/v1.0").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/selfimprove_work/sim_states/v1.0").expanduser())
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    jobs, where = [], {}
    for info in sorted(args.data.glob("*/*/*/*/lerobot/meta/info.json")):
        root = info.parent.parent                       # .../<split>/<kind>/<Task>/<date>/lerobot
        split, kind, task, date = root.relative_to(args.data).parts[:4]
        out_dir = args.out / split / task
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in sorted((root / "extras").glob("episode_*")):
            ep_id = f"{date}_{ep.name}"
            jobs.append((str(ep), str(out_dir / f"{ep_id}.npz"), ep_id))
            where[(split, task, ep_id)] = (kind, str(ep.relative_to(args.data)))
    print(f"{len(jobs)} episodes in {len({(s, t) for s, t, _ in where})} split/task pairs", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(episode, jobs, chunksize=64))
    by_task: dict[tuple[str, str], list[dict]] = {}
    for job, row in zip(jobs, rows):
        out_path = Path(job[1])
        split, task = out_path.parent.parent.name, out_path.parent.name
        kind, src = where[(split, task, row["id"])]
        by_task.setdefault((split, task), []).append({**row, "kind": kind, "source": src})
    manifest = {"source": str(args.data), "s3": S3_PREFIX, "created": time.strftime("%F %T"),
                "reset": "robocasa.scripts.dataset_scripts.playback_dataset.reset_to(env, {states: state, model: "
                         "gunzip(model_xml_gz), ep_meta: ep_meta})",
                "use": {"pretrain": "development rollouts (pretrain kitchens and objects)",
                        "target": "evaluation only: never for development or training"},
                "splits": {}}
    for (split, task), eps in sorted(by_task.items()):
        eps.sort(key=lambda r: r["id"])
        (args.out / split / task / "index.json").write_text(json.dumps(
            {"split": split, "task": task, "kind": eps[0]["kind"], "n_episodes": len(eps), "episodes": eps}, indent=1))
        s = manifest["splits"].setdefault(split, {"tasks": 0, "episodes": 0, "per_task": {}})
        s["tasks"] += 1
        s["episodes"] += len(eps)
        s["per_task"][task] = {"kind": eps[0]["kind"], "episodes": len(eps),
                               "layouts": len({r["layout_id"] for r in eps}),
                               "kitchens": len({(r["layout_id"], r["style_id"]) for r in eps})}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(json.dumps({k: {"tasks": v["tasks"], "episodes": v["episodes"]} for k, v in manifest["splits"].items()}),
          f"{size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    if args.upload:
        env = {**os.environ, "AWS_PROFILE": os.environ.get("AWS_PROFILE", "sagemaker")}
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", str(args.out), S3_PREFIX], check=True, env=env)
        print(f"uploaded to {S3_PREFIX}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

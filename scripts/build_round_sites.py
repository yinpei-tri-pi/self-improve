"""Build one round's two websites (the dashboard links them): its rollout site and its preprocessed-data site.

    .venv/bin/python scripts/build_round_sites.py --config configs/dev120.yaml --iteration 0 [--jobs 8]

- rollout site ``site/debug_<exp>_iter_NN/`` (scripts/build_debug_site.py): every episode, every turn, its S1 spans and
  S2 rows, from the round's local copy (<work_dir>/iter_NN/rollout, downloaded by the preprocessing stage);
- preprocessed data ``site/shards_<exp>_iter_NN/`` (scripts/build_shard_index.py): every S1 span and S2 row of the
  round's shards with the automatic checks, each linked into RoboAnnotator's shard inspectors, which read a view of the
  round's shards: <work root>/shard_view/{s1,s2}/<exp>_iter_NN (hard links to the shards, nothing copied; the
  inspectors' server is pointed at <work root>/shard_view by SYSTEM1_SHARDS_EXTRA / SYSTEM2_SHARDS_EXTRA).

Writes ``site/rounds/<exp>_iter_NN.json`` ({rollout_site, data_site, view names, built}) for the dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_view(src: Path, dst: Path, s2: bool) -> None:
    """dst/shards/*.tar hard links to src/shards/*.tar, dst/shards/manifest.jsonl (shard basenames), dst/meta.json."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "shards").mkdir(parents=True)
    for tar in sorted((src / "shards").glob("*.tar")):
        os.link(tar, dst / "shards" / tar.name)
    man = src / "shards" / "manifest.jsonl"
    if not man.exists():
        man = src / "manifest.jsonl"
    rows = [json.loads(x) for x in man.read_text().splitlines() if x.strip()]
    (dst / "shards" / "manifest.jsonl").write_text(
        "".join(json.dumps({"shard": Path(r["shard"]).name, "num_samples": r["num_samples"]}) + "\n" for r in rows))
    meta = json.loads((src / "meta.json").read_text())
    if s2:  # RoboAnnotator's System2 inspector lists datasets by this tag
        meta["source_dataset"], meta["dataset"] = meta.get("dataset"), "robocasa_system2"
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    norm = Path("~/openpi-si/assets/pi05_robocasa_system1/robocasa_system1/norm_stats.json").expanduser()
    if not s2 and norm.exists():
        shutil.copy(norm, dst / "norm_stats.json")


def main() -> int:
    from selfimprove.config import WORK_ROOT, Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--inspector", default="http://127.0.0.1:7891")
    ap.add_argument("--skip-rollout-site", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config, [])
    k = int(args.iteration)
    tag = f"{cfg.exp_id}_iter_{k:02d}"
    work = cfg.work_dir / f"iter_{k:02d}"
    py = sys.executable
    t0 = time.time()
    out = {"exp": cfg.exp_id, "iteration": k}
    if not args.skip_rollout_site:
        site = ROOT / "site" / f"debug_{tag}"
        subprocess.run([py, str(ROOT / "scripts/build_debug_site.py"), cfg.exp_id, "--iteration", str(k), "--out", str(site),
                        "--jobs", str(args.jobs), "--reuse-media"], check=True, cwd=ROOT)
        out["rollout_site"] = f"debug_{tag}/index.html"
    view = Path(WORK_ROOT) / "shard_view"
    s1, s2 = work / "s1_training_shard", work / "s2_training_shard"
    if (s1 / "shards").exists():
        make_view(s1, view / "s1" / tag, s2=False)
    if (s2 / "shards").exists():
        make_view(s2, view / "s2" / tag, s2=True)
    cmd = [py, str(ROOT / "scripts/build_shard_index.py"), "--view", str(view), "--out", str(ROOT / "site" / f"shards_{tag}"),
           "--s1", tag, "--s2", tag, "--s2-old", "", "--inspector", args.inspector, "--workers", "12"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    out.update(data_site=f"shards_{tag}/index.html", view=tag, inspector=args.inspector,
               built=time.strftime("%Y-%m-%d %H:%M"), seconds=round(time.time() - t0))
    (ROOT / "site" / "rounds").mkdir(parents=True, exist_ok=True)
    (ROOT / "site" / "rounds" / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build ``site/dashboard/``: one page with the whole self-improvement loop of an experiment.

    .venv/bin/python scripts/build_dashboard.py --mock [--exp si_sub50] [--out site/dashboard]

The page reads one state document (``state.json``, also inlined into ``index.html`` so the file opens without a
server; served over http it polls ``state.json`` every minute). The state lists the experiment's iterations, and per
iteration its stages with their status and facts:

- ``rollout``    System1 (the previous iteration's checkpoint) + HexaShell on the development tasks;
- ``preprocess`` the round's S1 shards and S2 rows;
- ``s1_train``   System1 continue-trained on the round's shards (the next rollout waits only for this);
- ``s2_train``   System2 trained on all rounds' rows so far (optional: only the evaluation needs it);
- ``eval``       System1 k + System2 k on the held-out target split.

``--mock`` writes placeholder numbers (real ones where the loop already has them, e.g. round 0's rollout); a collector
reading S3 (each stage's ``_status`` / ``_DONE``) and the training jobs fills the same document.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S3 = "s3://tri-ml-datasets-uw2/yinpeidai/self_improve"
S1_START = ("s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/f0717-270k-bs512-progact/pi05_robocasa_system1/"
            "f0717-270k-bs512-progact__progact_granfine_verbsimp_noexec/269999")


def curve(n_points: int, steps: int, start: float, end: float, noise: float, seed: int, upto: int | None = None):
    """A placeholder loss curve: an exponential decay from start toward end with noise, [[step, value], ...]."""
    rnd = random.Random(seed)
    out = []
    for i in range(n_points + 1):
        step = round(steps * i / n_points)
        if upto is not None and step > upto:
            break
        v = end + (start - end) * math.exp(-4.0 * step / steps) + rnd.gauss(0, noise)
        out.append([step, round(max(end * 0.7, v), 4)])
    return out


def mock_state(exp: str) -> dict:
    now = datetime(2026, 11, 25, 12, 0)
    it = lambda k: f"{S3}/{exp}/iter_{k:02d}"  # noqa: E731
    s1_loss_0 = curve(60, 15000, 0.110, 0.052, 0.002, 1, upto=9400)
    s1_val_0 = curve(12, 15000, 0.118, 0.061, 0.0015, 2, upto=9400)
    s2_loss_0 = curve(40, 6000, 1.45, 0.42, 0.02, 3, upto=2100)
    iters = [
        {"k": 0, "label": "round 0",
         "rollout": {"status": "done", "s1_ckpt": "start (270k)", "s1_ckpt_uri": S1_START,
                     "episodes": 1500, "planned": 1500, "success": 0.920, "success_note": "re-scored (fixture fix); 90.3% as recorded",
                     "by_set": {"atomic_seen": 0.841, "composite_seen": 0.921, "pretrain_atomic": 0.951, "pretrain_composite": 0.879},
                     "s1_step_share": 0.843, "turns_mean": 18.3, "prompt_tokens_mean": 566688, "cached_frac": 0.886,
                     "episode_minutes": 3.5, "nodes": 16, "hours": 1.61, "started": "2026-11-24 20:14", "finished": "2026-11-24 21:51",
                     "perturb": "off (recorded starts)", "uri": f"{it(0)}/rollout/", "site": "../debug_si_sub50/index.html"},
         "preprocess": {"status": "done", "finished": "2026-11-25 10:31",
                        "s1": {"spans": 17246, "samples": 945469, "shards": 462, "size_gb": 134.2, "episodes_used": 1355,
                               "recovery_spans": 737, "uri": f"{it(0)}/s1_training_shard/"},
                        "s2": {"rows": 26152, "decisions": {"system1": 17382, "expert": 7415}, "closing": 1355,
                               "uri": f"{it(0)}/s2_rows/"}},
         "s1_train": {"status": "running", "job": "yinpei-pi05-robocasa-mn-si-sub50-iter00", "instances": "2 x p5 (16 H100)",
                      "from_ckpt": "start (270k)", "batch": 256, "steps": 15000, "step": 9400, "new_frac": 0.8,
                      "data": "945k rollout samples (80%) + base set (20%)", "lr": "5e-5 flat, EMA 0.999",
                      "started": "2026-11-25 10:50", "eta": "2026-11-25 15:40", "sec_per_step": 1.15,
                      "loss": s1_loss_0, "val_loss": s1_val_0,
                      "ckpts": [{"step": 5000, "uri": f"{it(0)}/s1_ckpts/si-sub50-iter00/4999"}],
                      "ckpt_uri": f"{it(0)}/s1_ckpts/"},
         "s2_train": {"status": "running", "model": "Qwen3-VL-8B", "rows": 26152, "rounds": "0", "epochs": 2,
                      "steps": 6000, "step": 2100, "instances": "1 x p5", "started": "2026-11-25 11:05",
                      "eta": "2026-11-25 14:20", "loss": s2_loss_0, "ckpt_uri": f"{S3}/{exp}/s2_ckpts/iter_00/"},
         "eval": {"status": "waiting", "waits_for": ["s1_train", "s2_train"], "pair": "System1 round 0 + System2 round 0",
                  "split": "target (held out)", "episodes": 1500, "uri": f"{it(0)}/eval/"}},
        {"k": 1, "label": "round 1",
         "rollout": {"status": "waiting", "waits_for": ["s1_train of round 0"], "s1_ckpt": "round 0 (15k)",
                     "planned": 1500, "perturb": "+-30 cm / +-30 deg", "uri": f"{it(1)}/rollout/"},
         "preprocess": {"status": "queued"}, "s1_train": {"status": "queued", "steps": 15000},
         "s2_train": {"status": "queued", "model": "Qwen3-VL-8B", "rounds": "0-1"},
         "eval": {"status": "queued", "pair": "System1 round 1 + System2 round 1"}},
        {"k": 2, "label": "round 2",
         "rollout": {"status": "queued", "s1_ckpt": "round 1", "planned": 1500, "perturb": "+-30 cm / +-30 deg"},
         "preprocess": {"status": "queued"}, "s1_train": {"status": "queued", "steps": 15000},
         "s2_train": {"status": "queued", "model": "Qwen3-VL-8B", "rounds": "0-2"},
         "eval": {"status": "queued", "pair": "System1 round 2 + System2 round 2"}},
    ]
    baseline = {"label": "start: System1 270k + System2 (language steering)", "success": 0.537,
                "by_set": {"atomic_seen": 0.62, "composite_seen": 0.55, "composite_unseen": 0.41},
                "note": "placeholder from the fleet's target-split run; to be re-measured with the fixture fix"}
    # trends: the start point and each iteration (None: not yet)
    trends = {
        "iterations": ["start", "round 0", "round 1", "round 2"],
        "rollout_success": [None, 0.920, None, None],
        "s1_step_share": [None, 0.843, None, None],
        "prompt_tokens_k": [None, 566.7, None, None],
        "eval_success": [0.537, None, None, None],
    }
    return {"exp": exp, "mock": True, "updated": now.strftime("%Y-%m-%d %H:%M"),
            "title": "Self-improvement loop", "subtitle": "subset validation: 50 development tasks (17 target), 30 recorded episodes per task per round",
            "config": {"tasks": 50, "target_tasks": 17, "held_out": "16 composite_unseen (never trained)",
                       "episodes_per_round": 1500, "rounds": 3, "split": "demo (recorded starts; perturbed from round 1)",
                       "s1_start": S1_START, "s2": "Qwen3-VL-8B, single mode (judge, milestone, decision, subgoal)",
                       "train": "batch 256, 80% rollout / 20% base, 15k steps, ckpts every 5k",
                       "exp_uri": f"{S3}/{exp}/"},
            "now": {"running": [{"k": 0, "stage": "s1_train", "progress": 9400 / 15000, "eta": "15:40"},
                                {"k": 0, "stage": "s2_train", "progress": 2100 / 6000, "eta": "14:20"}],
                    "next": "round 1 rollout starts when System1 round 0 finishes (~15:40)"},
            "baseline": baseline, "trends": trends, "iterations": iters,
            "generated": (now + timedelta(minutes=0)).isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="si_sub50")
    ap.add_argument("--mock", action="store_true", help="placeholder numbers (the only mode so far)")
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "dashboard")
    args = ap.parse_args()
    state = mock_state(args.exp)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "state.json").write_text(json.dumps(state, indent=1))
    page = (ROOT / "scripts" / "dashboard_template.html").read_text()
    (args.out / "index.html").write_text(page.replace("/*STATE*/null", json.dumps(state)))
    print(f"{args.out / 'index.html'} ({'mock' if state['mock'] else 'live'}, {len(state['iterations'])} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

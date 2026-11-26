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


# -- live: the real state of an experiment (S3: state.json + every stage's _plan/_status/_DONE/meta, W&B loss) ------------
MAP = {"done": "done", "running": "running", "submitted": "running", "dry_run": "waiting", "failed": "failed",
       "skipped": "queued", "todo": "waiting", "pending": "queued"}


def _wandb_loss(cfg, k: int) -> list:
    """[[step, loss], ...] of iteration k's S1 training run (W&B, the loop's project; [] when not found)."""
    try:
        import wandb

        key = Path("~/.config/selfimprove/secrets/wandb-api-key").expanduser().read_text().strip()
        w = cfg.get("wandb") or {}
        api = wandb.Api(api_key=key, overrides={"base_url": w.get("base_url") or "https://api.wandb.ai"}, timeout=30)
        from selfimprove.train.launch import exp_name

        name = exp_name(cfg, k)
        runs = [r for r in api.runs(f"{w.get('entity', 'tri')}/{w.get('project', 's1s2-self-improve')}",
                                    filters={"group": cfg.exp_id}, per_page=50)
                if (r.job_type or "") == "s1-train" or name in (r.name or "")]
        if not runs:
            return []
        rows = [h for h in runs[-1].scan_history(keys=["_step", "loss"], page_size=1000) if h.get("loss") is not None]
        step = max(1, len(rows) // 200)  # at most ~200 points
        return [[int(h["_step"]), round(float(h["loss"]), 5)] for h in rows[::step]]
    except Exception:  # noqa: BLE001 - the dashboard shows what it can
        return []


def _size_gb(uri: str) -> float:
    from selfimprove import s3

    try:
        return round(sum(n for _, n in s3.iter_keys(uri.rstrip("/") + "/")) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return 0.0


def live_state(config: str) -> dict:
    import sys

    sys.path.insert(0, str(ROOT))
    from selfimprove import s3
    from selfimprove import tasks as T
    from selfimprove.config import Config
    from selfimprove.data.stage import s2_uri
    from selfimprove.train import launch

    cfg = Config.load(config, [])
    s3.configure(cfg.get("aws_profile"), cfg.region)
    st = s3.read_json(f"{cfg.exp_uri}/state.json", default=None) or {}
    sit = st.get("iterations") or {}
    r = cfg["rollout"] or {}
    plan0 = T.plan_round(cfg, 0, {})
    tasks = sorted({e.task for e in plan0})
    sets = {row["task"]: row.get("task_set") for row in T.rollout_tasks()}
    n_target = sum(sets.get(t) in ("atomic_seen", "composite_seen") for t in tasks)
    t = cfg["train"]
    iters, trends = [], {"iterations": ["start"], "rollout_success": [None], "s1_step_share": [None],
                         "prompt_tokens_k": [None], "eval_success": [None], "api_cost_usd": [None]}
    now_running = []
    for k in range(cfg.iterations):
        sk = sit.get(str(k)) or sit.get(k) or {}
        ro_s, pp_s, tr_s, ev_s = (sk.get(x) or {} for x in ("rollout", "preprocess", "train", "eval"))
        it_uri = f"{cfg.exp_uri}/iter_{k:02d}"
        plan = s3.read_json(f"{cfg.rollout_uri(k)}/_plan.json", default=None) or {}
        done = s3.read_json(f"{cfg.rollout_uri(k)}/_DONE", default=None)
        prog = done or s3.read_json(f"{cfg.rollout_uri(k)}/_status.json", default=None) or {}
        planned = int(prog.get("episodes_planned") or len(plan.get("episodes") or []) or len(plan0))
        eps = plan.get("episodes") or []
        pert = sum(e.get("start_salt") is not None for e in eps)
        ro_status = "done" if done else MAP.get(ro_s.get("status", "pending"), "queued")
        if ro_status == "queued" and (k == 0 or (sit.get(str(k - 1)) or {}).get("train", {}).get("status") == "done"):
            ro_status = "waiting"
        ckpt = plan.get("s1_ckpt") or ro_s.get("ckpt") or (cfg["start"]["s1_ckpt"] if k == 0 else None)
        rollout = {"status": ro_status, "s1_ckpt": "start (270k)" if k == 0 else f"round {k - 1}", "s1_ckpt_uri": ckpt,
                   "planned": planned, "episodes": prog.get("episodes_done"),
                   "failed": prog.get("episodes_failed"), "success": prog.get("success_rate"),
                   "by_set": {"atomic": prog.get("success_rate_atomic"), "composite": prog.get("success_rate_composite")}
                   if prog.get("success_rate") is not None else None,
                   "s1_step_share": prog.get("s1_step_share"), "turns_mean": prog.get("turns_mean"),
                   "episode_minutes": prog.get("episode_minutes_mean"), "prompt_tokens_mean": prog.get("tokens_prompt_mean"),
                   "cached_frac": prog.get("tokens_cached_frac"), "nodes": prog.get("nodes"), "hours": prog.get("elapsed_h"),
                   "api_cost_usd": prog.get("api_cost_usd"), "tokens_in": prog.get("tokens_input_total"),
                   "tokens_in_cached": prog.get("tokens_input_cached_total"), "tokens_out": prog.get("tokens_output_total"),
                   "started": ro_s.get("started"), "finished": ro_s.get("finished"),
                   "perturb": (f"{pert}/{len(eps)} perturbed (+-{r.get('start_shift', 0) * 100:.0f} cm, "
                               f"+-{r.get('start_turn', 0):g} deg)" if eps else None),
                   "uri": f"{cfg.rollout_uri(k)}/", "local": str(cfg.work_dir / f"iter_{k:02d}" / "rollout"),
                   "site": f"../debug_{cfg.exp_id}/index.html" if k == 0 else None}
        if ro_status == "running" and planned:
            now_running.append({"k": k, "stage": "rollout", "progress": (prog.get("episodes_done") or 0) / planned,
                                "eta": prog.get("eta") or ""})
        s1m = s3.read_json(f"{cfg.s1_shard_uri(k)}/meta.json", default=None)
        s2m = s3.read_json(f"{s2_uri(cfg, k)}/meta.json", default=None)
        pp_status = "done" if s1m else MAP.get(pp_s.get("status", "pending"), "queued")
        if pp_status == "running":
            now_running.append({"k": k, "stage": "preprocess", "progress": 0.5, "eta": ""})
        pre = {"status": pp_status, "started": pp_s.get("started"), "finished": pp_s.get("finished")}
        if s1m:
            use = s1m.get("segment_use") or {}
            pre["s1"] = {"spans": sum(v for a, v in use.items() if a.endswith("/kept")), "samples": s1m.get("samples"),
                         "shards": s1m.get("shards"), "size_gb": _size_gb(f"{cfg.s1_shard_uri(k)}/shards"),
                         "episodes_used": s1m.get("episodes_used"), "episodes": s1m.get("episodes"),
                         "recovery_spans": use.get("recovery/kept"), "kept": use,
                         "seconds": s1m.get("seconds"), "uri": f"{cfg.s1_shard_uri(k)}/",
                         "local": str(cfg.work_dir / f"iter_{k:02d}" / "s1_training_shard")}
        if s2m:
            dec = s2m.get("per_decision") or {}
            pre["s2"] = {"rows": s2m.get("num_samples"), "shards": s2m.get("num_shards"),
                         "decisions": {"model": dec.get("s1", 0), "expert": dec.get("harness", 0)},
                         "closing": (s2m.get("per_kind") or {}).get("closing"), "kinds": s2m.get("per_kind"),
                         "judges": s2m.get("per_judge"), "size_gb": _size_gb(f"{s2_uri(cfg, k)}/shards"),
                         "uri": f"{s2_uri(cfg, k)}/", "local": str(cfg.work_dir / f"iter_{k:02d}" / "s2_training_shard")}
        job = tr_s.get("job") or {}
        tr_status = MAP.get(tr_s.get("status", "pending"), "queued")
        if tr_status == "running" and job.get("status") in (None, "Queued"):
            tr_status = "waiting"
        ckpts = launch.find_checkpoints(cfg, k) if tr_status in ("running", "done") else []
        loss = _wandb_loss(cfg, k) if tr_status in ("running", "done") else []
        steps = int(t["steps"])
        s1_train = {"status": tr_status, "job": (tr_s.get("launch") or {}).get("job_name") or job.get("training_job"),
                    "instances": f"{t.get('instance_count', 2)} x {t.get('instance_type', 'p5')}",
                    "from_ckpt": rollout["s1_ckpt"], "batch": t["batch_size"], "steps": steps,
                    "step": loss[-1][0] if loss else (steps if tr_status == "done" else None),
                    "lr": f"{t.get('lr')} flat, EMA {t.get('ema_decay')}", "new_frac": t.get("new_data_fraction"),
                    "data": (f"{pre['s1']['samples']:,} rollout samples ({float(t.get('new_data_fraction', 0.8)):.0%} of "
                             f"each batch) + the base set" if pre.get("s1") else None),
                    "started": job.get("start") or tr_s.get("submitted"), "eta": job.get("end") or "",
                    "queue": f"{t.get('queue_full_name') or t.get('queue')} (priority {t.get('queue_priority')})",
                    "job_state": job.get("secondary") or job.get("batch_status"),
                    "loss": loss, "ckpts": [{"step": a, "uri": b} for a, b in ckpts], "ckpt_uri": f"{cfg.s1_ckpt_uri(k)}/"}
        if tr_status == "running" and loss:
            now_running.append({"k": k, "stage": "s1_train", "progress": loss[-1][0] / steps, "eta": ""})
        s2_train = {"status": "queued", "model": "Qwen3-VL-4B (full fine-tune)", "rounds": f"0-{k}",
                    "rows": pre.get("s2", {}).get("rows"), "note": "optional: only the evaluation needs it"}
        ev = {"status": MAP.get(ev_s.get("status", "pending"), "queued") if cfg["eval"].get("enabled", True) else "queued",
              "pair": f"System1 round {k} + System2", "split": "target (held out)", "uri": f"{cfg.eval_uri(k)}/",
              "success": (ev_s.get("summary") or {}).get("overall")}
        label = f"round {k}" + (" (recorded starts)" if eps and not pert else "")
        iters.append({"k": k, "label": label, "rollout": rollout, "preprocess": pre, "s1_train": s1_train,
                      "s2_train": s2_train, "eval": ev})
        trends["iterations"].append(f"round {k}")
        trends["rollout_success"].append(prog.get("success_rate"))
        trends["s1_step_share"].append(prog.get("s1_step_share"))
        trends["prompt_tokens_k"].append(round(prog["tokens_prompt_mean"] / 1000, 1) if prog.get("tokens_prompt_mean") else None)
        trends["eval_success"].append(ev["success"])
        trends["api_cost_usd"].append(prog.get("api_cost_usd"))
    nxt = next((f"{it['label']}: {s}" for it in iters for s in ("rollout", "preprocess", "s1_train")
                if it[s]["status"] in ("waiting", "queued")), "all rounds done")
    base = cfg["eval"].get("baseline") or {}
    return {"exp": cfg.exp_id, "mock": False, "updated": time_now(), "heartbeat": st.get("heartbeat"),
            "title": "Self-improvement loop",
            "subtitle": f"{len(tasks)} development tasks ({n_target} target), {cfg.seeds_per_task(0)} recorded episodes per "
                        f"task per round, {cfg.iterations} rounds",
            "config": {"tasks": len(tasks), "target_tasks": n_target, "held_out": "16 composite_unseen (never trained)",
                       "episodes_per_round": len(plan0), "rounds": cfg.iterations,
                       "split": (f"{r.get('start_draw', 'fresh')} draw of the recorded starts; "
                                 + ("perturbed only when a recording is played again" if r.get("augment_from_round", 1) is None
                                    else f"perturbed from round {r.get('augment_from_round', 1)}")
                                 + f" (+-{r.get('start_shift', 0) * 100:.0f} cm, +-{r.get('start_turn', 0):g} deg)"),
                       "s1_start": cfg["start"]["s1_ckpt"],
                       "s2": "Qwen3-VL-4B, single mode (judge, milestone, decision, subgoal); 50% new rows / 50% old set",
                       "train": f"batch {t['batch_size']}, {float(t.get('new_data_fraction', 0.8)):.0%} rollout / "
                                f"{1 - float(t.get('new_data_fraction', 0.8)):.0%} base, {int(t['steps']) // 1000}k steps, "
                                f"ckpts every {int(t['save_interval']) // 1000}k, {t.get('queue_full_name') or t.get('queue')}",
                       "exp_uri": f"{cfg.exp_uri}/", "local": str(cfg.work_dir),
                       "api_price": r.get("api_price")},
            "now": {"running": now_running, "next": nxt, "events": (st.get("events") or [])[-12:]},
            "baseline": {"label": base.get("label", "start: System1 270k + System2"), "success": base.get("overall"),
                         "by_set": {a: base[a] for a in ("atomic_seen", "composite_seen", "composite_unseen") if a in base},
                         "note": base.get("note", "")},
            "trends": trends, "iterations": iters, "generated": time_now()}


def time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="si_sub50")
    ap.add_argument("--mock", action="store_true", help="placeholder numbers")
    ap.add_argument("--config", help="live: the experiment's config (state from S3, W&B)")
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "dashboard")
    ap.add_argument("--every", type=int, default=0, help="live: rebuild every N seconds (0: once)")
    args = ap.parse_args()
    if not args.mock and not args.config:
        ap.error("--config CONFIG (live) or --mock")
    while True:
        state = mock_state(args.exp) if args.mock else live_state(args.config)
        write(state, args.out)
        if not args.every:
            return 0
        import time

        time.sleep(args.every)


def write(state: dict, out: Path) -> None:
    args = argparse.Namespace(out=out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "state.json").write_text(json.dumps(state, indent=1))
    page = (ROOT / "scripts" / "dashboard_template.html").read_text()
    (args.out / "index.html").write_text(page.replace("/*STATE*/null", json.dumps(state)))
    print(f"{args.out / 'index.html'} ({'mock' if state['mock'] else 'live'}, {len(state['iterations'])} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

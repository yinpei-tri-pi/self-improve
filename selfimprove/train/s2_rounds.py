"""Each round's System2 training (the design of 2026-11-26), next to the scheduler:

    .venv/bin/python -m selfimprove.train.s2_rounds --config configs/dev120.yaml [--every 600] [--dry-run] [--only K]

Round k's System2 continues from round k-1's (round 0 from the System2 base, ``s2.base.ckpt_uri``) on its round's S2
training shards (``iter_k/s2_training_shard``, 80% of each batch) + the old set (``s2.base.data_uri``, full0804_v4, 20%),
3 epochs over the round's TRUE sample count (its meta.json): steps = ceil(3 x N / (0.8 x 192)), LR 5e-6 / ViT 1e-6 /
aligner 5e-6, the final checkpoint only, into ``iter_k/s2_ckpts/<run>/checkpoint-<steps>/``. It submits a round once
its shards are done (``_DONE``) and the previous System2 is final (its checkpoint's ``full_model_verification.json``,
written last), and records the launch in ``iter_k/s2_ckpts/_launch.json`` (the dashboard reads it). A failed job is
recorded, not retried. Submission goes through sys2_train_eval's scripts/sagemaker/launch.py (--skip-build: the
image tag in ``s2.round.image_tag``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

from .. import s3
from ..config import Config

SYS2 = Path("~/sys2_train_eval").expanduser()
LAUNCH_PY = Path("~/openpi/.venv/bin/python").expanduser()  # the env with sagemaker + boto3 (as the base was launched)


def s2_shard_uri(cfg: Config, k: int) -> str:
    return f"{cfg.exp_uri}/iter_{int(k):02d}/s2_training_shard"


def ckpt_root(cfg: Config, k: int) -> str:
    return f"{cfg.exp_uri}/iter_{int(k):02d}/s2_ckpts"


def launch_record(cfg: Config, k: int) -> dict | None:
    return s3.read_json(f"{ckpt_root(cfg, k)}/_launch.json", default=None)


def final_of(run_uri: str, steps: int) -> str | None:
    """The run's final checkpoint folder (with a trailing /) once it is complete, else None."""
    ck = f"{run_uri.rstrip('/')}/checkpoint-{int(steps)}"
    return ck + "/" if s3.exists(f"{ck}/full_model_verification.json") else None


def init_of(cfg: Config, k: int) -> tuple[str | None, str]:
    """(round k's starting checkpoint or None when not final yet, its short name)."""
    if k == 0:
        b = cfg["s2"]["base"]
        return final_of(b["ckpt_uri"], int(b["steps"])), "sibase"
    prev = launch_record(cfg, k - 1)
    if not prev:
        return None, f"sir{k - 1}"
    return final_of(prev["checkpoint_uri"], int(prev["steps"])), f"sir{k - 1}"


def round_steps(n: int, r: dict) -> int:
    return math.ceil(float(r.get("epochs", 3)) * n / (float(r.get("new_fraction", 0.8)) * int(r.get("batch", 192))))


def sys2_config(cfg: Config, k: int, init: str, init_name: str, n: int) -> dict:
    r, b = cfg["s2"]["round"], cfg["s2"]["base"]
    steps = round_steps(n, r)
    lr, vit, al = (float(r["learning_rate"]), float(r["vit_lr"]), float(r["aligner_lr"]))
    e = lambda v: f"{v:.0e}".replace("e-0", "e-")  # noqa: E731 - 5e-6, as the base's tags
    tag = f"full-vitfull-lr{e(lr)}-vitlr{e(vit)}-alignerlr{e(al)}-zero2-2n-new{round(100 * float(r['new_fraction']))}-ep{r['epochs']}"
    return {
        "aws": {"region": cfg.region, "profile": cfg.get("aws_profile") or "sagemaker",
                "role_arn": r.get("role_arn", "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess")},
        "queue": {"full_name": r.get("queue_full_name", "fss-vla-p5-48xlarge-us-west-2"),
                  "priority": int(r.get("queue_priority", 150)), "share_identifier": "default"},
        "job": {"user": "yinpei", "name_prefix": "sys2", "instance_type": "ml.p5.48xlarge", "instance_count": 2,
                "volume_size": 300, "max_run_seconds": int(r.get("max_run_days", 2)) * 86400,
                "tags": [{"key": "tri.owner.email", "value": "yinpei.dai@tri.global"}, {"key": "tri.project", "value": "MM:PJ-0077"}]},
        "image": {"repository": "yinpeidai-system2-train", "tag": str(r.get("image_tag", "20261126-si-mixed-v2"))},
        "data": {"train_s3_uri": f"{s2_shard_uri(cfg, k)}/", "base_s3_uri": b["data_uri"], "model_s3_uri": init,
                 "input_mode": "FastFile"},
        "training": {"dataset_shortname": f"{cfg.exp_id.replace('_', '-')}-r{k:02d}-mix-{b.get('data_name', 'v4').split('_')[-1]}",
                     "job_dataset_shortname": f"si-r{k:02d}", "basemodel_shortname": f"qwen3vl-4b-{init_name}", "tags": tag,
                     "per_device_batch_size": 3, "gradient_accumulation_steps": 4, "max_steps": steps,
                     "save_steps": steps, "save_limit": 1, "workers": 2, "shuffle_buffer": 512,
                     "new_fraction": float(r["new_fraction"]), "tuner": "full", "parallelism": "zero2",
                     "attention": "flash_attention_2", "learning_rate": lr, "vit_lr": vit, "aligner_lr": al,
                     "max_length": 8192, "report_to": "wandb", "log_first_sample": True, "log_samples_per_rank": 4,
                     "log_step_timing": True, "freeze_vit": False, "freeze_aligner": False, "gradient_checkpointing": True},
        "wandb": {"project": "robocasa-system2"},
        "output": {"checkpoint_s3_root": ckpt_root(cfg, k), "artifacts_s3_root": f"{ckpt_root(cfg, k)}/_artifacts"},
    }


def submit(cfg: Config, k: int, *, dry_run: bool = False, log=print) -> dict | None:
    """Round k's System2 job, when it can start; None when it waits (or is already submitted)."""
    import yaml

    if launch_record(cfg, k):
        return None
    meta = s3.read_json(f"{s2_shard_uri(cfg, k)}/_DONE", default=None)
    if not meta:
        return None
    init, init_name = init_of(cfg, k)
    if not init:
        return None
    n = int(meta["num_samples"])
    conf = sys2_config(cfg, k, init, init_name, n)
    path = cfg.work_dir / "s2_launch" / f"iter_{k:02d}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    cmd = [str(LAUNCH_PY), "scripts/sagemaker/launch.py", "--config", str(path), "--skip-build"] + (["--dry-run"] if dry_run else [])
    out = subprocess.run(cmd, cwd=SYS2, capture_output=True, text=True)
    text = out.stdout + out.stderr
    if out.returncode != 0:
        log(f"round {k} S2: launch.py failed ({out.returncode}): {text[-800:]}")
        return {"error": text[-800:]}
    m = re.search(r"Queued (\S+) -> (\S+)", text)
    exp = re.search(r'"experiment": "([^"]+)"', text) or re.search(r'"SYS2_EXP_NAME": "([^"]+)"', text)
    rec = {"round": k, "job_name": m.group(1) if m else None, "queue": conf["queue"]["full_name"],
           "steps": conf["training"]["max_steps"], "samples": n, "init": init,
           "data": conf["data"]["train_s3_uri"], "base_data": conf["data"]["base_s3_uri"],
           "new_fraction": conf["training"]["new_fraction"], "epochs": cfg["s2"]["round"]["epochs"],
           "lr": [conf["training"][x] for x in ("learning_rate", "vit_lr", "aligner_lr")],
           "checkpoint_uri": f"{ckpt_root(cfg, k)}/{exp.group(1) if exp else ''}/",
           "config": str(path), "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": dry_run}
    if dry_run:
        log(f"round {k} S2 (dry run): {json.dumps(rec)}")
        return rec
    if not rec["job_name"]:
        log(f"round {k} S2: no 'Queued' line from launch.py: {text[-800:]}")
        return {"error": text[-800:]}
    s3.write_json(f"{ckpt_root(cfg, k)}/_launch.json", rec)
    s3.write_bytes(f"{ckpt_root(cfg, k)}/_config.yaml", path.read_bytes(), "text/yaml")
    log(f"round {k} S2 submitted: {rec['job_name']} ({rec['steps']} steps on {n} samples, from {init})")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--every", type=int, default=0, help="check every N seconds (0: once)")
    ap.add_argument("--only", type=int, default=None, help="only this round")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    while True:
        cfg = Config.load(args.config, [])
        s3.configure(cfg.get("aws_profile"), cfg.region)
        stamp = lambda m: print(time.strftime("%Y-%m-%d %H:%M:%S ") + m, flush=True)  # noqa: E731
        for k in ([args.only] if args.only is not None else range(cfg.iterations)):
            try:
                submit(cfg, k, dry_run=args.dry_run, log=stamp)
            except Exception as exc:  # noqa: BLE001 - keep watching; the next check tries again
                stamp(f"round {k} S2: {type(exc).__name__}: {exc}")
        if not args.every:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())

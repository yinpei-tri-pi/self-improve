"""Continue-train System1 for one round: from ckpt_k on the base data + the round's rollout shards.

SageMaker (the default): openpi-si's ``scripts/sagemaker/launch.py`` with the multi-node JAX entrypoint,
``base_ckpt`` = ckpt_k (its ``params/`` are the weights the trainer starts from and, with the
self-improvement objective, the frozen reference), the base shards as the ``robocasa`` channel and the
round's shards as a second stream (``ROBOCASA_MIX_SHARDS=<s3 shards>``, ``ROBOCASA_MIX_FRACTION`` =
``new_data_fraction``: openpi-si's MixedRoboCasaWebDataset interleaves the two by an exact quota, each with its
own shuffled passes and reservoir, so every batch has the fraction), checkpoints to ``iter_k/s1_ckpts/``, W&B in
the loop's project under the experiment's group.  ``local``: the same arguments to ``scripts/train.py`` on this
machine.

The checkpoint the next round uses is the last step dir under ``iter_k/s1_ckpts/`` (written to
``iter_k/s1_ckpts/_FINAL``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from .. import s3
from ..config import Config


def exp_name(cfg: Config, k: int) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", f"si-{cfg.exp_id}-iter{int(k):02d}").strip("-").lower()


def repeat_factor(new_samples: int, base_samples: int, fraction: float) -> int:
    """K so that K x new / (K x new + base) ~= fraction (at least 1)."""
    if new_samples <= 0 or fraction <= 0:
        return 0
    fraction = min(0.95, float(fraction))
    return max(1, round(fraction / (1.0 - fraction) * base_samples / new_samples))


def base_sample_count(cfg: Config) -> int:
    meta = s3.read_json(cfg["train"]["base_shards"].rstrip("/") + "/meta.json", default={}) or {}
    for key in ("num_frames", "frames", "samples", "num_samples", "total_samples"):
        if meta.get(key):
            return int(meta[key])
    return 34_200_000  # system1_fullset_0717


def round_seed(cfg: Config, k: int) -> int:
    """The round's training seed: openpi's webdataset loader orders the shards by seed + epoch, so a fixed seed would
    make every round read the base set in the same order (and a short run only its first part)."""
    import zlib

    return zlib.crc32(f"{cfg.exp_id}/{int(k)}".encode()) % 1_000_000


def trainer_args(cfg: Config, k: int = 0) -> str:
    t = cfg["train"]
    si = t.get("si") or {}
    objective = str(t.get("objective", "sl"))
    args = [
        f"--fsdp-devices={int(t['fsdp_devices'])}", f"--batch-size={int(t['batch_size'])}",
        f"--ema-decay={t['ema_decay']}", f"--num-train-steps={int(t['steps'])}",
        f"--save-interval={int(t['save_interval'])}", f"--keep-period={int(t['save_interval'])}",
        f"--lr-schedule.warmup-steps={int(t['warmup_steps'])}", f"--lr-schedule.peak-lr={t['lr']}",
        f"--lr-schedule.decay-steps={int(t['steps'])}", f"--lr-schedule.decay-lr={t['lr']}",
        "--data.shuffle-buffer=5000", f"--project-name={cfg['wandb']['project']}",
        str(t.get("model_flags") or ""), f"--seed={round_seed(cfg, k)}",
    ]
    if objective == "si":  # the old self-improvement objective (negatives, frozen reference); "sl": plain training
        args.append("--si.enabled")
        for key in ("lambda_sl", "lambda_pos", "lambda_neg", "beta_pos", "beta_neg"):
            if key in si:
                args.append(f"--si.{key.replace('_', '-')}={si[key]}")
    return " ".join(a for a in args if a)


def sagemaker_command(cfg: Config, k: int, ckpt_uri: str, extra_shards: str, *, image_tag: str | None,
                      dry_run: bool) -> list[str]:
    t = cfg["train"]
    openpi = Path(os.path.expanduser(t["openpi_dir"]))
    py = os.path.expanduser(t.get("openpi_python") or os.environ.get("SI_OPENPI_PYTHON", "~/openpi/.venv/bin/python"))
    cmd = [py, str(openpi / "scripts/sagemaker/launch.py"), f"--config={openpi / t['sagemaker_config']}"]
    if image_tag:
        cmd += ["--skip-build", f"--image.tag={image_tag}"]
    if dry_run:
        cmd.append("--dry-run")
    env = {"ROBOCASA_MIX_SHARDS": extra_shards, "ROBOCASA_MIX_FRACTION": str(float(t.get("new_data_fraction", 0.8))),
           "OPENPI_WANDB_GROUP": cfg.exp_id,
           "OPENPI_WANDB_TAGS": f"iter_{int(k):02d},s1-train", "OPENPI_WANDB_JOB_TYPE": "s1-train"}
    cmd += [
        f"--instance.type={t['instance_type']}", f"--instance.count={int(t['instance_count'])}",
        *([f"--queue.full_name={t['queue_full_name']}"] if t.get("queue_full_name") else [f"--queue.name={t['queue']}"]),
        f"--queue.priority={int(t['queue_priority'])}",
        f"--data.channels.robocasa.s3_uri={t['base_shards'].rstrip('/')}/",
        f"--data.channels.base_ckpt.s3_uri={ckpt_uri.rstrip('/')}/",
        f"--training.config={t['config']}", f"--training.exp_name={exp_name(cfg, k)}",
        f"--training.extra_args={trainer_args(cfg, k)}",
        f"--output.s3_prefix={cfg.s1_ckpt_uri(k)}/",
        f"--wandb.project={cfg['wandb']['project']}",
    ]
    cmd += [f"--training.env.{key}={value}" for key, value in env.items()]
    return cmd


def launch(cfg: Config, k: int, ckpt_uri: str, *, dry_run: bool = False, image_tag: str | None = None) -> dict:
    """Submit iteration k's training; returns {job_name, exp_name, extra_shards, command, output}."""
    t = cfg["train"]
    shard_meta = s3.read_json(f"{cfg.s1_shard_uri(k)}/meta.json", default={}) or {}
    new = int(shard_meta.get("samples") or 0)
    if not new:
        raise RuntimeError(f"no S1 samples in {cfg.s1_shard_uri(k)}/meta.json: build the round's shards first")
    extra = f"{cfg.s1_shard_uri(k)}/shards"  # the new stream (the base set is the robocasa channel)
    record = {"exp_name": exp_name(cfg, k), "base_ckpt": ckpt_uri, "mix_shards": extra, "new_samples": new,
              "new_data_fraction": float(t.get("new_data_fraction", 0.8)), "trainer_args": trainer_args(cfg, k),
              "backend": t.get("backend", "sagemaker")}
    if t.get("backend", "sagemaker") == "local":
        record["command"] = local_command(cfg, k, ckpt_uri, extra)
        if not dry_run:
            log = cfg.work_dir / f"train_iter{k:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(record["command"], cwd=os.path.expanduser(t["openpi_dir"]), stdout=open(log, "ab"),
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env=local_env(cfg, k, ckpt_uri, extra))
            record.update(pid=proc.pid, log=str(log))
        return record
    tag = image_tag or t.get("image_tag")
    cmd = sagemaker_command(cfg, k, ckpt_uri, extra, image_tag=tag, dry_run=dry_run)
    env = dict(os.environ, AWS_PROFILE=cfg.get("aws_profile") or "sagemaker", VIRTUAL_ENV="")
    out = subprocess.run(cmd, cwd=os.path.expanduser(t["openpi_dir"]), env=env, capture_output=True, text=True)
    record.update(command=" ".join(shlex.quote(c) for c in cmd), returncode=out.returncode,
                  output=(out.stdout + out.stderr)[-6000:])
    m = re.search(r"Queued (\S+) ->", out.stdout)
    if m:
        record["job_name"] = m.group(1)
    m = re.search(r"Image: (\S+)", out.stdout) or re.search(r"(\d+\.dkr\.ecr\.[^\s]+:[^\s]+)", out.stdout)
    if m:
        record["image"] = m.group(1)
    if out.returncode != 0:
        raise RuntimeError(f"SageMaker launch failed ({out.returncode}):\n{record['output']}")
    return record


def local_env(cfg: Config, k: int, ckpt_uri: str, extra: str) -> dict:
    t = cfg["train"]
    openpi = os.path.expanduser(t["openpi_dir"])
    shards = t["base_shards"].rstrip("/") + "/shards"
    return dict(os.environ, ROBOCASA_SHARDS_DIR=shards, ROBOCASA_MIX_SHARDS=extra,
                ROBOCASA_MIX_FRACTION=str(float(t.get("new_data_fraction", 0.8))),
                OPENPI_WEIGHT_LOADER_PARAMS_PATH=ckpt_uri.rstrip("/") + "/params",
                OPENPI_WANDB_GROUP=cfg.exp_id, OPENPI_WANDB_TAGS=f"iter_{int(k):02d},s1-train",
                OPENPI_WANDB_JOB_TYPE="s1-train", XLA_PYTHON_CLIENT_MEM_FRACTION="0.9",
                PYTHONPATH=f"{openpi}/src:{openpi}/packages/openpi-client/src")


def local_command(cfg: Config, k: int, ckpt_uri: str, extra: str) -> list[str]:
    t = cfg["train"]
    py = os.path.expanduser(os.environ.get("SI_OPENPI_PYTHON", "~/openpi/.venv/bin/python"))
    return [py, "scripts/train.py", t["config"], f"--exp-name={exp_name(cfg, k)}",
            f"--checkpoint-base-dir={cfg.work_dir / 's1_ckpts'}", *shlex.split(trainer_args(cfg, k))]


# -- status ----------------------------------------------------------------------------------------------
def job_status(cfg: Config, job_name: str) -> dict:
    """The job's state: {'status': Queued | InProgress | Completed | Failed | Stopped, ...}.

    launch.py submits through an AWS Batch queue of type SAGEMAKER_TRAINING: the service job carries our name, and
    the SageMaker training job it starts gets a generated one (AWSBatch...), found through the service job."""
    import boto3

    session = boto3.Session(profile_name=cfg.get("aws_profile") or None, region_name=cfg.region)
    t = cfg["train"]
    queue = t.get("queue_full_name") or f"fss-{t.get('queue', 'vla')}-{t.get('instance_type', 'p5')}-48xlarge-{cfg.region}"
    batch = session.client("batch")
    out: dict = {"queue": queue}
    jobs = batch.list_service_jobs(jobQueue=queue, filters=[{"name": "JOB_NAME", "values": [job_name]}]).get(
        "jobSummaryList", [])
    if not jobs:
        return {**out, "status": "Queued", "batch_status": "not listed yet"}
    job = sorted(jobs, key=lambda j: j.get("createdAt", 0))[-1]
    bstatus = str(job.get("status"))
    out.update(batch_status=bstatus, batch_reason=job.get("statusReason"), batch_job_id=job.get("jobId"))
    status = {"SUCCEEDED": "Completed", "FAILED": "Failed", "RUNNING": "InProgress"}.get(bstatus, "Queued")
    try:
        d = batch.describe_service_job(jobId=job["jobId"])
        arn = ((d.get("latestAttempt") or {}).get("serviceResourceId") or {}).get("value", "")
        if arn:
            name = arn.rsplit("/", 1)[-1]
            tj = session.client("sagemaker").describe_training_job(TrainingJobName=name)
            out.update(training_job=name, secondary=tj.get("SecondaryStatus"), failure=tj.get("FailureReason"),
                       start=str(tj.get("TrainingStartTime") or ""), end=str(tj.get("TrainingEndTime") or ""))
            status = {"Completed": "Completed", "Failed": "Failed", "Stopped": "Stopped"}.get(
                tj["TrainingJobStatus"], status)
    except Exception as exc:  # noqa: BLE001 - the batch status alone is enough to go on
        out["detail_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if status == "Failed" and not out.get("failure"):
        out["failure"] = out.get("batch_reason")
    out["status"] = status
    return out


def find_checkpoints(cfg: Config, k: int) -> list[tuple[int, str]]:
    """(step, step-dir uri) of every complete checkpoint under iter_k/s1_ckpts/ (has params/ and assets/)."""
    found: dict[str, set[str]] = {}
    root = cfg.s1_ckpt_uri(k) + "/"
    for uri, _ in s3.iter_keys(root):
        m = re.match(r"(.*/(\d+))/(params|assets)/", uri)
        if m:
            found.setdefault(m.group(1), set()).add(m.group(3))
    out = [(int(d.rsplit("/", 1)[1]), d) for d, parts in found.items() if parts >= {"params", "assets"}]
    return sorted(out)


def final_checkpoint(cfg: Config, k: int) -> str | None:
    marker = s3.read_json(f"{cfg.s1_ckpt_uri(k)}/_FINAL", default=None)
    if marker:
        return marker["uri"]
    ckpts = find_checkpoints(cfg, k)
    return ckpts[-1][1] if ckpts else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--image-tag", default=None)
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    s3.configure(cfg.get("aws_profile"), cfg.region)
    print(json.dumps(launch(cfg, args.iteration, args.ckpt, dry_run=args.dry_run, image_tag=args.image_tag), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

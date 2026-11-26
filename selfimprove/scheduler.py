"""The loop's scheduler: watches every stage, starts the next one, logs the whole experiment to W&B.

    selfimprove run    --config configs/<exp>.yaml      # loop until done (tick every poll_s)
    selfimprove tick   --config ...                     # one pass (e.g. from cron), then exit
    selfimprove status --config ...                     # print the state

Per iteration k (state in ``<exp>/state.json`` on S3, mirrored to ``<work_dir>/state.json``)::

    rollout(k)     Ray cluster: Astra + S1(ckpt_k) on the round's (task, seed) plan -> iter_k/rollout/_DONE
    preprocess(k)  here: download, PRM, select, S1 shards                          -> iter_k/s1_training_shard/_DONE
    train(k)       SageMaker: ckpt_k -> ckpt_{k+1} (AWR + unary, frozen ref = ckpt_k) -> iter_k/s1_ckpts/_FINAL
    eval(k)        Ray cluster (a second one): S1(ckpt_{k+1}) + S2 on the target split -> iter_k/eval/target/_DONE
                   (runs while rollout(k+1) does; the loop does not wait for it)

and after the last iteration ``s2``: the S2 data from every round's rollouts (a later stage; off by default).
Every check reads S3, so the scheduler can be stopped and restarted at any point, and a stage's own
driver is idempotent (a restarted rollout or eval plays only what is missing).

Where the Ray drivers run (``cluster.mode``): ``local`` runs them here (``--local``: this machine's GPUs);
``ray`` submits them as Ray jobs to the clusters' heads (``cluster.rollout_address`` /
``cluster.eval_address``, or found with ``ray get-head-ip`` from the cluster YAMLs, which ``ray up``
starts when they are not up).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import s3
from .config import HERE, Config
from .data import stage as data_stage
from .train import launch as train_launch
from .wandb_log import Loop

STAGES = ("rollout", "preprocess", "train", "eval")
#: the ray CLI of this venv (the scheduler runs as .venv/bin/python, often without the venv on PATH)
RAY = str(Path(sys.executable).with_name("ray"))


class Scheduler:
    def __init__(self, cfg: Config, *, wandb_on: bool = True, dry_run: bool = False, log=print):
        self.cfg = cfg
        self.dry_run = dry_run
        self.log_fn = log
        # a dry run plans from the real state but keeps its own (never blocks a real run)
        self.local_state = cfg.work_dir / ("state.dryrun.json" if dry_run else "state.json")
        self.state = self._load_state()
        self.loop = Loop(cfg, enabled=wandb_on)
        self.procs: dict[str, subprocess.Popen] = {}

    # -- state -------------------------------------------------------------------------------------------
    def _load_state(self) -> dict:
        st = None if (self.dry_run and self.local_state.exists()) else s3.read_json(self.cfg.state_uri, default=None)
        if st is None and self.local_state.exists():
            st = json.loads(self.local_state.read_text())
        if st is None:
            st = {"exp_id": self.cfg.exp_id, "created": time.strftime("%F %T"), "iterations": {}, "events": []}
            if not self.dry_run:
                s3.write_bytes(f"{self.cfg.exp_uri}/config.yaml", self.cfg.dump().encode(), "text/yaml")
        return st

    def save(self) -> None:
        self.state["updated"] = time.strftime("%F %T")
        self.local_state.parent.mkdir(parents=True, exist_ok=True)
        self.local_state.write_text(json.dumps(self.state, indent=1, default=str))
        if not self.dry_run:
            s3.write_json(self.cfg.state_uri, self.state)

    def it(self, k: int) -> dict:
        return self.state["iterations"].setdefault(str(k), {s: {"status": "pending"} for s in STAGES})

    def event(self, text: str) -> None:
        line = f"{time.strftime('%F %T')} {text}"
        self.state["events"] = (self.state.get("events") or [])[-200:] + [line]
        self.log_fn(line)
        self.loop.note(line)

    # -- checkpoints -------------------------------------------------------------------------------------
    def ckpt(self, k: int) -> str | None:
        """The S1 checkpoint iteration k rolls out with."""
        if k == 0:
            return self.cfg.start_s1_ckpt
        return self.it(k - 1)["train"].get("ckpt") or train_launch.final_checkpoint(self.cfg, k - 1)

    # -- stage runners -----------------------------------------------------------------------------------
    def _driver_cmd(self, module: str, k: int, ckpt: str, extra: list[str] | None = None, *,
                    remote: bool = False) -> list[str]:
        cmd = [sys.executable, "-m", module, "--iteration", str(k), "--ckpt", ckpt]
        path = self.cfg.get("_config_path")
        if path:
            # a Ray job runs in an uploaded copy of this repo: the config by its path inside it
            if remote:
                try:
                    path = str(Path(path).resolve().relative_to(HERE))
                except ValueError:
                    raise SystemExit(f"--config {path} must be inside {HERE} for cluster mode") from None
            cmd += ["--config", path]
        for item in self.cfg.get("_overrides") or []:
            cmd += ["--set", item]
        return cmd + (extra or [])

    def _start_driver(self, name: str, module: str, k: int, ckpt: str, cluster: str) -> dict:
        c = self.cfg.get("cluster") or {}
        mode = c.get("mode", "local")
        if mode == "local":
            cmd = self._driver_cmd(module, k, ckpt, ["--local"])
            log = self.cfg.work_dir / f"{name}_iter{k:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if self.dry_run:
                return {"mode": "local", "command": " ".join(cmd), "dry_run": True}
            env = dict(os.environ, **{str(a): str(b) for a, b in (c.get("local_env") or {}).items()})
            p = subprocess.Popen(cmd, cwd=str(HERE), stdout=open(log, "ab"), stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            self.procs[f"{name}{k}"] = p
            return {"mode": "local", "pid": p.pid, "log": str(log), "command": " ".join(cmd)}
        address = c.get(f"{cluster}_address") or self._head_address(c[f"{cluster}_config"])
        cmd = self._driver_cmd(module, k, ckpt, remote=True)
        runtime_env = {"working_dir": str(HERE), "excludes": [".venv", "wandb", "*.tar", ".git", "docs"],
                       "env_vars": {k2: str(v) for k2, v in (c.get("job_env") or {}).items()}}
        node_py = c.get("head_python", "python")
        submit = [RAY, "job", "submit", "--address", address, "--no-wait",
                  "--runtime-env-json", json.dumps(runtime_env), "--", node_py, *cmd[1:]]
        if self.dry_run:
            return {"mode": "ray", "command": " ".join(shlex.quote(x) for x in submit[:6]) + " ...", "dry_run": True}
        r = subprocess.run(submit, capture_output=True, text=True, check=True)
        out = r.stdout + r.stderr
        job = next((line.split("'")[1] for line in out.splitlines() if "submitted successfully" in line and "'" in line),
                   None)
        if job is None:
            raise RuntimeError(f"ray job submit gave no job id:\n{out[-2000:]}")
        return {"mode": "ray", "address": address, "job_id": job}

    def _head_address(self, cluster_yaml: str) -> str:
        path = HERE / cluster_yaml
        r = subprocess.run([RAY, "get-head-ip", str(path)], capture_output=True, text=True)
        ip = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        if not ip:
            if self.dry_run:
                return "http://<head>:8265"
            self.event(f"starting the cluster {cluster_yaml} (ray up)")
            subprocess.run([RAY, "up", "-y", str(path)], check=True, cwd=str(HERE))
            ip = subprocess.run([RAY, "get-head-ip", str(path)], capture_output=True, text=True,
                                check=True).stdout.strip().splitlines()[-1]
        return f"http://{ip}:8265"

    def _driver_alive(self, info: dict) -> bool:
        if info.get("mode") == "local":
            p = self.procs.get(info.get("_key", ""))
            if p is not None:
                return p.poll() is None
            pid = info.get("pid")
            try:
                os.kill(int(pid), 0)
                return True
            except (TypeError, ValueError, OSError):
                return False
        if info.get("job_id"):
            r = subprocess.run([RAY, "job", "status", "--address", info["address"], info["job_id"]],
                               capture_output=True, text=True)
            return any(x in (r.stdout + r.stderr) for x in ("RUNNING", "PENDING"))
        return False

    # -- one pass ------------------------------------------------------------------------------------------
    def tick(self) -> bool:
        """Advance whatever can advance; True when the whole experiment is done."""
        cfg = self.cfg
        n = cfg.iterations
        for k in range(n):
            it = self.it(k)
            self._eval_check(k)
            if it["train"]["status"] == "done":
                continue
            ckpt = self.ckpt(k)
            if ckpt is None:
                return False
            # rollout
            if it["rollout"]["status"] != "done":
                self._rollout(k, ckpt)
                return False
            # preprocess
            if it["preprocess"]["status"] != "done":
                self._preprocess(k)
                return False
            # train
            self._train(k, ckpt)
            return False
        # every round trained; wait for the evaluations, then the S2 stage
        if all(self.it(k)["eval"]["status"] in ("done", "skipped") for k in range(n)):
            self._s2()
            return True
        return False

    def _rollout(self, k: int, ckpt: str) -> None:
        it = self.it(k)["rollout"]
        done = s3.read_json(f"{self.cfg.rollout_uri(k)}/_DONE", default=None)
        if done is not None:
            it.update(status="done", summary=done, finished=time.strftime("%F %T"))
            self.loop.point(k, {f"rollout/{a}": b for a, b in done.items()})
            self.event(f"iter {k}: rollout done: {done.get('episodes_done')} episodes, "
                       f"success {done.get('success_rate')}")
            self.save()
            return
        status = s3.read_json(f"{self.cfg.rollout_uri(k)}/_status.json", default=None)
        if status:
            it["progress"] = status
        if it["status"] == "running" and self._driver_alive({**it.get("driver", {}), "_key": f"rollout{k}"}):
            return
        if it["status"] == "running":
            self.event(f"iter {k}: the rollout driver is gone without _DONE; starting it again (it resumes)")
        it["driver"] = self._start_driver("rollout", "selfimprove.rollout.driver", k, ckpt, "rollout")
        it.update(status="running", started=it.get("started") or time.strftime("%F %T"), ckpt=ckpt)
        self.event(f"iter {k}: rollout started with {ckpt}")
        self.save()

    def _preprocess(self, k: int) -> None:
        it = self.it(k)["preprocess"]
        done = s3.read_json(f"{self.cfg.s1_shard_uri(k)}/_DONE", default=None)
        if done is None:
            if self.dry_run:
                self.event(f"iter {k}: [dry-run] would preprocess the rollouts")
                return
            it.update(status="running", started=time.strftime("%F %T"))
            self.save()
            self.event(f"iter {k}: preprocessing (download, PRM, shards)")
            done = data_stage.run(self.cfg, k, log=self.log_fn)
        it.update(status="done", summary={a: b for a, b in done.items() if a != "filter"}, finished=time.strftime("%F %T"))
        self.loop.point(k, data_stage.summary_metrics(done))
        self.event(f"iter {k}: {done.get('samples')} S1 samples ({done.get('positive_samples')} positive, "
                   f"{done.get('negative_samples')} negative) in {done.get('shards')} shards")
        self.save()

    def _train(self, k: int, ckpt: str) -> None:
        it = self.it(k)["train"]
        dry = self.dry_run or bool(self.cfg["train"].get("dry_run"))  # train.dry_run: plan the job, never submit it
        if it["status"] == "pending":
            rec = train_launch.launch(self.cfg, k, ckpt, dry_run=dry)
            it.update(status="dry_run" if dry else "submitted", launch=rec, submitted=time.strftime("%F %T"))
            self.event(f"iter {k}: training submitted: {rec.get('job_name') or rec.get('exp_name')}")
            self.save()
            return
        if dry or it["status"] == "dry_run":
            return
        final = train_launch.final_checkpoint(self.cfg, k)
        job = (it.get("launch") or {}).get("job_name")
        st = train_launch.job_status(self.cfg, job) if job else {"status": "unknown"}
        it["job"] = st
        if st["status"] in ("Failed", "Stopped"):
            if it.get("retries", 0) < int(self.cfg["train"].get("max_retries", 1)):
                it["retries"] = it.get("retries", 0) + 1
                it["status"] = "pending"
                self.event(f"iter {k}: training {st['status']} ({st.get('failure')}); submitting again")
            else:
                it["status"] = "failed"
                self.event(f"iter {k}: training {st['status']} ({st.get('failure')}); the loop stops here")
            self.save()
            return
        if st["status"] == "Completed" or (st["status"] == "unknown" and final):
            if not final:
                self.event(f"iter {k}: training completed but no checkpoint under {self.cfg.s1_ckpt_uri(k)}")
                it["status"] = "failed"
                self.save()
                return
            s3.write_json(f"{self.cfg.s1_ckpt_uri(k)}/_FINAL", {"uri": final, "job": job})
            it.update(status="done", ckpt=final, finished=time.strftime("%F %T"))
            self.loop.point(k, {"train/steps": int(self.cfg["train"]["steps"])})
            self.event(f"iter {k}: training done: {final}")
            self._start_eval(k, final)
            self.save()

    def _start_eval(self, k: int, ckpt: str) -> None:
        it = self.it(k)["eval"]
        if not self.cfg["eval"].get("enabled", True):
            it["status"] = "skipped"
            return
        it["driver"] = self._start_driver("eval", "selfimprove.eval.target", k, ckpt, "eval")
        it.update(status="running", started=time.strftime("%F %T"), ckpt=ckpt)
        self.event(f"iter {k}: target-split eval started for {ckpt}")

    def _eval_check(self, k: int) -> None:
        it = self.it(k)["eval"]
        if it["status"] != "running":
            return
        done = s3.read_json(f"{self.cfg.eval_uri(k)}/_DONE", default=None)
        if done is not None:
            it.update(status="done", summary={a: b for a, b in done.items() if a != "per_task"},
                      finished=time.strftime("%F %T"))
            base = self.cfg["eval"].get("baseline") or {}
            pts = {f"eval/{g}": done.get(g) for g in ("overall", "atomic_seen", "composite_seen", "composite_unseen")
                   if done.get(g) is not None}
            pts.update({f"eval/delta_{g}": done[g] - base[g] for g in base if g in done})
            self.loop.point(k + 1, pts)  # x = rounds of self-improvement the evaluated checkpoint has had
            self.event(f"iter {k}: target eval done: overall {done.get('overall')} (baseline {base.get('overall')})")
            self.save()
            return
        status = s3.read_json(f"{self.cfg.eval_uri(k)}/_status.json", default=None)
        if status:
            it["progress"] = status
        if not self._driver_alive({**it.get("driver", {}), "_key": f"eval{k}"}):
            self.event(f"iter {k}: the eval driver is gone without _DONE; starting it again (it resumes)")
            it["driver"] = self._start_driver("eval", "selfimprove.eval.target", k, it["ckpt"], "eval")
            self.save()

    def _s2(self) -> None:
        st = self.state.setdefault("s2", {"status": "pending"})
        if st["status"] != "pending":
            return
        if not (self.cfg.get("s2") or {}).get("enabled"):
            st["status"] = "skipped"
            self.event("all iterations done; the S2 stage is off (s2.enabled)")
        else:
            st["status"] = "todo"
            self.event("all iterations done; the S2 stage (language annotation + Qwen3-VL training) is next: "
                       "python -m selfimprove.s2.build")
        self.save()

    def status_text(self) -> str:
        lines = [f"experiment {self.cfg.exp_id}  ({self.cfg.exp_uri})"]
        for k in range(self.cfg.iterations):
            it = self.it(k)
            prog = it["rollout"].get("progress") or {}
            lines.append(f"  iter {k:02d}: " + "  ".join(f"{s}={it[s]['status']}" for s in STAGES)
                         + (f"  (rollout {prog.get('episodes_done')}/{prog.get('episodes_planned')}, "
                            f"success {prog.get('success_rate')})" if prog and it["rollout"]["status"] == "running" else ""))
        lines += ["  recent:"] + [f"    {e}" for e in (self.state.get("events") or [])[-8:]]
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="selfimprove", description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "tick", "status"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="plan every step, start nothing, write nothing")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    cfg.raw["_config_path"] = os.path.abspath(args.config) if args.config else None
    cfg.raw["_overrides"] = list(args.set)
    s3.configure(cfg.get("aws_profile"), cfg.region)
    sched = Scheduler(cfg, wandb_on=not args.no_wandb and args.command != "status", dry_run=args.dry_run)
    if args.command == "status":
        print(sched.status_text())
        return 0
    if args.command == "tick":
        sched.tick()
        print(sched.status_text())
        return 0
    poll = float((cfg.get("cluster") or {}).get("poll_s", 300))
    while True:
        try:
            finished = sched.tick()
        except Exception as exc:  # noqa: BLE001 - a transient failure must not end a week-long loop
            sched.event(f"tick failed: {type(exc).__name__}: {exc}")
            finished = False
        print(sched.status_text(), flush=True)
        if finished:
            sched.loop.finish()
            return 0
        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())

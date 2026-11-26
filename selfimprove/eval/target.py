"""S1 (a round's checkpoint) + S2 on RoboCasa's target split, on a Ray cluster: the 53.7% run's protocol.

Units: the 1500 (task, episode) pairs of ``assets/target_eval_30.json`` (50 target tasks x 30 recorded
target-split demos, each reset to its first frame, the goal from its ep_meta).  One ``EvalNode`` per GPU
(a Ray actor): an S1 server for the checkpoint, the S2 vLLM server (sys2_train_eval's
serve_system2_vllm.sh, in the eval image), and openpi's ``examples/robocasa/combined_eval.py`` run per
unit with the fleet's flags (all rule tiers, last-milestone retry, the official horizons, max 400 steps a
segment).  Every episode folder goes to ``iter_k/eval/target/<task>/<episode>/``; ``summary.json`` has
the episode-level rate overall and per task group (atomic_seen 18, composite_seen 16, composite_unseen
16 tasks), the way openpi's extract_combine_results.py counts it.

Machine paths: ``SI_OPENPI_DIR`` (~/openpi-si), ``SI_OPENPI_PYTHON``, ``SI_EVAL_ROBOCASA_PYTHON``
(~/micromamba/envs/robocasa/bin/python), ``SI_SYS2_DIR`` (~/sys2_train_eval), ``SI_TARGET_DATA``
(a local copy of robocasa_dataset/v1.0/target; else synced from ``eval.target_data_uri``: data/, meta/,
extras/ only).
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .. import s3
from ..config import Config
from ..rollout.node import Proc, _free_port, _gpu_total_gb, _git_sha
from ..rollout.score import S1Socket
from ..wandb_log import Run

HOME = Path.home()
ASSETS = Path(__file__).resolve().parents[2] / "assets"
METHOD = "selfimprove"
#: the fleet's defaults (openpi examples/robocasa/run_combine_fleet.sh) for the 53.7% arm
FLEET_FLAGS = ["--s2-model", "system2-full", "--max-turns", "20", "--rollout-limit-mode", "max_official_steps",
               "--max-s2-calls-safety", "100", "--max-steps-cap", "400", "--no-highres-video", "--general-rules",
               "--task-rules", "--last-milestone-retry"]


def units() -> list[dict]:
    table = json.loads((ASSETS / "target_eval_30.json").read_text())["tasks"]
    split = {t: v.get("split") for t, v in table.items()}
    out = []
    for task in sorted(table):
        for ep in table[task]["episodes"]:
            out.append({"task": task, "episode": int(ep), "split": split.get(task)})
    return out


def task_groups() -> dict[str, str]:
    import sys

    from ..tasks import ASSETS as _A  # noqa: F401
    sys.path.insert(0, str(Path(os.path.expanduser(os.environ.get("SI_HEXASHELL_ROOT", "~/HexaShell-s1")))))
    try:
        from hexashell.benches.robocasa import protocol as P

        return {t: g for g in P.TARGET_SETS for t, _ in P.TASKS[g]}
    except Exception:  # noqa: BLE001
        return {}


class EvalNode:
    def __init__(self, cfg: dict, s1_ckpt: str, s2_ckpt: str, label: str):
        self.cfg = cfg
        self.e = cfg["eval"]
        s3.configure(os.environ.get("SELFIMPROVE_AWS_PROFILE"), cfg.get("aws_region"))
        self.openpi = Path(os.path.expanduser(os.environ.get("SI_OPENPI_DIR", self.e.get("openpi_dir", "~/openpi-si"))))
        self.openpi_python = os.path.expanduser(os.environ.get("SI_OPENPI_PYTHON", "~/openpi/.venv/bin/python"))
        self.robocasa_python = os.path.expanduser(os.environ.get("SI_EVAL_ROBOCASA_PYTHON",
                                                                 "~/micromamba/envs/robocasa/bin/python"))
        self.sys2 = Path(os.path.expanduser(os.environ.get("SI_SYS2_DIR", "~/sys2_train_eval")))
        scratch = os.environ.get("SI_NODE_DIR") or ("/mnt/data/selfimprove" if os.path.isdir("/mnt/data")
                                                     else str(HOME / "selfimprove_work" / "node"))
        self.scratch = Path(scratch)
        self.gpu = self._gpu()
        self.label = label
        self.name = f"{socket.gethostname()}-gpu{self.gpu}"
        self.dir = self.scratch / f"eval-{label}-{self.name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s1_dir = self._fetch(s1_ckpt)
        self.s2_dir = self._fetch(s2_ckpt)
        self.data = self._target_data()
        self.s1_port, self.s2_port = _free_port(), _free_port()
        self.s1 = self.s2 = None
        self._start()

    def _gpu(self) -> str:
        try:
            import ray

            if ray.is_initialized() and ray.get_gpu_ids():
                return str(ray.get_gpu_ids()[0])
        except Exception:  # noqa: BLE001
            pass
        return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"

    def _fetch(self, uri: str) -> Path:
        if not uri.startswith("s3://"):
            return Path(os.path.expanduser(uri))
        local = self.scratch / "ckpts" / hashlib.sha1(uri.encode()).hexdigest()[:12]
        if not (local / ".complete").exists():
            s3.sync_down(uri, local, excludes=("global_step*", "*.pt", "rng_state*"))
            (local / ".complete").write_text(uri)
        return local

    def _target_data(self) -> Path:
        local = os.environ.get("SI_TARGET_DATA")
        if local:
            data = Path(os.path.expanduser(local))
            if "target" not in data.parts:
                # combined_eval names an episode <task>__<split>__episode_N by the split it finds among the
                # data path's components ('target'): give it one
                alias = self.scratch / "robocasa_dataset" / "v1.0" / "target"
                alias.parent.mkdir(parents=True, exist_ok=True)
                if not alias.exists():
                    alias.symlink_to(data.resolve(), target_is_directory=True)
                data = alias
            return data
        dest = self.scratch / "robocasa_target"
        if not (dest / ".complete").exists():
            s3.sync_down(self.e["target_data_uri"], dest, excludes=("*/videos/*",))
            (dest / ".complete").write_text(self.e["target_data_uri"])
        return dest

    def _start(self) -> None:
        total = _gpu_total_gb(self.gpu)
        frac = min(0.5, float(self.e.get("s1_mem_gb", 20)) / total) if total else 0.32
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=f"{frac:.3f}",
                   PYTHONPATH=f"{self.openpi}/src:{self.openpi}/scripts:{self.openpi}/packages/openpi-client/src")
        self.s1 = Proc([self.openpi_python, str(self.openpi / "scripts/serve_policy.py"), "--port", str(self.s1_port),
                        "policy:checkpoint", "--policy.config=auto", "--policy.dir", str(self.s1_dir)], env=env,
                       cwd=self.openpi, log=self.dir / "s1_server.log")
        deadline = time.time() + 3600
        sock = S1Socket(f"ws://127.0.0.1:{self.s1_port}")
        while True:
            try:
                sock.connect()
                sock.close()
                break
            except Exception:  # noqa: BLE001
                if not self.s1.alive() or time.time() > deadline:
                    raise RuntimeError(f"S1 server failed (see {self.s1.log})") from None
                time.sleep(5)
        # S2 only once S1 holds its memory: vLLM sizes its KV cache from what is free when it starts
        env2 = dict(os.environ, MODEL_DIR=str(self.s2_dir), GPU=self.gpu, GPU_FRAC=str(self.e.get("s2_gpu_frac", 0.42)),
                    PORT=str(self.s2_port), NAME=f"si-s2-{self.label}-{self.s2_port}", MEDIA_DIR=str(self.dir / "media"),
                    LOG_DIR=str(self.dir))
        if self.e.get("s2_gpu_run_args"):  # e.g. snap / CDI docker: "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={gpu}"
            env2["GPU_RUN_ARGS"] = str(self.e["s2_gpu_run_args"]).format(gpu=self.gpu)
        self.s2 = Proc(["bash", str(self.sys2 / "scripts/serve_system2_vllm.sh")], env=env2, cwd=self.sys2,
                       log=self.dir / "s2_server.log")
        import urllib.request

        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.s2_port}/v1/models", timeout=10):
                    break
            except Exception:  # noqa: BLE001
                if not self.s2.alive() or time.time() > deadline:
                    raise RuntimeError(f"S2 vLLM failed (see {self.s2.log})") from None
                time.sleep(10)

    def info(self) -> dict:
        return {"name": self.name, "gpu": self.gpu, "s1": str(self.s1_dir), "s2": str(self.s2_dir)}

    def lerobot_dir(self, task: str) -> str:
        hits = sorted(glob.glob(str(self.data / "*" / task / "*" / "lerobot")))
        if not hits:
            raise FileNotFoundError(f"no target LeRobot data for {task} under {self.data}")
        return hits[0]

    def run_unit(self, task: str, episode: int, attempt: int = 1, *, upload_uri: str | None = None) -> dict:
        t0 = time.time()
        out_root = self.dir / "combine"
        flat = f"{task}__target__episode_{int(episode):06d}"
        for old in (out_root / METHOD).glob(f"{task}__*__episode_{int(episode):06d}"):
            shutil.rmtree(old, ignore_errors=True)
        cmd = [self.robocasa_python, "examples/robocasa/combined_eval.py", "--lerobot-dir", self.lerobot_dir(task),
               "--episodes", str(int(episode)), "--s1-dir", str(self.s1_dir), "--s2-dir", str(self.s2_dir),
               "--s1-port", str(self.s1_port), "--s2-port", str(self.s2_port),
               "--norm-stats", str(self.s1_dir / "assets/robocasa_system1/norm_stats.json"),
               "--out-root", str(out_root), "--method", METHOD, *FLEET_FLAGS,
               *str(self.e.get("combined_eval_flags") or "").split()]
        env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", CUDA_VISIBLE_DEVICES=self.gpu,
                   OPENPI_REPO=str(self.openpi), SYS2_REPO=str(self.sys2))
        res = {"task": task, "episode": int(episode), "attempt": attempt, "node": self.name}
        with open(self.dir / f"{flat}.log", "ab") as fh:
            try:
                subprocess.run(cmd, cwd=str(self.openpi), env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=float(self.e.get("episode_timeout_s", 7200)))
            except subprocess.TimeoutExpired:
                res["error"] = "timeout"
        found = sorted((out_root / METHOD).glob(f"{task}__*__episode_{int(episode):06d}/episode.json"))
        doc_path = found[0] if found else out_root / METHOD / flat / "episode.json"
        ep_dir = doc_path.parent
        if not doc_path.exists():
            return {**res, "status": "failed", "error": res.get("error", "no episode.json"),
                    "seconds": round(time.time() - t0, 1)}
        doc = json.loads(doc_path.read_text())
        res.update(status="error" if doc.get("error") else "ok", success=bool(doc.get("episode_success")),
                   termination=doc.get("termination"), turns=doc.get("n_turns"), seconds=doc.get("seconds"),
                   rule_interventions=doc.get("n_rule_interventions"), openpi_commit=_git_sha(self.openpi))
        if upload_uri and res["status"] == "ok":
            s3.sync_up(ep_dir, upload_uri, delete=True)
            res["uri"] = upload_uri
            shutil.rmtree(ep_dir, ignore_errors=True)
        return res

    def shutdown(self) -> None:
        for p in (self.s1, self.s2):
            if p is not None:
                p.stop()
        subprocess.run(["docker", "rm", "-f", f"si-s2-{self.label}-{self.s2_port}"], capture_output=True)


try:
    import ray

    RayEvalNode = ray.remote(num_gpus=1)(EvalNode)
except ImportError:  # pragma: no cover
    RayEvalNode = None


def summarize(results: list[dict], groups: dict[str, str]) -> dict:
    """Episode-level success overall and per task group (as extract_combine_results counts it)."""
    done = [r for r in results if r.get("status") == "ok"]
    by_group = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    for r in done:
        by_group[groups.get(r["task"], "unknown")].append(bool(r.get("success")))
        by_task[r["task"]].append(bool(r.get("success")))
    rate = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    out = {"episodes": len(done), "overall": round(rate([bool(r.get("success")) for r in done]), 4)}
    for g, xs in by_group.items():
        out[g] = round(rate(xs), 4)
        out[f"{g}_n"] = len(xs)
    out["per_task"] = {t: round(rate(xs), 4) for t, xs in sorted(by_task.items())}
    out["terminations"] = dict(collections.Counter(str(r.get("termination")) for r in done))
    return out


def run_eval(cfg: Config, k: int, s1_ckpt: str, *, local: bool = False, max_nodes: int | None = None,
             limit: int | None = None, tasks: list[str] | None = None, wandb_on: bool = True) -> dict:
    import ray

    from ..pool import run_pool

    e = cfg["eval"]
    all_units = [u for u in units() if not tasks or u["task"] in set(tasks)]
    prefix = cfg.eval_uri(k)
    done = {}
    for uri, _ in s3.iter_keys(prefix + "/"):
        if uri.endswith("/episode.json"):
            rel = uri[len(prefix) + 1:].split("/")
            if len(rel) == 3:
                done[f"{rel[0]}/{rel[1]}"] = uri
    results = []
    for key, uri in done.items():
        doc = s3.read_json(uri, default={}) or {}
        task, ep = key.split("/")
        results.append({"task": task, "episode": int(ep), "status": "ok", "success": bool(doc.get("episode_success")),
                        "termination": doc.get("termination"), "uri": uri.rsplit("/", 1)[0]})
    todo = [u for u in all_units if f"{u['task']}/{u['episode']}" not in done]
    if limit is not None:
        todo = todo[:limit]
    groups = task_groups()
    run = Run(cfg, name=f"iter_{k:02d}/eval", job_type="eval", iteration=k, enabled=wandb_on,
              extra_config={"s1_ckpt": s1_ckpt, "s2_ckpt": cfg.s2_ckpt, "units": len(all_units)})
    print(f"eval iteration {k}: {len(all_units)} units, {len(done)} done, {len(todo)} to run")
    env_vars = {"SELFIMPROVE_AWS_REGION": cfg.region}
    for name in ("SI_OPENPI_DIR", "SI_OPENPI_PYTHON", "SI_EVAL_ROBOCASA_PYTHON", "SI_SYS2_DIR", "SI_TARGET_DATA",
                 "SI_NODE_DIR", "SI_HEXASHELL_ROOT"):
        if os.environ.get(name):
            env_vars[name] = os.environ[name]
    if todo:
        if local:
            env_vars["SELFIMPROVE_AWS_PROFILE"] = cfg.get("aws_profile") or ""
            ray.init(runtime_env={"env_vars": env_vars}, include_dashboard=False)
        else:
            ray.init(address="auto")  # in a Ray job on the head; the nodes' paths come from their ~/.profile
        n = int(max_nodes or e.get("num_gpus", 8))
        if local:
            n = min(n, int(ray.cluster_resources().get("GPU", 0)) or 1)
        label = f"{cfg.exp_id}-i{k:02d}"
        t0 = time.time()

        def on_result(u, res):
            results.append(res)
            run.log_episode(res)

        def on_status(inflight, ready):
            s = summarize(results, groups)
            snap = {k2: v for k2, v in s.items() if isinstance(v, (int, float))}
            snap.update(progress=round(s["episodes"] / max(1, len(all_units)), 4), running=inflight, nodes=ready,
                        elapsed_h=(time.time() - t0) / 3600)
            s3.write_json(f"{prefix}/_status.json", {**snap, "updated": time.strftime("%F %T")})
            run.log(snap)

        run_pool(make_node=lambda: RayEvalNode.options(max_concurrency=int(e.get("stacks_per_gpu", 1)) + 2).remote(
                     cfg.raw, s1_ckpt, cfg.s2_ckpt, label),
                 units=todo, key=lambda u: f"{u['task']}/{u['episode']}",
                 submit=lambda node, u, a: node.run_unit.remote(u["task"], u["episode"], a,
                                                                upload_uri=f"{prefix}/{u['task']}/{u['episode']}"),
                 ok=lambda r: r.get("status") == "ok", n_nodes=n, slots=int(e.get("stacks_per_gpu", 1)),
                 on_result=on_result, on_status=on_status, request_gpus=not local)
    summary = summarize(results, groups)
    summary.update(s1_ckpt=s1_ckpt, s2_ckpt=cfg.s2_ckpt, units=len(all_units))
    s3.write_json(f"{prefix}/summary.json", summary)
    if summary["episodes"] >= len(all_units) and limit is None:
        s3.write_json(f"{prefix}/_DONE", summary)
    run.finish({k2: v for k2, v in summary.items() if isinstance(v, (int, float))}, table=results)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="the S1 checkpoint to evaluate")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default="", help="comma-separated subset (a check)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    s3.configure(cfg.get("aws_profile") if args.local else None, cfg.region)
    tasks = [t for t in args.tasks.split(",") if t]
    print(json.dumps(run_eval(cfg, args.iteration, args.ckpt, local=args.local, max_nodes=args.max_nodes,
                              limit=args.limit, tasks=tasks or None, wandb_on=not args.no_wandb), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

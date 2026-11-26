"""One GPU of the rollout cluster: an S1 server, a SAM3 service and ``slots`` warm RoboCasa simulators that
play HexaShell episodes (Astra + System1) one after another.

Used as a Ray actor (``RolloutNode``, one GPU, ``max_concurrency`` = slots): ``run_episode(task, seed)``
takes a free slot, plays the episode with ``python -m hexashell.benches.run`` on the slot's warm simulator
(``hexashell.benches.daemon``), scores its chunks with the S1 server (score.py), writes ``rollout.json``
and uploads the whole episode folder to ``rollout/<task>/<seed>/``.  The same class runs without Ray
(``python -m selfimprove.rollout.node --task T --seed S ...``) for one-off checks.

Machine paths come from HexaShell's machine settings file (``$HEXASHELL_ENV`` or
``<hexashell>/config/env.sh``: HEXASHELL_PYTHON, HEXASHELL_ROBOCASA_PYTHON, HEXASHELL_ROBOCASA_ENV,
SAM3_PYTHON, SAM3_CHECKPOINT_PATH) and these variables:

- ``SI_HEXASHELL_ROOT`` (default ~/HexaShell-s1), ``SI_OPENPI_DIR`` (~/openpi-si),
  ``SI_OPENPI_PYTHON`` (~/openpi/.venv/bin/python), ``SI_NODE_DIR`` (local scratch: /mnt/data/selfimprove
  when it exists, else ~/selfimprove_work/node)
- ``OPENAI_API_KEY`` for Astra when ``~/.config/hexashell/api-key`` does not exist (cluster nodes)
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import s3
from .score import S1Socket, score_episode

HOME = Path.home()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def machine_env(hexashell_root: Path) -> dict[str, str]:
    """HexaShell's machine settings (the export lines of its env file), read with bash."""
    env_file = os.environ.get("HEXASHELL_ENV") or str(hexashell_root / "config" / "env.sh")
    if not os.path.isfile(env_file):
        return {}
    out = subprocess.run(["bash", "-c", 'source "$1" >/dev/null 2>&1; env -0', "_", env_file],
                         capture_output=True, check=True).stdout
    env = dict(item.split("=", 1) for item in out.decode().split("\0") if "=" in item)
    keep = ("HEXASHELL_", "SAM3_", "ROBODOJO", "ACCEPT_EULA", "OMNI_", "PRIVACY", "PYTHONNOUSERSITE", "TMPDIR")
    return {k: v for k, v in env.items() if k.startswith(keep)}


@dataclass
class Paths:
    hexashell: Path
    openpi: Path
    openpi_python: str
    hexashell_python: str
    robocasa_python: str
    robocasa_env: str | None
    sam3_python: str
    sam3_ckpt: str
    scratch: Path
    machine: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Paths":
        root = Path(os.path.expanduser(os.environ.get("SI_HEXASHELL_ROOT", str(HOME / "HexaShell-s1"))))
        m = machine_env(root)
        get = lambda k, d=None: os.environ.get(k) or m.get(k) or d  # noqa: E731
        scratch = os.environ.get("SI_NODE_DIR") or ("/mnt/data/selfimprove" if os.path.isdir("/mnt/data")
                                                     else str(HOME / "selfimprove_work" / "node"))
        return cls(
            hexashell=root,
            openpi=Path(os.path.expanduser(os.environ.get("SI_OPENPI_DIR", str(HOME / "openpi-si")))),
            openpi_python=os.path.expanduser(os.environ.get("SI_OPENPI_PYTHON", str(HOME / "openpi/.venv/bin/python"))),
            hexashell_python=get("HEXASHELL_PYTHON", str(root / ".venvs/policy/bin/python")),
            robocasa_python=get("HEXASHELL_ROBOCASA_PYTHON", str(root / ".venvs/bench-robocasa/bin/python")),
            robocasa_env=get("HEXASHELL_ROBOCASA_ENV"),
            sam3_python=get("SAM3_PYTHON", str(root / ".venvs/policy/bin/python")),
            sam3_ckpt=get("SAM3_CHECKPOINT_PATH", str(root / "checkpoints/sam3.pt")),
            scratch=Path(scratch), machine=m)


def _read_env_file(path: str | None) -> dict[str, str]:
    """The ``export NAME=VALUE`` lines of a benchmark's env file (hexashell.benches.session.read_env_file)."""
    out: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return out
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = os.path.expandvars(v.strip().strip('"').strip("'"))
    return out


def sim_env(paths: Paths, egl_device: str) -> dict[str, str]:
    """The simulator server's environment, as hexashell.benches.session.server_env builds it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV", "CONDA"))}
    for key in [k for k in env if any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET"))]:
        env.pop(key)
    env.update(_read_env_file(paths.robocasa_env))
    # the machine file's HexaShell settings (env.cluster.sh: e.g. $HEXASHELL_ROBOCASA_DEMOS, the recorded episodes'
    # first states that split demo reads in this server): the raylet's own environment does not have them
    env.update({k: v for k, v in (paths.machine or {}).items() if k.startswith("HEXASHELL_")})
    env["PYTHONPATH"] = str(paths.hexashell)
    env["PYTHONUNBUFFERED"] = "1"
    env["HEXA_EGL_DEVICE"] = str(egl_device)
    return env


def _die_with_parent() -> None:
    """In the child, before exec: SIGTERM when the process that started it dies (a Ray actor killed hard would
    otherwise leave its S1 server holding ~20 GB of GPU memory)."""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


class Proc:
    """A child process in its own session (stopped with its whole group)."""

    def __init__(self, cmd, *, env, cwd, log: Path):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.fh = open(log, "ab")
        self.p = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=self.fh,
                                  stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, timeout: float = 60) -> None:
        import signal

        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.fh.close()


class Slot:
    def __init__(self, n: int, node: "RolloutNode"):
        self.n = n
        self.node = node
        self.daemon: Proc | None = None
        self.served = 0
        self.gen = 0
        self.sock = str(node.dir / f"sim{n}.sock")

    def ensure(self) -> None:
        if self.daemon is not None and self.daemon.alive() and self.served < self.node.recycle:
            return
        self.restart("recycle" if self.daemon is not None else "start")

    def restart(self, why: str) -> None:
        if self.daemon is not None:
            self.daemon.stop()
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        self.gen += 1
        p = self.node.paths
        self.daemon = Proc([p.robocasa_python, "-m", "hexashell.benches.daemon", "hexashell.benches.robocasa.server",
                            self.sock, "--idle-s", "3600"], env=sim_env(p, self.node.gpu), cwd=self.node.dir,
                           log=self.node.dir / "logs" / f"sim{self.n}-{self.gen}.log")
        self.served = 0
        deadline = time.time() + 1800
        while not os.path.exists(self.sock):
            if not self.daemon.alive():
                raise RuntimeError(f"simulator slot {self.n} exited while starting (see {self.daemon.log})")
            if time.time() > deadline:
                raise RuntimeError(f"simulator slot {self.n} did not listen within 30 min")
            time.sleep(1)
        self.node.note(f"slot{self.n} simulator ready ({why})")


class RolloutNode:
    """See the module docstring.  ``cfg`` is the experiment config's raw dict."""

    def __init__(self, cfg: dict, iteration: int, ckpt_uri: str, *, secrets: dict | None = None,
                 node_name: str | None = None):
        self.cfg = cfg
        self.r = cfg["rollout"]
        self.iteration = int(iteration)
        self.ckpt_uri = ckpt_uri.rstrip("/")
        self.paths = Paths.from_env()
        self._check_demos()
        self.secrets = dict(secrets or {})
        s3.configure(os.environ.get("SELFIMPROVE_AWS_PROFILE"), cfg.get("aws_region"))
        self.gpu = self._gpu()
        self.name = node_name or f"{socket.gethostname()}-gpu{self.gpu}"
        self.dir = self.paths.scratch / f"{cfg['exp_id']}-iter{self.iteration:02d}-{self.name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recycle = int(self.r.get("recycle_after", 20))
        self.slots_n = int(self.r.get("slots_per_gpu", 4))
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.ckpt_dir = self._fetch_ckpt()
        self.s1_port, self.sam3_port = _free_port(), _free_port()
        self.s1: Proc | None = None
        self.sam3: Proc | None = None
        self._start_services()
        self.free: queue.Queue[Slot] = queue.Queue()
        for i in range(self.slots_n):
            self.free.put(Slot(i + 1, self))

    # -- setup ---------------------------------------------------------------------------------------------
    def _gpu(self) -> str:
        try:
            import ray

            if ray.is_initialized():  # inside a Ray actor: the GPU Ray gave it (nvidia-smi numbering)
                ids = ray.get_gpu_ids()
                if ids:
                    return str(ids[0])
        except Exception:  # noqa: BLE001
            pass
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        return vis or "0"

    def note(self, text: str) -> None:
        line = f"{time.strftime('%F %T')} {self.name} {text}"
        with self.lock:
            self.events.append(line)
            with open(self.dir / "node.log", "a") as fh:
                fh.write(line + "\n")

    def _fetch_ckpt(self) -> Path:
        """The rollout checkpoint on local disk (an S3 step dir is synced once per node and round)."""
        if not self.ckpt_uri.startswith("s3://"):
            return Path(os.path.expanduser(self.ckpt_uri))
        digest = hashlib.sha1(self.ckpt_uri.encode()).hexdigest()[:12]
        local = self.paths.scratch / "ckpts" / digest
        marker = local / ".complete"
        if not marker.exists():
            self.note(f"downloading {self.ckpt_uri}")
            s3.sync_down(self.ckpt_uri, local)
            marker.write_text(self.ckpt_uri)
        return local

    def _start_services(self) -> None:
        p = self.paths
        gpu_total = _gpu_total_gb(self.gpu)
        frac = min(0.9, float(self.r.get("s1_mem_gb", 20)) / gpu_total) if gpu_total else 0.4
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=f"{frac:.3f}",
                   PYTHONPATH=f"{p.openpi}/src:{p.openpi}/scripts:{p.openpi}/packages/openpi-client/src",
                   PYTHONUNBUFFERED="1")
        self.s1 = Proc([p.openpi_python, str(p.openpi / "scripts" / "serve_policy.py"), "--port", str(self.s1_port),
                        "policy:checkpoint", "--policy.dir", str(self.ckpt_dir)], env=env, cwd=p.openpi,
                       log=self.dir / "logs" / "s1_server.log")
        if p.sam3_ckpt and os.path.exists(p.sam3_ckpt):
            env3 = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
            env3.update(CUDA_VISIBLE_DEVICES=self.gpu, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", MALLOC_ARENA_MAX="4")
            self.sam3 = Proc([p.sam3_python, str(p.hexashell / "services" / "sam3_server.py"), "--port",
                              str(self.sam3_port), "--checkpoint", p.sam3_ckpt], env=env3, cwd=p.hexashell,
                             log=self.dir / "logs" / "sam3.log")
        self._wait_s1()
        self._wait_sam3()
        self.note(f"services ready: s1 :{self.s1_port} (mem fraction {frac:.2f}), sam3 :{self.sam3_port}")

    def _wait_s1(self, timeout: float = 1800) -> None:
        sock = S1Socket(f"ws://127.0.0.1:{self.s1_port}")
        deadline = time.time() + timeout
        while True:
            if not self.s1.alive():
                raise RuntimeError(f"S1 server exited while loading (see {self.s1.log})")
            try:
                sock.connect()
                sock.close()
                return
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    raise RuntimeError(f"S1 server did not answer within {timeout:.0f} s") from None
                time.sleep(5)

    def _wait_sam3(self, timeout: float = 900) -> None:
        if self.sam3 is None:
            return
        import urllib.request

        deadline = time.time() + timeout
        while True:
            if not self.sam3.alive():
                raise RuntimeError(f"SAM3 exited while loading (see {self.sam3.log})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.sam3_port}/health", timeout=10):
                    return
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    raise RuntimeError("SAM3 did not answer") from None
                time.sleep(5)

    # -- episodes ------------------------------------------------------------------------------------------
    def _check_demos(self) -> None:
        """Split demo: the node must have the recorded episodes' first states, and its simulators must be told
        where (a node without them fails here, at start, not in every episode)."""
        if self.r.get("split") != "demo":
            return
        folder = (self.paths.machine or {}).get("HEXASHELL_ROBOCASA_DEMOS") or os.environ.get("HEXASHELL_ROBOCASA_DEMOS")
        if not folder or not Path(folder).is_dir():
            raise RuntimeError(f"split demo: no recorded episodes' first states on this node (HEXASHELL_ROBOCASA_DEMOS="
                               f"{folder!r}; cluster/setup_node.sh syncs them and writes it to the machine file)")
        missing = [t for t in (self.r.get("include_tasks") or []) if not (Path(folder) / t / "index.json").exists()]
        if missing:
            raise RuntimeError(f"split demo: {folder} has no recorded episodes of {missing}")

    def info(self) -> dict:
        return {"name": self.name, "gpu": self.gpu, "slots": self.slots_n, "s1_port": self.s1_port,
                "sam3_port": self.sam3_port, "ckpt": self.ckpt_uri, "dir": str(self.dir)}

    def hexashell_sets(self, start_salt: int | None = None) -> list[str]:
        sets = dict(self.r.get("hexashell_sets") or {})
        # the start perturbation is per episode (the plan's start_salt): rollout.start_shift / start_turn, off when None
        for k in ("start_shift", "start_turn", "start_salt"):
            sets.pop(k, None)
        if start_salt is not None and (self.r.get("start_shift") or self.r.get("start_turn")):
            sets.update(start_shift=self.r.get("start_shift") or 0, start_turn=self.r.get("start_turn") or 0,
                        start_salt=int(start_salt))
        sets["s1_url"] = f"ws://127.0.0.1:{self.s1_port}"
        if self.sam3 is not None:
            sets["sam3_url"] = f"http://127.0.0.1:{self.sam3_port}"
        return [f"{k}={v}" for k, v in sets.items()]

    def run_episode(self, task: str, seed: int, attempt: int = 1, *, upload_uri: str | None = None,
                    keep_local: bool = False, demo: str | None = None, start_salt: int | None = None) -> dict:
        slot = self.free.get()
        t0 = time.time()
        out = {"task": task, "seed": int(seed), "attempt": int(attempt), "node": self.name, "gpu": self.gpu,
               "slot": slot.n, "ckpt": self.ckpt_uri, "iteration": self.iteration, "exp_id": self.cfg["exp_id"],
               "start_salt": start_salt}
        try:
            slot.ensure()
            summary, record = self._play(slot, task, int(seed), start_salt)
            slot.served += 1
            out.update(summary=summary, play_s=round(time.time() - t0, 1))
            if record is None:
                slot.restart("no record")
                out["status"] = "no_record"
                return out
            score_cfg = self.r.get("score") or {}
            if score_cfg.get("enabled", True):
                try:
                    out["score"] = score_episode(record, f"ws://127.0.0.1:{self.s1_port}",
                                                 stride=int(score_cfg.get("stride", 16)),
                                                 num_draws=int(score_cfg.get("num_draws", 4)))
                except Exception as exc:  # noqa: BLE001 - the episode is still worth keeping
                    out["score_error"] = f"{type(exc).__name__}: {exc}"
            out.update(_episode_stats(record))
            if demo is not None:  # split demo: the node's recordings must be the plan's (same index order)
                played = (json.loads((record / "result.json").read_text()).get("demo") or {}).get("id")
                out["demo"] = played
                if played != demo:
                    out["errors"] = list(out.get("errors") or []) + [f"demo: seed {seed} started {played}, the plan has {demo}"]
            # an episode the simulator or the gateway ended is not the policy's failure: played again, and kept out
            # of the success rates (rollout.json keeps the errors either way)
            out["status"] = "error" if out.get("errors") else "ok"
            out["seconds"] = round(time.time() - t0, 1)
            out["hexashell_commit"] = _git_sha(self.paths.hexashell)
            out["openpi_commit"] = _git_sha(self.paths.openpi)
            (record / "rollout.json").write_text(json.dumps(out, indent=1, default=str))
            if out["status"] == "error":
                slot.restart("episode error")
            if upload_uri:
                s3.sync_up(record, upload_uri, delete=True)
                out["uri"] = upload_uri
                if not keep_local:
                    shutil.rmtree(record, ignore_errors=True)
            out["record"] = str(record)
            return out
        except Exception as exc:  # noqa: BLE001
            out.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - t0, 1))
            try:
                slot.restart("failure")
            except Exception as exc2:  # noqa: BLE001
                out["restart_error"] = str(exc2)
            return out
        finally:
            self.free.put(slot)

    def _play(self, slot: Slot, task: str, seed: int, start_salt: int | None = None) -> tuple[dict, Path | None]:
        p = self.paths
        out_dir = self.dir / "results"
        cmd = [p.hexashell_python, "-m", "hexashell.benches.run", "--bench", "robocasa", "--task", task, "--seed",
               str(seed), "--split", str(self.r.get("split", "pretrain")), "--out", str(out_dir), "--python",
               p.robocasa_python, "--sim-socket", slot.sock]
        if p.robocasa_env:
            cmd += ["--env-file", p.robocasa_env]
        for s in self.hexashell_sets(start_salt):
            cmd += ["--set", s]
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        env.update({k: v for k, v in p.machine.items() if k.startswith("HEXASHELL_")})
        env.update(self.secrets)
        env["PYTHONUNBUFFERED"] = "1"
        log = self.dir / "logs" / f"{task}_s{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            try:
                r = subprocess.run(cmd, env=env, cwd=str(p.hexashell), stdout=subprocess.PIPE, stderr=fh,
                                   timeout=float(self.r.get("episode_timeout_s", 9000)))
                stdout = r.stdout.decode(errors="replace")
            except subprocess.TimeoutExpired as exc:
                stdout = (exc.stdout or b"").decode(errors="replace")
        summary = {}
        for line in reversed(stdout.strip().splitlines()):
            try:
                summary = json.loads(line)
                break
            except ValueError:
                continue
        record = Path(summary["record"]) if summary.get("record") else None
        if record is not None and not (record / "result.json").exists():
            record = None
        return summary, record

    def shutdown(self) -> None:
        while not self.free.empty():
            slot = self.free.get_nowait()
            if slot.daemon is not None:
                slot.daemon.stop()
        for proc in (self.s1, self.sam3):
            if proc is not None:
                proc.stop()


def _gpu_total_gb(gpu: str) -> float:
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def _git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _episode_stats(record: Path) -> dict:
    """What the driver aggregates: outcome, length, cost, who moved."""
    r = json.loads((record / "result.json").read_text())
    out = {k: r.get(k) for k in ("success", "step", "step_limit", "end", "turns", "seconds", "task_set", "errors")}
    tok = r.get("tokens") or {}
    out["tokens_prompt"] = int(tok.get("prompt") or 0)
    out["tokens_cached"] = int(tok.get("cached") or 0)
    out["tokens_completion"] = int(tok.get("completion") or 0)
    tj = record / "trajectory" / "trajectory.json"
    if tj.exists():
        labels = json.loads(tj.read_text()).get("labels") or []
        n = int(out.get("step") or 0)
        s1 = 0
        for i, (start, _, by) in enumerate(labels):
            end = labels[i + 1][0] if i + 1 < len(labels) else n
            if by == "s1":
                s1 += max(0, end - start)
        out["s1_step_share"] = round(s1 / n, 4) if n else 0.0
    return out


try:  # the Ray actor (Ray is only needed on the cluster / with --ray)
    import ray

    RayRolloutNode = ray.remote(num_gpus=1)(RolloutNode)
except ImportError:  # pragma: no cover
    RayRolloutNode = None


def main(argv: list[str] | None = None) -> int:
    """One episode on this machine, without Ray: a check of the node's whole path."""
    import argparse

    from ..config import Config

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--ckpt", required=True, help="S1 checkpoint step dir (local or s3://)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--upload", action="store_true", help="upload to the experiment's rollout/<task>/<seed>/")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    s3.configure(cfg.get("aws_profile"), cfg.region)
    node = RolloutNode(cfg.raw, args.iteration, args.ckpt)
    try:
        uri = cfg.episode_uri(args.iteration, args.task, args.seed) if args.upload else None
        print(json.dumps(node.run_episode(args.task, args.seed, upload_uri=uri, keep_local=True), indent=1, default=str))
    finally:
        node.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

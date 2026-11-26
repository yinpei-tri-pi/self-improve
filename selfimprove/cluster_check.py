"""What a cluster GPU node must have, checked on the node itself (cluster/bring_up.sh validate).

    python -m selfimprove.cluster_check rollout|eval
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    return ok


def run(cmd: list[str], timeout: float = 600, cwd: str | None = None) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def main(argv: list[str] | None = None) -> int:
    role = (argv or sys.argv[1:] or ["rollout"])[0]
    from .rollout.node import Paths

    ok = True
    good, out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    ok &= check("GPU", good, out)
    if role == "rollout":
        p = Paths.from_env()
        for name, path in (("HexaShell policy python", p.hexashell_python), ("RoboCasa python", p.robocasa_python),
                           ("openpi python", p.openpi_python), ("SAM3 checkpoint", p.sam3_ckpt)):
            ok &= check(name, bool(path) and os.path.exists(path), str(path))
        good, out = run([p.robocasa_python, "-c", "import robocasa, robosuite, mujoco; print(robocasa.__file__)"])
        ok &= check("robocasa imports", good, out)
        good, out = run([p.hexashell_python, "-c", "import hexashell.benches.robocasa.protocol as P; print(len(P.HORIZON))"],
                        timeout=120, cwd=str(p.hexashell))
        ok &= check("HexaShell knows the pretrain300 tasks", good and out.strip().splitlines()[-1].isdigit() and int(out.strip().splitlines()[-1]) >= 300, out)
        ok &= check("OPENAI_API_KEY or key file", bool(os.environ.get("OPENAI_API_KEY"))
                    or Path("~/.config/hexashell/api-key").expanduser().exists())
    else:
        data = Path(os.environ.get("SI_TARGET_DATA", "/mnt/data/robocasa_dataset/v1.0/target"))
        ok &= check("target data", (data / ".complete").exists(), str(data))
        py = os.path.expanduser(os.environ.get("SI_EVAL_ROBOCASA_PYTHON", "~/micromamba/envs/robocasa/bin/python"))
        good, out = run([py, "-c", "import robocasa, mujoco; print('ok')"])
        ok &= check("combined_eval simulator env", good, out)
        good, out = run(["sudo", "docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
        ok &= check("S2 vLLM image", "system2" in out, out.splitlines()[0] if out else "")
    p = Paths.from_env()
    env = dict(os.environ, PYTHONPATH=f"{p.openpi}/src:{p.openpi}/packages/openpi-client/src")
    try:
        r = subprocess.run([p.openpi_python, "-c", "import openpi.policies.policy as m; print(hasattr(m, 'ScoringPolicy'))"],
                           capture_output=True, text=True, timeout=600, env=env)
        good, out = r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]
    except OSError as exc:
        good, out = False, str(exc)
    ok &= check("openpi-si (ScoringPolicy)", good and "True" in out, out)
    print("NODE OK" if ok else "NODE NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""One W&B project for the whole loop; one group per experiment; one run per stage of each iteration.

Runs (``group = exp_id``, name ``<exp_id>/<run>``):

- ``<exp>/scheduler``         the loop: one point per iteration and stage (x-axis ``iteration``):
                              ``loop/stage``, ``rollout/*`` (success, S1 share, cost), ``data/*`` (segments,
                              samples, positives/negatives), ``train/*`` (final losses), ``eval/*`` (target
                              split success overall and per task group, vs the 53.7% baseline)
- ``<exp>/iter_NN/rollout``   the round live: episodes done / running / failed, success, throughput, the
                              surrogate loss of S1 vs the harness; at the end a per-episode table
- ``<exp>/iter_NN/s1-train``  openpi's own training run (same project and group, OPENPI_WANDB_GROUP)
- ``<exp>/iter_NN/eval``      the target-split evaluation live + its per-task table

Run ids are derived from the names, so a restarted stage continues its run.  ``enabled=False`` (or
``WANDB_MODE=disabled``) makes every call a no-op.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any


def _wandb_credentials(w: dict) -> None:
    """W&B's host from the config and, when the environment has no key, the key from a 0600 file
    (~/.config/selfimprove/wandb-api-key: the cluster's file_mounts put it on the head)."""
    if w.get("base_url") and not os.environ.get("WANDB_BASE_URL"):
        os.environ["WANDB_BASE_URL"] = str(w["base_url"])
    key_file = os.path.expanduser("~/.config/selfimprove/wandb-api-key")
    if not os.environ.get("WANDB_API_KEY") and os.path.exists(key_file):
        os.environ["WANDB_API_KEY"] = open(key_file).read().strip()


def run_id(exp_id: str, name: str) -> str:
    return hashlib.sha1(f"{exp_id}/{name}".encode()).hexdigest()[:16]


class Run:
    def __init__(self, cfg, name: str, job_type: str, iteration: int | None = None, *, enabled: bool = True,
                 extra_config: dict | None = None, tags: list[str] | None = None, run_id_override: str | None = None,
                 project: str | None = None):
        self.cfg = cfg
        self.job_type = job_type
        self.iteration = iteration
        self.rows: list[dict] = []
        self.wb = None
        if not enabled or os.environ.get("WANDB_MODE") == "disabled":
            return
        import wandb

        w = cfg["wandb"] if hasattr(cfg, "raw") else cfg.get("wandb", {})
        _wandb_credentials(w)
        exp_id = cfg.exp_id if hasattr(cfg, "exp_id") else cfg["exp_id"]
        conf = {"exp_id": exp_id, "iteration": iteration, "stage": job_type, **(extra_config or {})}
        self.wb = wandb.init(project=project or w.get("project"), entity=w.get("entity"), group=exp_id,
                             name=f"{exp_id}/{name}", job_type=job_type, id=run_id_override or run_id(exp_id, name),
                             resume="allow", config=conf, tags=(tags or []) + ([f"iter_{iteration:02d}"]
                                                                               if iteration is not None else []),
                             reinit="create_new")
        self.wb.define_metric("elapsed_h")
        self.wb.define_metric(f"{job_type}/*", step_metric="elapsed_h")
        self.t0 = time.time()

    def log(self, metrics: dict[str, Any], *, prefix: str | None = None) -> None:
        if self.wb is None:
            return
        p = prefix if prefix is not None else self.job_type
        data = {f"{p}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        data["elapsed_h"] = metrics.get("elapsed_h", (time.time() - self.t0) / 3600)
        self.wb.log(data)

    def log_episode(self, row: dict) -> None:
        self.rows.append(row)

    def finish(self, summary: dict | None = None, table: list[dict] | None = None) -> None:
        if self.wb is None:
            return
        import wandb

        for k, v in (summary or {}).items():
            if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                self.wb.summary[f"{self.job_type}/{k}"] = v
        rows = table if table is not None else self.rows
        if rows:
            cols = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, str, bool))})
            t = wandb.Table(columns=cols)
            for r in rows:
                t.add_data(*[r.get(c) for c in cols])
            self.wb.log({f"{self.job_type}/episodes": t})
        self.wb.finish()
        self.wb = None


class Loop:
    """The scheduler's run: one point per (iteration, stage) on the ``iteration`` axis."""

    def __init__(self, cfg, *, enabled: bool = True, name: str = "scheduler", tags=None):
        self.run = Run(cfg, name, "loop", enabled=enabled, tags=tags)
        if self.run.wb is not None:
            for p in ("rollout", "data", "train", "eval", "loop"):
                self.run.wb.define_metric(f"{p}/*", step_metric="iteration")
            self.run.wb.define_metric("iteration")

    def point(self, iteration: int, metrics: dict[str, Any]) -> None:
        """``metrics`` keyed ``rollout/...``, ``eval/...`` etc.; x = ``iteration`` (eval of iteration k is
        logged when it finishes, possibly after iteration k+1's rollout points)."""
        if self.run.wb is None:
            return
        data = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        data["iteration"] = int(iteration)
        self.run.wb.log(data)

    def note(self, text: str) -> None:
        if self.run.wb is not None:
            self.run.wb.summary["loop/last_event"] = text

    def finish(self) -> None:
        self.run.finish()

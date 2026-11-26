"""The experiment configuration (one YAML file) and where everything of an experiment lives on S3.

Layout under ``{s3_root}/{exp_id}/``::

    config.yaml                       the configuration the experiment started with
    state.json                        the scheduler's state (stage of every iteration)
    iter_00/rollout/<task>/<seed>/    one HexaShell episode record (+ chunk_scores.jsonl, rollout.json)
    iter_00/rollout/_plan.json        the episodes of the round (task, seed) and the S1 checkpoint used
    iter_00/rollout/_status.json      progress, rewritten while the round runs
    iter_00/rollout/_DONE             the round is complete (summary)
    iter_00/s1_training_shard/        openpi webdataset shards (shards/*.tar, manifest.jsonl, meta.json,
                                      segments.jsonl = every segment with its PRM score, _DONE)
    iter_00/s1_ckpts/                 the continue-training run's checkpoints; _FINAL names the one to use
    iter_00/eval/target/...           S1(iter_00's checkpoint) + S2 on the target split; summary.json, _DONE
    s2_training_shard/                Qwen3-VL shards from all iterations' rollouts (once, at the end)
    s2_ckpts/

Iteration k's rollout uses the checkpoint ``ckpt_k``: the start checkpoint for k = 0, else iteration
k-1's ``_FINAL``.  Iteration k trains ``ckpt_{k+1}`` from ``ckpt_k`` on iteration k's shards (plus the
base data) and evaluates it on the target split while iteration k+1 rolls out.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = HERE / "configs" / "default.yaml"


def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


@dataclasses.dataclass
class Config:
    """The parsed YAML (``raw``) with typed accessors for what every stage needs."""

    raw: dict

    # -- loading ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> "Config":
        """``configs/default.yaml``, then the file at ``path`` over it, then ``key.sub=value`` overrides."""
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
        if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
            raw = _deep_update(raw, yaml.safe_load(Path(path).read_text()) or {})
        for item in overrides or []:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"override {item!r}: use key.sub=value")
            _set_dotted(raw, key.strip(), value)
        cfg = cls(raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.exp_id or "/" in self.exp_id:
            raise ValueError("exp_id: a non-empty name without '/'")
        if not str(self.s3_root).startswith("s3://"):
            raise ValueError("s3_root must be an s3:// prefix")
        for key in ("rollout", "filter", "train", "eval", "wandb"):
            if key not in self.raw:
                raise ValueError(f"the configuration has no '{key}' section")

    def dump(self) -> str:
        return yaml.safe_dump(self.raw, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # -- identity ---------------------------------------------------------------------------------
    @property
    def exp_id(self) -> str:
        return str(self.raw.get("exp_id") or "")

    @property
    def s3_root(self) -> str:
        return str(self.raw["s3_root"]).rstrip("/")

    @property
    def region(self) -> str:
        return str(self.raw.get("aws_region") or "us-west-2")

    @property
    def iterations(self) -> int:
        return int(self.raw.get("iterations", 1))

    @property
    def work_dir(self) -> Path:
        return Path(os.path.expanduser(str(self.raw.get("work_dir") or WORK_ROOT))) / self.exp_id

    # -- S3 layout ---------------------------------------------------------------------------------
    @property
    def exp_uri(self) -> str:
        return f"{self.s3_root}/{self.exp_id}"

    def iter_uri(self, k: int) -> str:
        return f"{self.exp_uri}/iter_{int(k):02d}"

    def rollout_uri(self, k: int) -> str:
        return f"{self.iter_uri(k)}/rollout"

    def episode_uri(self, k: int, task: str, seed: int) -> str:
        return f"{self.rollout_uri(k)}/{task}/{int(seed)}"

    def s1_shard_uri(self, k: int) -> str:
        return f"{self.iter_uri(k)}/s1_training_shard"

    def s1_ckpt_uri(self, k: int) -> str:
        return f"{self.iter_uri(k)}/s1_ckpts"

    def eval_uri(self, k: int) -> str:
        return f"{self.iter_uri(k)}/eval/target"

    @property
    def s2_shard_uri(self) -> str:
        return f"{self.exp_uri}/s2_training_shard"

    @property
    def s2_ckpt_uri(self) -> str:
        return f"{self.exp_uri}/s2_ckpts"

    @property
    def state_uri(self) -> str:
        return f"{self.exp_uri}/state.json"

    @property
    def start_s1_ckpt(self) -> str:
        return str(self.raw["start"]["s1_ckpt"]).rstrip("/")

    @property
    def s2_ckpt(self) -> str:
        return str(self.raw["start"]["s2_ckpt"]).rstrip("/")

    def seeds_per_task(self, k: int) -> int:
        r = self.raw["rollout"]
        if int(k) == 0 and r.get("pilot_seeds_per_task"):
            return int(r["pilot_seeds_per_task"])
        return int(r["seeds_per_task"])



#: where experiments keep what they download and make (configs/default.yaml work_dir); LEGACY_WORK_ROOT holds the runs
#: from before 2026-11-25 (sub50_check_20261125, si_v1, ...), still read by exp_dir()
WORK_ROOT = "/mnt/ssd1/self_improve_runs"
LEGACY_WORK_ROOT = "~/selfimprove_work"


def exp_dir(exp: str) -> Path:
    """An experiment's local folder: under the work root, or the legacy one for runs made before the move."""
    new = Path(WORK_ROOT) / exp
    old = Path(os.path.expanduser(LEGACY_WORK_ROOT)) / exp
    return old if (old.exists() and not new.exists()) else new

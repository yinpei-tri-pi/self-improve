"""A HexaShell RoboCasa episode record, read for System1: per-step state, action, pictures and labels.

``<episode>/trajectory/`` (hexashell/benches/robocasa/sim.py ``Trajectory``): ``trajectory.json``
(index: chunks, labels = runs of [first step, subgoal, by], facts, success) and ``steps_NNNNN.npz``
(per step before it: ``raw16`` System1's raw state, ``action`` the 12-d robosuite action stepped,
``level`` the gripper command, the three cameras' JPEGs end to end in ``jpeg`` with ``offsets``, and
``sim_state`` MuJoCo's full state).

The S1 math here (lean state, settle-padded chunks, Estimated-Length grid, progress, prompt fields) is
RoboAnnotator's ``producers/build_system1_shards.py`` + ``preprocess_robocasa_to_tar.py`` +
``robo_annotator/system2/build_samples.py``, copied so the rollout shards are byte-compatible with the
base data openpi trains on (tests/test_s1_parity.py checks it against RoboAnnotator itself).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

HORIZON = 20
#: our camera names -> the shards' / openpi's
CAMS = {"left": "scene_left", "right": "scene_right", "wrist": "wrist"}
TORSO_IDX = 3
LEAN_HOLD_DIMS = (3, 10)  # lean 11-d: control_mode, gripper_close
RAW_HOLD_DIMS = (4, 11)   # raw 12-d (LeRobot order): control_mode, gripper_close
SIM_GRIP_IDX = 6
_LEN_GRID = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1300]
_LEN_LO = [(_LEN_GRID[i - 1] + _LEN_GRID[i]) // 2 if i > 0 else 0 for i in range(len(_LEN_GRID))]
PROGRESS_CLASSES = 10


# -- RoboAnnotator's math ----------------------------------------------------------------------------
def sim_to_lerobot(a) -> np.ndarray:
    """robosuite's action order [eef pos 0:3, eef rot 3:6, grip 6, base 7:10, torso 10, mode 11] -> the
    LeRobot / shard order [base 0:3, torso 3, mode 4, eef pos 5:8, eef rot 8:11, grip 11] (the inverse of
    HexaShell's lerobot_action_to_sim)."""
    a = np.asarray(a, dtype=np.float32)
    out = np.empty(a.shape[:-1] + (12,), dtype=np.float32)
    out[..., 0:4] = a[..., 7:11]
    out[..., 4] = a[..., 11]
    out[..., 5:8] = a[..., 0:3]
    out[..., 8:11] = a[..., 3:6]
    out[..., 11] = a[..., 6]
    return out


def lean_action(raw_action) -> np.ndarray:
    a = np.asarray(raw_action, dtype=np.float32)
    return np.concatenate([a[..., :TORSO_IDX], a[..., TORSO_IDX + 1:]], axis=-1)


def _quat_xyzw_to_rot6d(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=-1)
    c1 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=-1)
    return np.concatenate([c0, c1], axis=-1)


def yaw_from_quat_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lean_state(raw_state, base_xy_ref, base_yaw_ref: float) -> np.ndarray:
    s = np.asarray(raw_state, dtype=np.float32)
    eef_pos = s[..., 7:10]
    eef_6d = _quat_xyzw_to_rot6d(s[..., 10:14])
    grip = (s[..., 14] - s[..., 15])[..., None]
    base_xy = s[..., 0:2] - np.asarray(base_xy_ref, dtype=np.float32)
    yaw = np.arctan2(2.0 * (s[..., 6] * s[..., 5] + s[..., 3] * s[..., 4]),
                     1.0 - 2.0 * (s[..., 4] ** 2 + s[..., 5] ** 2)) - np.float32(base_yaw_ref)
    yaw_sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)
    return np.concatenate([eef_pos, eef_6d, grip, base_xy, yaw_sincos], axis=-1).astype(np.float32)


def padded_chunk(action, t: int, span_end: int, horizon: int = HORIZON, hold_dims=()) -> tuple[np.ndarray, np.ndarray]:
    """Real actions t..min(span_end, T-1), then the settle pad: velocity dims 0, hold dims at the last real
    value; the mask marks the real steps."""
    T, A = action.shape
    limit = min(span_end, T - 1)
    chunk = np.zeros((horizon, A), dtype=np.float32)
    mask = np.zeros((horizon,), dtype=np.bool_)
    last_real = None
    for i in range(horizon):
        src = t + i
        if src <= limit:
            chunk[i] = action[src]
            mask[i] = True
            last_real = action[src]
        elif last_real is not None:
            for d in hold_dims:
                chunk[i, d] = last_real[d]
    return chunk, mask


def halt_chunk(last_action, horizon: int = HORIZON, hold_dims=()) -> np.ndarray:
    out = np.zeros((horizon, len(last_action)), dtype=np.float32)
    for d in hold_dims:
        out[:, d] = last_action[d]
    return out


def estimated_length(nframes: int) -> int:
    n = int(nframes)
    chosen = _LEN_GRID[0]
    for g, lo in zip(_LEN_GRID, _LEN_LO):
        if n >= lo:
            chosen = g
        else:
            break
    return chosen


def progress_class(frame: int, s: int, e: int, k: int = PROGRESS_CLASSES) -> int:
    n = e - s + 1
    off = max(0, min(frame, e) - s)
    if n < k:
        return 1 if off < n / 2.0 else k
    return int(min(k - 1, off * k // n)) + 1


def system1_prompt(task_goal: str, subgoal: str, quality: str, est_length: int = 0, gripper: str = "Open") -> str:
    task = (task_goal or "").strip().rstrip(".").lower()
    sg = (subgoal or "").strip().rstrip(".").lower()
    zeros = "0 0 0 ... 0"
    return (f"Task: {task}; Current Subgoal: {sg}\n"
            f"Quality: {quality}; Estimated Length: {est_length}; Executed Step: 0\n"
            f"Initial State: {zeros}; Current State: {zeros}; Current Gripper: {gripper};\n"
            f"Action:")


def progress_targets(t: int, s: int, e: int, horizon: int = HORIZON) -> np.ndarray:
    """openpi's progress-as-action target for frame t of span [s, e] (robocasa_webdataset._build_sample,
    linear branch): step i = min(1, frac0 + i/span_len), 1 past the span end; in [0, 1]."""
    span = e - s
    frac0 = min(1.0, max(0.0, (t - s) / max(1, span)))
    steps = np.arange(horizon, dtype=np.float32)
    prog = np.minimum(1.0, frac0 + steps / span) if span > 0 else np.full(horizon, frac0, np.float32)
    past = np.arange(horizon) + t > e
    prog[past] = 1.0
    return prog.astype(np.float32)


# -- the record ----------------------------------------------------------------------------------------
@dataclass
class Span:
    """One labelled run of control steps: [start, end] inclusive, its subgoal and who moved."""

    index: int
    start: int
    end: int
    subgoal: str | None
    by: str

    @property
    def steps(self) -> int:
        return self.end - self.start + 1


@dataclass
class Trajectory:
    folder: Path
    index: dict
    raw16: np.ndarray            # [T, 16]
    action_sim: np.ndarray       # [T, 12] robosuite order
    level: np.ndarray            # [T]
    _jpeg: np.ndarray = field(repr=False)
    _offsets: np.ndarray = field(repr=False)
    _cams: list = field(repr=False)
    has_sim_state: bool = False

    @classmethod
    def load(cls, episode_or_trajectory: str | Path) -> "Trajectory":
        folder = Path(episode_or_trajectory)
        if (folder / "trajectory").is_dir():
            folder = folder / "trajectory"
        index = json.loads((folder / "trajectory.json").read_text())
        raw, act, lev, blobs, offs = [], [], [], [], []
        base = 0
        has_state = True
        for name in index.get("chunks", []):
            with np.load(folder / name) as z:
                raw.append(z["raw16"]); act.append(z["action"]); lev.append(z["level"])
                blobs.append(z["jpeg"])
                o = z["offsets"].astype(np.int64)
                offs.append(o[:-1] + base)
                base += int(o[-1])
                has_state &= "sim_state" in z.files
        cams = list(index.get("cams") or CAMS)
        if not raw:
            empty = np.zeros((0,), np.float32)
            return cls(folder, index, np.zeros((0, 16), np.float32), np.zeros((0, 12), np.float32), empty,
                       np.zeros((0,), np.uint8), np.zeros((1,), np.int64), cams, False)
        offsets = np.concatenate(offs + [np.array([base], np.int64)])
        return cls(folder, index, np.concatenate(raw).astype(np.float32), np.concatenate(act).astype(np.float32),
                   np.concatenate(lev).astype(np.float32), np.concatenate(blobs), offsets, cams, has_state)

    # -- basics ---------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.raw16.shape[0])

    @property
    def facts(self) -> dict:
        return self.index.get("facts") or {}

    @property
    def task_goal(self) -> str:
        return str(self.facts.get("instruction") or "")

    @property
    def success(self) -> bool:
        return bool(self.index.get("success"))

    @cached_property
    def action_lr(self) -> np.ndarray:
        """[T, 12] in the shards' (LeRobot) order."""
        return sim_to_lerobot(self.action_sim)

    @cached_property
    def lean_act(self) -> np.ndarray:
        return lean_action(self.action_lr)

    @cached_property
    def base_ref(self) -> tuple[np.ndarray, float]:
        """The episode's frame-0 base pose (the base data's ``base_pos_ref`` / ``base_yaw_ref``)."""
        return self.raw16[0, 0:3].copy(), yaw_from_quat_xyzw(self.raw16[0, 3:7])

    @cached_property
    def lean(self) -> np.ndarray:
        pos, yaw = self.base_ref
        return lean_state(self.raw16, pos[0:2], yaw)

    def jpeg(self, step: int, cam: str) -> bytes:
        """The JPEG of camera ``cam`` (left/right/wrist) before control step ``step``."""
        i = int(step) * len(self._cams) + self._cams.index(cam)
        return self._jpeg[self._offsets[i]:self._offsets[i + 1]].tobytes()

    def image(self, step: int, cam: str) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(self.jpeg(step, cam))).convert("RGB"))

    def gripper_flag(self, t: int) -> str:
        """'Close' when the previous step commanded the gripper closed (frame 0: its own action)."""
        pt = max(0, int(t) - 1)
        return "Close" if float(self.action_sim[pt, SIM_GRIP_IDX]) > 0 else "Open"

    def spans(self) -> list[Span]:
        """The labelled runs of steps, in order (runs without a subgoal label included, subgoal None)."""
        labels = [(int(a), b, str(c)) for a, b, c in self.index.get("labels") or []]
        n = len(self)
        out = []
        for i, (start, subgoal, by) in enumerate(labels):
            end = (labels[i + 1][0] if i + 1 < len(labels) else n) - 1
            start, end = max(0, start), min(n - 1, end)
            if end >= start:
                out.append(Span(len(out), start, end, subgoal, by))
        return out

    # -- System1 requests (the S1 server's score op) ----------------------------------------------------
    def s1_element(self, t: int, span: Span, *, est_length: int, quality: str = "Success",
                   with_targets: bool = True) -> dict:
        """The request System1 would see at step ``t`` of ``span`` (HexaShell's s1.element, at the training
        image size), plus the executed actions and progress targets the loss scores."""
        el = {}
        for cam, key in CAMS.items():
            el[f"observation/{key}"] = self.image(t, cam)
            el[f"observation/anchor_{key}"] = self.image(span.start, cam)
        pos, yaw = self.base_ref
        el["observation/state"] = self.raw16[t].astype(np.float32)
        el["observation/anchor_state"] = self.raw16[span.start].astype(np.float32)
        el["observation/base_pos_ref"] = pos.astype(np.float32)
        el["observation/base_yaw_ref"] = np.float32(yaw)
        el.update(prompt=str(span.subgoal or ""), task_goal=self.task_goal, quality=quality,
                  est_length=int(est_length), executed_step=int(t - span.start), gripper_flag=self.gripper_flag(t))
        if with_targets:
            chunk, _ = padded_chunk(self.action_lr, t, span.end, HORIZON, RAW_HOLD_DIMS)
            el["actions"] = chunk
            el["progress_action"] = (2.0 * progress_targets(t, span.start, span.end) - 1.0).astype(np.float32)
        return el

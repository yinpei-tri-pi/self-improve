"""A synthetic HexaShell RoboCasa episode record (result.json, events.jsonl, trajectory/) for tests."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMS = ("left", "right", "wrist")


def _jpeg(value: int, size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value % 255, np.uint8)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_episode(folder: Path, *, task="OpenDrawer", seed=123, success=True, turns=None, chunk=50,
                  with_sim_state=True) -> Path:
    """``turns``: [(subgoal, by, steps, judge_of_previous)], the judge of the first turn is task_begin."""
    turns = turns or [("reach to the drawer handle", "s1", 40, "task_begin"),
                      ("grasp the drawer handle", "s1", 20, "subgoal_complete"),
                      ("pull the drawer open", "s1", 30, "subgoal_complete"),
                      ("pull the drawer open again", "agent", 25, "subgoal_failed"),
                      ("retract the gripper", "s1", 15, "subgoal_complete")]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trajectory").mkdir(exist_ok=True)
    n = sum(t[2] for t in turns)
    rng = np.random.default_rng(seed)
    raw16 = rng.normal(size=(n, 16)).astype(np.float32)
    for sl in (slice(3, 7), slice(10, 14)):
        raw16[:, sl] /= np.linalg.norm(raw16[:, sl], axis=1, keepdims=True)
    action = rng.normal(size=(n, 12)).astype(np.float32) * 0.1
    action[:, 6] = np.where(np.arange(n) > 50, 1.0, -1.0)
    action[:, 11] = -1.0
    events, labels, step = [], [], 0
    for k, (sub, by, steps, judge) in enumerate(turns):
        events.append({"t": k, "kind": "judge", "turn": k + 1, "judge": judge, "thought": f"turn {k + 1}",
                       "of": turns[k - 1][0] if k else None, "of_by": turns[k - 1][1] if k else None,
                       "next": sub, "step": step})
        events.append({"t": k, "kind": "subgoal", "text": sub, "by": by, "step": step})
        if by == "s1":
            events.append({"t": k, "kind": "s1", "subgoal": sub, "stop": "progress", "steps": steps,
                           "progress": 0.97, "est_length": 50})
        labels.append([step, sub, by])
        step += steps
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    chunks = []
    for c0 in range(0, n, chunk):
        idx = range(c0, min(n, c0 + chunk))
        blobs = [_jpeg(i * 3 + j) for i in idx for j in range(3)]
        offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        extra = {"sim_state": rng.normal(size=(len(idx), 30))} if with_sim_state else {}
        name = f"steps_{c0:05d}.npz"
        np.savez(folder / "trajectory" / name, step=np.array(list(idx), np.int32), raw16=raw16[c0:c0 + len(idx)],
                 action=action[c0:c0 + len(idx)], level=action[c0:c0 + len(idx), 6],
                 jpeg=np.frombuffer(b"".join(blobs), np.uint8), offsets=offsets, cams=np.array(CAMS), **extra)
        chunks.append(name)
    index = {"size": 32, "cams": list(CAMS), "chunks": chunks, "labels": labels, "steps": n, "success": success,
             "task": task, "facts": {"task": task, "seed": seed, "instruction": "Open the drawer."}}
    (folder / "trajectory" / "trajectory.json").write_text(json.dumps(index))
    result = {"bench": "robocasa", "task": task, "seed": seed, "split": "pretrain", "success": success, "step": n,
              "end": "episode_over", "turns": len(turns), "errors": [], "tokens": {"prompt": 1000, "completion": 50},
              "robot_calls": {"s1": sum(t[1] == "s1" for t in turns)}, "seconds": 60.0}
    (folder / "result.json").write_text(json.dumps(result))
    return folder

"""The per-chunk surrogate loss of the rollout checkpoint on an episode's executed chunks.

After an episode, the node sends every scored chunk to its own S1 server (openpi-si's ScoringPolicy:
``{"__op__": "score", "elements": [...]}``): the request System1 would have seen at that step (current
and anchor pictures at 256, raw state, base reference, subgoal, task goal, est_length, gripper flag) plus
the 20 actions actually executed from there (settle-padded at the segment's end) and the progress targets.
The answer is the flow-matching loss under fixed noise draws: ``loss_real`` over the 11 real action dims
(the unary objective's surrogate for -log pi(a|s)) and per step.  Chunks of both movers are scored: on
the harness's steps the loss says how far the correction is from what System1 would do.

Written to ``chunk_scores.jsonl`` in the episode folder: one row per chunk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..data.episode import load_episode
from ..data.trajectory import Span, estimated_length


def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class S1Socket:
    """A connection to one openpi websocket policy server (openpi_client's wire format)."""

    def __init__(self, url: str, timeout_s: float = 600.0):
        self.url = url if url.startswith("ws") else f"ws://{url}"
        self.timeout_s = timeout_s
        self.ws = None
        self.metadata: dict = {}

    def connect(self, wait_s: float = 0.0) -> dict:
        import msgpack
        import websockets.sync.client

        deadline = time.time() + wait_s
        while True:
            try:
                self.ws = websockets.sync.client.connect(self.url, compression=None, max_size=None, open_timeout=10)
                self.metadata = msgpack.unpackb(self.ws.recv(timeout=self.timeout_s), object_hook=_unpack)
                return self.metadata
            except Exception:  # noqa: BLE001
                self.ws = None
                if time.time() >= deadline:
                    raise
                time.sleep(5)

    def call(self, obj: dict) -> dict:
        import msgpack

        if self.ws is None:
            self.connect()
        self.ws.send(msgpack.packb(obj, default=_pack))
        reply = self.ws.recv(timeout=self.timeout_s)
        if isinstance(reply, str):
            raise RuntimeError(f"S1 server error: {reply.strip().splitlines()[-1][:500]}")
        return msgpack.unpackb(reply, object_hook=_unpack)

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        self.ws = None


def chunk_starts(start: int, end: int, stride: int) -> list[int]:
    return list(range(int(start), int(end) + 1, max(1, int(stride))))


def score_episode(folder: str | Path, url: str, *, stride: int = 16, num_draws: int = 4, batch: int = 16,
                  seed: int = 0, sock: S1Socket | None = None) -> dict:
    """Score every chunk of the episode at ``folder``; write chunk_scores.jsonl; return a summary."""
    folder = Path(folder)
    rec = load_episode(folder)
    if rec.traj is None or not len(rec.traj):
        return {"chunks": 0}
    own = sock is None
    sock = sock or S1Socket(url)
    rows, todo = [], []
    t0 = time.time()
    for seg in rec.segments:
        span = Span(seg.index, seg.start, seg.end, seg.subgoal, seg.by)
        est = int(seg.s1_est_length or estimated_length(seg.steps))
        for t in chunk_starts(seg.start, seg.end, stride):
            todo.append((seg, t, rec.traj.s1_element(t, span, est_length=est)))
    try:
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            reply = sock.call({"__op__": "score", "elements": [el for _, _, el in part], "num_draws": int(num_draws),
                               "seed": int(seed)})
            for (seg, t, _), sc in zip(part, reply["scores"]):
                row = {"segment": seg.index, "turn": seg.turn, "by": seg.by, "subgoal": seg.subgoal, "start": seg.start,
                       "end": seg.end, "t": int(t), "executed": int(t - seg.start),
                       "loss_real": round(float(sc["loss_real"]), 6), "loss_real_std": round(float(sc["loss_real_std"]), 6),
                       "loss_steps_real": [round(float(x), 5) for x in np.asarray(sc["loss_steps_real"]).reshape(-1)]}
                if "loss_steps_progress" in sc:
                    row["loss_progress"] = round(float(np.mean(sc["loss_steps_progress"])), 6)
                rows.append(row)
    finally:
        if own:
            sock.close()
    (folder / "chunk_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = {"chunks": len(rows), "seconds": round(time.time() - t0, 1)}
    for by in ("s1", "agent"):
        vals = [r["loss_real"] for r in rows if r["by"] == by]
        if vals:
            out[f"loss_real_{by}"] = float(np.mean(vals))
    return out

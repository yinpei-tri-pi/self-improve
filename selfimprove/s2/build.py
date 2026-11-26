"""Rollout turns -> System2 (Qwen3-VL) training rows, in sys2_train_eval's format (once, after the S1 rounds).

Every planner turn of a HexaShell episode is one S2 execution turn (sys1_sys2 ``sim/sys2_client.py``):

- system: ``SYS_EXEC`` (verbatim from sys2_client.py)
- user:   ``user_exec_turn(goal, plan, task_status, gripper_status)`` with ``<video>`` = the step just
          executed, the three cameras tiled 256 x 768 at 4 fps, x264 crf 26 (from the trajectory's own
          frames, every 5th 20 Hz step); turn 1 is ``user_exec_first`` with the opening scene as ``<image>``
- assistant: ``<thought>`` ``<judge>`` ``<plan_update>`` ``<estimated_step>`` ``<subgoal>`` ``<subgoal_detail>``

Astra's turns give thought, judge, subgoal and the step's length (the est_length it chose for System1, or
the steps it took); the two-level milestone plan, ``plan_update`` and ``subgoal_detail`` are not in the
rollouts: the cheap-LLM annotation (``annotate``, a later stage) fills them.  Until then the rows are written
as ``raw`` (``needs: [plan, plan_update, subgoal_detail]``) next to their clips:

    <exp>/s2_training_shard/raw/iter_NN/<task>/<seed>/turns.jsonl + clips/turnNN.mp4

and the annotated rows become webdataset shards (``<key>.json`` {"messages", "meta"} + ``<key>.videobin``)
by RoboAnnotator's producers/build_system2_shards.py layout.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .. import tasks as T
from ..data.episode import load_episode

CLIP_FPS = 4.0
SIM_HZ = 20.0
CLIP_CRF = 26
TILE = ("left", "right", "wrist")  # left-shoulder, right-shoulder, wrist (sys2's CAM_ORDER)

SYS_EXEC_SOURCE = "sys1_sys2/sim/sys2_client.py:SYS_EXEC"


def gripper_status(traj, end: int, window: int = 5) -> str:
    """open / close / unsure at the end of a clip: the commanded gripper, unsure while it just flipped."""
    a = traj.action_sim[max(0, end - window):end + 1, 6]
    if a.size == 0:
        return "open"
    signs = np.sign(a)
    if len(set(signs.tolist())) > 1:
        return "unsure"
    return "close" if signs[-1] > 0 else "open"


def write_clip(traj, start: int, end: int, path: Path) -> int:
    """The tiled 3-view clip of steps [start, end] at 4 fps; returns the frame count (at least 2)."""
    stride = int(round(SIM_HZ / CLIP_FPS))
    steps = list(range(start, end + 1, stride))
    if steps[-1] != end:
        steps.append(end)
    if len(steps) < 2:
        steps = [start, end]
    frames = [np.concatenate([traj.image(t, cam) for cam in TILE], axis=1) for t in steps]
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
           str(CLIP_FPS), "-i", "-", "-an", "-vcodec", "libx264", "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f, np.uint8).tobytes() for f in frames), check=True)
    return len(frames)


def episode_turns(folder: str | Path, out_dir: Path, *, iteration: int) -> list[dict]:
    """One raw S2 row per turn (and its clip); see the module docstring."""
    rec = load_episode(folder)
    if rec.result.get("task") in T.held_out_tasks():
        raise ValueError(f"{rec.result.get('task')} is a held-out composite_unseen task: not for S2 training data")
    traj = rec.traj
    if traj is None or not len(traj):
        return []
    judges = [e for e in rec.events if e.get("kind") == "judge"]
    s1_runs = [e for e in rec.events if e.get("kind") == "s1"]
    rows = []
    for k, j in enumerate(judges):
        step = int(j.get("step") or 0)
        prev = judges[k - 1] if k else None
        clip = None
        if prev is not None and step > int(prev.get("step") or 0):
            clip_path = out_dir / "clips" / f"turn{k:02d}.mp4"
            n = write_clip(traj, int(prev["step"]), min(step, len(traj) - 1), clip_path)
            clip = {"path": clip_path.name, "n_frames": n, "fps": CLIP_FPS, "span": [int(prev["step"]), step]}
        nxt = judges[k + 1] if k + 1 < len(judges) else None
        seg_len = (int(nxt["step"]) if nxt else len(traj)) - step
        est = next((int(r.get("est_length") or 0) for r in s1_runs if r.get("subgoal") == j.get("next")), 0) or seg_len
        rows.append({
            "episode_id": f"SelfImprove/{rec.task}/seed_{rec.seed}", "iteration": iteration, "turn": k + 1,
            "mode": "exec_first" if k == 0 else "exec_turn", "goal": traj.task_goal,
            "task_status": "ongoing",  # HexaShell's turns all come before the environment's success
            "gripper_status": gripper_status(traj, max(0, step - 1)) if k else "open",
            "clip": clip, "scene_step": step if k == 0 else None,
            "assistant": {"thought": j.get("thought") or "", "judge": j.get("judge"), "plan_update": None,
                          "estimated_step": int(est), "subgoal": j.get("next") or "", "subgoal_detail": None},
            "executed_by": None if nxt is None and not seg_len else next(
                (e.get("by") for e in rec.events if e.get("kind") == "subgoal" and int(e.get("step") or -1) == step), None),
            "episode_success": rec.success, "needs": ["plan", "plan_update", "subgoal_detail"],
        })
    if rec.success and judges:
        # the closing turn S2 is trained to give (the episode itself ends the moment the task is done): the
        # last step's clip, 'Current task status: finished', judge task_finish (thought: to annotate)
        last_step = int(judges[-1].get("step") or 0)
        end = len(traj) - 1
        if end > last_step:
            clip_path = out_dir / "clips" / f"turn{len(judges):02d}.mp4"
            n = write_clip(traj, last_step, end, clip_path)
            rows.append({"episode_id": f"SelfImprove/{rec.task}/seed_{rec.seed}", "iteration": iteration,
                         "turn": len(judges) + 1, "mode": "exec_turn", "goal": traj.task_goal, "task_status": "finished",
                         "gripper_status": gripper_status(traj, end), "scene_step": None,
                         "clip": {"path": clip_path.name, "n_frames": n, "fps": CLIP_FPS, "span": [last_step, end]},
                         "assistant": {"thought": None, "judge": "task_finish", "plan_update": None},
                         "executed_by": None, "episode_success": True, "synthesized": True,
                         "needs": ["plan", "plan_update", "thought"]})
    if rows and rows[0]["scene_step"] is not None:
        from PIL import Image

        img = np.concatenate([traj.image(rows[0]["scene_step"], cam) for cam in TILE], axis=1)
        (out_dir / "clips").mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_dir / "clips" / "scene00.jpg", quality=95)
    (out_dir / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def annotate(rows: list[dict], *, model: str | None = None) -> list[dict]:
    """The later stage: a cheap LLM writes each episode's milestone plan, every turn's plan_update and
    subgoal_detail from the goal, Astra's thoughts and the clips (RoboAnnotator's refine_plan_exec /
    plan_exec_claude producers do this for the dataset episodes)."""
    raise NotImplementedError("S2 language annotation is the next stage to build (see the module docstring)")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episode", help="a local episode folder (result.json, events.jsonl, trajectory/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    args = ap.parse_args(argv)
    rows = episode_turns(args.episode, Path(args.out), iteration=args.iteration)
    print(f"{len(rows)} turns -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

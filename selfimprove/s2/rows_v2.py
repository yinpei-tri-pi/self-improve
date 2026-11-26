"""S1 + harness rollouts -> rows for the new single-mode System2 (the design agreed on 2026-11-24).

Every planner turn of an episode (failed episodes too) is one row: System2 sees the task, the steps so far grouped by
milestone (each with who executed it and how it went), the last step's video and the gripper status (the first turn:
the opening scene), and answers with its reasoning, the judgement of the last step, the milestone, the decision (who
executes the next step: System1 or the harness, i.e. Astra) and the next step in System1's words. The targets come
straight from Astra's turn: ``planner_reasoning`` (the thought), ``judge``, ``milestone`` (+ ``milestones`` on the
first turn), the next ``subgoal`` and who executed it. The decision is Astra's actual choice (a step System1 tried and
fell short on stays ``s1``; the recovery turn after it is ``harness``; a turn whose code also called System1 is ``s1``). A successful episode gets one closing row: the
last step's video and ``task_finish``.

Rows: ``{"messages": [system, user, assistant], "meta": {...}}`` with ``<image>`` / ``<video>`` placeholders and the
media written next to them (``clips/turnNN.mp4``: the 3 cameras tiled 256 x 768 at 4 fps, at most 32 frames;
``clips/scene00.jpg``), as the old System2 shards carry them. ``episode_rows`` returns every row; a row whose video has
under MIN_FRAMES frames is marked ``meta.skipped`` and left out of ``rows.jsonl`` (``trained()``), its step still in the
later rows' history.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .. import tasks as T
from ..data.episode import base_step, load_episode

TILE = ("left", "right", "wrist")
CLIP_FPS, SIM_HZ, CLIP_CRF, CLIP_MAX = 4.0, 20.0, 26, 32
MOVER = {"s1": "the model", "agent": "the expert"}
#: the decision as the target writes it (fixed words, parsed at inference): who executes the next step
DECISION_WORD = {"s1": "model", "harness": "expert"}
#: the privileged status line (from the benchmark, as the old System2's ``Current task status``): the first turn, every
#: other turn, and the closing rows of a successful / a failed episode (success often fires before it shows: water that
#: has not run yet, a lid that has not visibly opened)
STATUS = {"first": "begin", "turn": "ongoing", "success": "success", "failed": "failed"}

SYSTEM = (
    "You are the high-level planner of a one-arm mobile robot (a Franka Panda arm with a parallel gripper on a wheeled "
    "base) in a home kitchen, seen from three cameras tiled side by side: left shoulder, right shoulder, wrist. Two "
    "controllers can execute a step: the model, a learned policy that is fast and cheap but fails on some steps, and "
    "the expert, which is reliable but slow and costly. Each turn you see the task, its milestones (each done, now or "
    "to do, with the recent steps of the current one and how each went: done, partial or failed), the video of the "
    "last step, the gripper status and the task status (begin, ongoing, success or failed). Judge the last step, then give the next step in the model's words "
    "and who should execute it: the model whenever it can do the step, the expert only when the model is likely to "
    "fail. Answer with <thought>, <judge>, <milestone>, <decision> (model or expert) and <subgoal>; when the task "
    "status is success or failed, only <thought> and <judge> (task_finish or task_failed).")


def natural(text: str) -> str:
    """Astra's planner_reasoning in S2's words: System1 is 'the model', the harness (Astra's own calls) 'the expert'."""
    t = re.sub(r"\b([Aa]) harness\b", lambda m: ("An" if m.group(1) == "A" else "an") + " expert", str(text or ""))
    t = re.sub(r"\bHarness\b", "Expert", t)
    t = re.sub(r"\bharness\b", "expert", t)
    t = re.sub(r"\b(?:System1|S1)\b", "the model", t)
    # a sentence (or the text) that now starts with "the model": capitalised
    return re.sub(r"(^|[.!?]\s+|\n)the model", lambda m: m.group(1) + "The model", t)


def gripper_status(traj, end: int, window: int = 5) -> str:
    a = traj.action_sim[max(0, end - window):end + 1, 6]
    if a.size == 0:
        return "open"
    signs = np.sign(a)
    if len(set(signs.tolist())) > 1:
        return "unsure"
    return "close" if signs[-1] > 0 else "open"


def clip_steps(start: int, end: int) -> list[int]:
    """Control steps of a clip: every 5th (4 fps), at most CLIP_MAX evenly spread, both ends included; at least two
    frames (a one-step step repeats its frame: Qwen-VL's video input wants two)."""
    n = min(CLIP_MAX, max(2, int(round((end - start) / (SIM_HZ / CLIP_FPS))) + 1))
    steps = sorted({int(round(x)) for x in np.linspace(start, end, n)})
    return steps * 2 if len(steps) == 1 else steps


def write_clip(traj, start: int, end: int, path: Path) -> int:
    """The clip of steps start..end; an existing file of that name is kept (clips are named for their steps)."""
    if path.exists():
        return len(clip_steps(start, end))
    frames = [np.concatenate([traj.image(t, cam) for cam in TILE], axis=1) for t in clip_steps(start, end)]
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
           str(CLIP_FPS), "-i", "-", "-an", "-vcodec", "libx264", "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f, np.uint8).tobytes() for f in frames), check=True)
    return len(frames)


#: a row whose video has fewer frames than this is not trained (too little to judge); its step stays in the history
MIN_FRAMES = 4


def trained(rows: list[dict]) -> list[dict]:
    """The rows that go into the S2 training data (``meta.skipped`` unset)."""
    return [r for r in rows if not r["meta"].get("skipped")]


RECENT = 5  # the current milestone's last steps shown
MERGE_SHORT = 50  # as System1's harness fragments (selfimprove/data/s1_select.py MERGE_SHORT)
_MID = re.compile(r"^\s*(M\d+)\s*[:.)-]\s*(.*)$")


def _mid(label: str) -> str:
    m = _MID.match(str(label or ""))
    return m.group(1) if m else str(label or "").strip()


def parse_milestones(text: str) -> list[tuple[str, str]]:
    """'M1: ...' lines -> [(id, full line)]."""
    out = []
    for line in str(text or "").splitlines():
        m = _MID.match(line)
        if m:
            out.append((m.group(1), f"{m.group(1)}: {m.group(2).strip()}"))
    return out


def checklist_text(done: list[dict], current: str, milestones: list[tuple[str, str]]) -> str:
    """The milestones, each done / now / to do, and the current one's recent steps (the design of 2026-11-25).

    ``done``: the steps before the last one ({milestone, subgoal, outcome}), in order; ``current``: the last step's
    milestone. Milestones before the current one are done, those after it to do. Going back (M3 -> M2) makes the
    earlier one "now, again", marks the return in its step list ("(back from M3)") and empties the later ones, which
    become "to do, tried before". The current milestone shows its last RECENT steps (the attempts at one step, X /
    "continue to X" / "X again", are one line with how the latest went)."""
    order = [mid for mid, _ in milestones]
    text = dict(milestones)
    for d in done + [{"milestone": current}]:  # labels missing from the list: appended in the order they appear
        mid = _mid(d["milestone"])
        if mid not in text:
            order.append(mid)
            text[mid] = str(d["milestone"]).strip() or "(no milestone)"
    idx = {m: i for i, m in enumerate(order)}
    steps = {m: [] for m in order}
    again, tried = set(), set()
    prev = None

    def move(to: str) -> None:
        if prev is not None and to != prev and idx.get(to, 0) < idx.get(prev, 0):  # back to an earlier milestone
            for m in order:
                if idx[m] > idx[to] and steps[m]:
                    tried.add(m)
                    steps[m] = []
            steps[to].append({"mark": f"(back from {prev})"})
            again.add(to)

    for d in done:
        mid = _mid(d["milestone"])
        move(mid)
        key = base_step(d["subgoal"]) or d["subgoal"]
        lst = steps[mid]
        if lst and lst[-1].get("key") == key:
            lst[-1]["outcome"] = d["outcome"]
        else:
            words = d["subgoal"] if base_step(d["subgoal"]) == " ".join(str(d["subgoal"]).split()).lower() else key
            lst.append({"key": key, "text": words, "outcome": d["outcome"]})
        prev = mid
    cur = _mid(current)
    move(cur)
    lines = []
    for m in order:
        i, c = idx[m], idx.get(cur, 0)
        if i < c:
            lines.append(f"  {text[m]} (done)")
        elif i > c:
            lines.append(f"  {text[m]} (to do" + (", tried before)" if m in tried else ")"))
        else:
            lines.append(f"  {text[m]} (now" + (", again)" if m in again or m in tried else ")"))
            shown = steps[m][-RECENT:]
            for st in shown:
                lines.append(f"    {st['mark']}" if "mark" in st else f"    - {st['text']} ({st['outcome']})")
    return "\n".join(lines) if lines else "(none yet)"


def merge_fragments(turns: list[dict], lean_act) -> list[dict]:
    """Consecutive short harness turns joined as System1's harness fragments are (rule A, 2026-11-25): each under
    MERGE_SHORT steps, back to back, judged done or partway, all navigation or all manipulation, until the merged step
    reaches MERGE_SHORT steps, never across a milestone. The merged step keeps the first turn's row fields (its judge, its reasoning) with the
    last turn's words and verdict: one System2 row, decision expert."""
    from ..data.s1_select import is_navigation, primitive

    ok = ("subgoal_complete", "subgoal_incomplete", "task_done")
    nav = lambda t: is_navigation(primitive(t["subgoal"]), lean_act[t["start"]:t["end"] + 1])  # noqa: E731
    # one step first, as System1's harness spans: X then "continue to X" while partway (its turns stay rows of their own)
    units, i = [], 0
    while i < len(turns):
        u = [turns[i]]
        while (i + len(u) < len(turns) and u[-1]["by"] != "s1" and turns[i + len(u)]["by"] != "s1"
               and u[-1]["judge_of_this"] == "subgoal_incomplete"
               and base_step(turns[i + len(u)]["subgoal"]) == base_step(u[0]["subgoal"])):
            u.append(turns[i + len(u)])
        units.append(u)
        i += len(u)
    out, i = [], 0
    turns = [{**u[0], "end": u[-1]["end"], "judge_of_this": u[-1]["judge_of_this"], "unit": u} for u in units]
    while i < len(turns):
        t = turns[i]
        steps = t["end"] - t["start"] + 1
        if t["by"] == "s1" or steps >= MERGE_SHORT or t["judge_of_this"] not in ok:
            out.extend(t["unit"])
            i += 1
            continue
        group, total, j = [t], steps, i + 1
        while total < MERGE_SHORT and j < len(turns):
            u = turns[j]
            us = u["end"] - u["start"] + 1
            if (u["by"] == "s1" or us >= MERGE_SHORT or u["judge_of_this"] not in ok or u["start"] != group[-1]["end"] + 1
                    or nav(u) != nav(t) or _mid(u["milestone"]) != _mid(t["milestone"])):
                break
            group.append(u)
            total += us
            j += 1
        if len(group) == 1:
            out.extend(t["unit"])
            i += 1
            continue
        first, last = group[0]["unit"][0], group[-1]["unit"][-1]
        out.append({**first, "end": last["end"], "subgoal": group[-1]["subgoal"], "judge_of_this": last["judge_of_this"],
                    "merged": [x["turn"] for g in group for x in g["unit"]]})
        i = j
    return [{k: v for k, v in t.items() if k != "unit"} for t in out]


def merge_as_s1(turns: list[dict], rec) -> list[dict] | None:
    """A successful episode's turns joined exactly as System1's merged harness spans are (s1_select: kind
    harness_merged), so that System2's merged subgoal is the one System1 learns - and nothing else joined."""
    from ..data import s1_select

    groups = [sp for sp in s1_select.select(rec) if sp.kind == "harness_merged" and sp.kept]
    if not groups:
        return turns  # System1 merged nothing here (its fragments dropped, or none): neither does System2
    by_turn = {t["turn"]: t for t in turns}
    out, used = [], set()
    for t in turns:
        if t["turn"] in used:
            continue
        g = next((sp for sp in groups if sp.turns[0] == t["turn"]), None)
        members = [by_turn[n] for n in (g.turns if g else []) if n in by_turn]
        if g is None or len(members) < 2 or any(m["by"] == "s1" for m in members):
            out.append(t)
            continue
        used.update(m["turn"] for m in members)
        out.append({**t, "end": members[-1]["end"], "subgoal": g.subgoal, "judge_of_this": members[-1]["judge_of_this"],
                    "merged": [m["turn"] for m in members]})
    return out


OUTCOME = {"subgoal_complete": "done", "subgoal_incomplete": "partial", "subgoal_failed": "failed",
           "task_done": "done", None: "?"}


def latest_milestones(turns: list[dict]) -> list[tuple[str, str]]:
    """The milestone list as the latest of these turns gave it (the first turn's, or a later change)."""
    for t in reversed(turns):
        got = parse_milestones(t.get("milestones"))
        if got:
            return got
    return []


def user_first(goal: str) -> str:
    """The first row's user text (the opening scene as one tiled picture)."""
    return (f"The task is: {goal}\n\nNo step has run yet. Here is the scene:\n<image>\n"
            f"Current task status: {STATUS['first']}.\n\n"
            "List the milestones, then give the first step and who should execute it.")


def user_turn(goal: str, plan: str, last: str, mover: str, gripper: str, status: str) -> str:
    """Every other row's user text: the milestone view, the last step and its video, the gripper and task status."""
    return (f"The task is: {goal}\n\nMilestones:\n{plan}\n\nThe last step was '{last}', "
            f"executed by {MOVER.get(mover, mover)}. Here is its video:\n<video>\n"
            f"Current gripper: {gripper}.\nCurrent task status: {status}.\n\n"
            "Judge the last step, then give the next step and who should execute it.")


def target_turn(thought: str, judge: str, milestone: str, decision: str, subgoal: str, milestones: str = "") -> str:
    out = [f"<thought>{natural(thought)}</thought>", f"<judge>{judge}</judge>"]
    if milestones:
        out.append(f"<milestones>\n{milestones}\n</milestones>")
    out += [f"<milestone>{milestone}</milestone>", f"<decision>{DECISION_WORD[decision]}</decision>",
            f"<subgoal>{subgoal}</subgoal>"]
    return "\n".join(out)


def target_closing(thought: str, judge: str) -> str:
    return f"<thought>{thought}</thought>\n<judge>{judge}</judge>"


def episode_rows(folder: str | Path, out_dir: Path, *, make_media: bool = True) -> list[dict]:
    rec = load_episode(folder)
    if rec.result.get("task") in T.held_out_tasks():
        raise ValueError(f"{rec.result.get('task')} is a held-out composite_unseen task: not for S2 training data")
    traj = rec.traj
    planner = {}
    for e in rec.events:  # the planner fields ride on the judge events (or 'planner' after a turn that moved nothing)
        if e.get("kind") in ("judge", "planner") and int(e.get("turn") or 0):
            planner[int(e["turn"])] = e
    # one entry per turn that moved the robot: its span, who moved first, and how the next turn judged it
    turns: list[dict] = []
    for sg in rec.segments:
        if turns and turns[-1]["turn"] == sg.turn:
            # one turn, two movers (a few harness motions, then robot.s1()): System1 executed the step
            turns[-1]["end"] = sg.end
            if sg.by == "s1":
                turns[-1]["by"] = "s1"
            continue
        p = planner.get(sg.turn, {})
        turns.append({"turn": sg.turn, "start": sg.start, "end": sg.end, "subgoal": sg.subgoal, "by": sg.by,
                      "judge_of_this": sg.judge, "milestone": str(p.get("milestone") or ""),
                      "milestones": str(p.get("milestones") or ""), "reasoning": str(p.get("planner_reasoning") or ""),
                      "judge_given": p.get("judge")})
    # short harness turns joined as System1's fragments are: its own groups in a successful episode, the same rule otherwise
    merged = merge_as_s1(turns, rec) if rec.success else None
    turns = merged if merged is not None else merge_fragments(turns, traj.lean_act)
    rows = []
    goal = traj.task_goal
    for k, t in enumerate(turns):
        prev = turns[k - 1] if k else None
        meta = {"episode_id": f"SelfImprove/{rec.task}/seed_{rec.seed}", "task": rec.task, "seed": rec.seed,
                "turn": t["turn"], "step": t["start"], "episode_success": rec.success, "kind": "first" if k == 0 else "turn",
                "task_status": STATUS["first"] if k == 0 else STATUS["turn"], "merged_turns": t.get("merged"),
                "step_span": [t["start"], t["end"]],  # the step this row plans (its turns' steps, merged or not)
                "decision": "s1" if t["by"] == "s1" else "harness"}
        if prev is None:
            media = f"scene{k:02d}.jpg"
            if make_media:
                from PIL import Image

                (out_dir / "clips").mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.concatenate([traj.image(t["start"], c) for c in TILE], axis=1)).save(
                    out_dir / "clips" / media, quality=95)
            user = user_first(goal)
            meta.update(image=f"clips/{media}")
        else:
            media = f"clip_{prev['start']:05d}_{prev['end']:05d}.mp4"
            n = write_clip(traj, prev["start"], prev["end"], out_dir / "clips" / media) if make_media else len(
                clip_steps(prev["start"], prev["end"]))
            done = [{"milestone": x["milestone"], "subgoal": x["subgoal"], "by": x["by"],
                     "outcome": OUTCOME.get(x["judge_of_this"], x["judge_of_this"])} for x in turns[:k - 1]]
            plan = checklist_text(done, prev["milestone"], latest_milestones(turns[:k]))
            user = user_turn(goal, plan, prev["subgoal"], prev["by"], gripper_status(traj, prev["end"]), STATUS["turn"])
            meta.update(clip=f"clips/{media}", clip_frames=n, clip_fps=CLIP_FPS, clip_span=[prev["start"], prev["end"]])
        judge = "task_begin" if prev is None else (t["judge_given"] or prev["judge_of_this"] or "subgoal_complete")
        out = target_turn(t["reasoning"], judge, t["milestone"], meta["decision"], t["subgoal"],
                          t["milestones"] if prev is None else "")
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                                  {"role": "assistant", "content": out}], "meta": meta,
                     "labels": {"judge": judge, "decision": meta["decision"], "milestone": t["milestone"],
                                "subgoal": t["subgoal"], "thought": natural(t["reasoning"])}})
    if turns:  # the closing row: the last step's video, the task status -> task_finish (success) / task_failed
        last = turns[-1]
        status = STATUS["success"] if rec.success else STATUS["failed"]
        media = f"clip_{last['start']:05d}_{last['end']:05d}.mp4"
        n = write_clip(traj, last["start"], last["end"], out_dir / "clips" / media) if make_media else len(
            clip_steps(last["start"], last["end"]))
        done = [{"milestone": x["milestone"], "subgoal": x["subgoal"], "by": x["by"],
                 "outcome": OUTCOME.get(x["judge_of_this"], x["judge_of_this"])} for x in turns[:-1]]
        plan = checklist_text(done, last["milestone"], latest_milestones(turns))
        user = user_turn(goal, plan, last["subgoal"], last["by"], gripper_status(traj, last["end"]), status)
        judge = "task_finish" if rec.success else "task_failed"
        thought = ("The task is complete." if rec.success else
                   "The episode has ended without the task done.")
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                                  {"role": "assistant", "content": target_closing(thought, judge)}],
                     "meta": {"episode_id": f"SelfImprove/{rec.task}/seed_{rec.seed}", "task": rec.task, "seed": rec.seed,
                              "turn": max(sg.turn for sg in rec.segments) + 1, "step": last["end"], "episode_success": bool(rec.success),
                              "kind": "closing", "task_status": status, "decision": None, "clip": f"clips/{media}",
                              "clip_frames": n, "clip_fps": CLIP_FPS, "clip_span": [last["start"], last["end"]],
                              "synthesized": True},
                     "labels": {"judge": judge, "decision": None, "milestone": None, "subgoal": None, "thought": ""}})
    for r in rows:  # too short a video to judge: not trained (the site still shows it)
        n = r["meta"].get("clip_frames")
        if n is not None and n < MIN_FRAMES:
            r["meta"]["skipped"] = f"video of {n} frames, under {MIN_FRAMES}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trained(rows)))
    return rows

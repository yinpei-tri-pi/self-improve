"""A HexaShell + System1 episode as the PRM sees it: one segment per executed step (a turn's motions).

Every reply of the model is one planner turn (``events.jsonl``): a ``judge`` event (turn k: its judgement
``judge`` of the previous turn's step ``of`` / ``of_by``, the control ``step`` it starts at, the subgoal
``next`` it issues), a ``subgoal`` event (``text``, ``by``: ``s1`` or ``agent``) and, for System1, an
``s1`` event per call (stop reason, steps, progress, the est_length used).  Turn k's step runs from its
``step`` to turn k+1's; turn k+1's ``judge`` says how it went.  Within a turn the trajectory's labels tell
which control steps System1 made and which the harness made; a segment is one such run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .trajectory import Span, Trajectory


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def base_step(subgoal: str | None) -> str:
    s = " ".join(str(subgoal or "").split()).strip().rstrip(".").lower()
    if s.startswith("continue to "):
        s = s[len("continue to "):]
    if s.endswith(" again"):
        s = s[: -len(" again")]
    return s


@dataclass
class Segment:
    episode: str                 # "<task>/<seed>"
    turn: int                    # the turn whose code made these steps
    index: int                   # position in the episode
    start: int                   # control steps [start, end], inclusive
    end: int
    by: str                      # s1 | agent
    subgoal: str
    judge: str | None            # the next turn's judgement of this turn's step (None: no next turn)
    judge_thought: str = ""      # the next turn's thought (Astra's reasoning about how it went)
    next: str = ""               # moved_on | continued | retried | taken_over | task_done | ended
    episode_success: bool = False
    task: str = ""
    seed: int = 0
    task_goal: str = ""
    # System1 calls in this turn (est_length used, stop reason, steps, final progress)
    s1_est_length: int | None = None
    s1_stop: str | None = None
    s1_progress: float | None = None
    s1_calls: int = 0
    # filled by the PRM / filter
    score: str | None = None
    advantage: float | None = None
    use: str | None = None       # positive | negative | None (dropped)
    reason: str = ""
    surrogate: dict = field(default_factory=dict)  # chunk scores over this segment (mean, n)

    @property
    def steps(self) -> int:
        return self.end - self.start + 1

    def span(self, index: int = 0) -> Span:
        return Span(index, self.start, self.end, self.subgoal, self.by)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeRecord:
    folder: Path
    result: dict
    events: list[dict]
    traj: Trajectory | None
    segments: list[Segment]

    @property
    def task(self) -> str:
        return str(self.result.get("task") or "")

    @property
    def seed(self) -> int:
        return int(self.result.get("seed") or 0)

    @property
    def success(self) -> bool:
        return bool(self.result.get("success"))

    @property
    def errors(self) -> list:
        return list(self.result.get("errors") or [])

    @property
    def end(self) -> str:
        return str(self.result.get("end") or "")


def _turns(events: list[dict]) -> list[dict]:
    """One dict per turn: its judge event, subgoal event and the s1 events that followed it."""
    turns: list[dict] = []
    for e in events:
        kind = e.get("kind")
        if kind == "judge":
            turns.append({"judge": e, "subgoal": None, "s1": []})
        elif kind == "subgoal" and turns and turns[-1]["subgoal"] is None:
            turns[-1]["subgoal"] = e
        elif kind == "s1" and turns:
            turns[-1]["s1"].append(e)
    return turns


def _next_kind(seg_base: str, seg_by: str, nxt: dict | None, episode_success: bool, last: bool) -> str:
    if last:
        return "task_done" if episode_success else "ended"
    if nxt is None:
        return "ended"
    text = str((nxt.get("subgoal") or {}).get("text") or nxt["judge"].get("next") or "")
    by = str((nxt.get("subgoal") or {}).get("by") or "")
    if base_step(text) != seg_base:
        return "moved_on"
    if text.strip().lower().startswith("continue to "):
        return "continued"
    if by and by != seg_by:
        return "taken_over"
    return "retried"


def load_episode(folder: str | Path, *, with_trajectory: bool = True) -> EpisodeRecord:
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    events = _jsonl(folder / "events.jsonl")
    traj = Trajectory.load(folder) if with_trajectory and (folder / "trajectory" / "trajectory.json").exists() else None
    rec = EpisodeRecord(folder, result, events, traj, [])
    rec.segments = segments_of(rec)
    return rec


def segments_of(rec: EpisodeRecord) -> list[Segment]:
    turns = _turns(rec.events)
    n_steps = len(rec.traj) if rec.traj is not None else int(rec.result.get("step") or 0)
    label_runs = rec.traj.spans() if rec.traj is not None else []
    success = rec.success
    key = f"{rec.task}/{rec.seed}"
    goal = rec.traj.task_goal if rec.traj is not None else str(rec.result.get("instruction") or "")
    out: list[Segment] = []
    for k, turn in enumerate(turns):
        j = turn["judge"]
        start = int(j.get("step") or 0)
        nxt = turns[k + 1] if k + 1 < len(turns) else None
        end = (int(nxt["judge"].get("step") or 0) if nxt else n_steps) - 1
        if end < start:
            continue  # the turn moved nothing
        sub = turn["subgoal"] or {}
        subgoal = str(sub.get("text") or j.get("next") or "")
        turn_by = str(sub.get("by") or "agent")
        verdict = nxt["judge"].get("judge") if nxt else ("task_done" if success else None)
        thought = str(nxt["judge"].get("thought") or "") if nxt else ""
        s1_runs = turn["s1"]
        # who moved when, within the turn (the trajectory's labels); one segment per run
        pieces = [(max(start, r.start), min(end, r.end), r.by) for r in label_runs if r.end >= start and r.start <= end]
        if not pieces:
            pieces = [(start, end, turn_by)]
        merged: list[list] = []
        for a, b, by in pieces:
            if merged and merged[-1][2] == by and merged[-1][1] + 1 >= a:
                merged[-1][1] = b
            else:
                merged.append([a, b, by])
        last_turn = nxt is None
        for a, b, by in merged:
            seg = Segment(episode=key, turn=int(j.get("turn") or k + 1), index=len(out), start=a, end=b, by=by,
                          subgoal=subgoal, judge=verdict, judge_thought=thought,
                          next=_next_kind(base_step(subgoal), by, nxt, success, last_turn), episode_success=success,
                          task=rec.task, seed=rec.seed, task_goal=goal)
            if by == "s1" and s1_runs:
                seg.s1_calls = len(s1_runs)
                seg.s1_est_length = int(s1_runs[0].get("est_length") or 0) or None
                seg.s1_stop = str(s1_runs[-1].get("stop") or "")
                p = s1_runs[-1].get("progress")
                seg.s1_progress = None if p is None else float(p)
            out.append(seg)
    return out

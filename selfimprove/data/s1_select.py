"""Which spans of an S1 + harness rollout become System1 training data (the rules agreed on 2026-11-24).

- Failed episodes: nothing.
- System1's own steps, as they are: a step the next turn judged complete (or the last step of a successful episode)
  is kept with the subgoal System1 was given ("continue to X" / "X again" included); a run that hit its step budget
  (stop ``budget``: the timeout) is dropped, and so is any step judged incomplete or failed.
- The harness's steps: consecutive harness steps on the same step (X, "continue to X", "X again") are merged into one
  span with the step's own wording; a failed attempt cuts the run (the retry starts a new run, worded "X again"). A run
  is kept when its last step was judged complete - or incomplete (it got partway: kept without its halt frame, so
  System1 does not learn to stop there) - and it is S1-friendly: no whole-body motion (the base turning while the arm moves, a drive about a point), and not very jerky (the
  demonstrations' extreme jerk for that primitive, p99.9 in ``assets/demo_motion_stats.json``, exceeded on more than
  JERK_MAX_SHARE of its steps, or by more than JERK_MAX_RATIO times on any step; from the 2026-11-24 debug round:
  harness steps are often recoveries System1 should learn, and a short motion's two speed changes are no jerk). A
  partial grip (the gripper held between open and closed, e.g. a reach with ``grip=.8``: 174 spans of round 0) is
  kept with the gripper relabeled to the demonstrations' convention, which only ever command it fully open or
  closed: -1 where it is held on the open side, +1 on the closed side (``grip_relabel``; the state keeps the real
  finger opening; 2026-11-25, OpenDrawer l32's reaches). A merged
  manipulation run longer than ``max_steps`` is cut into pieces: "X", then "continue to X"; a navigation run (go to /
  navigate / reposition the base / search for, or the base driving with the arm still) is kept whole.
- Recovery within the step (the user, 2026-11-25): System1's step X judged failed or incomplete (or stopped at its
  budget), then X was finished - by the harness ("X again" / "continue to X") in at most RECOVERY_TURNS turns, the miss
  and the fix together under RECOVERY_TOTAL steps (no bound for a navigation: a search or a drive), or by System1's own
  retry of X (a harness preparation in its turn, such as opening the gripper, included; no bound), kept by the rules above: one more span from System1's start to the end of
  the fix (kind ``recovery_within``), worded X and
  anchored at System1's start, so System1 learns to recover by itself within its step. After a failed attempt or a
  timeout, System1's own frames are context only (the anchor, the progress): only the harness's frames are trained
  on; after an incomplete one (right, but short) all frames are. The harness's part alone stays a span too (kind
  ``recovery_request``: recovery when System2 asks for it).
- Harness fragments (the user, 2026-11-25): consecutive kept harness spans (no System1 step and no dropped step
  between) each under MERGE_SHORT steps are merged until the merged span reaches MERGE_SHORT steps, worded as the
  last one (kind ``harness_merged``: "grasp X" covers reach + grasp, "retract the arm" release + retract); never a
  navigation with a manipulation, never across a milestone, never a recovery_request fix, and not when the merged
  motion is very jerky.
- Long harness struggles: in a run of consecutive harness steps with more than STRUGGLE_STEPS manipulation steps and a
  miss among them (a step judged failed, or left partway), the manipulation steps up to the last miss are dropped:
  only the clean tail that ends the struggle is kept. A run without a miss keeps all its steps (each its own span:
  Astra doing steps by hand), and navigation is always kept.
- Spans shorter than 10 trained steps are dropped (the user, 2026-11-25; the base set's builder drops under 8).
- Estimated Length of every kept span: the grid value at or below its length, at least 50 (System1's own stop criterion
  can make a run longer than the step needs).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .episode import EpisodeRecord, Segment, base_step
from .trajectory import _LEN_GRID

ASSETS = Path(__file__).resolve().parents[2] / "assets"
MAX_STEPS = 400
#: shorter spans are dropped (as the base set's builder does)
MIN_STEPS = 10
#: (display only) the share of a span's steps with a motion metric over the demonstrations' p99 for its primitive
SMOOTH_MAX_SHARE = 0.05
METRICS = ("pos", "rot", "base", "pos_jerk", "rot_jerk")
#: a harness span is very jerky (dropped) when a jerk metric exceeds the demonstrations' p99.9 (for its primitive) on
#: more than JERK_MAX_SHARE of its steps, or on any step by more than JERK_MAX_RATIO times
JERK_METRICS = ("pos_jerk", "rot_jerk")
JERK_MAX_SHARE = 0.10
JERK_MAX_RATIO = 3.0
JERK_FLOOR = 1e-3  # a demos' p99.9 below this (the arm still in their 'go' steps) counts as this
#: recovery within the step: the harness's fix of System1's miss in at most this many turns, the miss + the fix under
#: RECOVERY_TOTAL steps
RECOVERY_TURNS = 2
RECOVERY_TOTAL = 125
#: harness fragments: spans under this many steps are merged with the next ones until they reach it
MERGE_SHORT = 50
#: a harness streak with more than this many manipulation steps and a miss is a struggle (only its clean tail kept)
STRUGGLE_STEPS = 2


@dataclass
class S1Span:
    start: int
    end: int
    subgoal: str
    by: str                       # s1 | harness | recovery (System1's miss + the harness's fix, one step)
    turns: list[int]
    kept: bool
    reason: str
    est_length: int = 0
    primitive: str = ""
    smooth: dict = field(default_factory=dict)   # metric -> share of steps over the demos' p99
    halt: bool = True             # a halt frame after it (not for a step left partway)
    cut: bool = False             # a piece of one step cut at max_steps (X, "continue to X", ...)
    grip_relabel: bool = False    # its held partial gripper levels become the demos' -1 / +1 (relabel_grip)
    train_from: int | None = None  # the first trained frame (None: start); earlier frames are context (anchor, progress)
    #: s1 | harness | harness_merged | recovery_within | recovery_request (set by select)
    kind: str = ""

    @property
    def trained_steps(self) -> int:
        return self.end - max(self.start, self.train_from if self.train_from is not None else self.start) + 1

    @property
    def steps(self) -> int:
        return self.end - self.start + 1

    def to_json(self) -> dict:
        return {**asdict(self), "steps": self.steps}


def floor_length(steps: int) -> int:
    """The length grid's value at or below ``steps``, at least 50."""
    below = [g for g in _LEN_GRID if g <= int(steps)]
    return max(50, below[-1] if below else 50)


def primitive(subgoal: str) -> str:
    s = base_step(subgoal)
    return s.split()[0] if s else "-"


NAV_VERBS = ("go", "navigate", "reposition", "search")


def is_navigation(prim: str, lean_act: np.ndarray) -> bool:
    """A navigation step: its verb, or the base driving in most of its steps with the arm still (``move the base``)."""
    if prim in NAV_VERBS:
        return True
    a = np.asarray(lean_act, float)
    if not len(a):
        return False
    base = np.abs(a[:, 0:3]).max(1) > 0.02
    arm = np.abs(a[:, 4:10]).max(1) > 1e-3
    return bool(base.mean() >= 0.8 and arm.mean() <= 0.1)


def load_motion_stats(path: Path | None = None) -> dict:
    path = path or ASSETS / "demo_motion_stats.json"
    return json.loads(path.read_text())["primitives"] if path.exists() else {}


def motion(a: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step motion of [T, 11] lean actions (scripts/demo_motion_stats.py computes the same on the demos)."""
    out = {"pos": np.linalg.norm(a[:, 4:7], axis=1), "rot": np.linalg.norm(a[:, 7:10], axis=1),
           "base": np.linalg.norm(a[:, 0:3], axis=1)}
    out["pos_jerk"] = np.linalg.norm(np.diff(a[:, 4:7], n=2, axis=0), axis=1) if len(a) > 2 else np.zeros(0)
    out["rot_jerk"] = np.linalg.norm(np.diff(a[:, 7:10], n=2, axis=0), axis=1) if len(a) > 2 else np.zeros(0)
    return out


def smoothness(lean_act: np.ndarray, prim: str, stats: dict) -> dict[str, float]:
    """metric -> the share of the span's steps above the demos' p99 for the primitive (a verb the demos have too few
    steps of: their "go" steps for a span that drives, their "reach" steps otherwise)."""
    m = motion(np.asarray(lean_act, float))
    ref = _reference(m, prim, stats)  # a verb the demos have too few steps of: a navigation or an arm step like theirs
    if not ref:
        return {}
    return {k: round(float((v > ref[k]["p99"]).mean()), 3) if len(v) else 0.0 for k, v in m.items() if k in ref}


def _reference(m: dict, prim: str, stats: dict) -> dict | None:
    return stats.get(prim) or stats.get("go" if len(m["base"]) and float(m["base"].mean()) > 0.02 else "reach")


def jerkiness(lean_act: np.ndarray, prim: str, stats: dict) -> dict[str, tuple[float, float]]:
    """jerk metric -> (the share of steps over the demos' p99.9 for the primitive, the largest step / that p99.9)."""
    m = motion(np.asarray(lean_act, float))
    ref = _reference(m, prim, stats)
    out = {}
    for k in JERK_METRICS:
        v = m[k]
        if not ref or k not in ref or "p99.9" not in ref[k] or not len(v):
            continue
        top = max(float(ref[k]["p99.9"]), JERK_FLOOR)
        out[k] = (round(float((v > top).mean()), 3), round(float(v.max() / top), 2))
    return out


def very_jerky(jerk: dict[str, tuple[float, float]]) -> str:
    """'' or why the span is very jerky."""
    bad = [f"{k} {share:.0%} of steps over the demos' p99.9, at most {ratio:.1f}x it" for k, (share, ratio) in jerk.items()
           if share > JERK_MAX_SHARE or ratio > JERK_MAX_RATIO]
    return "; ".join(bad)


def span_flags(traj, s: int, e: int) -> dict:
    """A partial grip (the gripper's target held between open and closed for 3+ steps without a command) and whole-body
    motion (the base moving while the arm moves, in mode -1, for 5+ steps)."""
    lev = np.asarray(traj.level[s:e + 1], float)
    a = np.asarray(traj.lean_act[s:e + 1], float)
    if not len(a):
        return {"partial_grip": False, "whole_body": False}
    held = np.abs(a[:, 10]) < 1e-6
    base = np.abs(a[:, 0:3]).max(1) > 0.02
    arm = np.abs(a[:, 4:10]).max(1) > 1e-3
    return {"partial_grip": bool(((np.abs(lev) < 0.98) & held).sum() >= 3),
            "whole_body": bool(((base & arm) & (a[:, 3] < 0)).sum() >= 5)}


def relabel_grip(lean_act: np.ndarray, raw_action: np.ndarray, level: np.ndarray, s: int, e: int,
                 lean_dim: int = 10, raw_dim: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the actions with steps s..e in the demonstrations' gripper convention: a step that holds the gripper
    at a partial level (command 0) commands the side it is held on (-1 open, +1 closed; ``level``: the gripper's
    target before each step, +1 closed); the steps that move it keep their sign."""
    lean, raw = np.array(lean_act, float, copy=True), np.array(raw_action, float, copy=True)
    for t in range(max(0, s), min(e, len(lean) - 1) + 1):
        if abs(lean[t, lean_dim]) < 1e-6:
            side = 1.0 if float(level[t]) >= 0.0 else -1.0
            lean[t, lean_dim] = side
            raw[t, raw_dim] = side
    return lean, raw


def _completed(seg: Segment) -> bool:
    return seg.judge in ("subgoal_complete", "task_done")


def _pieces(span: S1Span, max_steps: int) -> list[S1Span]:
    """A span longer than max_steps cut into "X", then "continue to X"; only the last piece can halt."""
    if span.steps <= max_steps:
        return [span]
    out, a, first = [], span.start, True
    base = base_step(span.subgoal)
    while a <= span.end:
        b = min(span.end, a + max_steps - 1)
        word = span.subgoal if first else f"continue to {base}"
        out.append(S1Span(a, b, word, span.by, span.turns, span.kept, span.reason, floor_length(b - a + 1), span.primitive,
                          span.smooth, halt=span.halt and b == span.end, grip_relabel=span.grip_relabel, cut=True))
        a, first = b + 1, False
    return out


def select(rec: EpisodeRecord, stats: dict | None = None, max_steps: int = MAX_STEPS) -> list[S1Span]:
    """Every span of the episode with its verdict (kept or not, and why); kept spans carry their Estimated Length."""
    stats = load_motion_stats() if stats is None else stats
    spans = _select(rec, stats, max_steps)
    if rec.success:
        _drop_struggles(rec, spans)
        spans += _recoveries(rec, spans)
        spans += _merge_fragments(rec, spans, stats)
        spans.sort(key=lambda sp: (sp.start, sp.by == "recovery", sp.kind == "harness_merged"))
    for sp in spans:
        if not sp.kind:
            sp.kind = {"s1": "s1", "recovery": "recovery_within"}.get(sp.by, "harness")
    for sp in spans:
        if sp.kept and sp.trained_steps < MIN_STEPS:
            sp.kept, sp.reason, sp.est_length = False, f"too short (under {MIN_STEPS} steps)", 0
    return spans


def _segments(rec: EpisodeRecord) -> list[Segment]:
    """The episode's segments with each two-part turn - a few harness motions (opening the gripper, a small move), then
    robot.s1() in the same turn - as one System1 step under the turn's subgoal, the preparation included (WashLettuce
    l81 of the check run: `gripper('open')` then System1's `grasp the colander again`; split by mover, the 14 steps of
    the opening alone were labeled "grasp ... again"). The joined step is marked ``prepped``."""
    cached = getattr(rec, "_s1_segments", None)
    if cached is not None:
        return cached
    out: list[Segment] = []
    segs = rec.segments
    i = 0
    while i < len(segs):
        sg = segs[i]
        if sg.by != "s1" and i + 1 < len(segs) and segs[i + 1].turn == sg.turn and segs[i + 1].by == "s1":
            nxt = replace(segs[i + 1], start=sg.start)
            nxt.s1_stop = getattr(segs[i + 1], "s1_stop", None)
            nxt.prepped = True
            out.append(nxt)
            i += 2
            continue
        out.append(sg)
        i += 1
    try:
        rec._s1_segments = out
    except AttributeError:
        pass
    return out


def _harness_runs(segs: list[Segment]) -> list[list[Segment]]:
    """The episode's streaks of consecutive harness segments (no System1 step between)."""
    runs, cur = [], []
    for sg in segs:
        if sg.by == "s1":
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(sg)
    return runs + ([cur] if cur else [])


def _drop_struggles(rec: EpisodeRecord, spans: list[S1Span]) -> None:
    """In a harness streak with more than STRUGGLE_STEPS manipulation steps and a miss among them (failed, or left
    partway), drop the manipulation spans up to the last miss: the clean tail that ends the struggle stays."""
    lean = rec.traj.lean_act
    for run in _harness_runs(_segments(rec)):
        lo, hi = run[0].start, run[-1].end
        mine = [sp for sp in spans if sp.by == "harness" and lo <= sp.start <= hi
                and not is_navigation(sp.primitive, lean[sp.start:sp.end + 1])]
        # the pieces of one step cut at max_steps are one step, and a piece before the last is no miss (its step went on)
        steps = {(tuple(sp.turns), base_step(sp.subgoal)) for sp in mine}
        if len(steps) <= STRUGGLE_STEPS:
            continue
        last_piece = {}
        for sp in mine:
            k = (tuple(sp.turns), base_step(sp.subgoal))
            last_piece[k] = max(last_piece.get(k, -1), sp.end)
        misses = [sp for sp in mine if sp.reason.startswith("judged subgoal_failed")
                  or (sp.kept and not sp.halt and not (sp.cut and sp.end < last_piece[(tuple(sp.turns), base_step(sp.subgoal))]))]
        if not misses:
            continue
        last = max(sp.start for sp in misses)
        for sp in mine:
            if sp.start <= last and sp.kept:
                sp.kept, sp.est_length = False, 0
                sp.reason = (f"before the last miss of a long harness struggle ({len(steps)} manipulation steps, "
                             f"{len(misses)} missed): only its clean tail is kept")


def _recoveries(rec: EpisodeRecord, spans: list[S1Span]) -> list[S1Span]:
    """System1's miss at step X and the harness's short fix of it, as one more span worded X (see the module doc)."""
    segs, out = _segments(rec), []
    for i, sg in enumerate(segs):
        timeout = (sg.s1_stop or "") == "budget"
        if sg.by != "s1" or not (timeout or sg.judge in ("subgoal_failed", "subgoal_incomplete")):
            continue
        # the fix: System1's own retry of the step (the next step, by System1 - a harness preparation in its turn, such
        # as opening the gripper, included), or the harness's steps on it
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        by_s1 = nxt is not None and nxt.by == "s1" and base_step(nxt.subgoal) == base_step(sg.subgoal)
        run, j = ([nxt] if by_s1 else []), i + 1
        while not by_s1 and j < len(segs) and segs[j].by != "s1" and base_step(segs[j].subgoal) == base_step(sg.subgoal):
            run.append(segs[j])
            j += 1
        if not run or not _completed(run[-1]):
            continue
        steps = run[-1].end - run[0].start + 1
        nav = is_navigation(primitive(sg.subgoal), rec.traj.lean_act[sg.start:run[-1].end + 1])
        if not by_s1 and (len({r.turn for r in run}) > RECOVERY_TURNS
                          or (not nav and run[-1].end - sg.start + 1 >= RECOVERY_TOTAL)):
            continue  # (bounds for a harness fix; a search or a drive may be long; System1's own retry is never bound)
        fix = [sp for sp in spans if sp.by in ("harness", "s1") and sp.start == run[0].start and sp.end == run[-1].end
               and sp.kept and sp.kind not in ("recovery_within",)]
        if not fix:  # the fix itself is not S1-friendly (jerky, whole-body) or dropped as part of a struggle
            continue
        context = timeout or sg.judge == "subgoal_failed"
        fix[0].kind = "recovery_request"
        fix[0].reason += ("; System1's own retry of its miss: recovery on request" if by_s1
                          else "; the fix of System1's miss: recovery on request")
        fixer = "System1's own retry" if by_s1 else "the harness's fix"
        why = (f"recovery within the step: System1's {'timeout' if timeout else sg.judge} + {fixer} "
               f"({len(run)} turn{'s' if len(run) > 1 else ''}, {steps} steps); "
               + ("System1's frames are context, the fix's are trained" if context else "all frames trained"))
        out.append(S1Span(sg.start, run[-1].end, sg.subgoal, "recovery", [sg.turn] + [r.turn for r in run], True, why,
                          floor_length(run[-1].end - sg.start + 1), primitive(sg.subgoal), fix[0].smooth,
                          grip_relabel=fix[0].grip_relabel, train_from=run[0].start if context else None,
                          kind="recovery_within"))
    return out


def _merge_fragments(rec: EpisodeRecord, spans: list[S1Span], stats: dict) -> list[S1Span]:
    """Consecutive short harness spans merged until they reach MERGE_SHORT steps (see the module doc). The merged
    spans are returned; their parts stay in ``spans``, dropped with the reason "merged into ..."."""
    lean = rec.traj.lean_act
    milestone = {}  # turn -> its milestone id (the planner fields), so that no merge crosses one
    for e in getattr(rec, "events", None) or []:
        if e.get("kind") in ("judge", "planner") and e.get("turn") and e.get("milestone"):
            milestone[int(e["turn"])] = str(e["milestone"]).split(":")[0].strip()
    ms = lambda sp: milestone.get(sp.turns[0])  # noqa: E731
    cand = sorted((sp for sp in spans if sp.by == "harness" and sp.kept and sp.kind != "recovery_request"),
                  key=lambda sp: sp.start)
    nav = {id(sp): is_navigation(sp.primitive, lean[sp.start:sp.end + 1]) for sp in cand}
    out, i = [], 0
    while i < len(cand):
        group, total = [cand[i]], cand[i].steps
        j = i + 1
        while (total < MERGE_SHORT and group[-1].steps < MERGE_SHORT and j < len(cand) and cand[j].steps < MERGE_SHORT
               and cand[j].start == group[-1].end + 1 and nav[id(cand[j])] == nav[id(group[0])]
               and ms(cand[j]) == ms(group[0])):
            group.append(cand[j])
            total += cand[j].steps
            j += 1
        if len(group) > 1 and group[0].steps < MERGE_SHORT:
            s, e, last = group[0].start, group[-1].end, group[-1]
            jerky = very_jerky(jerkiness(lean[s:e + 1], last.primitive, stats))
            if not jerky:
                why = (f"harness fragments merged ({len(group)} turns' steps, {e - s + 1} steps), worded as the last: "
                       + " + ".join(f"'{g.subgoal}'" for g in group))
                out.append(S1Span(s, e, last.subgoal, "harness", [t for g in group for t in g.turns], True, why,
                                  floor_length(e - s + 1), last.primitive, {}, halt=last.halt,
                                  grip_relabel=any(g.grip_relabel for g in group), kind="harness_merged"))
                for g in group:
                    g.kept, g.est_length = False, 0
                    g.reason = f"merged into the harness span {s}-{e} ('{last.subgoal}')"
                i = j
                continue
        i += 1
    return out


def _select(rec: EpisodeRecord, stats: dict, max_steps: int) -> list[S1Span]:
    traj, segs = rec.traj, _segments(rec)
    out: list[S1Span] = []
    if not rec.success:
        return [S1Span(sg.start, sg.end, sg.subgoal, "s1" if sg.by == "s1" else "harness", [sg.turn], False,
                       "failed episode") for sg in segs]
    i = 0
    while i < len(segs):
        sg = segs[i]
        if sg.by == "s1":
            if (sg.s1_stop or "") == "budget":
                out.append(S1Span(sg.start, sg.end, sg.subgoal, "s1", [sg.turn], False, "S1 timeout (step budget)"))
            elif not _completed(sg):
                out.append(S1Span(sg.start, sg.end, sg.subgoal, "s1", [sg.turn], False, f"judged {sg.judge or 'nothing'}"))
            else:
                out.append(S1Span(sg.start, sg.end, sg.subgoal, "s1", [sg.turn], True, "S1 step completed",
                                  floor_length(sg.end - sg.start + 1), primitive(sg.subgoal)))
            i += 1
            continue
        # a run of harness steps on the same step; a failed attempt ends a run (the retry starts the next one)
        run = [sg]
        while (i + len(run) < len(segs) and segs[i + len(run)].by != "s1" and run[-1].judge == "subgoal_incomplete"
               and base_step(segs[i + len(run)].subgoal) == base_step(sg.subgoal)):
            run.append(segs[i + len(run)])
        i += len(run)
        s, e, last = run[0].start, run[-1].end, run[-1]
        # the run's words are its first turn's, as Astra gave them: X (merged with its "continue to X"), "X again" (a
        # retry), or "continue to X" (finishing System1's partway step)
        word = " ".join(str(run[0].subgoal).split()).strip().rstrip(".")
        turns = [r.turn for r in run]
        prim = primitive(word)
        partway = last.judge == "subgoal_incomplete"
        if not _completed(last) and not partway:
            out.append(S1Span(s, e, word, "harness", turns, False, f"judged {last.judge or 'nothing'}", primitive=prim))
            continue
        flags = span_flags(traj, s, e)
        regrip = flags["partial_grip"]
        if flags["whole_body"]:
            out.append(S1Span(s, e, word, "harness", turns, False, "whole-body motion", primitive=prim))
            continue
        sm = smoothness(traj.lean_act[s:e + 1], prim, stats)
        jerky = very_jerky(jerkiness(traj.lean_act[s:e + 1], prim, stats))
        if jerky:
            out.append(S1Span(s, e, word, "harness", turns, False, f"very jerky ({jerky})", primitive=prim, smooth=sm))
            continue
        why = (("harness step partway (judged subgoal_incomplete; no halt frame)" if partway else
                "harness step completed") + ", not jerky" + (f" (merged {len(run)} turns)" if len(run) > 1 else "")
               + ("; gripper held partly open/closed, relabeled -1/+1" if regrip else ""))
        span = S1Span(s, e, word, "harness", turns, True, why, floor_length(e - s + 1), prim, sm, halt=not partway,
                      grip_relabel=regrip)
        # a long manipulation is cut at max_steps; a navigation stays whole (a drive has no natural midpoint)
        out.extend([span] if is_navigation(prim, traj.lean_act[s:e + 1]) else _pieces(span, max_steps))
    return out

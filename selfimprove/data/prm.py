"""Astra as the process reward model: a five-level score per segment and what it is used for.

The score of a segment (one turn's run of System1's or the harness's control steps) comes from the
next turn's judgement of it, what the planner did next, the episode's outcome and the time it took:

- very_good  the step was judged complete (or the task finished during it) and the planner moved on,
             in an episode that succeeded, within ``slow_factor`` x the step's expected length
- good       complete and moved on, but the episode failed or the step was slow (or System1 stopped at
             its step budget rather than by its own progress)
- neutral    partway (subgoal_incomplete), judged complete yet redone, or never judged (the episode ended)
- bad        judged subgoal_failed in an episode that still succeeded
- very_bad   judged subgoal_failed in an episode that failed

Use: System1's very_good/good -> positive (AWR-weighted SL + unary-positive), System1's bad/very_bad ->
negative (unary-negative), the harness's very_good/good -> positive (DAgger: the expert's correction in a
state System1 reached).  Everything else is left out.  The thresholds and the lists are the config's
``filter`` section.
"""

from __future__ import annotations

import math

from .episode import EpisodeRecord, Segment
from .trajectory import estimated_length

LEVELS = ("very_good", "good", "neutral", "bad", "very_bad")
COMPLETE = ("subgoal_complete", "task_finish", "task_done")


def score_segment(seg: Segment, fcfg: dict) -> tuple[str, str]:
    """(level, reason)."""
    slow = float(fcfg.get("slow_factor", 1.5))
    expected = seg.s1_est_length if (seg.by == "s1" and seg.s1_est_length) else estimated_length(seg.steps)
    if seg.judge in COMPLETE:
        if seg.next not in ("moved_on", "task_done"):
            return "neutral", f"judged {seg.judge} but then {seg.next}"
        flaws = []
        if not seg.episode_success:
            flaws.append("episode failed")
        if seg.by == "s1" and seg.steps > slow * max(1, expected):
            flaws.append(f"slow ({seg.steps} steps vs {expected} expected)")
        if seg.by == "s1" and seg.s1_stop == "budget":
            flaws.append("System1 ran out of its step budget")
        if flaws:
            return "good", "; ".join(flaws)
        return "very_good", "complete, moved on, episode succeeded"
    if seg.judge == "subgoal_failed":
        return ("bad", "failed (episode succeeded)") if seg.episode_success else ("very_bad", "failed, episode failed")
    if seg.judge == "subgoal_incomplete":
        return "neutral", "partway"
    return "neutral", f"judge {seg.judge!r}, then {seg.next}"


def assign(rec: EpisodeRecord, fcfg: dict) -> list[Segment]:
    """Score every segment of an episode and decide its use (in place; returns the segments)."""
    adv = fcfg.get("advantage") or {}
    s1_pos = set(fcfg.get("use_s1_positive") or ())
    s1_neg = set(fcfg.get("use_s1_negative") or ())
    h_pos = set(fcfg.get("use_harness_positive") or ())
    min_steps = int(fcfg.get("min_segment_steps", 8))
    bad_episode = bool(fcfg.get("drop_error_episodes", True)) and (rec.errors or rec.end in (
        "gateway_error", "simulator_failed", "error"))
    for seg in rec.segments:
        seg.score, seg.reason = score_segment(seg, fcfg)
        seg.advantage = float(adv.get(seg.score, 0.0))
        seg.use = None
        if bad_episode:
            seg.reason += "; dropped: the episode ended in an error"
            continue
        if seg.steps < min_steps:
            seg.reason += f"; dropped: {seg.steps} < {min_steps} steps"
            continue
        if not seg.subgoal:
            seg.reason += "; dropped: no subgoal"
            continue
        if seg.by == "s1":
            if seg.score in s1_pos:
                seg.use = "positive"
            elif seg.score in s1_neg:
                seg.use = "negative"
        elif seg.score in h_pos:
            seg.use = "positive"
    return rec.segments


def awr_weights(segments: list[Segment], fcfg: dict) -> dict[int, float]:
    """id(segment) -> its SL weight exp(A / beta), clipped, normalised to a frame-weighted mean of 1 over
    the positives (so the rollout data's SL term is on the same scale as the base data's)."""
    beta = float(fcfg.get("awr_beta", 1.0))
    cap = float(fcfg.get("awr_max_weight", 5.0))
    pos = [s for s in segments if s.use == "positive"]
    raw = {id(s): min(cap, math.exp(float(s.advantage or 0.0) / max(beta, 1e-6))) for s in pos}
    frames = sum(s.steps for s in pos)
    if not frames:
        return {}
    mean = sum(raw[id(s)] * s.steps for s in pos) / frames
    return {k: min(cap, v / mean) for k, v in raw.items()}

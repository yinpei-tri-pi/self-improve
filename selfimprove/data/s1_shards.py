"""Rollout segments -> System1 training shards, in the exact format openpi's RoboCasa loader reads.

One sample per control step of a selected segment (``<key>.{scene_left,scene_right,wrist}.jpg`` and
``.anchor_<cam>.jpg``, ``.arrays.npz``, ``.meta.json``), as RoboAnnotator's
``producers/build_system1_shards.py`` writes them, plus the self-improvement fields in meta.json:

- ``si_label``   +1 positive / -1 negative (the loss: TrainConfig.si)
- ``si_weight``  the AWR weight (positives; 1 for negatives, which the SL term never sees)
- ``si_score``, ``si_advantage``, ``si_source`` (s1 | harness), ``si_judge``, ``si_turn``, ``si_exp``,
  ``si_iter``, ``si_ref_loss`` (the rollout checkpoint's surrogate loss on the chunk at this step, when
  scored)

The JPEGs are the record's own bytes (no re-encoding).  Shards are written in two passes: every
episode's samples go to one local tar with an index of member offsets, then a global shuffle of the
sample keys deals them out to ``shard_NNNNNN.tar`` files of ``samples_per_shard`` samples, shuffled
within each shard as well (the base data is shuffled the same two ways).
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import tasks as T
from .episode import Segment, load_episode
from .prm import assign, awr_weights
from .trajectory import (CAMS, HORIZON, LEAN_HOLD_DIMS, RAW_HOLD_DIMS, Trajectory, estimated_length, halt_chunk,
                         padded_chunk, progress_class, system1_prompt)

PROGRESS_CLASSES = 10


@dataclass
class SpanPlan:
    """One selected segment, as the shard writer needs it (picklable, no arrays)."""

    seg: dict
    variant: str
    quality: str
    est_length: int
    label: float
    weight: float
    halt: int


def episode_id(exp_id: str, iteration: int, task: str, seed: int) -> str:
    return f"SelfImprove/{exp_id}/iter_{int(iteration):02d}/{task}/seed_{int(seed)}"


def plan_episode(folder: str, fcfg: dict) -> tuple[list[Segment], list[SpanPlan]]:
    """Score the episode's segments and plan the spans to write (positives and negatives)."""
    rec = load_episode(folder, with_trajectory=False)
    segs = assign(rec, fcfg)
    weights = awr_weights(segs, {**fcfg, "awr_max_weight": 1e9})  # per-episode; renormalised globally later
    plans = []
    for s in segs:
        if s.use is None:
            continue
        if s.use == "positive":
            slow = s.score == "good" and "slow" in s.reason
            quality = str(fcfg.get("slow_quality", "Suboptimal")) if slow else "Success"
            est = estimated_length(s.steps)
            label, weight, halt = 1.0, float(weights.get(id(s), 1.0)), int(fcfg.get("halt_frames", 1))
            variant = f"rollout_{'s1' if s.by == 's1' else 'harness'}"
        else:
            quality = str(fcfg.get("negative_quality", "Success"))
            est = int(s.s1_est_length or estimated_length(s.steps))
            label, weight, halt = -1.0, 1.0, 0
            variant = "rollout_s1_neg"
        plans.append(SpanPlan(s.to_json(), variant, quality, est, label, weight, halt))
    return segs, plans


def plan_episode_v2(folder: str, stats: dict | None = None) -> tuple[list[dict], list[SpanPlan]]:
    """The agreed rules (selfimprove/data/s1_select.py): System1's completed steps as they are, the harness's steps
    (completed, or partway without a halt frame) merged per step unless very jerky; positives only (label 1, weight 1),
    Quality Success, Estimated Length floored."""
    from . import s1_select

    rec = load_episode(folder)
    spans = s1_select.select(rec, stats)
    plans = []
    for sp in spans:
        if not sp.kept:
            continue
        seg = {"start": sp.start, "end": sp.end, "subgoal": sp.subgoal,
               "by": {"s1": "s1", "recovery": "recovery"}.get(sp.by, "agent"), "train_from": sp.train_from,
               "kind": sp.kind,
               "turn": sp.turns[0], "turns": sp.turns, "judge": "subgoal_complete", "score": None, "advantage": None,
               "grip_relabel": bool(sp.grip_relabel)}
        plans.append(SpanPlan(seg, f"rollout_{sp.by}", "Success", int(sp.est_length), 1.0, 1.0, 1 if sp.halt else 0))
    return [sp.to_json() for sp in spans], plans


def _chunk_losses(folder: Path) -> dict[int, float]:
    """step -> the scored chunk's loss_real (rollout/<task>/<seed>/chunk_scores.jsonl)."""
    path = folder / "chunk_scores.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out[int(r["t"])] = float(r["loss_real"])
            except (ValueError, KeyError):
                pass
    return out


def span_samples(traj: Trajectory, plan: SpanPlan, *, ep_id: str, extra_meta: dict, ref_losses: dict[int, float],
                 slim: bool = False):
    """Yield (key, members) for every frame of one planned span (build_system1_shards._gen_span_frames)."""
    seg = plan.seg
    n = len(traj)
    s, e = max(0, int(seg["start"])), min(n - 1, int(seg["end"]))
    halt = max(0, min(plan.halt, n - 1 - e))
    flat = ep_id.replace("/", "__")
    vtag = plan.variant.replace(":", "-")
    raw_state, raw_action = traj.raw16, traj.action_lr
    lean_state, lean_act = traj.lean, traj.lean_act
    if seg.get("grip_relabel"):  # a harness span held the gripper partly open/closed: the demos' -1 / +1 instead
        from .s1_select import relabel_grip

        lean_act, raw_action = relabel_grip(lean_act, raw_action, traj.level, s, e)
    base_pos, base_yaw = traj.base_ref
    anchor_jpgs = {key: traj.jpeg(s, cam) for cam, key in CAMS.items()}
    scored = sorted(ref_losses)
    first = max(s, int(seg["train_from"])) if seg.get("train_from") is not None else s  # earlier: context only
    for t in range(first, e + halt + 1):
        is_halt = t > e
        key = f"{flat}__{vtag}__s{s:06d}-{e:06d}__f{t:06d}"
        members = {}
        for cam, ck in CAMS.items():
            members[f"{key}.{ck}.jpg"] = traj.jpeg(t, cam)
        for ck, data in anchor_jpgs.items():
            members[f"{key}.anchor_{ck}.jpg"] = data
        raw_ep, mask_ep = padded_chunk(raw_action, t, n - 1, HORIZON, RAW_HOLD_DIMS)
        lean_ep, _ = padded_chunk(lean_act, t, n - 1, HORIZON, LEAN_HOLD_DIMS)
        raw_sg, mask_sg = padded_chunk(raw_action, t, e, HORIZON, RAW_HOLD_DIMS)
        lean_sg, _ = padded_chunk(lean_act, t, e, HORIZON, LEAN_HOLD_DIMS)
        if is_halt:
            raw_ep = halt_chunk(raw_action[e], HORIZON, RAW_HOLD_DIMS)
            lean_ep = halt_chunk(lean_act[e], HORIZON, LEAN_HOLD_DIMS)
            mask_ep = np.zeros((HORIZON,), dtype=bool)
            raw_sg, mask_sg, lean_sg = raw_ep, mask_ep, lean_ep
        pt = max(0, t - 1)
        last_raw, last_lean = raw_action[pt], lean_act[pt]
        gripper = "Close" if float(last_raw[11]) > 0 else "Open"
        buf = io.BytesIO()
        if slim:  # only what openpi's production loader reads (robocasa_webdataset._build_sample)
            np.savez(buf, state=raw_state[t], state_anchor=raw_state[s], lean_state=lean_state[t],
                     lean_state_anchor=lean_state[s], lean_action_subgoal_pad=lean_sg, action_pad_mask_subgoal=mask_sg)
        else:
            np.savez(buf, state=raw_state[t], state_anchor=raw_state[s], lean_state=lean_state[t],
                     lean_state_anchor=lean_state[s], last_action=last_raw, last_action_lean=last_lean,
                     action_subtask=raw_ep, action_subtask_mask=mask_sg, lean_action_subtask=lean_ep,
                     action_subgoal_pad=raw_sg, lean_action_subgoal_pad=lean_sg, action_pad_mask_episode=mask_ep,
                     action_pad_mask_subgoal=mask_sg)
        members[f"{key}.arrays.npz"] = buf.getvalue()
        prog = 1.0 if is_halt else min(1.0, max(0.0, (t - s) / max(1, e - s)))
        if slim:
            meta = {"episode_id": ep_id, "variant": plan.variant, "span": [s, e], "frame_index": t,
                    "task_goal": traj.task_goal, "subgoal": seg["subgoal"], "quality": plan.quality,
                    "est_length": int(plan.est_length), "gripper_flag": gripper,
                    "base_pos_ref": [float(x) for x in base_pos], "base_yaw_ref": float(base_yaw),
                    "subgoal_progress_frac": prog, "frame_kind": "halt" if is_halt else "real",
                    # provenance (not read by the loader)
                    "si_source": {"s1": "s1", "recovery": "recovery"}.get(seg["by"], "harness"),
                    "si_turns": seg.get("turns", [seg["turn"]]), "si_kind": seg.get("kind"),
                    **extra_meta}
            members[f"{key}.meta.json"] = json.dumps(meta).encode()
            yield key, members
            continue
        pclass = max(1, min(PROGRESS_CLASSES, int(prog * PROGRESS_CLASSES) + 1))
        # the chunk score nearest at or before this step (chunks are scored every `stride` steps)
        ref = None
        if scored:
            i = int(np.searchsorted(scored, t, side="right")) - 1
            if i >= 0 and scored[i] >= s:
                ref = ref_losses[scored[i]]
        meta = {
            "episode_id": ep_id, "episode_len": n, "frame_index": t, "task_goal": traj.task_goal,
            "base_pos_ref": [float(x) for x in base_pos], "base_yaw_ref": float(base_yaw),
            "variant": plan.variant, "quality": plan.quality,
            "prompt": system1_prompt(traj.task_goal, seg["subgoal"], plan.quality, plan.est_length, gripper=gripper),
            "est_length": int(plan.est_length), "executed_step": t - s, "gripper_flag": gripper,
            "anchor_key": f"{flat}/f{s:06d}", "subgoal": seg["subgoal"], "subgoal_detail": "",
            "milestone_text": seg["subgoal"], "milestone_span": [s, e], "span": [s, e],
            "progress_frac": prog, "subgoal_progress_frac": prog, "milestone_progress_frac": prog,
            "subgoal_progress_class": pclass, "milestone_progress_class": progress_class(t, s, e),
            "progress_classes": PROGRESS_CLASSES, "image_base": "", "image_root": "",
            "leak_lead": 0, "leak_halt": halt, "fail_end": -1, "frame_kind": "halt" if is_halt else "real",
            # self-improvement
            "si_label": plan.label, "si_weight": plan.weight, "si_score": seg["score"],
            "si_advantage": seg["advantage"], "si_source": "s1" if seg["by"] == "s1" else "harness",
            "si_judge": seg["judge"], "si_turn": seg["turn"], "si_ref_loss": ref, **extra_meta,
        }
        members[f"{key}.meta.json"] = json.dumps(meta).encode()
        yield key, members


def _write_episode(args) -> dict:
    """Pass 1 for one episode: its samples into ``tmp/<flat>.tar`` + an index of member offsets."""
    folder, plans, ep_id, extra_meta, tmp_dir, slim = args
    folder = Path(folder)
    traj = Trajectory.load(folder)
    ref = _chunk_losses(folder)
    out = Path(tmp_dir) / (ep_id.replace("/", "__") + ".tar")
    keys: list[tuple[str, list[str]]] = []
    counts = {"positive_frames": 0, "negative_frames": 0}
    with tarfile.open(out, "w") as tar:
        for plan in plans:
            for key, members in span_samples(traj, plan, ep_id=ep_id, extra_meta=extra_meta, ref_losses=ref, slim=slim):
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                keys.append((key, list(members)))
                counts["positive_frames" if plan.label > 0 else "negative_frames"] += 1
    # where each member's bytes sit (read back: the headers of long names vary in size)
    with tarfile.open(out, "r") as tar:
        where = {m.name: (m.offset_data, m.size) for m in tar.getmembers()}
    index = [{"key": key, "members": [[n, *where[n]] for n in names]} for key, names in keys]
    return {"tar": str(out), "index": index, **counts}


def _write_shard(args) -> dict:
    """Pass 2: one output shard from (tar, entry) references, in the given (already shuffled) order."""
    path, refs = args
    handles: dict[str, object] = {}
    n = 0
    try:
        with tarfile.open(path, "w") as out:
            for tar_path, entry in refs:
                fh = handles.get(tar_path)
                if fh is None:
                    fh = handles[tar_path] = open(tar_path, "rb")
                for name, offset, size in entry["members"]:
                    fh.seek(offset)
                    data = fh.read(size)
                    info = tarfile.TarInfo(name)
                    info.size = size
                    info.mtime = 0
                    out.addfile(info, io.BytesIO(data))
                n += 1
    finally:
        for fh in handles.values():
            fh.close()
    return {"shard": Path(path).name, "num_samples": n}


def build(episodes: list[tuple[str, str, int]], out_dir: str | Path, *, exp_id: str, iteration: int, fcfg: dict,
          seed: int = 0, workers: int = 8, log=print) -> dict:
    """``episodes``: (local folder, task, seed) of every downloaded rollout.  Writes ``out_dir``/shards/*.tar,
    manifest.jsonl (both layouts), meta.json, segments.jsonl and README.md; returns the summary."""
    out_dir = Path(out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    plans_by_ep, all_segments = [], []
    held = sorted({task for _, task, _ in episodes} & T.held_out_tasks())
    if held:
        raise ValueError(f"held-out composite_unseen tasks may not enter S1 training data: {held}")
    v2 = str(fcfg.get("planner", "v2")) == "v2"
    slim = bool(fcfg.get("slim", v2))
    stats = None
    if v2:
        from . import s1_select

        stats = s1_select.load_motion_stats()
    for folder, task, s in episodes:
        try:
            if v2:
                segs_json, plans = plan_episode_v2(folder, stats)
            else:
                segs, plans = plan_episode(folder, fcfg)
                segs_json = [x.to_json() for x in segs]
        except (OSError, ValueError, KeyError) as exc:
            log(f"skip {folder}: {type(exc).__name__}: {exc}")
            continue
        all_segments.extend({**x, "episode": f"{task}/{s}"} for x in segs_json)
        if plans:
            plans_by_ep.append((folder, task, s, plans))
    # renormalise the AWR weights over the whole round (frame-weighted mean 1 over the positives; v2: all weights 1)
    pos = [(p, p.seg["end"] - p.seg["start"] + 1) for _, _, _, ps in plans_by_ep for p in ps if p.label > 0]
    frames = sum(n for _, n in pos)
    if frames and not v2:
        mean = sum(p.weight * n for p, n in pos) / frames
        cap = float(fcfg.get("awr_max_weight", 5.0))
        for p, _ in pos:
            p.weight = float(min(cap, p.weight / mean))
    extra = {"si_exp": exp_id, "si_iter": int(iteration)}
    jobs = [(folder, plans, episode_id(exp_id, iteration, task, s), {**extra, "task": task, "seed": int(s)}, str(tmp), slim)
            for folder, task, s, plans in plans_by_ep]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, r in enumerate(ex.map(_write_episode, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0:
                log(f"pass 1: {i + 1}/{len(jobs)} episodes")
    refs = [(r["tar"], e) for r in results for e in r["index"]]
    rng = random.Random(seed)
    rng.shuffle(refs)
    per = int(fcfg.get("samples_per_shard", 2048))
    shard_jobs = []
    for i in range(0, len(refs), per):
        part = refs[i:i + per]
        rng.shuffle(part)
        shard_jobs.append((str(out_dir / "shards" / f"shard_{i // per:06d}.tar"), part))
    manifest = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in ex.map(_write_shard, shard_jobs, chunksize=1):
            manifest.append(r)
    manifest.sort(key=lambda r: r["shard"])
    (out_dir / "shards" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps({"shard": f"shards/{r['shard']}", "num_samples": r["num_samples"]}) + "\n" for r in manifest))
    (out_dir / "segments.jsonl").write_text("".join(json.dumps(s) + "\n" for s in all_segments))
    for r in results:
        os.unlink(r["tar"])
    tmp.rmdir()
    pos_frames = sum(r["positive_frames"] for r in results)
    neg_frames = sum(r["negative_frames"] for r in results)
    from collections import Counter

    if v2:  # kept / dropped and why, per mover
        used = Counter((s["by"], "kept" if s["kept"] else "dropped") for s in all_segments)
        scores = Counter((s["by"], s["reason"].split(" (")[0]) for s in all_segments)
    else:
        used = Counter((s["by"], s["use"]) for s in all_segments)
        scores = Counter((s["by"], s["score"]) for s in all_segments)
    meta = {
        "dataset": f"selfimprove/{exp_id}/iter_{int(iteration):02d}/s1_training_shard",
        "format": ("openpi RoboCasa System1 webdataset, the fields the production loader reads + provenance (v2 rules)"
                   if v2 else "openpi RoboCasa System1 webdataset (RoboAnnotator build_system1_shards layout) + si_* meta"),
        "episodes": len(episodes), "episodes_used": len(plans_by_ep), "segments": len(all_segments),
        "samples": pos_frames + neg_frames, "positive_samples": pos_frames, "negative_samples": neg_frames,
        "shards": len(manifest), "samples_per_shard": per, "horizon": HORIZON,
        "segment_use": {f"{a}/{b}": c for (a, b), c in sorted(used.items(), key=str)},
        "segment_scores": {f"{a}/{b}": c for (a, b), c in sorted(scores.items(), key=str)},
        "filter": fcfg, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "README.md").write_text(README.format(**{k: meta[k] for k in (
        "dataset", "episodes", "episodes_used", "segments", "samples", "positive_samples", "negative_samples",
        "shards")}))
    return meta


README = """# {dataset}

System1 (openpi pi0.5, RoboCasa) training shards from one self-improvement rollout round: HexaShell episodes
(Astra plans, System1 executes, the harness takes over when System1 fails), scored segment by segment with
Astra as the process reward model (selfimprove/data/prm.py).

- episodes: {episodes} ({episodes_used} with at least one selected segment), segments: {segments}
- samples: {samples} ({positive_samples} positive, {negative_samples} negative) in {shards} shards

Each sample is RoboAnnotator's build_system1_shards layout (current + anchor JPEGs, arrays.npz, meta.json) plus
`si_label` (+1 positive: AWR-weighted SL + unary-positive; -1 negative: unary-negative only), `si_weight` (AWR),
`si_score`, `si_source` (s1 | harness), `si_judge`, `si_ref_loss`. Train with openpi's `--si.enabled` and pool
these shards with the base data (`ROBOCASA_SHARDS_DIR=<base>,<these>*K`). segments.jsonl lists every segment of
the round with its score and use, selected or not.
"""

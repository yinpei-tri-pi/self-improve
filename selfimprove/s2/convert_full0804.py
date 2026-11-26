"""The old System2 set (system2_full_0804: 1.1M records from 30,931 pretrain demos) -> the new single-mode S2 rows.

    .venv/bin/python -m selfimprove.s2.convert_full0804 --out ~/data/old_s2/full0804_v2 [--workers 8] [--limit N]

What is converted (the rest is dropped):

- ``plan_exec_v2 · execution`` and the ``failure_v2`` pairs (failure+recovery, incomplete+continue): one new row each.
  The old two-level plan becomes the milestone view: ``[x]`` milestones done, the ``[~]`` one ongoing, ``[ ]`` to do;
  the ``[x]`` fine steps of the ongoing milestone are its steps (done); its ``[~]`` fine step is the last step (the
  video), listed last as ongoing.
  The recovery of a failure pair (role y) shows the failed try in the steps and "X again" as the last step; the
  continuation of an incomplete+continue pair (role y) the partial try and "continue to X". Rows whose status says
  success without a task_finish target are dropped (they contradict the system prompt). The
  target keeps the old thought and judge; the milestone is the ``[~]`` one after ``<plan_update>``; the decision is
  always the model (the demos had no expert); the subgoal as it was. ``task_finish`` rows become closing rows
  (status success, ``<thought>`` + ``<judge>`` only). Rows without a ``[~]`` fine step (the old plan does not say
  which step ran) are dropped, as are videos under rows_v2.MIN_FRAMES frames.
- The first turn (``execution · 01_unroll``, the opening scene): the new first row; the thought is the cold plan's
  (``plan_exec_v2 · no memory``, the milestones' rationale) followed by the unroll's (the first step), the milestones
  the plan it was handed.
- Dropped: ``summary_v2`` (no counterpart), the plans with recalled recipes, the ``__rNN`` repeats of failure records
  (failure_repeat 10), composite_unseen tasks (none expected: the set is pretrain only).

The media bytes pass through unchanged (h264 768x256 at 4 fps, as rows_v2 writes; tiled 768x256 jpg for the scene).
Phase 1 converts each source shard (streamed from S3, deleted after) into one output shard and puts the first-turn
candidates aside; phase 2 joins them with their cold plans and appends them across the output shards. Output: the
``system2-shards`` layout (``shards/*.tar``, ``manifest.jsonl``, ``meta.json``) that sys2_train_eval's plugin reads.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import subprocess
import tarfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = "s3://tri-ml-datasets-uw2/yinpeidai/robocasa_final_training_shard/system2_full_0804"
_LINE = re.compile(r"^(\s*)[-*]\s*\[( |~|x)\]\s*(M\d+(?:\.\d+)?)\s*:\s*(.*?)\s*$")
_REPEAT = re.compile(r"__r\d\d$")


def parse_plan(text: str) -> list[dict]:
    """'- [x] M1: ...' / '  * [~] M1.2: ...' lines -> [{id, text, mark, steps: [{id, text, mark}]}]."""
    ms: list[dict] = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        _, mark, mid, body = m.groups()
        if "." in mid:
            if ms:
                ms[-1]["steps"].append({"id": mid, "text": body, "mark": mark})
        else:
            ms.append({"id": mid, "text": body, "mark": mark, "steps": []})
    return ms


def apply_update(plan: list[dict], update: list[dict]) -> list[dict]:
    """The plan after ``<plan_update>``: each updated milestone replaced (its fine steps too, when given); new ones
    appended in their order."""
    out = [dict(m, steps=list(m["steps"])) for m in plan]
    pos = {m["id"]: i for i, m in enumerate(out)}
    for u in update:
        if u["id"] in pos:
            cur = out[pos[u["id"]]]
            cur.update(text=u["text"], mark=u["mark"])
            if u["steps"]:
                cur["steps"] = list(u["steps"])
        else:
            pos[u["id"]] = len(out)
            out.append(dict(u, steps=list(u["steps"])))
    return out


def _tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


#: the old thoughts' talk about <estimated_step> (gone from the new target): "..., so let me predict a long duration ..."
_DURATION = re.compile(r"\s*[^.!?]*\bduration\b[^.!?]*[.!?]", re.I)


def clean_thought(text: str | None) -> str:
    return _DURATION.sub("", str(text or "")).strip()


def _goal(user: str) -> str | None:
    m = re.match(r"The goal is:\s*(.*?)\n\n", user, re.S)
    return m.group(1).strip() if m else None


def _status(user: str) -> str:
    m = re.search(r"Current task status:\s*(\w+)", user)
    return {"finished": "success"}.get(m.group(1), m.group(1)) if m else "ongoing"


def _gripper(user: str) -> str:
    m = re.search(r"Current gripper status:\s*(\w+)", user)
    return m.group(1) if m else "unsure"


def _task(meta: dict) -> str:
    parts = str(meta.get("episode_id", "")).split("/")
    return parts[2] if len(parts) > 2 else ""


def convert_turn(rec: dict) -> tuple[dict | None, str]:
    """One old execution record (with a video) -> (the new row, '') or (None, why dropped)."""
    from . import rows_v2 as R
    from ..data.episode import base_step

    meta, (_, user, asst) = rec["meta"], rec["messages"]
    u, a = user["content"], asst["content"]
    goal = _goal(u)
    plan = parse_plan(u.split("The controller just carried out")[0])
    judge = _tag(a, "judge")
    if not goal or not plan or not judge:
        return None, "unparsed"
    if (meta.get("clip_frames") or 0) < R.MIN_FRAMES:
        return None, f"video under {R.MIN_FRAMES} frames"
    cur = next((m for m in plan if m["mark"] == "~"), None)
    last = next((s for s in (cur or {}).get("steps", []) if s["mark"] == "~"), None)
    if cur is None or last is None:
        return None, "no [~] fine step: the step that ran is unknown"
    label = lambda m: f"{m['id']}: {m['text']}"  # noqa: E731
    milestones = [(m["id"], label(m)) for m in plan]
    done = [{"milestone": label(cur), "subgoal": s["text"], "outcome": "done"} for s in cur["steps"] if s["mark"] == "x"]
    last_words = last["text"]
    if str(meta.get("variant", "")).startswith("failure_augmented") and meta.get("role") == "y":
        # the recovery of a failure pair: the failed try in the steps, "X again" as the last step
        done.append({"milestone": label(cur), "subgoal": last["text"], "outcome": "failed"})
        last_words = f"{base_step(last['text']) or last['text']} again"
    elif meta.get("variant") == "incomplete" and meta.get("role") == "y":
        # the continuation of an incomplete+continue pair (its video is the continuation): the partial try in the
        # steps, "continue to X" as the last step - as S2 sees a continuation at inference
        done.append({"milestone": label(cur), "subgoal": last["text"], "outcome": "partial"})
        last_words = f"continue to {base_step(last['text']) or last['text']}"
    # the old plan's [x] marks after the current milestone (rare): keep the order, checklist_text decides by position
    view = R.checklist_text(done, label(cur), milestones, last_words)
    status = _status(u)
    if status == "success" and judge != "task_finish":
        return None, "status success without task_finish (contradicts the system prompt)"
    user_new = R.user_turn(goal, view, last_words, "s1", _gripper(u), status)
    thought = clean_thought(_tag(a, "thought"))
    out_meta = {"episode_id": meta.get("episode_id"), "task": _task(meta), "source": "full0804",
                "old_sub_mode": meta.get("sub_mode"), "old_variant": meta.get("variant"),
                "clip_frames": meta.get("clip_frames"), "clip_fps": meta.get("clip_fps")}
    if judge == "task_finish":
        target = R.target_closing(R.natural(thought), "task_finish")
        user_new = R.user_turn(goal, view, last_words, "s1", _gripper(u), "success")
        out_meta.update(kind="closing", decision=None, task_status="success")
        labels = {"judge": judge, "decision": None, "milestone": None, "subgoal": None}
    else:
        after = apply_update(plan, parse_plan(_tag(a, "plan_update") or ""))
        now = next((m for m in after if m["mark"] == "~"), None) or next((m for m in after if m["mark"] == " "), None)
        subgoal = _tag(a, "subgoal")
        if now is None or not subgoal:
            return None, "no current milestone or subgoal after the update"
        target = R.target_turn(thought, judge, label(now), "s1", subgoal)
        out_meta.update(kind="turn", decision="s1", task_status=status)
        labels = {"judge": judge, "decision": "s1", "milestone": label(now), "subgoal": subgoal}
    return {"messages": [{"role": "system", "content": R.SYSTEM}, {"role": "user", "content": user_new},
                         {"role": "assistant", "content": target}], "meta": out_meta, "labels": labels}, ""


def convert_first(rec: dict, cold_thought: str | None) -> tuple[dict | None, str]:
    """The old first turn (01_unroll: the handed plan + the opening scene) -> the new first row."""
    from . import rows_v2 as R

    meta, (_, user, asst) = rec["meta"], rec["messages"]
    u, a = user["content"], asst["content"]
    goal = _goal(u)
    plan = parse_plan(u)
    after = apply_update(plan, parse_plan(_tag(a, "plan_update") or ""))
    now = next((m for m in after if m["mark"] == "~"), None) or (after[0] if after else None)
    subgoal = _tag(a, "subgoal")
    if not goal or not plan or now is None or not subgoal:
        return None, "unparsed first turn"
    thought = " ".join(x for x in (clean_thought(cold_thought), clean_thought(_tag(a, "thought"))) if x)
    label = lambda m: f"{m['id']}: {m['text']}"  # noqa: E731
    target = R.target_turn(thought, "task_begin", label(now), "s1", subgoal, "\n".join(label(m) for m in plan))
    return {"messages": [{"role": "system", "content": R.SYSTEM}, {"role": "user", "content": R.user_first(goal)},
                         {"role": "assistant", "content": target}],
            "meta": {"episode_id": meta.get("episode_id"), "task": _task(meta), "source": "full0804",
                     "old_sub_mode": meta.get("sub_mode"), "kind": "first", "decision": "s1",
                     "task_status": "begin", "cold_plan_thought": cold_thought is not None},
            "labels": {"judge": "task_begin", "decision": "s1", "milestone": label(now), "subgoal": subgoal}}, ""


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0
    tar.addfile(info, io.BytesIO(data))


def _groups(path: Path):
    with tarfile.open(path, "r|*") as tar:
        cur, group = None, {}
        for m in tar:
            if not m.isfile():
                continue
            key, suffix = m.name.split(".", 1)
            if cur is not None and key != cur:
                yield cur, group
                group = {}
            cur = key
            group[suffix] = tar.extractfile(m).read()
        if group and cur is not None:
            yield cur, group


def phase1(job: tuple[str, str, str]) -> dict:
    """One source shard -> shards/<name>.tar (converted turns) + pending/<name>.tar (first turns) + cold thoughts."""
    from .. import tasks as T

    shard, out, tmp = job
    out, tmp = Path(out), Path(tmp)
    name = Path(shard).name
    done_marker = out / "pending" / f"{name}.done.json"
    if done_marker.exists():  # resumable
        return json.loads(done_marker.read_text())
    local = tmp / name
    subprocess.run(["aws", "s3", "cp", f"{SRC}/{shard}", str(local), "--quiet"], check=True,
                   env={**__import__("os").environ, "AWS_PROFILE": "sagemaker"})
    held = T.held_out_tasks()
    stats = collections.Counter()
    tasks = collections.Counter()
    cold: dict[str, str] = {}
    n_out = 0
    with tarfile.open(out / "shards" / f"{name}.partial", "w") as dst, \
            tarfile.open(out / "pending" / f"{name}.partial", "w") as pend:
        for key, g in _groups(local):
            if "json" not in g:
                continue
            rec = json.loads(g["json"])
            sub = str(rec["meta"].get("sub_mode", ""))
            if _REPEAT.search(key):
                stats["drop: a failure_repeat copy"] += 1
                continue
            if _task(rec["meta"]) in held:
                stats["drop: composite_unseen task"] += 1
                continue
            if sub.endswith("no memory"):
                cold[rec["meta"]["episode_id"]] = _tag(rec["messages"][2]["content"], "thought") or ""
                stats["cold plan (joined to the first turn)"] += 1
                continue
            if sub.startswith("summary_v2") or sub.endswith("with memory"):
                stats[f"drop: {sub.split(' · ')[0]}{' · with memory' if sub.endswith('with memory') else ''}"] += 1
                continue
            if "videobin" not in g:
                if "imagebin0" in g and rec["meta"].get("judge") == "task_begin":
                    _add(pend, f"{key}.json", g["json"])
                    _add(pend, f"{key}.imagebin0", g["imagebin0"])
                    stats["first turn (pending)"] += 1
                else:
                    stats["drop: no video"] += 1
                continue
            row, why = convert_turn(rec)
            if row is None:
                stats[f"drop: {why}"] += 1
                continue
            _add(dst, f"{key}.json", json.dumps(row).encode())
            _add(dst, f"{key}.videobin", g["videobin"])
            stats[f"row: {row['meta']['kind']} {row['labels']['judge']}"] += 1
            tasks[row["meta"]["task"]] += 1
            n_out += 1
    (out / "shards" / f"{name}.partial").rename(out / "shards" / name)
    (out / "pending" / f"{name}.partial").rename(out / "pending" / name)
    local.unlink()
    res = {"shard": name, "rows": n_out, "stats": dict(stats), "cold": cold, "tasks": dict(tasks)}
    done_marker.write_text(json.dumps(res))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N source shards (a check)")
    args = ap.parse_args()
    out = args.out.expanduser()
    for d in ("shards", "pending", "_tmp"):
        (out / d).mkdir(parents=True, exist_ok=True)
    man = subprocess.run(["aws", "s3", "cp", f"{SRC}/manifest.jsonl", "-"], check=True, capture_output=True, text=True,
                         env={**__import__("os").environ, "AWS_PROFILE": "sagemaker"}).stdout
    shards = [json.loads(x)["shard"] for x in man.splitlines() if x.strip()]
    if args.limit:
        shards = shards[:args.limit]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(phase1, [(s, str(out), str(out / "_tmp")) for s in shards]), 1):
            results.append(r)
            if i % 25 == 0 or i == len(shards):
                print(f"phase 1: {i}/{len(shards)} shards, {time.time() - t0:.0f}s", flush=True)
    # phase 2: the first rows, joined with their cold plans, appended across the output shards
    cold = {k: v for r in results for k, v in r["cold"].items()}
    names = [r["shard"] for r in results]
    first_rows: dict[str, list] = collections.defaultdict(list)
    stats = collections.Counter()
    for r in results:
        stats.update(r["stats"])
    for name in names:
        for key, g in _groups(out / "pending" / name):
            rec = json.loads(g["json"])
            row, why = convert_first(rec, cold.get(rec["meta"]["episode_id"]))
            if row is None:
                stats[f"drop: {why}"] += 1
                continue
            stats["row: first task_begin" + ("" if row["meta"]["cold_plan_thought"] else " (no cold plan)")] += 1
            dest = names[zlib.crc32(key.encode()) % len(names)]
            first_rows[dest].append((key, json.dumps(row).encode(), g["imagebin0"]))
    counts = {r["shard"]: r["rows"] for r in results}
    for dest, items in first_rows.items():
        with tarfile.open(out / "shards" / dest, "a") as tar:
            for key, js, img in items:
                _add(tar, f"{key}.json", js)
                _add(tar, f"{key}.imagebin0", img)
        counts[dest] += len(items)
    for name in names:
        (out / "pending" / name).unlink(missing_ok=True)
    (out / "manifest.jsonl").write_text("".join(json.dumps({"shard": f"shards/{n}", "num_samples": counts[n]}) + "\n"
                                                for n in names))
    per_task = collections.Counter()
    for r in results:
        per_task.update(r.get("tasks", {}))
    for items in first_rows.values():
        per_task.update(json.loads(js)["meta"]["task"] for _, js, _ in items)
    per_kind = {k.split(": ", 1)[1]: v for k, v in stats.items() if k.startswith("row: ")}
    meta = {"dataset": "selfimprove/s2_base/full0804_v2",
            "format": "system2-shards (sys2_train_eval): <key>.json {messages, meta, labels} + <key>.videobin | "
                      "<key>.imagebin0; the new single-mode S2 rows (selfimprove/s2/rows_v2.py)",
            "source": SRC, "num_shards": len(names), "num_samples": sum(counts.values()),
            "rows": per_kind, "dropped": {k[6:]: v for k, v in stats.items() if k.startswith("drop: ")},
            "cold_plans": len(cold), "clip": {"fps": 4.0, "member": "videobin", "min_frames": 4,
                                              "loader_max_frames": 32, "size": "768x256"},
            "per_task": dict(per_task), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("num_shards", "num_samples", "rows", "dropped", "cold_plans")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

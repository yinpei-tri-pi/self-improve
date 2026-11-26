"""Build ``site/formats/``: a static page that sets our S1 and S2 training data side by side with the rollout records
the loop now writes, from real samples, as the basis for deciding how rollouts become training data.

    .venv/bin/python scripts/build_format_site.py --rollouts <local episode record> [...] --s2-shard <local S2 tar>
        [--s1-sample <cache dir>] [--out site/formats]

- S1: the first sample of the base set (system1_fullset_0717 shard 0, streamed from S3 once into ``--s1-sample``):
  its six images, the prompt openpi renders from it (state digits included: the quantile norm stats of
  openpi-si/assets, then 256 bins), its arrays and meta, each marked used / unused by the production loader.
- S2: one record of each mode from a full0804 shard (PLAN, EXECUTION, a failure+recovery pair, SUMMARY).
- Rollout: one HexaShell episode record (``rollout/<task>/<seed>/`` downloaded): its files, the per-step arrays, the
  turns (who moved, subgoal, judgement, System1's stop), and what our preprocessors make of it today: an S1
  sample from an S1 step and from a harness step, and a draft S2 turn with its clip.
- The field-by-field mappings and the open decisions.

Open ``site/formats/index.html`` in a browser (all assets are relative files).
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402
from selfimprove.data.episode import load_episode  # noqa: E402
from selfimprove.data.trajectory import CAMS, estimated_length  # noqa: E402

S1_SHARD = ("s3://tri-ml-datasets-uw2/yinpeidai/robocasa_final_training_shard/system1_fullset_0717/shards/"
            "shard_w00_000000.tar")
NORM = Path("~/openpi-si/assets/pi05_robocasa_system1/robocasa_system1/norm_stats.json").expanduser()
CAM_ORDER = ("left", "right", "wrist")
S1_IMG = {"left": "scene_left", "right": "scene_right", "wrist": "wrist"}

# what openpi's production loader (pi05_robocasa_system1 + the three flags) reads of a base-set sample
S1_META_USE = {
    "task_goal": "prompt: Task", "subgoal": "prompt: Current Subgoal", "quality": "prompt: Quality",
    "est_length": "prompt: Estimated Length", "gripper_flag": "prompt: Current Gripper",
    "base_pos_ref": "lean state (base xy relative to the episode start)",
    "base_yaw_ref": "lean state (yaw relative to the episode start)", "span": "progress target",
    "frame_index": "progress target", "subgoal_progress_frac": "progress target",
    "episode_id": "progress overlay key", "variant": "progress overlay key",
}
S1_META_UNUSED = {
    "executed_step": "dropped by --data.no-include-executed-step (still sent at inference, ignored)",
    "prompt": "stale placeholder string; the loader rebuilds the prompt", "subgoal_detail": "never read",
    "milestone_text": "progress-head era", "milestone_span": "progress-head era",
    "milestone_progress_frac": "progress-head era", "milestone_progress_class": "progress-head era",
    "progress_classes": "progress-head era (10-way classes)", "subgoal_progress_class": "progress-head era",
    "progress_frac": "duplicate of subgoal_progress_frac", "anchor_key": "old anchor lookup",
    "episode_len": "never read", "image_base": "provenance", "image_root": "provenance",
    "leak_lead": "retired", "leak_halt": "retired", "fail_end": "retired", "frame_kind": "never read",
}
S1_ARR_USE = {
    "state": "raw 16-d current: lean state recomputed from it", "state_anchor": "raw 16-d anchor",
    "lean_state": "baked lean state: only for the drift assert", "lean_state_anchor": "only for the drift assert",
    "lean_action_subgoal_pad": "THE action label (20 x 11)", "action_pad_mask_subgoal": "progress target mask",
}
S1_ARR_UNUSED = {
    "last_action": "never read", "last_action_lean": "never read", "action_subtask": "raw 12-d, pad ablation",
    "action_subtask_mask": "pad ablation", "lean_action_subtask": "episode-padded, pad ablation",
    "action_subgoal_pad": "raw 12-d version of the label", "action_pad_mask_episode": "pad ablation",
}
LEAN_STATE = ["eef x", "eef y", "eef z", "eef rot6d 1", "eef rot6d 2", "eef rot6d 3", "eef rot6d 4", "eef rot6d 5",
              "eef rot6d 6", "gripper width", "base dx (from start)", "base dy (from start)", "sin yaw (from start)",
              "cos yaw (from start)"]
RAW16 = ["base x", "base y", "base z", "base qx", "base qy", "base qz", "base qw", "eef x (base frame)", "eef y", "eef z",
         "eef qx", "eef qy", "eef qz", "eef qw", "finger 1", "finger 2"]
LEAN_ACT = ["base vx", "base vy", "base yaw", "mode", "eef dx", "eef dy", "eef dz", "eef drx", "eef dry", "eef drz",
            "gripper"]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def badge(kind: str) -> str:
    return f"<span class='b {kind}'>{kind}</span>"


def table(header: list[str], rows: list[list], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def pre(text, cls: str = "") -> str:
    return f"<pre class='{cls}'>{esc(text)}</pre>"


def details(summary: str, body: str, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{summary}</summary>{body}</details>"


# -- the S1 prompt, as openpi renders it ------------------------------------------------------------------
def s1_prompt(task: str, subgoal: str, quality: str, est: int, cur14, anc14, grip: str, norm: dict) -> str:
    """build_prompt (include_task_goal, conditioning quality + est_length) + the pi05 tokenizer's state block:
    [current14, anchor14] quantile-normalized together, 256 bins, the anchor rendered first."""
    task = str(task).strip().lower().replace("\n", " ").rstrip(".!; ")
    sub = str(subgoal).strip().lower().replace("\n", " ")
    text = f"{task}; Current Subgoal: {sub}\nQuality: {quality}; Estimated Length: {int(est)}".strip().replace("_", " ")
    q01, q99 = np.asarray(norm["q01"], float), np.asarray(norm["q99"], float)
    state = np.concatenate([np.asarray(cur14, float), np.asarray(anc14, float)])
    x = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    d = np.clip(np.digitize(x, bins=np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255)
    return (f"Task: {text}\nInitial State: {' '.join(map(str, d[14:]))}; "
            f"Current State: {' '.join(map(str, d[:14]))}; Current Gripper: {grip};\nAction: ")


def state_digits(cur14, anc14, norm: dict) -> np.ndarray:
    q01, q99 = np.asarray(norm["q01"], float), np.asarray(norm["q99"], float)
    x = (np.concatenate([np.asarray(cur14, float), np.asarray(anc14, float)]) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return np.clip(np.digitize(x, bins=np.linspace(-1, 1, 257)[:-1]) - 1, 0, 255)


def chunk_table(lean20x11: np.ndarray, progress: np.ndarray | None, n: int = 8) -> str:
    rows = []
    for i in range(min(n, len(lean20x11))):
        r = [str(i)] + [f"{v:+.3f}" for v in lean20x11[i]]
        if progress is not None:
            r.append(f"{progress[i]:+.3f}")
        rows.append(r)
    head = ["step"] + LEAN_ACT + (["progress (2p-1)"] if progress is not None else [])
    return table(head, rows, "num") + f"<p class='note'>first {min(n, len(lean20x11))} of {len(lean20x11)} rows</p>"


# -- S1 base sample ------------------------------------------------------------------------------------------
def fetch_s1_sample(cache: Path) -> Path:
    if (cache / "meta.json").exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    bucket, key = s3.split(S1_SHARD)
    body = s3.client().get_object(Bucket=bucket, Key=key)["Body"]
    first = None
    with tarfile.open(fileobj=body, mode="r|") as tf:
        for m in tf:
            k, ext = m.name.split(".", 1)
            first = first or k
            if k != first:
                break
            (cache / ext).write_bytes(tf.extractfile(m).read())
    (cache / "key.txt").write_text(first)
    body.close()
    return cache


def section_s1(src: Path, assets: Path, norm: dict) -> str:
    meta = json.loads((src / "meta.json").read_text())
    with np.load(src / "arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    for name in ("scene_left", "scene_right", "wrist"):
        for pre_ in ("", "anchor_"):
            shutil.copy(src / f"{pre_}{name}.jpg", assets / f"s1_{pre_}{name}.jpg")
    imgs = "".join(
        f"<figure><img src='assets/s1_{p}{n}.jpg'><figcaption>{lab} - {n.replace('scene_', '')}</figcaption></figure>"
        for p, lab in (("", "current (frame t)"), ("anchor_", "anchor (span start)")) for n in S1_IMG.values())
    prompt = s1_prompt(meta["task_goal"], meta["subgoal"], meta["quality"], meta["est_length"], arr["lean_state"],
                       arr["lean_state_anchor"], meta["gripper_flag"], norm)
    s, e = (meta.get("span") or [None, None])[:2]
    t = meta.get("frame_index")
    prog = None
    if s is not None and t is not None:
        frac0 = min(1.0, max(0.0, (t - s) / max(1, e - s)))
        p = np.minimum(1.0, frac0 + np.arange(20) / max(1, e - s))
        p[np.arange(20) + t > e] = 1.0
        prog = 2 * p - 1
    meta_rows = []
    for k in sorted(meta):
        v = meta[k]
        sv = json.dumps(v) if not isinstance(v, str) else v
        if k in S1_META_USE:
            meta_rows.append([f"<code>{esc(k)}</code>", esc(sv[:160]), badge("used"), esc(S1_META_USE[k])])
        else:
            meta_rows.append([f"<code>{esc(k)}</code>", esc(sv[:160]), badge("unused"), esc(S1_META_UNUSED.get(k, ""))])
    arr_rows = []
    for k, a in arr.items():
        use = k in S1_ARR_USE
        arr_rows.append([f"<code>{esc(k)}</code>", esc(f"{tuple(a.shape)} {a.dtype}"), badge("used" if use else "unused"),
                         esc(S1_ARR_USE.get(k) or S1_ARR_UNUSED.get(k, ""))])
    key = (src / "key.txt").read_text().strip()
    digits = state_digits(arr["lean_state"], arr["lean_state_anchor"], norm)
    st_rows = [[esc(n), f"{arr['lean_state'][i]:+.4f}", str(digits[i]), f"{arr['lean_state_anchor'][i]:+.4f}", str(digits[14 + i])]
               for i, n in enumerate(LEAN_STATE)]
    raw_rows = [[esc(n), f"{arr['state'][i]:+.4f}", f"{arr['state_anchor'][i]:+.4f}"] for i, n in enumerate(RAW16)]
    io_panel = f"""<div class='io'><div class='in'><h3>INPUT</h3>
<h4>images (6, 224 px): current frame t and the anchor (the subgoal span's first frame), 3 cameras each</h4>
<div class='grid3 small'>{imgs}</div>
<h4>state: lean 14-d for current and anchor, normalized (checkpoint quantile stats) and cut into 256 bins, written into the prompt</h4>
{table(['lean dim', 'current', 'bin', 'anchor', 'bin'], st_rows, 'num')}
{details('raw 16-d state shipped with the sample (the lean state is recomputed from it)', table(['raw dim', 'current', 'anchor'], raw_rows, 'num'))}
<h4>prompt (text tokens)</h4>{pre(prompt, 'prompt')}
{table(['prompt field', 'value here', 'from'], [
        ['Task', esc(meta['task_goal']), 'the episode goal, lowercased'],
        ['Current Subgoal', esc(meta['subgoal']), 'the Gemini fine-step text of this span'],
        ['Quality', esc(meta['quality']), 'Success / Suboptimal; always Success at inference'],
        ['Estimated Length', esc(meta['est_length']), 'span length snapped to the grid 50..1300'],
        ['Initial State / Current State', '28 bins (above)', 'anchor first, then current'],
        ['Current Gripper', esc(meta['gripper_flag']), 'sign of the previous gripper command'],
    ])}</div>
<div class='out'><h3>OUTPUT (the training target)</h3>
<h4>action chunk: 20 future steps x 11 lean dims (+ the progress dim)</h4>
{chunk_table(arr['lean_action_subgoal_pad'], prog, n=20)}
{table(['lean action dim', 'meaning (official 12-d action, torso dropped)'], [
        ['base vx, vy, yaw', 'base joint velocities (normalized)'], ['mode', '+1 base driving (arm held), -1 arm motion'],
        ['eef dx, dy, dz', 'arm OSC position delta, robot-base frame (+-1 = +-5 cm)'],
        ['eef drx, dry, drz', 'arm OSC rotation delta (+-1 = +-0.5 rad)'], ['gripper', '-1 open, +1 close (held)'],
        ['progress (2p-1)', 'how far through the subgoal: p ramps to 1 at the span end; at inference S1 stops at p >= 0.95'],
    ])}
<p class='note'>Past the span end the chunk is a settle pad (velocities 0, mode / gripper held). Loss: flow matching on the
(normalized) chunk. At inference HexaShell executes 16 of the 20 actions per query.</p></div></div>"""
    return f"""
<section id='s1'><h2>1. S1 training sample (base set <code>system1_fullset_0717</code>)</h2>
<p>One sample = one control step <i>t</i> of an annotated subgoal span [s, e] of a demo, plus one halt frame
after each span. 34.2M samples from 30,897 demo episodes (16,736 shards of 2,048, 2.15 TB). Sample <code>{esc(key)}</code>.</p>
{io_panel}
<h3>Everything stored in the sample (what the loader reads, and what it never reads)</h3>
{table(['arrays.npz', 'shape', '', 'note'], arr_rows)}
{details('meta.json: every field', table(['meta.json', 'value', '', 'note'], meta_rows), True)}
</section>"""


# -- S2 base samples -----------------------------------------------------------------------------------------
def video_strip(mp4: Path, out: Path, n: int = 6, width: int = 384) -> int:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), f"{tmp}/f%04d.png"], check=True)
        frames = sorted(Path(tmp).glob("f*.png"))
        if not frames:
            return 0
        pick = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(n, len(frames)))]
        ims = [Image.open(f).convert("RGB") for f in pick]
        h = int(ims[0].height * width / ims[0].width)
        cols = 2
        rows_ = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * width + (cols - 1) * 4, rows_ * h + (rows_ - 1) * 4), "white")
        for i, im in enumerate(ims):
            sheet.paste(im.resize((width, h)), ((i % cols) * (width + 4), (i // cols) * (h + 4)))
        sheet.save(out, quality=88)
        return len(frames)


def s2_text(msg) -> str:
    c = msg["content"]
    if isinstance(c, list):
        return "".join(x.get("text") if x.get("type") == "text" else f"<{x.get('type')}>" for x in c)
    return str(c)


S2_TAG_USE = {"thought": "logged only", "plan": "kept as the plan checklist (the only memory between turns)",
              "judge": "task_finish ends the episode; subgoal_failed read by 2 task rules; complete vs incomplete unused",
              "plan_update": "merged into the plan", "estimated_step": "S1's budget min(400, 2 est) and length tag (rules override)",
              "subgoal": "S1's prompt (Current Subgoal)", "subgoal_detail": "unused", "narration": "unused by the default loop",
              "summary": "unused by the default loop"}
S2_MODE_WHEN = {"PLAN (cold)": "called once at the episode start", "EXECUTION": "called every turn after a step ran",
                "EXECUTION, failure + recovery pair": "EXECUTION records built from injected failures and their recoveries",
                "SUMMARY (narrate)": "not called by the default eval loop"}


def section_s2(shard: Path, assets: Path) -> str:
    want = [("PLAN (cold)", "plan_exec_v2 · no memory"), ("EXECUTION", "plan_exec_v2 · execution"),
            ("EXECUTION, failure + recovery pair", "failure_v2 · failure+recovery [early_grasp] · pair 1y"),
            ("SUMMARY (narrate)", "summary_v2")]
    tf = tarfile.open(shard)
    keys: dict[str, list[str]] = {}
    for n in tf.getnames():
        k, ext = n.split(".", 1)
        keys.setdefault(k, []).append(ext)
    cards = []
    for title, sub in want:
        for k, exts in keys.items():
            rec = json.loads(tf.extractfile(f"{k}.json").read())
            if rec["meta"].get("sub_mode") != sub:
                continue
            slug = sub.split(" ")[0].replace("_v2", "") + ("_pair" if "pair" in sub else "")
            media = ""
            if "imagebin0" in exts:
                (assets / f"s2_{slug}.jpg").write_bytes(tf.extractfile(f"{k}.imagebin0").read())
                media = f"<img class='tile' src='assets/s2_{slug}.jpg'><p class='note'>&lt;image&gt;: one 256x768 tile (left | right | wrist)</p>"
            if "videobin" in exts:
                mp4 = assets / f"s2_{slug}.mp4"
                mp4.write_bytes(tf.extractfile(f"{k}.videobin").read())
                nf = video_strip(mp4, assets / f"s2_{slug}_strip.jpg")
                media = (f"<video src='assets/s2_{slug}.mp4' controls loop muted class='tile'></video>"
                         f"<p class='note'>&lt;video&gt;: {nf} frames at {esc(rec['meta'].get('clip_fps'))} fps, 256x768 tiles; "
                         f"some frames:</p><img class='strip' src='assets/s2_{slug}_strip.jpg'>")
            sysm, user, asst = (s2_text(m) for m in rec["messages"])
            meta = {kk: rec["meta"][kk] for kk in ("sub_mode", "episode_id", "clip_frames", "clip_fps", "variant")
                    if kk in rec["meta"]}
            tags = re.findall(r"<(\w+)>(.*?)</\1>", asst, re.S)
            tag_rows = [[f"&lt;{esc(t)}&gt;", pre(v.strip()), esc(S2_TAG_USE.get(t, ""))] for t, v in tags]
            cards.append(f"""<div class='card'><h3>{esc(title)}</h3>
<p class='note'><code>{esc(k)}</code> - {esc(S2_MODE_WHEN.get(title, ''))}</p>
<div class='io'><div class='in'><h3>INPUT</h3>{details('system prompt (the same for every record of this mode)', pre(sysm))}
<h4>user message</h4>{pre(user)}<h4>visual input</h4>{media}
{details('record meta', pre(json.dumps(meta, indent=1, ensure_ascii=False)))}</div>
<div class='out'><h3>OUTPUT (the training target)</h3>{pre(asst, 'target')}
<h4>the target, field by field</h4>{table(['tag', 'value', 'used by the eval loop'], tag_rows)}</div></div></div>""")
            break
    used = table(["output tag", "values", "used by the eval loop?"], [
        ["&lt;plan&gt; (PLAN)", "milestone checklist M1..Mn", badge("used") + " the only memory between turns"],
        ["&lt;thought&gt;", "free text", badge("unused") + " logged only"],
        ["&lt;judge&gt;", "task_begin, subgoal_complete, subgoal_incomplete, subgoal_failed, task_finish",
         badge("partly") + " task_finish ends the episode (never right: the loop stops on env success first); "
         "subgoal_failed read by 2 task rules; complete vs incomplete never branched on"],
        ["&lt;plan_update&gt;", "checklist lines by milestone id", badge("used") + " merged into the plan"],
        ["&lt;estimated_step&gt;", "50..1300 grid", badge("partly") + " S1 budget min(400, 2 est) + S1's length tag; "
         "steering rules override (max of proposals)"],
        ["&lt;subgoal&gt;", "S1-style fine step", badge("used") + " S1's prompt"],
        ["&lt;subgoal_detail&gt;", "longer description", badge("unused")],
        ["&lt;narration&gt; / &lt;summary&gt; (SUMMARY)", "", badge("unused") + " default loop never calls it"],
    ])
    return f"""
<section id='s2'><h2>2. S2 training samples (<code>system2-full-0804</code>, Qwen3-VL-4B)</h2>
<p>1.10M records from 30,931 demo episodes on 299 pretrain tasks: PLAN 62k (5.6%), EXECUTION 640k (58%: normal 389k,
incomplete+continue pairs 190k, failure+recovery 62k = 6.2k unique x10), SUMMARY 397k (36%). Each record is one
[system, user, assistant] triple; no chat history. Privileged inputs: <code>Current task status</code> (ongoing/finished, from the
env's success) and <code>Current gripper status</code> (from the gripper commands).</p>
{''.join(cards)}
<h3>Which outputs the eval loop uses</h3>{used}
<p class='note'>Also: about 22 task rules plus general rules (repeat caps, regrasp recovery, estimate pins, plan inserts) rewrite
S2's output at inference. Train vs inference mismatches: clip frame sampling (linspace vs static-drop + every 5th),
goal casing (lowercased vs raw).</p>
</section>"""


# -- our rollout ---------------------------------------------------------------------------------------------
def tile(traj, t: int, size: int = 128):
    from PIL import Image

    ims = [Image.fromarray(traj.image(t, cam)).resize((size, size)) for cam in CAM_ORDER]
    out = Image.new("RGB", (size * 3, size))
    for i, im in enumerate(ims):
        out.paste(im, (i * size, 0))
    return out


def folder_sizes(folder: Path) -> list[list]:
    rows = []
    for p in sorted(folder.iterdir()):
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        rows.append([f"<code>{esc(p.name)}{'/' if p.is_dir() else ''}</code>", f"{size / 1e6:.2f} MB", str(len(files))])
    return rows


ROLLOUT_FILES = {
    "trajectory/": "every control step (20 Hz): 3 cameras 256x256 JPEG q95, raw16 state, 12-d action (robosuite order), "
                   "gripper level, camera poses/focals, full MuJoCo state; once: model.xml.gz + scene.json",
    "events.jsonl": "the episode's events: per turn the judge (Astra's verdict on the last step + thought), subgoal (text, by "
                    "s1|agent), System1 runs (est_length, stop, steps, progress), robot calls",
    "cells.jsonl": "every code cell: Astra's thought, code, output, timings, steps before/after",
    "transcript.json": "the full conversation with Astra (requests and replies)",
    "pictures/": "every picture Astra was shown (render size)", "frames/": "the 3 cameras after each cell that moved the robot",
    "prompt.md": "the system prompt + task card Astra saw", "result.json": "outcome, steps, tokens, settings, the recorded episode started",
    "rollout.json": "the node's summary (status, timing, scores, commits)",
    "chunk_scores.jsonl": "S1's surrogate loss on every 16th executed chunk (S1's and the harness's)",
    "states.jsonl": "the robot's state after each cell", "cells/": "the cells as .py files", "server.log": "simulator log",
    "worker.err": "sandbox stderr",
}


S1_MAP = [
    ["6 images (current, anchor)", "trajectory JPEGs at t and at the span start (256 px, as the base set)", "direct"],
    ["raw16 state, base refs, lean state", "trajectory raw16; refs = frame 0 of the episode; lean recomputed", "direct"],
    ["Task", "result.json instruction (the recording's ep_meta lang)", "direct"],
    ["Current Subgoal", "the turn's subgoal (Astra words its own steps in S1's vocabulary too)", "direct decide"],
    ["Quality", "Success for every kept span (no negatives are trained)", "decided"],
    ["Estimated Length", "the grid value at or below the span's length, at least 50 (S1's own stop can make runs longer)", "derived decided"],
    ["Current Gripper", "sign of the previous gripper command", "derived"],
    ["20 x 11 actions + settle pad + halt", "executed actions t..e, reordered; padded_chunk / halt_chunk (RoboAnnotator math)", "derived"],
    ["progress (2p-1)", "linear ramp over the span (the base set's linear branch)", "derived"],
    ["which spans are kept", "successful episodes only; S1's completed steps as they are (timeouts dropped); the harness's "
     "completed steps merged per step (cut at 400 steps), kept when smooth (the demos' p99 per primitive), no partial grip, "
     "no whole-body turn; under 8 steps dropped", "decided"],
    ["objective", "the normal training objective (no negatives, no frozen reference); 70% rollout / 30% base data, the base "
     "shards in a new random order each round (a per-round seed)", "decided"],
    ["meta to write", "only what the loader reads + provenance (exp, iteration, task, seed, mover, turns): checked with "
     "openpi's loader", "decided"],
]
S2_MAP = [
    ["goal", "result.json instruction", "direct"],
    ["steps done (memory instead of a plan)", "the subgoals of the turns judged complete so far", "derived decide"],
    ["last step + its clip", "the last turn's subgoal; trajectory frames of its span at 4 fps, 256x768", "derived"],
    ["current scene", "the tile at the turn (first turn: the recording's start)", "derived"],
    ["gripper status", "gripper commands over the last frames (as the base set)", "derived decide"],
    ["task status line", "privileged; made task_finish trivial", "decide"],
    ["&lt;thought&gt;", "Astra's thought (mentions its own tools: rewrite with a short LLM pass?)", "decide"],
    ["&lt;judge&gt;", "Astra's judge of the last step", "direct"],
    ["&lt;decision&gt; s1 | harness", "the harness label: who executed the next step (Astra's own calls = harness). "
     "Open: at a turn where S1 was chosen and then failed, label it s1 (what Astra did) or harness (what would have worked)?",
     "decided decide"],
    ["which turns", "all turns, failures included (S2 learns from S1's failures when to call Astra)", "decided"],
    ["&lt;subgoal&gt;", "Astra's next step, in S1's wording", "direct"],
    ["&lt;estimated_step&gt;", "S1's est_length, or the per-subgoal table at inference", "decide"],
    ["plan / plan_update / subgoal_detail / SUMMARY", "not in rollouts", "decide"],
]


def map_table(rows) -> str:
    return table(["field", "from the rollout", "status"],
                 [[r[0], esc(r[1]) if "&lt;" not in r[1] else r[1], " ".join(badge(b) for b in r[2].split())] for r in rows])


def motion_stats(traj, s: int, e: int) -> dict:
    """Per span: the size of the commanded motion (lean action units: the controller's normalized deltas)."""
    a = np.asarray(traj.lean_act[s:e + 1], float)
    if len(a) < 3:
        return {}
    pos, rot, base = a[:, 4:7], a[:, 7:10], a[:, 0:3]
    jerk = np.linalg.norm(np.diff(pos, n=2, axis=0), axis=1)
    grip = np.sign(a[:, 10])
    return {"modes": sorted({round(float(v), 1) for v in a[:, 3]}), "grips": sorted({round(float(v), 1) for v in a[:, 10]}),
            "base_steps": int((np.abs(base).max(1) > 1e-3).sum()), "arm_steps": int((np.abs(a[:, 4:10]).max(1) > 1e-3).sum()),
            "rot_max": float(np.linalg.norm(rot, axis=1).max()), "rot_mean": float(np.linalg.norm(rot, axis=1).mean()),
            "pos_max": float(np.linalg.norm(pos, axis=1).max()), "jerk_max": float(jerk.max()),
            "base_max": float(np.linalg.norm(base, axis=1).max()), "grip_toggles": int((np.diff(grip) != 0).sum())}


def timeline(segs, n_steps: int, link: bool = False) -> str:
    """Segmented bar of the episode's steps, one segment per step span, colored by who moved (hover: the labels;
    ``link``: a click jumps to the turn's row, ``#turn<N>``)."""
    w, h, top = 1200, 34, 22
    xs = lambda step: 4 + (w - 8) * step / max(1, n_steps)  # noqa: E731
    rects = []
    for sg in segs:
        x0, x1 = xs(sg.start), xs(sg.end + 1)
        color = "var(--s1)" if sg.by == "s1" else "var(--harness)"
        tip = esc(f"turn {sg.turn}: {sg.subgoal}  [{'S1' if sg.by == 's1' else 'harness'}, steps {sg.start}-{sg.end}] "
                  f"judged {sg.judge or '-'}, then {sg.next or '-'}")
        rects.append(f"<rect class='seg' x='{x0 + 1:.1f}' y='{top}' width='{max(1.0, x1 - x0 - 2):.1f}' height='{h}' rx='4' "
                     f"fill='{color}'><title>{tip}</title></rect>")
        mark = {"subgoal_complete": "", "subgoal_incomplete": "~", "subgoal_failed": "x"}.get(sg.judge or "", "")
        label = f"{sg.turn}{mark}" if x1 - x0 > 24 else mark
        if label and x1 - x0 > 10:
            rects.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{top + h / 2 + 4}' class='mk'>{label}</text>")
        if link:  # the segment's rect (and its label) as a link to the turn's row
            k = 2 if label and x1 - x0 > 10 else 1
            rects[-k:] = [f"<a href='#turn{sg.turn}'>{''.join(rects[-k:])}</a>"]
    ticks = "".join(f"<text x='{xs(t):.1f}' y='{top + h + 16}' class='tk'>{t}</text>" for t in range(0, n_steps + 1, 100))
    legend = ("<div class='legend'><span><i style='background:var(--s1)'></i>System1 moved</span>"
              "<span><i style='background:var(--harness)'></i>harness (Astra's own calls) moved</span>"
              "<span>numbers: the turn; ~ judged incomplete, x judged failed (next turn); hover a span for its labels"
              + ("; click it for the turn" if link else "") + "</span></div>")
    return (f"{legend}<svg class='tl' viewBox='0 0 {w} {top + h + 22}' role='img' aria-label='episode timeline'>"
            f"<text x='4' y='14' class='tk'>control step</text>{''.join(rects)}{ticks}</svg>")


def span_flags(traj, s: int, e: int) -> dict:
    """What the S1 filter rules look at in a harness span: a partial grip (the gripper target held between open and
    closed) and whole-body motion (the base moving while the arm moves, with mode -1: a turn about a point)."""
    lev = np.asarray(traj.level[s:e + 1], float)
    a = np.asarray(traj.lean_act[s:e + 1], float)
    held = np.abs(a[:, 10]) < 1e-6 if len(a) else np.zeros(0, bool)
    partial = bool(((np.abs(lev) < 0.98) & held).sum() >= 3)
    base = np.abs(a[:, 0:3]).max(1) > 0.02 if len(a) else np.zeros(0, bool)
    arm = np.abs(a[:, 4:10]).max(1) > 1e-3 if len(a) else np.zeros(0, bool)
    whole = bool(((base & arm) & (a[:, 3] < 0)).sum() >= 5) if len(a) else False
    return {"partial_grip": partial, "whole_body": whole}


def section_rollout(folder: Path, assets: Path, norm: dict, prefix: str = "r") -> tuple[str, str, str, dict]:
    from PIL import Image

    from selfimprove.config import Config
    from selfimprove.data.s1_shards import plan_episode

    rec = load_episode(folder)
    traj = rec.traj
    result = rec.result
    tidx = traj.index
    events = [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines() if x.strip()]
    cells = [json.loads(x) for x in (folder / "cells.jsonl").read_text().splitlines() if x.strip()]
    s1_runs = [e for e in events if e.get("kind") == "s1"]
    cs_path = folder / "chunk_scores.jsonl"  # absent when the round ran without chunk scoring
    chunks = [json.loads(x) for x in cs_path.read_text().splitlines() if x.strip()] if cs_path.exists() else []
    scored, _ = plan_episode(str(folder), Config.load()["filter"])
    verdict = {(sg.start, sg.end): sg for sg in scored}
    rollout = json.loads((folder / "rollout.json").read_text())

    files = [[n, sz, c, esc(ROLLOUT_FILES.get(n.replace("<code>", "").replace("</code>", ""), ""))]
             for n, sz, c in folder_sizes(folder)]
    with np.load(traj.folder / tidx["chunks"][0]) as z:
        arr_rows = [[f"<code>{k}</code>", esc(f"{tuple(z[k].shape)} {z[k].dtype}")] for k in z.files]
    used_runs = set()
    n_flags = {"s1_keep": 0, "partial": 0, "whole": 0}
    from selfimprove.data import s1_select
    s1_spans = s1_select.select(rec)
    kept_spans = [sp for sp in s1_spans if sp.kept]
    span_rows = [[f"{sp.start}-{sp.end}", str(sp.steps), badge("s1" if sp.by == "s1" else "harness"), esc(sp.subgoal),
                  str(sp.est_length), esc(", ".join(map(str, sp.turns))), esc(sp.reason)] for sp in kept_spans]
    seg_rows, detail_cards = [], []
    for sg in rec.segments:
        img = f"{prefix}_turn{sg.index:02d}.jpg"
        tile(traj, sg.start).save(assets / img, quality=85)
        run = None
        if sg.by == "s1":
            run = next((i for i, r in enumerate(s1_runs) if i not in used_runs and r.get("subgoal") == sg.subgoal), None)
            if run is not None:
                used_runs.add(run)
        r = s1_runs[run] if run is not None else {}
        s1 = (f"{sg.s1_calls} call(s); est {r.get('est_length')} (from {r.get('est_from')}); budget {r.get('budget')}; "
              f"stop {r.get('stop')}; steps {r.get('steps')}; progress {r.get('progress')}" if sg.by == "s1" else "")
        losses = [c["loss_real"] for c in chunks if int(c.get("start", -1)) == sg.start and int(c.get("end", -1)) == sg.end]
        loss = f"{np.mean(losses):.3f} ({len(losses)})" if losses else "-"
        v = verdict.get((sg.start, sg.end))
        filt = f"{badge(v.use or 'dropped') if v else ''} {esc(v.score) if v else ''}<br><span class='note'>{esc(v.reason) if v else ''}</span>"
        m = motion_stats(traj, sg.start, sg.end)
        mot = (f"rot max {m['rot_max']:.2f} mean {m['rot_mean']:.2f}<br>pos max {m['pos_max']:.2f}, jerk max {m['jerk_max']:.2f}"
               f"<br>base max {m['base_max']:.2f}; grip toggles {m['grip_toggles']}") if m else "-"
        conv = (f"mode {m['modes']}<br>gripper {m['grips']}<br>base moving {m['base_steps']}, arm moving {m['arm_steps']} steps"
                if m else "-")
        fl = span_flags(traj, sg.start, sg.end)
        dec = [sp for sp in s1_spans if sp.start <= sg.start <= sp.end or sg.start <= sp.start <= sg.end]
        if dec and any(sp.kept for sp in dec):
            kept = [sp for sp in dec if sp.kept]
            s1use = badge("keep") + " " + esc(kept[0].reason) + "".join(
                f"<br><span class='note'>training span {sp.start}-{sp.end} '{esc(sp.subgoal)}', Estimated Length {sp.est_length}</span>"
                for sp in kept)
        elif dec:
            s1use = badge("skip") + " " + esc(dec[0].reason)
        else:
            s1use = "-"
        n_flags["s1_keep"] += "keep" in s1use
        n_flags["partial"] += fl["partial_grip"] and sg.by != "s1"
        n_flags["whole"] += fl["whole_body"] and sg.by != "s1"
        seg_rows.append([str(sg.turn), f"{sg.start}-{sg.end}<br>({sg.steps})", badge("s1" if sg.by == "s1" else "harness"),
                         esc(sg.subgoal), s1, esc(sg.judge), esc(sg.next), loss, mot, conv, s1use, badge("keep"),
                         f"<img class='thumb' src='assets/{img}'>"])
        body = [f"<h4>next turn's thought (Astra's reasoning about how this step went)</h4>{pre(sg.judge_thought or '-')}"]
        if sg.by == "s1" and r.get("prompts"):
            body.append(f"<h4>the prompt System1 actually received (first query)</h4>{pre(r['prompts'][0], 'prompt')}")
        if sg.by != "s1":
            code = "\n\n".join(c.get("code") or "" for c in cells if c.get("turn") == sg.turn)
            body.append(f"<h4>Astra's code for this step (the harness moves)</h4>{pre(code[:3000] or '-')}")
        detail_cards.append(details(f"turn {sg.turn}: {esc(sg.subgoal)} ({'S1' if sg.by == 's1' else 'harness'}, steps "
                                    f"{sg.start}-{sg.end})", "".join(body)))
    ep = {k: result.get(k) for k in ("task", "seed", "split", "instruction", "success", "end", "step", "step_limit", "turns",
                                     "layout_id", "style_id", "demo")}
    ep.update({k: rollout.get(k) for k in ("s1_step_share", "tokens_prompt", "tokens_cached", "tokens_completion", "seconds")})
    ep["chunk_loss_mean"] = (rollout.get("score") or {})
    s1_table_path = Path("~/HexaShell-s1/skills/robocasa_s1").expanduser() / f"{result.get('task')}.md"
    s1_table = s1_table_path.read_text() if s1_table_path.exists() else "-"
    sec_r = f"""
<section id='rollout'><h2>Rollout record (HexaShell, Astra + S1, started from a recorded episode)</h2>
<p>One record = one whole episode: <b>{esc(result.get('task'))}</b>, seed {esc(result.get('seed'))} (recording
<code>{esc((result.get('demo') or {}).get('id'))}</code>), from <code>{esc(rollout.get('exp_id'))}</code>, recorded with the
harness in the demonstrations' action convention.
About 64 MB per episode; 3,600 episodes ~ 230 GB. The unit the filters act on is the <b>step span</b>: the control steps
one turn executed, with one subgoal and one mover (System1 or the harness).</p>
<h3>Episode labels</h3>{pre(json.dumps(ep, indent=1, ensure_ascii=False))}
<h3>Timeline</h3>{timeline(rec.segments, len(traj))}
<h3>Every step span with its labels</h3>
<p class='note'>judged = the next turn's verdict on this step; then = what Astra did next (moved_on, continued, retried,
taken_over, task_done, ended); chunk loss = S1's surrogate loss on its executed chunks of this span (mean, count; low = S1-like
actions); motion = the commanded motion in the controller's normalized units (for the smoothness filter on harness spans:
thresholds: the demos' p99 per primitive); S1 data = the agreed rules (selfimprove/data/s1_select.py), kept or not and why;
S2 data = every turn.</p>
{table(['turn', 'steps', 'moved', 'subgoal', 'System1 run', 'judged', 'then', 'chunk loss', 'motion', 'action convention', 'S1 data (proposed rule)', 'S2 data', 'start (left|right|wrist)'], seg_rows, 'turns')}
<h3>The S1 training spans this episode gives (after the rules: merged, cut, filtered)</h3>
<p class='note'>{len(kept_spans)} spans, {sum(sp.steps for sp in kept_spans)} control steps (+1 halt frame each) of {len(traj)}. Rules:
successful episodes only; System1's completed steps as they are (timeouts dropped); the harness's completed steps merged per
step (cut at 400 steps into "X" / "continue to X"), kept when smooth (within the demos' p99 per primitive) and without a partial
grip or a whole-body turn; spans under 8 steps dropped; Estimated Length = the grid value at or below the length, at least 50.</p>
{table(['steps', 'length', 'moved', 'subgoal (the prompt)', 'Estimated Length', 'turns', 'why kept'], span_rows) if span_rows else "<p>none (a failed episode)</p>"}
<h3>Per span: Astra's reasoning, S1's actual prompt, the harness code</h3>{''.join(detail_cards)}
<div class='two'><div><h3>Files</h3>{table(['file', 'size', 'files', 'what'], files)}</div>
<div><h3>Per control step (<code>trajectory/steps_NNNNN.npz</code>, 100 steps per chunk)</h3>{table(['array', 'shape'], arr_rows)}
<p class='note'>{esc(tidx.get('action_layout'))}<br>{esc(tidx.get('state_layout'))}<br>{esc(tidx.get('sim_state_layout'))}</p></div></div>
<h3>How Astra knew what to ask System1: the task's System1 table on its card</h3>
<p class='note'>skills/robocasa_s1/{esc(result.get('task'))}.md: the steps in the wordings S1 was trained on for this task
(drafted from S1's training subgoals, then edited), who does each, what to check, and the other wordings S1 knows.</p>
{details('the System1 table of this task', pre(s1_table))}
</section>"""

    cards = []
    for by in ("s1", "agent"):
        seg = next((sg for sg in rec.segments if sg.by == by and sg.steps >= 20 and sg.judge == "subgoal_complete"), None)
        if seg is None:
            continue
        span = seg.span()
        t = seg.start + seg.steps // 3
        est = estimated_length(seg.steps)
        el = traj.s1_element(t, span, est_length=est)
        names = []
        for cam in CAM_ORDER:
            for pre_, step in (("", t), ("anchor_", span.start)):
                n = f"{prefix}_s1_{by}_{pre_}{cam}.jpg"
                Image.fromarray(traj.image(step, cam)).save(assets / n, quality=90)
                names.append((pre_, cam, n))
        grid = "".join(f"<figure><img src='assets/{n}'><figcaption>{'anchor' if p else 'current'} - {c}</figcaption></figure>"
                       for p in ("", "anchor_") for (pp, c, n) in names if pp == p)
        prompt = s1_prompt(traj.task_goal, seg.subgoal, "Success", est, traj.lean[t], traj.lean[span.start],
                           traj.gripper_flag(t), norm)
        lean = np.delete(np.asarray(el["actions"]), 3, axis=1)  # LeRobot order minus torso = the lean 11
        who = "System1" if by == "s1" else "harness"
        cards.append(f"""<div class='card'><h3>From a {who} step: turn {seg.turn}, "{esc(seg.subgoal)}"
(steps {seg.start}-{seg.end}), frame t = {t}</h3>
<div class='grid3'>{grid}</div>{pre(prompt, 'prompt')}{chunk_table(lean, el['progress_action'])}</div>""")
    sec_s1 = f"""
<section id='to_s1'><h2>Rollout to S1 sample (what <code>selfimprove/data/s1_shards.py</code> makes today)</h2>
<p>Each kept step span becomes one sample per control step (plus a halt frame), in the base set's layout. Two real
examples from this episode:</p>{''.join(cards)}
</section>"""

    from selfimprove.s2.build import episode_turns

    with tempfile.TemporaryDirectory() as tmp:
        rows = episode_turns(folder, Path(tmp), iteration=0)
        pick = next((r for r in rows if r.get("clip") and r.get("executed_by") == "agent"), None) or next(
            (r for r in rows if r.get("clip")), None)
        clip_html = ""
        if pick:
            shutil.copy(Path(tmp) / "clips" / pick["clip"]["path"], assets / f"{prefix}_s2_clip.mp4")
            nf = video_strip(assets / f"{prefix}_s2_clip.mp4", assets / f"{prefix}_s2_strip.jpg")
            clip_html = (f"<video src='assets/{prefix}_s2_clip.mp4' controls loop muted class='tile'></video>"
                         f"<p class='note'>{nf} frames, 4 fps, 256x768 (from the trajectory's own frames)</p>"
                         f"<img class='strip' src='assets/{prefix}_s2_strip.jpg'>")
    k = rows.index(pick) if pick else 0
    done_before = [r["assistant"].get("subgoal") for r in rows[:max(0, k - 1)] if r["assistant"].get("subgoal")]
    last = rows[k - 1]["assistant"].get("subgoal") if k else "-"
    draft_in = (f"The goal is: {traj.task_goal}\n\nDone so far:\n" + "".join(f"{i + 1}. {x}\n" for i, x in enumerate(done_before))
                + f"\nLast step: {last}\n<video>\nCurrent gripper status: {pick.get('gripper_status')}\n\nWhat next?") if pick else ""
    nxt_by = rows[k].get("executed_by") if pick else None
    draft_out = (f"<thought>{(pick['assistant'].get('thought') or '')[:400]}</thought>\n<judge>{pick['assistant'].get('judge')}</judge>\n"
                 f"<decision>{'s1' if nxt_by == 's1' else 'harness'}</decision>\n<subgoal>{pick['assistant'].get('subgoal')}</subgoal>"
                 ) if pick else ""
    turn_rows = [[str(r["turn"]), esc(r["mode"]), esc(r["assistant"].get("judge")), esc(r["assistant"].get("subgoal")),
                  badge("s1" if r.get("executed_by") == "s1" else ("harness" if r.get("executed_by") == "agent" else "-")),
                  esc(r.get("gripper_status")), esc((r.get("clip") or {}).get("n_frames")),
                  esc(r["assistant"].get("estimated_step"))] for r in rows]
    sec_s2 = f"""
<section id='to_s2'><h2>Rollout to S2 turn (draft; <code>selfimprove/s2/build.py</code> + the proposed single mode)</h2>
<p>Every Astra turn is a planner turn: it judged the step just executed (watching it), chose the next step and who executes
it. <code>build.py</code> cuts each step's clip from the trajectory (4 fps, 256x768) and keeps Astra's thought, judge, next
subgoal and mover. The turns of this episode:</p>
{table(['turn', 'mode', 'judge (of the last step)', 'next subgoal', 'next step by', 'gripper', 'clip frames', 'est'], turn_rows)}
<h3>One turn in the proposed format</h3>
<div class='two'><div><h4>user (draft)</h4>{pre(draft_in)}<h4>assistant (target, from Astra)</h4>{pre(draft_out, 'target')}</div>
<div>{clip_html}</div></div>
</section>"""
    summary = {"task": result.get("task"), "seed": result.get("seed"), "demo": (result.get("demo") or {}).get("id"),
               "success": result.get("success"), "step": result.get("step"), "limit": result.get("step_limit"),
               "turns": result.get("turns"), "s1_share": rollout.get("s1_step_share"),
               "spans": len(rec.segments), "s1_spans": sum(sg.by == "s1" for sg in rec.segments),
               "harness_spans": sum(sg.by != "s1" for sg in rec.segments), **n_flags}
    return sec_r, sec_s1, sec_s2, summary


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1d2433;background:#f6f7f9}
header{background:#1d2433;color:#fff;padding:18px 28px}header h1{margin:0 0 4px;font-size:22px}header p{margin:0;opacity:.8}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde1e7;padding:8px 28px;z-index:2}
nav a{margin-right:16px;color:#2a5bd7;text-decoration:none;font-size:14px}
main{max-width:1280px;margin:0 auto;padding:12px 28px 60px}
section{background:#fff;border:1px solid #dde1e7;border-radius:8px;padding:14px 20px;margin:18px 0}
h2{font-size:19px;margin:4px 0 10px}h3{font-size:16px;margin:16px 0 8px}h4{font-size:14px;margin:10px 0 4px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
th,td{border:1px solid #dde1e7;padding:4px 6px;vertical-align:top;text-align:left}th{background:#eef1f5}
table.num td{font-family:ui-monospace,Menlo,monospace;text-align:right}table.num td:first-child{text-align:left}
pre{white-space:pre-wrap;word-break:break-word;background:#f3f5f8;border:1px solid #e3e7ed;border-radius:6px;padding:8px;
font-size:12px;max-height:420px;overflow:auto;margin:4px 0}
pre.prompt{background:#fffbea;border-color:#efe2a8}pre.target{background:#eefaf1;border-color:#bfe3c8}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.grid3 img{width:100%;border-radius:4px}
figure{margin:0}figcaption{font-size:12px;color:#5a6475;text-align:center}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.io{display:grid;grid-template-columns:1fr 1fr;gap:14px}.io>div{border-radius:8px;padding:6px 12px}
.io .in{background:#f5f8ff;border:1px solid #c9d8f5}.io .out{background:#f3fbf5;border:1px solid #c3e6cc}
.io h3{margin:6px 0;font-size:14px;letter-spacing:.05em}.grid3.small img{max-height:170px;object-fit:cover}
.card{border:1px solid #e3e7ed;border-radius:8px;padding:10px 14px;margin:12px 0;background:#fcfcfd}
.card.warn{border-color:#efb48a;background:#fffaf5}
img.tile,video.tile{width:100%;border-radius:4px}img.strip{width:100%;border:1px solid #e3e7ed}
img.thumb{width:240px}table.turns td{font-size:12px}
.note{font-size:12px;color:#5a6475;margin:2px 0}
.b{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;font-weight:600;border:1px solid}
.b.used,.b.direct{color:#156c35;border-color:#8fd0a5;background:#ecf8f0}
.b.unused{color:#8a1f1f;border-color:#e2a3a3;background:#fbeeee}
.b.partly,.b.derived{color:#7a5400;border-color:#e6c97a;background:#fdf6e3}
.b.decide{color:#5b2a91;border-color:#c4a8e6;background:#f4eefb}
.b.decided{color:#0b4f6c;border-color:#8cc7de;background:#eaf6fb}
.b.keep{color:#156c35;border-color:#8fd0a5;background:#ecf8f0}.b.skip{color:#8a1f1f;border-color:#e2a3a3;background:#fbeeee}
.b.s1{color:#1c4fa8;border-color:#9db8ea;background:#eef3fc}.b.harness{color:#9a3d00;border-color:#efb48a;background:#fdf1e8}
.b.positive{color:#156c35;border-color:#8fd0a5;background:#ecf8f0}.b.negative,.b.dropped{color:#8a1f1f;border-color:#e2a3a3;background:#fbeeee}
:root{--s1:#2a78d6;--harness:#eb6834}
svg.tl{width:100%;height:auto;background:#fcfcfb}svg.tl .tk{font-size:11px;fill:#52514e}svg.tl .mk{font-size:13px;fill:#fff;font-weight:700;text-anchor:middle}
svg.tl rect.seg:hover{stroke:#0b0b0b;stroke-width:1.5}
.legend{display:flex;gap:18px;font-size:12px;color:#52514e;margin:4px 0}.legend i{display:inline-block;width:14px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}
ul.tight li{margin:2px 0;font-size:13.5px}details summary{cursor:pointer;font-size:13px;color:#2a5bd7}
"""


def overview() -> str:
    rows = [
        ["unit", "one control step (frame) of an annotated subgoal span", "one planner turn (or the episode's plan)",
         "one whole episode: every turn and every control step"],
        ["source", "demos + Gemini fine-step annotation (+ cut / failure-injected variants)",
         "demos + Gemini annotation + LLM refinement + synthetic negatives", "Astra + S1 playing, from a recorded start state"],
        ["vision", "6 images 224 px: 3 cameras at t + 3 at the span start", "256x768 tile: start image, or a 4 fps clip of the last step (<=32 frames)",
         "every step: 3 cameras 256 px; Astra's pictures; full MuJoCo state (re-renderable)"],
        ["state", "lean 14-d x2 (current, anchor) as 256-bin text", "gripper status line (+ privileged task status)",
         "raw16 per step, 12-d action per step, camera poses, MuJoCo state"],
        ["language in", "Task; Current Subgoal; Quality; Estimated Length; Current Gripper", "goal + plan checklist (+ status lines)",
         "instruction; per turn Astra's thought, judge, subgoal, mover; the card"],
        ["target", "20 x 11 actions + progress", "thought, judge, plan_update, estimated_step, subgoal, subgoal_detail",
         "labels to derive: step spans, judge, mover, outcome, S1's chunk losses"],
        ["size", "34.2M samples, 2.15 TB", "1.1M records, 109 GB", "~64 MB / episode; ~230 GB / 3,600-episode round"],
    ]
    return f"""<section id='overview'><h2>0. At a glance</h2>
{table(['', 'S1 training sample', 'S2 training sample', 'our rollout record'], [[f'<b>{r[0]}</b>'] + [esc(c) for c in r[1:]] for r in rows])}
<p class='note'>Badges: {badge('used')} read by training/inference today, {badge('unused')} carried but never read,
{badge('direct')} copied from the rollout, {badge('derived')} computed from it, {badge('decided')} decided,
{badge('decide')} needs a decision.</p></section>"""


def decisions() -> str:
    items = [
        ("S2 memory without a plan", "Last subgoal + last clip cannot track progress through a composite task. Replace the plan "
         "with the list of steps judged complete (text, from Astra's turns)? Or keep a plan?"),
        ("S2 output", "<thought> + <judge> + <decision> (s1 | harness) + <subgoal> in S1's wording; drop plan_update, "
         "subgoal_detail, task_finish, SUMMARY."),
        ("S2 estimated length", "Drop it from S2 and use the per-subgoal table (as HexaShell), or keep it?"),
        ("S2 status lines", "Drop the privileged task status; keep the gripper status?"),
        ("S2 delegate labels", "Astra's own choice per turn; or outcome-based (S1 tried and failed -> harness, S1 succeeded -> s1); "
         "or with a cost term (the economic part)."),
        ("S2 thought text", "Astra's thoughts mention its own tools (look, to_world, measure). Rewrite into planner-style "
         "reasoning with a cheap LLM pass, keep as-is, or drop the thought?"),
        ("S1 prompt (decided)", "The template unchanged; Quality Success; Estimated Length floored to the grid, at least 50; "
         "only the fields the loader reads are written."),
        ("S1 data (decided)", "Successful episodes; S1's completed steps as they are (timeouts dropped); harness steps merged "
         "per step (cut at 400), smooth within the demos' p99 per primitive, no partial grip or whole-body turn."),
        ("S2 data (decided)", "All turns, failures included; the harness label (who executed each step) teaches S2 when to "
         "call Astra. Open: a turn where S1 was chosen and failed: label s1 (Astra's choice) or harness (what worked)?"),
        ("S1 objective and mix (decided)", "The normal objective; 70% rollout / 30% base data, base shards reshuffled per "
         "round. Training steps: estimated after the first 30-episode round. Task balance (e.g. atomic 40 / composite "
         "20-25 episodes): later."),
        ("Harness action convention (decided, fixed)", "The harness now follows the demos' convention (gripper +-1 held, base "
         "drives in mode +1 with arm deltas 0). Partial grips and turns about a point are skipped for S1; older rollouts' "
         "harness drives are left out of S1's data."),
        ("Rollout storage", "Keep the trajectory at 256 (as now); drop frames/ (repeats the trajectory); pictures/ at 256 or "
         "dropped; JPEG quality 95 or 90."),
    ]
    rows = [[str(i + 1), f"<b>{esc(a)}</b>", esc(b)] for i, (a, b) in enumerate(items)]
    return f"<section id='decide'><h2>6. Decisions to discuss</h2>{table(['#', 'topic', 'question'], rows)}</section>"


RESULT_MEANING = {
    "bench": "benchmark", "task": "RoboCasa task", "seed": "split demo: the position of the recording in the task's index",
    "split": "demo = started from a recorded pretrain episode", "task_set": "atomic_seen / composite_seen / pretrain_atomic / ...",
    "layout_id": "kitchen layout (the recording's)", "style_id": "kitchen style (the recording's)",
    "instruction": "the task's language goal (the recording's ep_meta lang)", "success": "the environment's success check fired",
    "step": "control steps used", "step_limit": "the task's horizon", "verdict": "the benchmark's verdict",
    "end": "how it ended: episode_over (success or horizon), finish (Astra stopped), turn_limit, error",
    "finish": "Astra's finish call (status, summary)", "turns": "Astra's replies", "cells": "code cells run",
    "robot_calls": "robot API calls by name", "tokens": "Astra tokens (prompt, completion, cached, reasoning)",
    "pictures": "pictures shown / dropped from the context", "seconds": "wall time", "settings": "HexaShell settings of the run",
    "demo": "the recorded episode it started from (id, source, frames)", "card": "what the task card held (notes, S1 table)",
    "errors": "simulator / gateway errors (empty when clean)", "model": "Astra model", "s1": "System1 mode",
}
EVENT_MEANING = {
    "sim_open": "the episode opened: facts (task, kitchen, instruction, horizon) and the start pose",
    "worker_start": "the code sandbox started", "start": "the run's settings",
    "request": "one request to Astra (messages, pictures, tokens)", "reply": "Astra's reply (text, tool calls, usage, time)",
    "judge": "Astra's verdict on the step just executed (judge, thought) and the next step (next, of_by, step)",
    "subgoal": "the label set for the coming control steps: subgoal text, by (s1 = System1, agent = Astra's own calls), step",
    "s1": "one robot.s1() run: subgoal, est_length (+ options), budget, stop reason, steps, final progress, holding, prompts",
    "s1_query": "one query to the System1 server inside a run (its inputs and timing)",
    "drive": "a base drive (target, reached, blocked, steps)", "goto": "an arm move (target, reached, steps)",
    "gripper": "a gripper command (opening, holding)", "carry": "a carry move", "measure": "a measurement (sizes, centre)",
    "sam3": "a SAM3 segmentation query", "look": "a picture taken", "pictures_dropped": "old pictures dropped from the context",
    "finish": "Astra ended the episode (status, summary)",
}
STEP_MEANING = {
    "step": "control step index (20 Hz)",
    "raw16": "System1's raw 16-d state before the step: base pos (3), base quat xyzw (4), eef pos in the base frame (3), eef quat (4), finger qpos (2)",
    "action": "the 12-d official action stepped, robosuite order: [0:3] eef dpos, [3:6] eef drot, [6] gripper, [7:10] base, [10] torso, [11] mode",
    "level": "the gripper's commanded target level after the step (-1 open .. +1 closed)",
    "jpeg": "the 3 cameras' JPEGs before the step, end to end (256 px, quality 95)", "offsets": "byte offsets into jpeg (3 per step)",
    "cams": "camera order (left, right, wrist)", "cam_pose": "each camera's pose in the start frame, per step", "cam_focal": "focal length in pixels at 256",
    "sim_state": "MuJoCo state before the step (time, qpos, qvel): with model.xml.gz + scene.json the episode replays and re-renders",
}


def short(v, n: int = 220) -> str:
    t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return t if len(t) <= n else t[:n] + f" ... ({len(t)} chars)"


def save_format_page(folder: Path, assets: Path, scored: Path | None) -> str:
    from PIL import Image

    rec_traj = load_episode(folder).traj
    result = json.loads((folder / "result.json").read_text())
    rollout = json.loads((folder / "rollout.json").read_text())
    files = [[n, sz, c, esc(ROLLOUT_FILES.get(n.replace("<code>", "").replace("</code>", ""), ""))] for n, sz, c in folder_sizes(folder)]
    res_rows = [[f"<code>{esc(k)}</code>", esc(short(v)), esc(RESULT_MEANING.get(k, ""))] for k, v in result.items()]
    roll_rows = [[f"<code>{esc(k)}</code>", esc(short(v))] for k, v in rollout.items()]
    events = [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines() if x.strip()]
    kinds: dict[str, list] = {}
    for e in events:
        kinds.setdefault(e.get("kind"), []).append(e)
    ev_rows = [[f"<code>{esc(k)}</code>", str(len(v)), esc(EVENT_MEANING.get(k, "")), esc(", ".join(sorted(v[0])))] for k, v in kinds.items()]
    ev_examples = "".join(details(f"example <code>{esc(k)}</code> event",
                                  pre(json.dumps({kk: (short(vv, 400) if isinstance(vv, str) else vv) for kk, vv in v[0].items()},
                                                 indent=1, ensure_ascii=False, default=str)[:5000])) for k, v in kinds.items())
    cells = [json.loads(x) for x in (folder / "cells.jsonl").read_text().splitlines() if x.strip()]
    cell = next((c for c in cells if "robot.goto" in (c.get("code") or "")), cells[0])
    cell_ex = pre(json.dumps({k: (short(v, 600) if isinstance(v, str) else v) for k, v in cell.items()}, indent=1, ensure_ascii=False,
                             default=str)[:6000])
    tr = json.loads((folder / "transcript.json").read_text())
    roles = {}
    for m in tr:
        roles[m.get("role")] = roles.get(m.get("role"), 0) + 1
    asst = next((m for m in tr if m.get("role") == "assistant" and m.get("tool_calls")), {})
    tool = next((m for m in tr if m.get("role") == "tool"), {})
    tidx = rec_traj.index
    with np.load(rec_traj.folder / tidx["chunks"][0]) as z:
        arr_rows = []
        for k in z.files:
            a = z[k]
            ex = a[0] if a.ndim >= 1 and k not in ("jpeg", "offsets", "cams") else a[:6]
            exs = np.array2string(np.asarray(ex).ravel()[:16], precision=4, separator=", ", max_line_width=200)
            arr_rows.append([f"<code>{esc(k)}</code>", esc(f"{tuple(a.shape)} {a.dtype}"), esc(STEP_MEANING.get(k, "")), f"<code>{esc(exs)}</code>"])
    tile(rec_traj, 0, size=256).save(assets / "fmt_step0.jpg", quality=90)
    t_keys = [[f"<code>{esc(k)}</code>", esc(short(v if k != "labels" else v[:6]))] for k, v in tidx.items() if k != "chunks"]
    t_keys.append(["<code>chunks</code>", esc(f"{len(tidx.get('chunks', []))} files: {tidx.get('chunks', [])[:3]} ...")])
    scene = json.loads((rec_traj.folder / "scene.json").read_text())
    scene_rows = [[f"<code>{esc(k)}</code>", esc(short(v, 300))] for k, v in scene.items()]
    states = (folder / "states.jsonl").read_text().splitlines()[:3]
    pics = sorted(p.name for p in (folder / "pictures").iterdir())
    frames = sorted(p.name for p in (folder / "frames").iterdir())
    shutil.copy(folder / "pictures" / pics[0], assets / "fmt_picture0.jpg")
    chunk_ex = ""
    if scored and (scored / "chunk_scores.jsonl").exists():
        row = json.loads((scored / "chunk_scores.jsonl").read_text().splitlines()[0])
        chunk_ex = pre(json.dumps(row, indent=1))
    body = f"""
<section><h2>What one S1 + harness rollout saves</h2>
<p>Record <code>s3://.../self_improve/&lt;exp&gt;/iter_NN/rollout/&lt;task&gt;/&lt;seed&gt;/</code> of one episode: here <b>{esc(result['task'])}</b>
seed {esc(result['seed'])} from <code>{esc(rollout.get('exp_id'))}</code>. Every field below is the real value of this episode.</p>
<h3>Folder</h3>{table(['file', 'size', 'files', 'what'], files)}</section>
<section><h2>result.json (the episode's outcome and settings)</h2>{table(['key', 'value', 'meaning'], res_rows)}</section>
<section><h2>rollout.json (the Ray node's summary)</h2>{table(['key', 'value'], roll_rows)}</section>
<section><h2>events.jsonl (everything that happened, in order; the labels per turn come from here)</h2>
{table(['kind', 'count', 'meaning', 'fields'], ev_rows)}{ev_examples}</section>
<section><h2>cells.jsonl (Astra's code cells: thought, code, output, steps)</h2>{cell_ex}</section>
<section><h2>transcript.json (the full conversation with Astra)</h2>
<p>{len(tr)} messages: {esc(roles)}. The system message is the prompt (also in prompt.md), the first user message the task card
with the start pictures; each assistant message is one python tool call whose arguments hold thought / judge / subgoal / code;
each tool message is the cell's output.</p>
{details('an assistant message (tool call)', pre(short(asst.get('tool_calls'), 3000)))}
{details('a tool message (the cell output)', pre(short(tool.get('content'), 3000)))}</section>
<section><h2>trajectory/ (every control step: what S1's training data is cut from)</h2>
<h3>trajectory.json (the index)</h3>{table(['key', 'value'], t_keys)}
<h3>steps_NNNNN.npz (100 steps each)</h3>{table(['array', 'shape', 'meaning', 'step 0 (first values)'], arr_rows)}
<p class='note'>Step 0, the 3 cameras as stored (256 px):</p><img class='tile' src='assets/fmt_step0.jpg' style='max-width:768px'>
<h3>scene.json (RoboCasa's episode meta: with model.xml.gz it rebuilds the scene)</h3>{table(['key', 'value'], scene_rows)}</section>
<section><h2>states.jsonl (the robot's state after each cell)</h2>{pre(chr(10).join(states))}</section>
<section><h2>pictures/, frames/, cells/</h2>
<p><b>pictures/</b> ({len(pics)}): every picture Astra was shown, e.g. {esc(', '.join(pics[:5]))}. <b>frames/</b> ({len(frames)}):
the cameras after each cell that moved the robot, e.g. {esc(', '.join(frames[:3]))}. <b>cells/</b>: each cell's code as NNN.py.</p>
<img src='assets/fmt_picture0.jpg' style='max-width:320px'></section>
<section><h2>chunk_scores.jsonl (when the round scores chunks)</h2>
<p>S1's surrogate loss on every 16th executed chunk (S1's and the harness's alike), from another episode of a scored round:</p>{chunk_ex}</section>"""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>S1 + harness rollout: the saved data format</title>
<style>{CSS}</style></head><body><header><h1>S1 + harness rollout: the saved data format</h1>
<p>Every file and field one episode writes, with its real values.</p></header>
<nav><a href='index.html'>&larr; comparison</a></nav><main>{body}</main></body></html>"""


CONVENTION = f"""<div class='card'><h3>The action convention (official RoboCasa 12-d action, as the demonstrations use it)</h3>
{table(['dim', 'S1 training data (demos)', "S1's own rollout spans", 'harness spans (since 2026-11-24)', 'harness spans before'], [
    ['mode (11)', '+1 while the base drives (arm deltas 0), -1 for arm motion', 'as trained', '<b>as the demos</b>: +1 for plain drives, arm 0',
     '-1 always, arm-hold deltas while driving'],
    ['gripper (6)', '+-1 held (incremental: 0.2 per step toward the sign, clipped)', 'as trained', '<b>+-1 held</b>; starts open; 0 only at a partial grip',
     '0 when not opening or closing'],
])}
<p class='note'>Checked: drives as accurate as before (0.8 cm, 0.3 deg) with less hand drift on the base (12-13 mm vs 15-23 mm).
<b>S1 filter rules:</b> harness spans with a partial grip (3.9% of harness turns) or a turn of the base about a point (whole-body,
0.3%) are not used for S1 (S2 keeps them); rollouts recorded before the fix keep their harness drives out of S1's data.</p></div>"""


def episode_page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head><body>
<header><h1>{esc(title)}</h1><p>One rollout of the S1 + harness data collection, with every label the S1 and S2 filters use.</p></header>
<nav><a href='index.html'>&larr; comparison</a><a href='#rollout'>Rollout</a><a href='#to_s1'>To S1</a><a href='#to_s2'>To S2</a></nav>
<main>{body}</main></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, nargs="+", required=True, help="local episode records (rollout/<task>/<seed>/)")
    ap.add_argument("--s2-shard", type=Path, required=True)
    ap.add_argument("--s1-sample", type=Path, default=Path("~/selfimprove_work/site_src/s1_base").expanduser())
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "formats")
    ap.add_argument("--scored-example", type=Path, default=None, help="an episode record with chunk_scores.jsonl")
    args = ap.parse_args()
    s3.configure("sagemaker", "us-west-2")
    assets = args.out / "assets"
    if args.out.exists():
        shutil.rmtree(args.out)
    assets.mkdir(parents=True)
    norm = json.loads(NORM.read_text())["norm_stats"]["state"]
    s1 = section_s1(fetch_s1_sample(args.s1_sample), assets, norm)
    s2 = section_s2(args.s2_shard, assets)
    rows = []
    for folder in args.rollouts:
        res = json.loads((folder / "result.json").read_text())
        slug = f"{res['task']}_{res['seed']}"
        sec_r, sec_to_s1, sec_to_s2, summ = section_rollout(folder, assets, norm, prefix=slug)
        (args.out / f"rollout_{slug}.html").write_text(episode_page(f"{res['task']} seed {res['seed']}: S1 + harness rollout",
                                                                    sec_r + sec_to_s1 + sec_to_s2))
        rows.append([f"<a href='rollout_{slug}.html'><b>{esc(summ['task'])}</b> seed {esc(summ['seed'])}</a>",
                     f"<code>{esc(summ['demo'])}</code>", badge("keep") if summ["success"] else badge("skip"),
                     f"{summ['step']} / {summ['limit']}", str(summ["turns"]), f"{summ['s1_share']:.2f}",
                     f"{summ['spans']} ({summ['s1_spans']} S1, {summ['harness_spans']} harness)", str(summ["s1_keep"]),
                     str(summ["partial"]), str(summ["whole"])])
    (args.out / "rollout_format.html").write_text(save_format_page(args.rollouts[1] if len(args.rollouts) > 1 else args.rollouts[0],
                                                                   assets, args.scored_example))
    rollouts = f"""<section id='rollout'><h2>3. Our rollouts: S1 + harness (four example episodes)</h2>
<p><b><a href='rollout_format.html'>The saved data format, file by file and field by field</a></b> (one episode's real values).</p>
<p>Astra + S1 in HexaShell, each episode started from a recorded pretrain episode (split demo), the harness in the demonstrations'
action convention. Each page shows the episode's timeline, every step span with all its labels (who moved, System1's run and actual
prompt, the next turn's judgement and thought, chunk losses, motion and action convention, the proposed S1 / S2 data rules), Astra's
code for the harness steps, the System1 table on Astra's card, and what our preprocessors make of it.</p>
{table(['episode', 'recording', 'success', 'steps', 'turns', 'S1 step share', 'step spans', 'spans S1 would keep', 'harness spans with a partial grip', 'whole-body turns'], rows)}
</section>
<section id='to_s1'><h2>4. Rollout to S1: the mapping</h2>{CONVENTION}{map_table(S1_MAP)}</section>
<section id='to_s2'><h2>5. Rollout to S2: the mapping</h2>{map_table(S2_MAP)}</section>"""
    nav = "".join(f"<a href='#{i}'>{t}</a>" for i, t in (("overview", "At a glance"), ("s1", "S1: input / output"),
                                                          ("s2", "S2: input / output"), ("rollout", "Rollouts"),
                                                          ("to_s1", "Rollout to S1"), ("to_s2", "Rollout to S2"), ("decide", "Decisions")))
    nav += "<a href='rollout_format.html'><b>Rollout save format</b></a>"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>S1 / S2 training data vs rollouts</title>
<style>{CSS}</style></head><body>
<header><h1>S1 / S2 training data vs our S1 + harness rollouts</h1>
<p>Real samples: the S1 base set, the S2 full0804 set, and four S1 + harness rollouts. Built by
selfimprove/scripts/build_format_site.py.</p></header><nav>{nav}</nav><main>
{overview()}{s1}{s2}{rollouts}{decisions()}</main></body></html>"""
    (args.out / "index.html").write_text(page)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"{args.out / 'index.html'} + {len(args.rollouts)} episode pages ({size / 1e6:.1f} MB with assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

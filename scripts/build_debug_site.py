"""Build ``site/debug_<exp>/``: every episode of a rollout round, every turn with its labels: what the turn did, whether
its span goes into System1's training data (and why not), and the System2 row it becomes (input, target).

    .venv/bin/python scripts/build_debug_site.py EXP [--iteration 0] [--cache ~/selfimprove_work/<EXP>/rollout]
        [--out site/debug_<EXP>] [--jobs 8]

Downloads the round's records once (without pictures/ and frames/), then per episode: the S1 spans
(selfimprove/data/s1_select.py: kept or dropped, why, the training span and its Estimated Length) and the S2 rows
(selfimprove/s2/rows_v2.py: the user text, the clip, the target). Open index.html (all paths are relative).
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402
from selfimprove.config import Config  # noqa: E402

_spec = importlib.util.spec_from_file_location("fmt", ROOT / "scripts" / "build_format_site.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
esc, badge, table, pre, details, CSS = F.esc, F.badge, F.table, F.pre, F.details, F.CSS

EXTRA_CSS = """
.filter{margin:8px 0}.filter input{padding:4px 8px;width:280px}
table.turns td{font-size:11.5px}td.reason{max-width:340px}
.b.s1d{color:#1c4fa8;border-color:#9db8ea;background:#eef3fc}
.b.k-s1{color:#1c4fa8;border-color:#9db8ea;background:#eef3fc}.b.k-harness{color:#9a3d00;border-color:#efb48a;background:#fdf1e8}
.b.k-harness_merged{color:#0f6e4c;border-color:#8fd9bd;background:#e8f7f1}
.b.k-recovery_within{color:#4c3db8;border-color:#b9b0ee;background:#f0eefc}
.b.k-recovery_request{color:#a3305c;border-color:#efb3c9;background:#fcedf3}.b.recovery{color:#5b2a86;border-color:#c3a6e0;background:#f5eefb}.b.hd{color:#9a3d00;border-color:#efb48a;background:#fdf1e8}
video.clip{width:300px;border-radius:4px;background:#111;display:block}td a.open{font-size:11px}
tr:has(span:target) td{background:#fff7d6}
div.turn{border:1px solid #dde3ea;border-radius:8px;padding:8px 12px 6px;margin:12px 0;background:#fff}
div.turn:target{box-shadow:0 0 0 3px #f2c94c}div.turn .th{font-weight:600;font-size:14px;margin-bottom:6px}
.s2box,.s1box{padding:6px 10px;border-radius:6px;margin:6px 0;font-size:12.5px;line-height:1.5}
.s2box{background:#f4f1fb;border-left:4px solid #7a5cc7}.s2box.nomove{background:#f6f6f6;border-left-color:#aaa}
.s1box{background:#eef5fb;border-left:4px solid #2c6fb7}.lab{font-weight:700;margin-right:8px}.why{color:#222;margin-top:3px}
div.exec{margin:8px 0 4px}video.wide{width:768px;max-width:100%;border-radius:4px;background:#111}
div.turn details summary{font-size:12px;color:#46607a;cursor:pointer;margin-top:4px}
div.astra pre{max-height:420px;overflow:auto;font-size:11.5px;white-space:pre-wrap}div.astra h4{margin:6px 0 2px;font-size:12px}
"""
VIDEO_FPS, VIDEO_EVERY, VIDEO_CRF = 10, 2, 28  # every 2nd control step (20 Hz) at 10 fps: real time


def write_video(traj, start: int, end: int, path: Path) -> None:
    """Steps start..end (both included) as an H.264 clip, the 3 cameras tiled 768 x 256, in real time."""
    import subprocess

    import numpy as np

    steps = list(range(start, end + 1, VIDEO_EVERY))
    if steps[-1] != end:
        steps.append(end)
    if len(steps) == 1:
        steps *= 2
    frames = [np.concatenate([traj.image(t, cam) for cam in F.CAM_ORDER], axis=1) for t in steps]
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
           str(VIDEO_FPS), "-i", "-", "-an", "-vcodec", "libx264", "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f, np.uint8).tobytes() for f in frames), check=True)


#: the videos' own controls sit under them (the browser's overlay covers the picture while it plays): a play/pause
#: button, a seek bar and the time; a click on the picture plays or pauses too
VIDEO_CSS = """
.vid{display:inline-block}.vid video{display:block;cursor:pointer}
.vbar{display:flex;align-items:center;gap:6px;margin-top:3px;font-size:11px}
.vbar button{width:30px;height:22px;padding:0;border:1px solid #9aa6b2;border-radius:4px;background:#f4f6f8;cursor:pointer}
.vbar input[type=range]{flex:1;min-width:60px}.vbar .vtime{color:#555;min-width:72px}
"""
VIDEO_JS = """<script>
function vidOf(el){return el.closest('.vid').querySelector('video');}
function vidToggle(el){const v=vidOf(el); if(v.paused){v.play();}else{v.pause();}}
function vidSeek(r){const v=vidOf(r); if(v.duration){v.currentTime=r.value/1000*v.duration;}}
function vidBar(v){return v.closest('.vid')?v.closest('.vid').querySelector('.vbar'):null;}
document.addEventListener('play',e=>{const b=vidBar(e.target); if(b){b.querySelector('button').textContent='\u275a\u275a';}},true);
document.addEventListener('pause',e=>{const b=vidBar(e.target); if(b){b.querySelector('button').textContent='\u25b6';}},true);
document.addEventListener('timeupdate',e=>{const v=e.target,b=vidBar(v); if(b&&v.duration){b.querySelector('input').value=1000*v.currentTime/v.duration;
  b.querySelector('.vtime').textContent=v.currentTime.toFixed(1)+' / '+v.duration.toFixed(1)+' s';}},true);
</script>"""


def video(src: str, cls: str, poster: str = "", note: str = "") -> str:
    """A video with its controls under it (VIDEO_CSS / VIDEO_JS), an 'open' link and a note."""
    pos = f" poster='{poster}'" if poster else ""
    return (f"<div class='vid'><video class='{cls}' src='{src}'{pos} muted playsinline preload='none' "
            f"onclick='vidToggle(this)'></video><div class='vbar'><button onclick='vidToggle(this)' title='play / pause'>"
            f"&#9654;</button><input type='range' min='0' max='1000' value='0' oninput='vidSeek(this)'>"
            f"<span class='vtime'></span><a class='open' href='{src}' target='_blank'>open</a></div>"
            + (f"<span class='note'>{note}</span>" if note else "") + "</div>")


def clip(src: str, poster: str, note: str) -> str:
    return video(src, "clip", poster, note)
FILTER_JS = """<script>
function filt(){const q=document.getElementById('q').value.toLowerCase();
for(const r of document.querySelectorAll('table.eps tbody tr')){r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';}}
</script>"""


def fetch(cfg: Config, k: int, cache: Path, jobs: int = 16) -> list[Path]:
    """The round's finished episodes (those with a result.json), downloaded in parallel into the cache."""
    from concurrent.futures import ThreadPoolExecutor

    plan = s3.read_json(f"{cfg.rollout_uri(k)}/_plan.json", default=None)
    if plan is None:  # the round's records are gone from S3 (deleted): the episodes downloaded before
        return sorted(p.parent for p in cache.glob("*/*/.fetched"))

    def one(e) -> Path | None:
        d = cache / e["task"] / str(e["seed"])
        uri = cfg.episode_uri(k, e["task"], e["seed"])
        if not (d / ".fetched").exists():  # written after a whole download (a stopped one leaves no marker)
            if not s3.exists(f"{uri}/result.json"):
                return None
            s3.sync_down(uri, d, excludes=("pictures/*", "frames/*", "cells/*"))
            (d / ".fetched").touch()
        return d

    with ThreadPoolExecutor(jobs) as pool:
        return [d for d in pool.map(one, plan["episodes"]) if d is not None]


ASTRA_FIELDS = ("thought", "judge", "subgoal", "milestone", "milestones", "planner_reasoning")


DECISION_LABEL = {"s1": "model", "harness": "expert"}  # as the S2 target writes the decision
KINDS = {"s1": "System1's own", "harness": "harness", "harness_merged": "harness merged",
         "recovery_within": "recovery within the step", "recovery_request": "recovery on request"}


def kind_badge(kind: str) -> str:
    """The S1 span's kind, in its own colour (EXTRA_CSS .k-*)."""
    return f"<span class='b k-{esc(kind)}'>{esc(KINDS.get(kind, kind))}</span>"


def astra_fold(turn: int, cells: list[dict]) -> str:
    """Astra's whole turn, folded: the reply's fields, the code, and what it printed (the result Astra read)."""
    body = []
    for c in cells:
        fields = "\n".join(f"{k}: {c.get(k)}" for k in ASTRA_FIELDS if c.get(k) not in (None, ""))
        out = str(c.get("output") or "")
        if len(out) > 8000:
            out = out[:8000] + f"\n... ({len(out) - 8000} more characters)"
        err = f"<h4>error</h4>{pre(c['error'])}" if c.get("error") else ""
        steps = f"steps {c.get('step_before')} -> {c.get('step_after')}, {c.get('seconds')} s"
        body.append(f"<div class='astra'><h4>reply</h4>{pre(fields)}<h4>code</h4>{pre(c.get('code') or '')}"
                    f"<h4>output ({esc(steps)})</h4>{pre(out)}{err}</div>")
    return details(f"Astra's turn {turn}: reply, code and output", "".join(body))


def astra_table(header: list[str], rows: list[tuple[int, list[str]]], cells: dict[int, list[dict]]) -> str:
    """The turn table, each turn followed by a full-width row with its fold (astra_fold); two rows of one turn (two
    movers) share one fold, after the second."""
    th = "".join(f"<th>{h}</th>" for h in header)
    out = []
    for i, (turn, r) in enumerate(rows):
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        last_of_turn = i + 1 == len(rows) or rows[i + 1][0] != turn
        if last_of_turn and cells.get(turn):
            out.append(f"<tr class='fold'><td colspan='{len(header)}'>{astra_fold(turn, cells[turn])}</td></tr>")
    return f"<table class='turns'><thead><tr>{th}</tr></thead><tbody>{''.join(out)}</tbody></table>"


def episode(args) -> dict:
    """One episode's page; returns its summary row."""
    folder, out, reuse = (list(args) + [False])[:3]
    folder, out = Path(folder), Path(out)
    sys.path.insert(0, str(ROOT))
    from selfimprove.data import s1_select
    from selfimprove.data.episode import load_episode
    from selfimprove.s2 import rows_v2

    rec = load_episode(folder)
    res, traj = rec.result, rec.traj
    slug = f"{rec.task}_{rec.seed}"
    assets = out / "assets" / slug
    assets.mkdir(parents=True, exist_ok=True)
    spans = s1_select.select(rec)
    rows = rows_v2.episode_rows(folder, assets, make_media=True)  # its clips are named for their steps: kept if there
    have = (lambda p: reuse and p.exists())  # noqa: E731  (--reuse-media: the pictures and clips of an earlier build)
    turn_cells: dict[int, list[dict]] = {}
    if (folder / "cells.jsonl").exists():
        for line in (folder / "cells.jsonl").read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                turn_cells.setdefault(int(c.get("turn") or c.get("cell") or 0), []).append(c)
    by_turn = {r["meta"]["turn"]: r for r in rows}
    execs: dict[int, list[str]] = {}
    seg_video = {}  # (start, end) -> the clip, so an S1 span that is one turn's step reuses it
    for sg in rec.segments:
        img = f"assets/{slug}/t{sg.index:02d}.jpg"
        if not have(out / img):
            F.tile(traj, sg.start, size=128).save(out / img, quality=80)
        vid = f"assets/{slug}/t{sg.index:02d}.mp4"
        if not have(out / vid):
            write_video(traj, sg.start, sg.end, out / vid)
        seg_video[(sg.start, sg.end)] = (vid, img)
        dec = [sp for sp in spans if sp.start <= sg.end and sg.start <= sp.end]  # every span this step is part of
        lines = []
        for sp in dec:
            kind = kind_badge(sp.kind)
            if sp.kept:
                trained = (f", trained from step {sp.train_from} (before: context)" if sp.train_from is not None else "")
                lines.append(f"{badge('keep')} {kind} {sp.start}-{sp.end} <i>{esc(sp.subgoal)}</i>, Estimated Length "
                             f"{sp.est_length}{trained} <span class='note'>{esc(sp.reason)}</span>")
            else:
                lines.append(f"{badge('skip')} {kind} {sp.start}-{sp.end}: {esc(sp.reason)}")
        data = "<br>".join(lines) or "-"
        execs.setdefault(sg.turn, []).append(
            f"<div class='exec'>{video(vid, 'wide', img, f'{sg.steps} steps, real time')}"
            f"<div class='s1box'><span class='lab'>S1</span>{badge('s1' if sg.by == 's1' else 'harness')} executed "
            f"<b>{esc(sg.subgoal)}</b> &middot; steps {sg.start}-{sg.end} ({sg.steps})"
            f"<div>judged <b>{esc(sg.judge)}</b> by the next turn, then {esc(sg.next)}</div>"
            f"<div>S1 training data:<br>{data}</div></div></div>")
    turns = sorted({*execs, *(t for t in turn_cells if t > 0)})
    merged_into = {n: r["meta"]["turn"] for r in rows for n in (r["meta"].get("merged_turns") or [])[1:]}
    blocks = []
    for t in turns:
        r = by_turn.get(t)
        c = (turn_cells.get(t) or [{}])[0]
        joined = merged_into.get(t)
        if r is not None:
            lab = r["labels"]
            dcls = "s1d" if lab["decision"] == "s1" else "hd"
            span = r["meta"].get("merged_turns")
            one = (f"<div class='note'>one S2 row for turns {', '.join(map(str, span))}: short harness steps joined as "
                   f"System1's harness merged span, worded as the last</div>" if span else "")
            if r["meta"].get("skipped"):
                one += f"<div>{badge('skip')} <span class='note'>not trained: {esc(r['meta']['skipped'])}</span></div>"
            s2 = (f"<div class='s2box'><span class='lab'>S2</span>{one}decision <span class='b {dcls}'>{esc(DECISION_LABEL.get(lab['decision'], lab['decision']))}</span>"
                  f" &middot; judge <b>{esc(lab['judge'])}</b> &middot; <span class='note'>{esc(lab['milestone'])}</span>"
                  f"<div>subgoal <b>{esc(lab['subgoal'])}</b></div><div class='why'>{esc(lab['thought'])}</div></div>")
        elif joined is not None:  # a turn inside a merged step: its S2 row is the step's first turn's
            s2 = (f"<div class='s2box nomove'><span class='lab'>S2</span><span class='note'>no row of its own: part of "
                  f"the merged step whose S2 row is at <a href='#turn{joined}'>turn {joined}</a></span></div>")
        else:  # a turn that moved nothing (looked, measured): no S2 row; Astra's fields as they were
            s2 = (f"<div class='s2box nomove'><span class='lab'>S2</span><span class='note'>no motion this turn (it looked "
                  f"or measured): not an S2 row</span> &middot; judge <b>{esc(c.get('judge'))}</b> &middot; "
                  f"<span class='note'>{esc(c.get('milestone') or '')}</span><div>subgoal <b>{esc(c.get('subgoal') or '')}"
                  f"</b></div><div class='why'>{esc(c.get('planner_reasoning') or '')}</div></div>")
        movers = execs.get(t, [])
        head = (f"Turn {t}" + (f" &middot; {len(movers)} motions" if len(movers) > 1 else "")
                + ("" if movers else " &middot; <span class='note'>no motion</span>"))
        fold = astra_fold(t, turn_cells[t]) if turn_cells.get(t) else ""
        blocks.append(f"<div class='turn' id='turn{t}'><div class='th'>{head}</div>{s2}{''.join(movers)}{fold}</div>")
    turn_table = "".join(blocks)
    cards = []
    for r in rows:
        m = r["meta"]
        media = ""
        if m.get("clip"):
            media = (video(f"assets/{slug}/{m['clip']}", "tile") +
                     f"<p class='note'>{m.get('clip_frames')} frames, steps {m.get('clip_span')}</p>")
        elif m.get("image"):
            media = f"<img class='tile' src='assets/{slug}/{m['image']}'>"
        lengths = []
        if m.get("clip_span"):
            a, b = m["clip_span"]
            lengths.append(f"judges steps {a}-{b} ({b - a + 1} steps, {m.get('clip_frames')} frames)")
        elif m.get("image"):
            lengths.append("the opening scene")
        if m.get("step_span"):
            a, b = m["step_span"]
            what = f"turns {', '.join(map(str, m['merged_turns']))}" if m.get("merged_turns") else f"turn {m['turn']}"
            lengths.append(f"plans {what}: steps {a}-{b} ({b - a + 1} steps)")
        dec = DECISION_LABEL.get(m.get("decision"), m.get("decision"))
        skip = f"{badge('skip')} <span class='note'>not trained: {esc(m['skipped'])}</span> &middot; " if m.get("skipped") else ""
        cards.append(details(f"{skip}S2 row, turn {m['turn']} ({m['kind']}): decision <b>{esc(dec)}</b>, "
                             f"judge <b>{esc(r['labels']['judge'])}</b> &middot; <span class='note'>{esc(' · '.join(lengths))}</span>",
                             f"<div class='io'><div class='in'><h3>INPUT</h3>{pre(r['messages'][1]['content'])}{media}</div>"
                             f"<div class='out'><h3>OUTPUT (target)</h3>{pre(r['messages'][2]['content'], 'target')}</div></div>"))
    kept_sp = [sp for sp in spans if sp.kept]
    sp_rows = []
    for sp in kept_sp:
        if (sp.start, sp.end) in seg_video:
            vid, img = seg_video[(sp.start, sp.end)]
        else:  # merged turns, or a long step cut at 400: its own clip
            # named for its exact steps: a span whose end changes with the rules gets its own clip, never a stale one
            vid, img = f"assets/{slug}/s{sp.start:05d}-{sp.end:05d}.mp4", f"assets/{slug}/s{sp.start:05d}.jpg"
            if not have(out / vid):
                write_video(traj, sp.start, sp.end, out / vid)
            if not have(out / img):
                F.tile(traj, sp.start, size=128).save(out / img, quality=80)
        sp_rows.append([f"{sp.start}-{sp.end}" + (f"<br><span class='note'>trained from {sp.train_from}</span>"
                                                  if sp.train_from is not None else ""), str(sp.trained_steps),
                        kind_badge(sp.kind), esc(sp.subgoal),
                        str(sp.est_length), esc(", ".join(map(str, sp.turns))), esc(sp.reason),
                        clip(vid, img, "the training span, real time")])
    fin = res.get("finish") or {}
    facts = {k: res.get(k) for k in ("task", "seed", "instruction", "success", "end", "step", "step_limit", "turns", "demo")}
    sp_ = res.get("start_perturb") or {}
    facts["start"] = (f"perturbed: {sp_.get('forward_m', 0):+.2f} m forward, {sp_.get('side_m', 0):+.2f} m sideways, "
                      f"{sp_.get('turn_deg', 0):+.1f} deg ({sp_.get('draws', 1)} draws, salt {sp_.get('salt')})"
                      if sp_ else "the recorded start")
    if fin:
        facts["finish"] = fin
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(slug)}</title><style>{CSS}{EXTRA_CSS}{VIDEO_CSS}</style>{VIDEO_JS}</head>
<body><header><h1>{esc(rec.task)} seed {rec.seed}: {'success' if rec.success else 'FAILED'}</h1>
<p>{esc(res.get('instruction'))}</p></header><nav><a href='index.html'>&larr; all episodes</a></nav><main>
<section><h2>Episode</h2>{pre(json.dumps(facts, indent=1, ensure_ascii=False))}<h3>Timeline</h3>{F.timeline(rec.segments, len(traj), link=True)}</section>
<section><h2>Every turn: S2's label, the execution, S1's label</h2>
<p class='note'><b>S2</b>: the turn's System2 target - decision (who executes this turn's step: System1 or the harness), judge
(this turn's verdict on the last step), milestone, subgoal, and the reasoning (Astra's planner_reasoning). Then the
<b>execution</b> (the video, real time) and <b>S1</b>: who executed which subgoal, how the next turn judged it, and whether it goes
into System1's training data (as which span, with which Estimated Length, or why not). The fold below each turn has Astra's whole
reply, its code and what it printed.</p>
{turn_table}</section>
<section><h2>S1 training spans ({len(kept_sp)}, {sum(sp.steps for sp in kept_sp)} steps + 1 halt frame each, none after the last frame or a step left partway)</h2>
{table(['steps', 'length', 'moved', 'subgoal (prompt)', 'Estimated Length', 'turns', 'why', 'video'], sp_rows) if sp_rows else '<p>none</p>'}</section>
<section><h2>S2 rows ({len(rows_v2.trained(rows))} trained, {len(rows) - len(rows_v2.trained(rows))} skipped)</h2>{''.join(cards)}</section></main></body></html>"""
    (out / f"{slug}.html").write_text(page)
    reasons = collections.Counter(sp.reason.split(" (")[0] for sp in spans if not sp.kept)
    dec = collections.Counter(r["meta"].get("decision") for r in rows if r["meta"]["kind"] != "closing")
    judges = collections.Counter(r["labels"]["judge"] for r in rows)
    consist = collections.Counter()
    for r in rows:
        th, d = r["labels"]["thought"].lower(), r["meta"].get("decision")
        s1 = bool(re.search(r"(system1|the model) (should|can|will|could)", th))
        h = bool(re.search(r"(harness|by hand|expert) (should|can|will|must)", th))
        if d and (s1 or h):
            consist["ok" if (d == "s1" and s1 and not h) or (d == "harness" and h and not s1) else "mismatch"] += 1
    ex = []  # (kind, turn) of the episode's examples of the data rules, for the index's "examples to check"
    for sp in spans:
        if sp.kind == "harness_merged" and sp.kept:
            ex.append(("harness fragments merged (worded as the last)", sp.turns[0]))
        if sp.kind == "recovery_request" and sp.kept:
            ex.append(("recovery on request (the fix alone)", sp.turns[0]))
        if sp.by == "recovery":
            ex.append(("recovery within the step, after a failure (System1's frames context)" if sp.train_from is not None
                       else "recovery within the step, after an incomplete step (all frames)", sp.turns[0]))
        elif sp.reason.startswith("before the last miss"):
            ex.append(("long harness struggle: dropped before its last miss", sp.turns[0]))
        elif sp.kept and sp.by == "harness" and sp.grip_relabel:
            ex.append(("partial grip kept, gripper relabeled", sp.turns[0]))
        elif sp.kept and sp.by == "harness" and sp.steps > 250 and s1_select.is_navigation(sp.primitive, traj.lean_act[sp.start:sp.end + 1]):
            ex.append(("long navigation span kept whole", sp.turns[0]))
    from selfimprove.data.s1_select import _harness_runs, is_navigation
    for run in _harness_runs(rec.segments) if rec.success else []:
        man = [x for x in run if not is_navigation(s1_select.primitive(x.subgoal), traj.lean_act[x.start:x.end + 1])]
        if len({x.turn for x in man}) > s1_select.STRUGGLE_STEPS and all(x.judge in ("subgoal_complete", "task_done") for x in man):
            ex.append(("clean long harness streak, every step kept", run[0].turn))
    return {"slug": slug, "task": rec.task, "seed": rec.seed, "examples": ex, "kind": res.get("task_set"), "success": rec.success,
            "step": res.get("step"), "limit": res.get("step_limit"), "turns": res.get("turns"),
            "s1_share": json.loads((folder / "rollout.json").read_text()).get("s1_step_share"),
            "s1_spans": len(kept_sp), "s1_steps": sum(sp.steps for sp in kept_sp), "s1_samples": sum(sp.trained_steps + (sp.halt and sp.end < len(traj) - 1) for sp in kept_sp),
            "dropped": dict(reasons), "s2_rows": len(rows_v2.trained(rows)),
            "s2_short": len(rows) - len(rows_v2.trained(rows)), "decisions": dict(dec), "judges": dict(judges),
            "consistency": dict(consist)}


RULES_HTML = """
<h3>System1 (selfimprove/data/s1_select.py, s1_shards.py)</h3>
<p class='note'>Every span carries its kind (si_kind in the shard meta), shown on the pages in its colour:
<span class='b k-s1'>System1's own</span> <span class='b k-harness'>harness</span> <span class='b k-harness_merged'>harness merged</span>
<span class='b k-recovery_within'>recovery within the step</span> <span class='b k-recovery_request'>recovery on request</span></p>
<ul>
<li><b>Successful episodes only.</b> A span is a step's control steps with its subgoal as the prompt, the anchor picture at its
start, Estimated Length = the length grid's value at or below its length (at least 50), Quality Success, one halt frame
after it (none after the episode's last frame or a step left partway).</li>
<li><b>System1's own steps:</b> kept as they are when the next turn judged them complete (the words it was given, "continue
to X" / "X again" included); dropped when judged incomplete or failed, or stopped at the step budget.</li>
<li><b>The harness's steps:</b> consecutive turns on one step (X, "continue to X", "X again") are one span, in Astra's
words of its first turn; kept when completed, or left partway (then without a halt frame); dropped when judged failed,
very jerky (jerk over the demos' p99.9 on more than 10% of steps or 3x it once), whole-body (base and arm together) or
shorter than 10 steps. A gripper held partly open/closed is relabeled to the demos' -1 / +1. A manipulation over 400
steps is cut into X and "continue to X"; navigation stays whole.</li>
<li><b>Harness fragments merged:</b> consecutive kept harness spans (nothing failed or System1's between) each under 50
steps are joined until the span reaches 50 steps, worded as the last one ("grasp X" = reach + grasp, "retract the arm" =
release + retract); a navigation is never joined to a manipulation, nothing is joined across a milestone, nor a
recovery fix.</li>
<li><b>A two-part turn</b> (a few harness motions, e.g. opening the gripper, then robot.s1() in the same turn) is one System1
step under the turn's subgoal, the preparation included.</li>
<li><b>Recovery:</b> System1's miss at X (failed, incomplete or a timeout), then X finished by System1's own retry (no
bound) or by the harness in at most 2 turns, the
miss and the fix together under 125 steps (no bound for a navigation: a search or a drive), gives two examples: <i>within the step</i> (System1's start to the fix's end,
worded X; after a failure or a timeout System1's frames are context only - anchor, progress - and only the fix's frames
are trained; after an incomplete step all are) and <i>on request</i> (the fix alone, "X again" / "continue to X").</li>
<li><b>Long harness struggles:</b> a run of harness steps with more than 2 manipulation steps and a miss: only the clean
tail after the last miss is kept. Runs without a miss keep every step.</li>
<li>Training: 80% these spans, 20% the base data, batch 256, 15k steps from the 270k checkpoint.</li>
</ul>
<h3>System2 (selfimprove/s2/rows_v2.py)</h3>
<ul>
<li><b>Every turn that moved the robot, of every episode</b> (failures too) is one row; short harness turns are joined
as System1's fragments are (one row, decision expert, the merged words). Its input: the task; the <b>milestones</b>, each
done / now / to do, with only the current one's last 5 steps ("(done)", "(partial)", "(failed)"; the attempts at one step
are one line) - going back (M3 -> M2) shows "now, again" with a "(back from M3)" marker and empties M3 ("to
do, tried before"); the last step's video (4 fps, 3 cameras, at most 32 frames; the first turn: the opening scene); the
gripper status and the task status from the benchmark: "begin", "ongoing", "success" / "failed" (the closing rows).</li>
<li><b>The target:</b> &lt;thought&gt; (Astra's planner_reasoning: System1 is "the model", the harness "the expert"),
&lt;judge&gt;, &lt;milestones&gt; (first turn), &lt;milestone&gt;, &lt;decision&gt; (model | expert: who executed the
step; a turn that called System1 is model) and &lt;subgoal&gt;. Each episode ends with a closing row: the last step's video
and its status, success -> task_finish, failed -> task_failed.</li>
</ul>"""


def compare_section(summ: list[dict], other: str) -> str:
    """This round against ``other``'s site summary on the same recordings (task, seed): per task set and per task."""
    path = ROOT / "site" / f"debug_{other}" / "summary.json"
    if not path.exists():
        return f"<section><h2>Compared with {esc(other)}</h2><p>no {esc(str(path))}</p></section>"
    old = {(e["task"], e["seed"]): e for e in json.loads(path.read_text())["episodes"]}
    pairs = [(e, old[(e["task"], e["seed"])]) for e in summ if (e["task"], e["seed"]) in old]
    if not pairs:
        return ""

    def stats(ps):
        n = len(ps)
        a, b = sum(bool(o["success"]) for _, o in ps), sum(bool(e["success"]) for e, _ in ps)
        lost = sum(bool(o["success"]) and not e["success"] for e, o in ps)
        won = sum(bool(e["success"]) and not o["success"] for e, o in ps)
        share = lambda xs: sum((x["s1_share"] or 0) for x in xs) / max(1, len(xs))  # noqa: E731
        return [str(n), f"{a}/{n} ({a / n:.0%})", f"{b}/{n} ({b / n:.0%})", f"{b - a:+d}", f"{won} / {lost}",
                f"{share([o for _, o in ps]):.2f} &rarr; {share([e for e, _ in ps]):.2f}",
                f"{sum(o['step'] for _, o in ps) / n:.0f} &rarr; {sum(e['step'] for e, _ in ps) / n:.0f}"]
    head = ["", "recordings", f"{esc(other)}", "this run", "change", "won / lost", "S1 step share", "mean steps"]
    by_kind = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    for e, o in pairs:
        by_kind[e["kind"] or ""].append((e, o))
        by_task[(e["kind"] or "", e["task"])].append((e, o))
    rows = [[esc(k)] + stats(v) for k, v in sorted(by_kind.items())] + [["<b>all</b>"] + stats(pairs)]
    trows = []
    for (k, t), v in sorted(by_task.items(), key=lambda kv: (sum(bool(e["success"]) for e, _ in kv[1]) -
                                                             sum(bool(o["success"]) for _, o in kv[1]), kv[0])):
        trows.append([f"{esc(k)}: <a href='#' onclick=\"document.getElementById('q').value='{esc(t)}';filt();"
                      f"document.getElementById('q').scrollIntoView();return false;\">{esc(t)}</a>"] + stats(v))
    return (f"<section><h2>Compared with {esc(other)} on the same recordings</h2>"
            f"<p class='note'>{len(pairs)} episodes of this run started from recordings {esc(other)} also played; won / lost: "
            f"recordings only this run / only {esc(other)} succeeded on. Per task: largest losses first.</p>"
            f"{table(head, rows)}{details('per task', table(['task'] + head[1:], trows))}</section>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--index-only", action="store_true", help="redraw index.html from summary.json (pages unchanged)")
    ap.add_argument("--reuse-media", action="store_true", help="keep the pictures and clips of an earlier build (pages redone)")
    ap.add_argument("--compare", default=None, help="another experiment's site (site/debug_<EXP>): compare on the same recordings")
    args = ap.parse_args()
    cfg = Config.load(None, [f"exp_id={args.exp}"])
    s3.configure(cfg.get("aws_profile"), cfg.region)
    from selfimprove.config import exp_dir

    legacy = exp_dir(args.exp) / "rollout"  # where the debug round's records are
    # the preprocessing stage's download folder (selfimprove/data/stage.py), so a round is downloaded once
    cache = args.cache or (legacy if legacy.exists() else exp_dir(args.exp) / f"iter_{args.iteration:02d}" / "rollout")
    out = args.out or ROOT / "site" / f"debug_{args.exp}"
    if args.index_only:
        summ = json.loads((out / "summary.json").read_text())["episodes"]
    else:
        if out.exists() and not args.reuse_media:
            shutil.rmtree(out)
        (out / "assets").mkdir(parents=True, exist_ok=True)
        folders = fetch(cfg, args.iteration, cache)
        with ProcessPoolExecutor(args.jobs) as pool:
            summ = list(pool.map(episode, [(str(f), str(out), bool(args.reuse_media)) for f in folders]))
    summ.sort(key=lambda s: (s["kind"] or "", s["task"], s["seed"]))
    tot = collections.Counter()
    drop, dec, judges, cons = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for s in summ:
        tot["episodes"] += 1
        tot["success"] += bool(s["success"])
        for k in ("s1_spans", "s1_steps", "s1_samples", "s2_rows", "s2_short"):
            tot[k] += s[k]
        drop.update(s["dropped"]); dec.update(s["decisions"]); judges.update(s["judges"]); cons.update(s["consistency"])
    rows = [[f"<a href='{esc(s['slug'])}.html'>{esc(s['task'])}</a>", str(s["seed"]), esc(s["kind"]),
             "<span class='b keep'>success</span>" if s["success"] else "<span class='b skip'>failed</span>", f"{s['step']}/{s['limit']}", str(s["turns"]),
             f"{(s['s1_share'] or 0):.2f}", f"{s['s1_spans']} ({s['s1_samples']})", str(s["s2_rows"]),
             esc(", ".join(f"{k} {v}" for k, v in s["decisions"].items())),
             esc(", ".join(f"{k} {v}" for k, v in sorted(s["dropped"].items(), key=lambda kv: -kv[1])))] for s in summ]
    agg = table(["", "value"], [
        ["episodes / successes", f"{tot['episodes']} / {tot['success']} ({tot['success'] / max(1, tot['episodes']):.1%})"],
        ["S1 training spans / samples", f"{tot['s1_spans']} / {tot['s1_samples']} (steps + 1 halt frame each, none after the episode's last frame or a step left partway)"],
        ["S1 spans dropped, by reason", esc(", ".join(f"{k} {v}" for k, v in drop.most_common()))],
        ["S2 rows (trained)", f"{tot['s2_rows']} (+ {tot['s2_short']} not trained: a video under 4 frames)"], ["S2 decisions", esc(", ".join(f"{k} {v}" for k, v in dec.most_common()))],
        ["S2 judges", esc(", ".join(f"{k} {v}" for k, v in judges.most_common()))],
        ["reasoning vs decision (where the reasoning names an executor)", esc(", ".join(f"{k} {v}" for k, v in cons.items()))],
    ])
    # per task: success, System1's share, length, the training data it gives
    per = collections.defaultdict(list)
    for s in summ:
        per[(s["kind"] or "", s["task"])].append(s)
    trows, kinds = [], collections.defaultdict(list)
    for (kind, task), eps in sorted(per.items(), key=lambda kv: (kv[0][0], sum(bool(e["success"]) for e in kv[1]) / len(kv[1]))):
        ok = sum(bool(e["success"]) for e in eps)
        kinds[kind].extend(eps)
        mean = lambda key: sum((e[key] or 0) for e in eps) / len(eps)  # noqa: E731
        trows.append([esc(kind), f"<a href='#' onclick=\"document.getElementById('q').value='{esc(task)}';filt();"
                      f"document.getElementById('q').scrollIntoView();return false;\">{esc(task)}</a>",
                      f"{ok}/{len(eps)} ({ok / len(eps):.0%})", f"{mean('s1_share'):.2f}", f"{mean('step'):.0f}",
                      f"{mean('turns'):.1f}", str(sum(e["s1_samples"] for e in eps)),
                      esc(", ".join(f"{e['seed']}" for e in eps if not e["success"]))])
    krows = []
    for kind, eps in sorted(kinds.items()):
        ok = sum(bool(e["success"]) for e in eps)
        krows.append([esc(kind), str(len({e["task"] for e in eps})), f"{ok}/{len(eps)} ({ok / len(eps):.1%})",
                      f"{sum((e['s1_share'] or 0) for e in eps) / len(eps):.2f}", str(sum(e["s1_samples"] for e in eps))])
    exm = collections.defaultdict(list)
    for s_ in summ:
        for kind, turn in s_.get("examples") or []:
            exm[kind].append((s_["slug"], s_["task"], s_["seed"], turn))
    ex_rows = []
    for kind, items in sorted(exm.items()):
        links = ", ".join(f"<a href='{esc(sl)}.html#turn{t}'>{esc(tk)} {sd} t{t}</a>" for sl, tk, sd, t in items[:10])
        ex_rows.append([esc(kind), str(len(items)), links])
    examples = table(["data rule", "cases", "examples (first 10; the link opens the turn)"], ex_rows) if ex_rows else ""
    per_task = (table(["set", "tasks", "success", "S1 step share", "S1 samples"], krows) +
                "<p class='note'>per task, worst first within each set; click a task to list its episodes below</p>" +
                table(["set", "task", "success", "S1 step share", "mean steps", "mean turns", "S1 samples", "failed seeds"],
                      trows))
    compared = compare_section(summ, args.compare) if args.compare else ""
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(args.exp)} iteration {args.iteration}</title><style>{CSS}{EXTRA_CSS}</style>{FILTER_JS}</head>
<body><header><h1>{esc(args.exp)}, iteration {args.iteration}: every episode with its S1 and S2 labels</h1>
<p>S1 + harness rollouts (Astra + System1, started from recorded episodes); S1 data by the agreed rules, S2 rows in the new single-mode
format. Built by selfimprove/scripts/build_debug_site.py. The data formats: <a href='../formats/index.html'>comparison site</a>.</p></header>
<main><section><h2>Totals</h2>{agg}</section>
<section><h2>Results per task set and per task</h2>{per_task}</section>
{compared}
<section><h2>How the training data is built</h2>{RULES_HTML}</section>
<section><h2>S1 data rules: examples to check</h2>{examples}</section>
<section><h2>Episodes</h2><div class='filter'><input id='q' placeholder='filter (task, seed, success...)' oninput='filt()'></div>
{table(['task', 'seed', 'set', 'success', 'steps', 'turns', 'S1 step share', 'S1 spans (samples)', 'S2 rows', 'S2 decisions', 'S1 drops'], rows, 'eps')}
</section></main></body></html>"""
    (out / "index.html").write_text(page)
    if args.index_only:
        print(f"{out / 'index.html'}: redrawn ({tot['episodes']} episodes)")
        return 0
    (out / "summary.json").write_text(json.dumps({"totals": tot, "dropped": drop, "decisions": dec, "judges": judges,
                                                  "consistency": cons, "episodes": summ}, indent=1, default=str))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out / 'index.html'}: {tot['episodes']} episodes ({size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

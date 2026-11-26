"""An index of the built training shards (S1 + S2) that links every span / row into RoboAnnotator's shard inspectors.

    .venv/bin/python scripts/build_shard_index.py --view ~/selfimprove_work/shard_view --out site/shards_sub50_check_20261125 \\
        [--inspector http://127.0.0.1:7891]

The inspectors (RoboAnnotator ``/system1_training_sample``, ``/system2_training_sample``) read the shard tars exactly as the
trainers do, one sample at a time by a global index (the manifest's shard order, then the tar order of a shard's
``.meta.json`` / ``.json`` members). This page computes those indices, so each S1 span (its first sample, its halt
frame) and each S2 row is one click away, and runs the checks a reader would do by eye on every span: contiguous
frames, the halt frame, progress 0 -> 1, one subgoal and one anchor per span.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfimprove.data.shard_checks import s1_scan, s1_spans  # noqa: E402

_S1_KEY = re.compile(r"__(?P<task>[A-Za-z0-9]+)__seed_(?P<seed>\d+)__(?P<variant>[a-z0-9_]+)__s(?P<s>\d+)-(?P<e>\d+)__f(?P<f>\d+)$")


def esc(x) -> str:
    return html.escape(str(x))


def manifest(root: Path) -> list[tuple[str, int]]:
    rows = [json.loads(x) for x in (root / "shards" / "manifest.jsonl").read_text().splitlines() if x.strip()]
    return [(Path(r["shard"]).name, int(r["num_samples"])) for r in rows]




def s2_scan(path: str) -> list[dict]:
    out = []
    with tarfile.open(path) as t:
        names = set(t.getnames())
        for m in t.getmembers():
            if m.name.endswith(".json"):
                key = m.name[:-len(".json")]
                rec = json.loads(t.extractfile(m).read())
                meta, lab = rec.get("meta", {}), rec.get("labels", {})
                media = "video" if f"{key}.videobin" in names else ("image" if f"{key}.imagebin0" in names else "-")
                out.append({"key": key, "task": meta.get("task"), "seed": meta.get("seed"), "turn": meta.get("turn"),
                            "kind": meta.get("kind"), "judge": lab.get("judge"), "decision": meta.get("decision"),
                            "frames": meta.get("clip_frames"), "media": media, "old": meta.get("old_sub_mode"),
                            "ep": meta.get("episode_id"), "subgoal": lab.get("subgoal"), "success": meta.get("episode_success")})
    return out


def indexed(root: Path, scan, workers: int, limit: int = 0) -> list[dict]:
    shards = manifest(root)
    if limit:
        shards = shards[:limit]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(scan, [str(root / "shards" / s) for s, _ in shards]))
    rows, i = [], 0
    for (name, n), part in zip(shards, parts):
        if len(part) != n:
            print(f"warning: {name} manifest says {n}, the tar has {len(part)}")
        for r in part:
            r["i"], r["shard"] = i, name
            i += 1
            rows.append(r)
    return rows




CSS = """body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#1d2330;max-width:1500px}
h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #e3e6ec;padding-bottom:4px}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0}th,td{border:1px solid #e3e6ec;padding:3px 7px;vertical-align:top}
th{background:#f4f6f9;position:sticky;top:0}td.n{text-align:right;font-variant-numeric:tabular-nums}
.b{display:inline-block;padding:0 6px;border-radius:9px;font-size:11.5px;color:#fff}
.s1{background:#2f7d4f}.harness{background:#b06a10}.harness_merged{background:#8a4fb0}.recovery_within{background:#1f6fb2}
.recovery_request{background:#0f8a8a}.bad{background:#c0392b}.ok{background:#2f7d4f}.note{color:#68707d}
input.f{padding:4px 8px;width:260px;margin:4px 8px 4px 0}a{color:#1f5fbf;text-decoration:none}a:hover{text-decoration:underline}
.cards{display:flex;gap:14px;flex-wrap:wrap}.card{border:1px solid #e3e6ec;border-radius:8px;padding:10px 14px;min-width:200px}
.card b{font-size:18px}"""

FILTER_JS = """<script>
function filt(inp, id){const q=inp.value.toLowerCase().split(/\\s+/).filter(Boolean);
for(const tr of document.querySelectorAll('#'+id+' tbody tr')){const t=tr.textContent.toLowerCase();
tr.style.display=q.every(w=>t.includes(w))?'':'none';}}
</script>"""


def table(head, rows, tid=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td{' class=n' if str(c)[:1].isdigit() and len(str(c)) < 12 else ''}>{c}</td>" for c in r)
                + "</tr>" for r in rows)
    return f"<table id='{tid}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inspector", default="http://127.0.0.1:7891")
    ap.add_argument("--s1", default="si_sub50check_iter00")
    ap.add_argument("--s2", default="si_sub50check_rollout")
    ap.add_argument("--s2-old", default="", help="an old-set view to show too (e.g. old_full0804_v4); none by default")
    ap.add_argument("--old-shards", type=int, default=24, help="old-set shards scanned for the example links")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    I1 = f"{args.inspector}/system1_training_sample?dataset={args.s1}&i="
    I2 = lambda ds: f"{args.inspector}/system2_training_sample?dataset={ds}&i="  # noqa: E731

    s1 = indexed(args.view / "s1" / args.s1, s1_scan, args.workers)
    spans = s1_spans(s1)
    s2 = indexed(args.view / "s2" / args.s2, s2_scan, args.workers)
    has_old = bool(args.s2_old) and (args.view / "s2" / args.s2_old / "shards").exists()
    old = indexed(args.view / "s2" / args.s2_old, s2_scan, args.workers, limit=args.old_shards) if has_old else []
    old_meta = json.loads((args.view / "s2" / args.s2_old / "meta.json").read_text()) if has_old else {}

    s1_meta = json.loads((args.view / "s1" / args.s1 / "meta.json").read_text())
    partial = len(manifest(args.view / "s1" / args.s1)) < int(s1_meta.get("shards") or 0)
    if partial:  # samples are shuffled over every shard: a span is whole only in all of them -> no per-span verdict here
        for sp in spans:
            sp["issues"] = []
    full_checks = s1_meta.get("span_checks") or {}
    kinds = collections.Counter(sp["kind"] for sp in spans)
    samples_by_kind = collections.Counter(r["kind"] for r in s1)
    bad = [sp for sp in spans if sp["issues"]]
    issue_count = collections.Counter(i.split(" ")[0] + " " + i.split(" ")[1] for sp in bad for i in sp["issues"])
    span_rows = []
    for sp in spans:
        link = f"<a href='{I1}{sp['first']}' target=_blank>#{sp['first']}</a>"
        hl = f"<a href='{I1}{sp['halt_i']}' target=_blank>#{sp['halt_i']}</a>" if sp["halt_i"] is not None else "-"
        chk = (f"<span class='b bad'>{esc('; '.join(sp['issues']))}</span>" if sp["issues"] else "<span class='b ok'>ok</span>")
        trained = f"{sp['from']}-{sp['e']}" + (f" <span class=note>(context {sp['s']}-{sp['from'] - 1})</span>"
                                               if sp["from"] is not None and sp["from"] > sp["s"] else "")
        span_rows.append([esc(sp["task"]), sp["seed"], f"<span class='b {esc(sp['kind'])}'>{esc(sp['kind'])}</span>",
                          esc(sp["sub"]), f"{sp['s']}-{sp['e']}", trained, sp["real"], sp["halt"], sp["est"],
                          f"{sp['p0']:.2f} → {sp['p1']:.2f}" if sp["p0"] is not None else "-", esc(sp["grip"]),
                          esc(", ".join(map(str, sp["turns"] or []))), link, hl, chk])
    s2_rows = [[r["i"] and f"<a href='{I2(args.s2)}{r['i']}' target=_blank>#{r['i']}</a>" or
                f"<a href='{I2(args.s2)}0' target=_blank>#0</a>", esc(r["task"]), r["seed"], r["turn"], esc(r["kind"]),
                esc(r["judge"]), esc({"s1": "model", "harness": "expert"}.get(r["decision"], r["decision"] or "-")),
                r["media"] + (f" ({r['frames']} fr)" if r["frames"] else ""), esc(r["subgoal"] or ""),
                "success" if r["success"] else "failed"] for r in sorted(s2, key=lambda r: (r["task"], int(r["seed"]), int(r["turn"])))]
    old_by = collections.defaultdict(list)
    for r in old:
        old_by[(r["old"], r["kind"], r["judge"])].append(r)
    old_rows = [[esc(m), esc(k), esc(j), len(rs), " ".join(f"<a href='{I2(args.s2_old)}{r['i']}' target=_blank>#{r['i']}</a>"
                                                         for r in rs[:12])]
                for (m, k, j), rs in sorted(old_by.items(), key=lambda kv: -len(kv[1]))]
    s2k = collections.Counter((r["kind"], r["judge"]) for r in s2)
    s2d = collections.Counter({"s1": "model", "harness": "expert"}.get(r["decision"], "-") for r in s2)
    cards = lambda items: "<div class=cards>" + "".join(f"<div class=card><b>{v}</b><br>{k}</div>" for k, v in items) + "</div>"  # noqa: E731
    page = f"""<!doctype html><meta charset=utf-8><title>Training shards: {esc(args.s1)}</title><style>{CSS}</style>{FILTER_JS}
<h1>Training shards: S1 <code>{esc(args.s1)}</code>, S2 <code>{esc(args.s2)}</code></h1>
<p class=note>Every link opens RoboAnnotator's shard inspector on that sample, read from the tar exactly as the trainer reads it
(<a href='{args.inspector}/system1_training_sample?dataset={args.s1}' target=_blank>S1 inspector</a>,
<a href='{args.inspector}/system2_training_sample?dataset={args.s2}' target=_blank>S2 inspector</a>; both also offer the
reference sets <code>ref_base_system1_midset_0717_val</code> and <code>ref_old_system2_target_0804</code> in the old formats
for comparison). In training, S1 batches are 80% these spans / 20% the base set; S2 batches 50% these rows / 50% the converted
old set.</p>

<h2>S1: {len(s1):,} samples in {len(spans):,} spans ({len(manifest(args.view / 's1' / args.s1))} shards)</h2>
{cards([(f"{k} spans", f"{v:,} ({samples_by_kind[k]:,} samples)") for k, v in kinds.most_common()] +
        [("spans failing a check", len(bad))])}
<p>Checks on every span: frames contiguous and ending at the span end; the first frame at the span start (a recovery within the
step starts at its fix, the frames before are context: anchor and progress only); progress monotone in [0, 1]; one subgoal;
at most one halt frame. """ + ((f"<b>This page reads {len(manifest(args.view / 's1' / args.s1))} of the round's "
      f"{s1_meta.get('shards')} shards</b> (the rest are on S3): samples are shuffled over every shard, so a span is whole only "
      f"in all of them and the per-span checks ran on all shards at preprocessing: "
      + (f"<b>{full_checks.get('spans', 0):,} spans, {full_checks.get('failing', 0):,} failing</b>"
         + (f" ({esc(full_checks.get('issues'))})" if full_checks.get('failing') else "") if full_checks else
         "not recorded for this round (preprocessed before the checks were added; round 0 passed them on all shards: 41,159 spans, "
         "0 failing, 2026-11-26 09:17)")
      + ". The table lists the parts of spans in these shards.") if partial else
     (('Failing: ' + esc(dict(issue_count))) if bad else 'All spans pass.')) + """</p>
<input class=f placeholder='filter spans (task, kind, subgoal...)' oninput="filt(this,'s1t')">
{table(['task', 'seed', 'kind', 'subgoal (prompt)', 'span', 'trained frames', 'frames', 'halt', 'Est. Length', 'progress', 'gripper', 'turns', 'first sample', 'halt sample', 'check'], span_rows, 's1t')}

<h2>S2 rollout rows: {len(s2):,} rows ({len(manifest(args.view / 's2' / args.s2))} shards)</h2>
{cards([(f"{k} · {j}", v) for (k, j), v in s2k.most_common()] + [(f"decision {k}", v) for k, v in s2d.most_common()])}
<input class=f placeholder='filter rows (task, judge, expert...)' oninput="filt(this,'s2t')">
{table(['row', 'task', 'seed', 'turn', 'kind', 'judge', 'decision', 'media', 'subgoal (target)', 'episode'], s2_rows, 's2t')}

""" + (f"""
<h2>S2 old set, converted: {old_meta.get('num_samples', 0):,} rows ({old_meta.get('num_shards')} shards)</h2>
<p>Built from <code>system2_full_0804</code> by <code>selfimprove/s2/convert_full0804.py</code>: rows {esc(old_meta.get('rows'))};
dropped {esc(old_meta.get('dropped'))}. Example links below come from the first {args.old_shards} shards ({len(old):,} rows),
grouped by the old mode they came from.</p>
{table(['old mode', 'kind', 'judge', 'rows (scanned)', 'examples'], old_rows)}
""" if has_old else "")
    (out / "index.html").write_text(page)
    print(f"{out / 'index.html'}: {len(spans)} S1 spans ({len(bad)} failing), {len(s2)} S2 rows, {len(old)} old rows scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

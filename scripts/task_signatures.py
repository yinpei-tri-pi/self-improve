"""Per task: its goal texts and the steps Astra issued (verb + things), from the pilot rollouts (pretrain tasks) and
the local HexaShell runs (target tasks).  Writes assets/task_signatures.json for the task selection."""

import collections
import glob
import json
import re
import sys
from pathlib import Path

STOP = {"the", "a", "an", "of", "to", "in", "on", "into", "onto", "from", "and", "then", "it", "its", "with", "at", "by",
        "for", "is", "are", "be", "that", "this", "up", "down", "out", "off", "over", "back", "again", "continue",
        "open", "closed", "near", "next", "left", "right", "front", "top", "inside", "under", "toward", "towards", "away",
        "you", "your", "any", "all", "one", "two", "each", "other", "some", "so", "as", "if", "or", "until"}


def words(text):
    return [w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in STOP and len(w) > 2]


def steps_of(events_path):
    out = []
    for line in Path(events_path).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "subgoal" and e.get("text"):
            t = str(e["text"]).lower().strip()
            t = re.sub(r"^continue to\s+", "", t)
            t = re.sub(r"\s+again$", "", t)
            out.append((t.split(" ", 1)[0], t, e.get("by")))
    return out


def collect(folders):
    tasks = collections.defaultdict(lambda: {"goals": set(), "verbs": collections.Counter(), "things": collections.Counter(),
                                             "pairs": collections.Counter(), "episodes": 0, "success": 0, "s1": []})
    for f in folders:
        f = Path(f)
        try:
            r = json.loads((f / "result.json").read_text())
        except (OSError, ValueError):
            continue
        t = tasks[r["task"]]
        t["episodes"] += 1
        t["success"] += bool(r.get("success"))
        if r.get("instruction"):
            t["goals"].add(str(r["instruction"]).strip())
        for verb, text, _by in steps_of(f / "events.jsonl") if (f / "events.jsonl").exists() else []:
            t["verbs"][verb] += 1
            for w in words(text)[1:]:
                t["things"][w] += 1
                t["pairs"][f"{verb}:{w}"] += 1
    return tasks


def main():
    pilot = glob.glob("/home/yinpeidai/selfimprove_work/si_v1/iter_00/rollout/*/*")
    local = [p for run in ("full1_s1first", "full2_s1first", "full3_s1first")
             for p in glob.glob(f"/home/yinpeidai/HexaShell-s1/results/{run}/*/")]
    out = {}
    for src, folders in (("pilot", pilot), ("local", local)):
        for task, t in collect(folders).items():
            if task in out and src == "local":
                continue  # a target task in both: the pilot (pretrain kitchens) wins
            out[task] = {"source": src, "goals": sorted(t["goals"])[:6], "episodes": t["episodes"], "success": t["success"],
                         "verbs": dict(t["verbs"].most_common(12)), "things": dict(t["things"].most_common(25)),
                         "pairs": dict(t["pairs"].most_common(40))}
    dest = Path(__file__).resolve().parents[1] / "assets" / "task_signatures.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} tasks -> {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()

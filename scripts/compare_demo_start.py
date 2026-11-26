"""Side by side: the first video frame of a recorded RoboCasa episode (top row) and the start pictures of the
HexaShell episode that split demo started from it (bottom row), for the left, right and wrist cameras.

    python scripts/compare_demo_start.py EXP [--iteration 0] [--data ~/data/robocasa_dataset/v1.0]
        [--out ~/selfimprove_work/<EXP>/compare]

Reads the plan and each played episode's record from S3 (``rollout/<task>/<seed>/``: result.json for the recorded
episode it started, pictures/00N_start_<cam>.jpg) and the recording's videos from the local datasets. Writes one PNG
per episode (``<task>_<demo id>.png``) and ``index.html`` with all of them. Needs ffmpeg and Pillow.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfimprove import s3  # noqa: E402
from selfimprove.config import Config  # noqa: E402

CAMS = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}


def first_frame(video: Path, size: int):
    from PIL import Image

    png = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB").resize((size, size))


def main() -> int:
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--data", type=Path, default=Path("~/data/robocasa_dataset/v1.0").expanduser())
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    cfg = Config.load(None, [f"exp_id={args.exp}"])
    s3.configure(cfg.get("aws_profile"), cfg.region)
    out = args.out or (Path(cfg["work_dir"]).expanduser() / args.exp / "compare")
    out.mkdir(parents=True, exist_ok=True)
    plan = s3.read_json(f"{cfg.rollout_uri(args.iteration)}/_plan.json")
    rows = []
    for e in plan["episodes"]:
        uri = cfg.episode_uri(args.iteration, e["task"], e["seed"])
        result = s3.read_json(f"{uri}/result.json", default=None)
        if result is None:
            print(f"{e['task']}/{e['seed']}: not played yet")
            continue
        demo = result.get("demo") or {}
        src = args.data / demo["source"]          # .../<Task>/<date>/lerobot/extras/episode_N
        lerobot, ep = src.parent.parent, src.name
        names = {Path(k).name: k for k, _ in s3.iter_keys(f"{uri}/pictures/")}
        s = args.size
        sheet = Image.new("RGB", (3 * s, 2 * s + 60), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (cam, name) in enumerate(CAMS.items()):
            sheet.paste(first_frame(lerobot / "videos" / "chunk-000" / f"observation.images.{name}" / f"{ep}.mp4", s),
                        (i * s, 30))
            pic = next((k for n, k in names.items() if n.endswith(f"_start_{cam}.jpg")), None)
            if pic:
                sheet.paste(Image.open(io.BytesIO(s3.read_bytes(pic))).convert("RGB").resize((s, s)), (i * s, s + 60))
        draw.text((6, 8), f"recording {demo['id']} (first frame)   kitchen {result.get('layout_id')}/{result.get('style_id')}",
                  fill="black")
        draw.text((6, s + 38), f"HexaShell {e['task']} seed {e['seed']} start   success {result.get('success')}", fill="black")
        name = f"{e['task']}_{demo['id']}.png"
        sheet.save(out / name)
        rows.append((e["task"], demo["id"], result.get("instruction"), result.get("success"), name))
        print(f"{e['task']}: {demo['id']} -> {out / name}")
    html = ["<html><body style='font-family:sans-serif'>", f"<h2>{args.exp}: recording (top) vs HexaShell start (bottom)</h2>"]
    for task, did, instr, ok, name in rows:
        html += [f"<h3>{task} {did}: {instr} (success {ok})</h3>", f"<img src='{name}' width='100%'>"]
    (out / "index.html").write_text("\n".join(html + ["</body></html>"]))
    print(f"{len(rows)} comparisons in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

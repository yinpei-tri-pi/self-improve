"""docs/pipeline.png + docs/pipeline.svg: the self-improvement loop, minimal text.

    <python with matplotlib> docs/make_pipeline_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK = "#1f2430"
MUTED = "#6a7080"
FILL = {"astra": "#F2C4A0", "s1": "#A9C6EA", "eval": "#C9E4C5", "s2": "#F6DE9C"}
EDGE = {"astra": "#C98B5E", "s1": "#5B8BC9", "eval": "#6FA86A", "s2": "#C9A646"}


def box(ax, cx, cy, w, h, kind, title, sub):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=FILL[kind], ec=EDGE[kind], lw=2))
    ax.text(cx, cy + 0.025, title, ha="center", va="center", fontsize=22, weight="bold", color=INK)
    ax.text(cx, cy - 0.04, sub, ha="center", va="center", fontsize=14, color=INK)


def arrow(ax, a, b, label="", rad=0.0, color=INK, ls="-", dx=0.0, dy=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=24, lw=2.4, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))
    if label:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", va="center", fontsize=14,
                color=color, style="italic")


def main() -> None:
    fig = plt.figure(figsize=(14, 6.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2.12)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.06, 0.93, "S1 / S2 self-improvement loop", fontsize=24, weight="bold", color=INK, va="center")

    w, h, top, bot = 0.52, 0.2, 0.62, 0.22
    xs = (0.36, 1.06, 1.76)
    box(ax, xs[0], top, w, h, "astra", "① Rollout", "Astra + S1  ·  Ray")
    box(ax, xs[1], top, w, h, "astra", "② Filter", "Astra as PRM  →  S1 shards")
    box(ax, xs[2], top, w, h, "s1", "③ Train S1", "AWR + unary  ·  SageMaker")
    box(ax, xs[2], bot, w, h, "eval", "④ Target eval", "S1 + S2  ·  Ray")
    box(ax, xs[1], bot, w, h, "s2", "⑤ Train S2", "once, at the end")

    arrow(ax, (xs[0] + w / 2, top), (xs[1] - w / 2, top))
    arrow(ax, (xs[1] + w / 2, top), (xs[2] - w / 2, top))
    arrow(ax, (xs[2], top + h / 2), (xs[0], top + h / 2), "ckpt k+1", rad=0.17, color=EDGE["s1"], dy=0.13)
    arrow(ax, (xs[2], top - h / 2), (xs[2], bot + h / 2), "ckpt k+1", color=EDGE["eval"], dx=0.09)
    arrow(ax, (xs[1], top - h / 2), (xs[1], bot + h / 2), "all rounds", color=EDGE["s2"], ls="--", dx=0.1)

    ax.text(0.36, bot, "state on S3\nprogress in W&B", ha="center", va="center", fontsize=14, color=MUTED,
            linespacing=1.6)
    fig.savefig(OUT / "pipeline.png", dpi=150)
    fig.savefig(OUT / "pipeline.svg")
    print(OUT / "pipeline.png")


if __name__ == "__main__":
    main()

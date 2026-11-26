"""``selfimprove <command>``: the loop (run / tick / status) and each stage on its own.

    selfimprove run|tick|status  --config configs/<exp>.yaml [--set key=value] [--dry-run]
    selfimprove rollout          --iteration K --ckpt CKPT [--local] [--limit N]     (selfimprove.rollout.driver)
    selfimprove preprocess       --iteration K                                       (selfimprove.data.stage)
    selfimprove train            --iteration K --ckpt CKPT [--dry-run]               (selfimprove.train.launch)
    selfimprove eval             --iteration K --ckpt CKPT [--local] [--limit N]     (selfimprove.eval.target)
    selfimprove node             --ckpt CKPT --task T --seed S [--upload]            (one episode, no Ray)
    selfimprove mock-wandb       [--exp-id mock_demo]                                (a simulated experiment in W&B)
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("run", "tick", "status"):
        from .scheduler import main as m

        return m([cmd, *rest])
    table = {"rollout": "selfimprove.rollout.driver", "preprocess": "selfimprove.data.stage",
             "train": "selfimprove.train.launch", "eval": "selfimprove.eval.target", "node": "selfimprove.rollout.node",
             "mock-wandb": "selfimprove.mock_wandb"}
    if cmd not in table:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    import importlib

    return importlib.import_module(table[cmd]).main(rest)


if __name__ == "__main__":
    raise SystemExit(main())

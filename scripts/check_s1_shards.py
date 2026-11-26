"""Read rollout shards through openpi's own RoboCasa loader + transforms, exactly as training does.

    <openpi venv python> scripts/check_s1_shards.py SHARDS_DIR [--ckpt CKPT_DIR] [--batches 2] [--batch-size 8]

Run with the openpi-si checkout on the path (PYTHONPATH=<openpi-si>/src:<openpi-si>/scripts). The config
is the checkpoint's own (scripts/train.py resolve_robocasa_config), so the transforms - lean-state
recompute (asserted equal to the baked lean), Normalize with the checkpoint's norm stats, the prompt,
progress-as-action - are the ones the next round trains with.  Prints the batch shapes and si_* fields.
"""

import argparse
import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("--ckpt", default="/home/yinpeidai/data/sys1_ckpts/f0717-270k-bs512-progact_granfine_verbsimp_noexec/269999")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    os.environ["ROBOCASA_SHARDS_DIR"] = args.shards

    import numpy as np
    import train as _train  # openpi-si/scripts/train.py

    from openpi.training import data_loader as _data_loader

    cfg = _train.resolve_robocasa_config(args.ckpt)
    if cfg is None:
        print("could not resolve the checkpoint's config", file=sys.stderr)
        return 2
    data = dataclasses.replace(cfg.data, shards=args.shards) if hasattr(cfg.data, "shards") else cfg.data
    # norm stats: the openpi checkout's assets/<config>/robocasa_system1 (byte-identical to the checkpoint's)
    cfg = dataclasses.replace(cfg, data=data, batch_size=args.batch_size, num_workers=0)
    loader = _data_loader.create_data_loader(cfg, shuffle=True, num_batches=args.batches, skip_norm_stats=False)
    n = 0
    for obs, act in loader:
        n += 1
        print(f"batch {n}: actions {tuple(act.shape)}, state {tuple(obs.state.shape)}, "
              f"images {sorted(obs.images)}, prompt tokens {tuple(obs.tokenized_prompt.shape)}")
        print(f"  si_label {np.asarray(obs.si_label).tolist()}")
        print(f"  si_weight {np.round(np.asarray(obs.si_weight), 3).tolist()}")
        print(f"  progress dim (action[..., 11]) first sample: {np.round(np.asarray(act[0, :, 11]), 2).tolist()}")
        if n >= args.batches:
            break
    print("OK" if n else "NO BATCHES")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())

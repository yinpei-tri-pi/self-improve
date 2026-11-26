"""A simulated experiment in the loop's W&B project: what a real run will show, with made-up numbers.

    python -m selfimprove.mock_wandb [--exp-id mock_demo] [--iterations 5]

Every run is tagged ``mock`` and lives in its own group (``--exp-id``).  It writes the same runs and keys
as the real stages (selfimprove/wandb_log.py): ``<exp>/scheduler`` (one point per iteration),
``<exp>/iter_NN/rollout`` (the round live), ``<exp>/iter_NN/s1-train`` (openpi's training metrics incl.
the self-improvement terms) and ``<exp>/iter_NN/eval`` (the target split live).  The numbers follow a
plausible trajectory: the target-split rate climbs from the 53.7% baseline, the rollout success and the
share of steps System1 executes rise as S1 improves, the harness's corrections shrink.
"""

from __future__ import annotations

import argparse
import math
import random

GROUPS = {"atomic_seen": (18, 0.794), "composite_seen": (16, 0.544), "composite_unseen": (16, 0.240)}


def main(argv: list[str] | None = None) -> int:
    import wandb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-id", default="mock_demo")
    ap.add_argument("--project", default="s1s2-self-improve")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    exp = args.exp_id
    common = dict(project=args.project, entity=args.entity, group=exp, tags=["mock"], reinit="create_new")

    loop = wandb.init(name=f"{exp}/scheduler", job_type="loop", config={"exp_id": exp, "mock": True}, **common)
    for p in ("rollout", "data", "train", "eval", "loop"):
        loop.define_metric(f"{p}/*", step_metric="iteration")
    loop.define_metric("iteration")
    base = {g: r for g, (_, r) in GROUPS.items()}
    weights = {g: n * 30 for g, (n, _) in GROUPS.items()}
    loop.log({"iteration": 0, "eval/overall": 0.537, **{f"eval/{g}": r for g, r in base.items()},
              **{f"eval/delta_{g}": 0.0 for g in base}, "eval/delta_overall": 0.0})

    s1_share, succ, harness_loss = 0.80, 0.78, 0.74
    for k in range(args.iterations):
        # ---- rollout round k ----------------------------------------------------------------------
        planned = 574 if k == 0 else 2870
        run = wandb.init(name=f"{exp}/iter_{k:02d}/rollout", job_type="rollout",
                         config={"exp_id": exp, "iteration": k, "episodes_planned": planned, "mock": True},
                         **{**common, "tags": ["mock", f"iter_{k:02d}"]})
        run.define_metric("elapsed_h")
        run.define_metric("rollout/*", step_metric="elapsed_h")
        hours = 3.0 if k == 0 else 14.0
        done = 0
        for i in range(1, 41):
            h = hours * i / 40
            done = int(planned * min(1.0, (i / 40) ** 1.1))
            run.log({"elapsed_h": h, "rollout/episodes_done": done, "rollout/episodes_planned": planned,
                     "rollout/progress": done / planned, "rollout/episodes_running": 0 if i == 40 else 128,
                     "rollout/nodes": 0 if i == 40 else 32,
                     "rollout/success_rate": succ + rng.gauss(0, 0.01) * (1 - i / 40),
                     "rollout/success_rate_atomic": min(0.99, succ + 0.12 + rng.gauss(0, 0.01)),
                     "rollout/success_rate_composite": succ - 0.03 + rng.gauss(0, 0.01),
                     "rollout/s1_step_share": s1_share + rng.gauss(0, 0.005),
                     "rollout/surrogate_loss_s1": 0.042 * (0.93 ** k) + rng.gauss(0, 0.001),
                     "rollout/surrogate_loss_agent": harness_loss + rng.gauss(0, 0.01),
                     "rollout/episodes_per_hour": done / max(h, 1e-6),
                     "rollout/tokens_prompt_mean": 480000 * (0.95 ** k), "rollout/tokens_cached_frac": 0.9})
        run.summary.update({"rollout/success_rate": succ, "rollout/s1_step_share": s1_share})
        run.finish()
        loop.log({"iteration": k, "rollout/success_rate": succ, "rollout/s1_step_share": s1_share,
                  "rollout/episodes_done": planned, "rollout/surrogate_loss_agent": harness_loss,
                  "rollout/elapsed_h": hours})

        # ---- preprocess ---------------------------------------------------------------------------
        frames = planned * 900
        pos = int(frames * (0.62 + 0.03 * k))
        neg = int(frames * (0.06 - 0.008 * k))
        loop.log({"iteration": k, "data/episodes": planned, "data/segments": planned * 21, "data/samples": pos + neg,
                  "data/positive_samples": pos, "data/negative_samples": neg, "data/shards": (pos + neg) // 2048,
                  "data/use/s1/positive": int(planned * 21 * s1_share * 0.85),
                  "data/use/s1/negative": int(planned * 21 * s1_share * 0.07),
                  "data/use/agent/positive": int(planned * 21 * (1 - s1_share) * 0.8)})

        # ---- S1 training (openpi's run) -----------------------------------------------------------
        tr = wandb.init(name=f"{exp}/iter_{k:02d}/s1-train", job_type="s1-train",
                        config={"exp_id": exp, "iteration": k, "steps": 10000, "batch_size": 512, "mock": True,
                                "si": {"lambda_pos": 0.1, "lambda_neg": 0.1, "beta_pos": 10, "beta_neg": 10}},
                        **{**common, "tags": ["mock", f"iter_{k:02d}", "s1-train"]})
        for step in range(0, 10001, 100):
            f = step / 10000
            tr.log({"loss": 0.034 - 0.004 * f + rng.gauss(0, 0.0015), "flow_loss_real": 0.021 - 0.003 * f,
                    "si/loss_sl": 0.033 - 0.004 * f + rng.gauss(0, 0.001),
                    "si/loss_pos": math.log(2) * 0.31 * (1 - 0.22 * f), "si/loss_neg": math.log(2) * 0.03 * (1 - 0.5 * f),
                    "si/margin_pos": 0.0035 * f + rng.gauss(0, 0.0003), "si/margin_neg": 0.03 * f + rng.gauss(0, 0.002),
                    "si/delta_base": rng.gauss(0, 0.0004), "si/frac_pos": 0.31, "si/frac_neg": 0.03,
                    "learning_rate": 5e-5 * min(1.0, step / 200), "grad_norm": 0.6 + rng.gauss(0, 0.05)}, step=step)
        tr.finish()
        loop.log({"iteration": k, "train/steps": 10000, "train/si/margin_pos": 0.0035, "train/si/margin_neg": 0.03})

        # ---- target-split eval of ckpt_{k+1} -------------------------------------------------------
        gain = 0.035 * (0.8 ** k)
        new = {"atomic_seen": min(0.95, base["atomic_seen"] + gain * 0.6),
               "composite_seen": min(0.9, base["composite_seen"] + gain * 1.1),
               "composite_unseen": min(0.8, base["composite_unseen"] + gain * 1.3)}
        ev = wandb.init(name=f"{exp}/iter_{k:02d}/eval", job_type="eval",
                        config={"exp_id": exp, "iteration": k, "units": 1500, "mock": True},
                        **{**common, "tags": ["mock", f"iter_{k:02d}", "eval"]})
        ev.define_metric("elapsed_h")
        ev.define_metric("eval/*", step_metric="elapsed_h")
        for i in range(1, 31):
            frac = i / 30
            ev.log({"elapsed_h": 9.0 * frac, "eval/episodes": int(1500 * frac), "eval/progress": frac,
                    **{f"eval/{g}": new[g] + rng.gauss(0, 0.04) * (1 - frac) for g in new},
                    "eval/overall": sum(new[g] * weights[g] for g in new) / 1500 + rng.gauss(0, 0.03) * (1 - frac)})
        overall = sum(new[g] * weights[g] for g in new) / 1500
        ev.summary.update({"eval/overall": overall, **{f"eval/{g}": v for g, v in new.items()}})
        table = wandb.Table(columns=["task_group", "baseline", "this_checkpoint", "delta"])
        for g in new:
            table.add_data(g, GROUPS[g][1], new[g], new[g] - GROUPS[g][1])
        ev.log({"eval/groups": table})
        ev.finish()
        loop.log({"iteration": k + 1, "eval/overall": overall, **{f"eval/{g}": v for g, v in new.items()},
                  "eval/delta_overall": overall - 0.537,
                  **{f"eval/delta_{g}": new[g] - GROUPS[g][1] for g in new}})
        base = new
        s1_share = min(0.95, s1_share + 0.03)
        succ = min(0.95, succ + 0.025)
        harness_loss *= 0.88
    loop.finish()
    print(f"mock experiment {exp!r} written to W&B project {args.project!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

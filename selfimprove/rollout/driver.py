"""A rollout round on a Ray cluster: every planned (task, seed) episode played once with the round's S1.

Run on the cluster's head (``ray.init(address="auto")``) or locally (``--local``: a Ray instance on this
machine).  The round's plan (``rollout/_plan.json``) is written once and reused on a restart; an episode
whose ``rollout/<task>/<seed>/rollout.json`` exists is done, so the driver can be stopped and started
again at any time.  Episodes go to the nodes as slots free up (a dynamic queue: composite episodes take
10x as long as atomic ones), a failed episode is played again up to ``max_attempts`` times, and a node
that dies is replaced (its episodes go back to the queue).  Progress goes to ``rollout/_status.json``
and the W&B run ``<exp>/iter_NN/rollout``; the end of the round writes ``rollout/_DONE``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import asdict

from .. import s3, tasks as T
from ..config import Config
from ..wandb_log import Run


def used_seeds(cfg: Config, iteration: int) -> dict[str, set[int]]:
    """Seeds earlier iterations of the experiment played, per task (never drawn again)."""
    out: dict[str, set[int]] = collections.defaultdict(set)
    for k in range(int(iteration)):
        plan = s3.read_json(f"{cfg.rollout_uri(k)}/_plan.json", default=None)
        for e in (plan or {}).get("episodes", []):
            out[e["task"]].add(int(e["seed"]))
    return out


def load_or_write_plan(cfg: Config, iteration: int, ckpt: str) -> list[T.Episode]:
    uri = f"{cfg.rollout_uri(iteration)}/_plan.json"
    plan = s3.read_json(uri, default=None)
    if plan is None:
        episodes = T.plan_round(cfg, iteration, used_seeds(cfg, iteration))
        plan = {"exp_id": cfg.exp_id, "iteration": int(iteration), "s1_ckpt": ckpt, "created": time.strftime("%F %T"),
                "seeds_per_task": cfg.seeds_per_task(iteration), "split": cfg["rollout"].get("split", "pretrain"),
                "tasks": sorted({e.task for e in episodes}), "episodes": [asdict(e) for e in episodes]}
        s3.write_json(uri, plan)
    elif plan.get("s1_ckpt") != ckpt:
        raise SystemExit(f"rollout/_plan.json of iteration {iteration} was made for {plan.get('s1_ckpt')}, not {ckpt}")
    return [T.Episode(**e) for e in plan["episodes"]]


def done_episodes(cfg: Config, iteration: int) -> dict[str, dict]:
    """task/seed -> its rollout.json, for the episodes already uploaded."""
    out = {}
    prefix = cfg.rollout_uri(iteration)
    for uri, _ in s3.iter_keys(prefix + "/"):
        if uri.endswith("/rollout.json"):
            rel = uri[len(prefix) + 1:].split("/")
            if len(rel) == 3:
                out[f"{rel[0]}/{rel[1]}"] = uri
    return out


class Stats:
    """The round's running numbers (for _status.json and W&B)."""

    def __init__(self, planned: list[T.Episode]):
        self.planned = {e.key: e for e in planned}
        self.results: dict[str, dict] = {}
        self.failed: dict[str, dict] = {}
        self.t0 = time.time()

    def add(self, r: dict) -> None:
        key = f"{r['task']}/{r['seed']}"
        if r.get("status") == "ok" and r.get("uri"):
            self.results[key] = r
            self.failed.pop(key, None)
        else:
            self.failed[key] = r

    def snapshot(self, running: int, nodes: int) -> dict:
        res = list(self.results.values())
        kinds = collections.defaultdict(list)
        for r in res:
            kinds[self.planned[f"{r['task']}/{r['seed']}"].kind].append(bool(r.get("success")))
        elapsed = time.time() - self.t0
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
        out = {
            "episodes_planned": len(self.planned), "episodes_done": len(res), "episodes_running": running,
            "episodes_failed": len(self.failed), "nodes": nodes, "elapsed_h": round(elapsed / 3600, 3),
            "progress": round(len(res) / max(1, len(self.planned)), 4),
            "success_rate": round(mean([bool(r.get("success")) for r in res]), 4),
            "success_rate_atomic": round(mean(kinds["atomic"]), 4),
            "success_rate_composite": round(mean(kinds["composite"]), 4),
            "s1_step_share": round(mean([float(r.get("s1_step_share") or 0) for r in res]), 4),
            "turns_mean": round(mean([float(r.get("turns") or 0) for r in res]), 2),
            "episode_minutes_mean": round(mean([float(r.get("seconds") or 0) for r in res]) / 60, 2),
            "tokens_prompt_mean": round(mean([float(r.get("tokens_prompt") or 0) for r in res])),
            "tokens_cached_frac": round(sum(float(r.get("tokens_cached") or 0) for r in res)
                                        / max(1.0, sum(float(r.get("tokens_prompt") or 0) for r in res)), 4),
            "episodes_per_hour": round(len(res) / max(elapsed / 3600, 1e-6), 2),
        }
        for by in ("s1", "agent"):
            vals = [r["score"][f"loss_real_{by}"] for r in res if f"loss_real_{by}" in (r.get("score") or {})]
            if vals:
                out[f"surrogate_loss_{by}"] = round(mean(vals), 6)
        return out


def run_round(cfg: Config, iteration: int, ckpt: str, *, local: bool = False, address: str | None = None,
              max_nodes: int | None = None, limit: int | None = None, wandb_on: bool = True,
              ) -> dict:
    import ray

    from .node import RayRolloutNode

    r = cfg["rollout"]
    planned = load_or_write_plan(cfg, iteration, ckpt)
    keys = {e.key for e in planned}
    done = {k: v for k, v in done_episodes(cfg, iteration).items() if k in keys}  # a stray folder is not the plan's
    stats = Stats(planned)
    for key, uri in list(done.items()):
        rj = s3.read_json(uri, default=None)
        if rj and rj.get("status") == "ok":
            rj["uri"] = uri.rsplit("/", 1)[0]
            stats.add(rj)
        else:  # an episode that ended in a simulator / gateway error is played again
            done.pop(key)
    todo = [e for e in T.interleave(planned) if e.key not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"iteration {iteration}: {len(planned)} planned, {len(done)} done, {len(todo)} to play with {ckpt}")
    run = Run(cfg, name=f"iter_{iteration:02d}/rollout", job_type="rollout", iteration=iteration,
              enabled=wandb_on, extra_config={"s1_ckpt": ckpt, "episodes_planned": len(planned)})
    if not todo:
        summary = stats.snapshot(0, 0)
        s3.write_json(f"{cfg.rollout_uri(iteration)}/_DONE", summary)
        run.finish(summary)
        return summary

    env_vars = {"SELFIMPROVE_AWS_REGION": cfg.region}
    for k in ("SI_HEXASHELL_ROOT", "SI_OPENPI_DIR", "SI_OPENPI_PYTHON", "SI_NODE_DIR", "HEXASHELL_ENV",
              "HEXASHELL_ROBOCASA_DEMOS"):
        if os.environ.get(k):
            env_vars[k] = os.environ[k]
    if local:
        env_vars["SELFIMPROVE_AWS_PROFILE"] = cfg.get("aws_profile") or ""
        ray.init(runtime_env={"env_vars": env_vars}, include_dashboard=False)
    else:
        # inside a Ray job on the head: the nodes' paths come from their ~/.profile (setup_node.sh); a
        # runtime_env here would have to merge with the job's (Ray refuses overlapping env_vars)
        ray.init(address=address or "auto")
    n_nodes = int(max_nodes or r.get("num_gpus", 8))
    if local:
        n_nodes = min(n_nodes, int(ray.cluster_resources().get("GPU", 0)) or 1)
    slots = int(r.get("slots_per_gpu", 4))

    def make_node():
        return RayRolloutNode.options(max_concurrency=slots + 2).remote(cfg.raw, iteration, ckpt)

    def submit(node, e: T.Episode, attempt: int):
        return node.run_episode.remote(e.task, e.seed, attempt, upload_uri=cfg.episode_uri(iteration, e.task, e.seed),
                                       demo=e.demo, start_salt=e.start_salt)

    def on_result(e: T.Episode, res: dict) -> None:
        res.setdefault("task", e.task)
        res.setdefault("seed", e.seed)
        stats.add(res)
        run.log_episode(res)

    def on_status(inflight: int, ready: int) -> None:
        snap = stats.snapshot(inflight, ready)
        s3.write_json(f"{cfg.rollout_uri(iteration)}/_status.json", {**snap, "updated": time.strftime("%F %T")})
        run.log(snap)
        print(json.dumps(snap), flush=True)

    from ..pool import run_pool

    pool = run_pool(make_node=make_node, units=todo, key=lambda e: e.key, submit=submit,
                    ok=lambda res: res.get("status") == "ok" and bool(res.get("uri")), n_nodes=n_nodes,
                    slots=slots, max_attempts=int(r.get("max_attempts", 2)), on_result=on_result,
                    on_status=on_status,
                    # rollout.cluster_gpus: the GPUs of all drivers sharing the cluster (an A/B's arms), else ours
                    request_gpus=False if local else int(r.get("cluster_gpus") or n_nodes))
    summary = stats.snapshot(0, 0)
    summary.update(pool)
    summary["failed"] = {k: v.get("status") for k, v in stats.failed.items()}
    if limit is None:
        s3.write_json(f"{cfg.rollout_uri(iteration)}/_DONE", summary)
    run.finish(summary, table=list(stats.results.values()))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--local", action="store_true", help="a Ray instance on this machine (its GPUs)")
    ap.add_argument("--address", default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="play at most this many episodes (a check)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    s3.configure(cfg.get("aws_profile") if args.local else None, cfg.region)
    # Astra's key never travels through Ray (runtime envs and task arguments show up in Ray's errors and
    # dashboard): each node's HexaShell reads its own ~/.config/hexashell/api-key (a 0600 file the cluster's
    # file_mounts put there) or, locally, the $OPENAI_API_KEY its processes inherit
    summary = run_round(cfg, args.iteration, args.ckpt, local=args.local, address=args.address,
                        max_nodes=args.max_nodes, limit=args.limit, wandb_on=not args.no_wandb)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

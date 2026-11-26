"""A pool of GPU node actors working through a queue of units (episodes), shared by rollout and eval.

``make_node()`` returns a new Ray actor handle that has ``info()`` (ready check) and ``shutdown()``;
``submit(node, unit, attempt)`` starts one unit on a node and returns its ObjectRef.  Each node runs up
to ``slots`` units at once; a unit whose result ``ok(result)`` rejects is queued again until
``max_attempts``; a node that dies is replaced and its units requeued; ``on_result`` sees every
result and ``on_status`` is called about once a minute with (in_flight, nodes_ready).
"""

from __future__ import annotations

import collections
import time
from typing import Any, Callable


def run_pool(*, make_node: Callable[[], Any], units: list, key: Callable[[Any], str],
             submit: Callable[[Any, Any, int], Any], ok: Callable[[dict], bool], n_nodes: int, slots: int,
             max_attempts: int = 2, on_result: Callable[[Any, dict], None] = lambda u, r: None,
             on_status: Callable[[int, int], None] = lambda i, n: None, status_every_s: float = 60.0,
             request_gpus: bool | int = False, max_start_failures: int | None = None, log=print) -> dict[str, int]:
    """``request_gpus``: ask the autoscaler for GPU nodes up front (True: ``n_nodes``; an int: that many - the
    request is the cluster's total, so drivers sharing a cluster must all ask for their sum)."""
    import ray

    max_start_failures = max_start_failures if max_start_failures is not None else max(3, n_nodes)
    if request_gpus:
        from ray.autoscaler.sdk import request_resources

        request_resources(bundles=[{"GPU": 1}] * (n_nodes if request_gpus is True else int(request_gpus)))
    nodes = {i: make_node() for i in range(n_nodes)}
    starting = {i: n.info.remote() for i, n in nodes.items()}
    ready: dict[int, Any] = {}
    queue = collections.deque(units)
    attempts: dict[str, int] = collections.Counter()
    inflight: dict[Any, tuple[int, Any]] = {}
    last = 0.0
    replaced = 0
    start_failures = 0
    try:
        while queue or inflight:
            if starting:
                done_refs, _ = ray.wait(list(starting.values()), num_returns=len(starting), timeout=0)
                for i, ref in list(starting.items()):
                    if ref not in done_refs:
                        continue
                    starting.pop(i)
                    try:
                        log(f"node {i} ready: {ray.get(ref)}")
                        ready[i] = nodes[i]
                    except Exception as exc:  # noqa: BLE001
                        start_failures += 1
                        if start_failures > max_start_failures:
                            raise RuntimeError(f"{start_failures} nodes failed to start (last: {str(exc)[:800]}); "
                                               "fix the node setup before the round goes on") from None
                        log(f"node {i} failed to start ({start_failures}/{max_start_failures}): {str(exc)[:500]}; "
                            "replacing")
                        replaced += 1
                        nodes[i] = make_node()
                        starting[i] = nodes[i].info.remote()
                if not ready and not inflight:
                    time.sleep(5)
                    continue
            for i, node in ready.items():
                busy = sum(1 for j, _ in inflight.values() if j == i)
                while busy < slots and queue:
                    u = queue.popleft()
                    attempts[key(u)] += 1
                    inflight[submit(node, u, attempts[key(u)])] = (i, u)
                    busy += 1
            finished = ray.wait(list(inflight), num_returns=1, timeout=30)[0] if inflight else []
            if not inflight:
                time.sleep(5)
            for ref in finished:
                i, u = inflight.pop(ref)
                try:
                    res = ray.get(ref)
                except Exception as exc:  # noqa: BLE001 - the actor died: replace it, requeue its units
                    res = {"status": "node_lost", "error": str(exc)[:500]}
                    if i in ready:
                        ready.pop(i)
                        for other, (j, u2) in list(inflight.items()):
                            if j == i:
                                inflight.pop(other)
                                queue.appendleft(u2)
                        replaced += 1
                        nodes[i] = make_node()
                        starting[i] = nodes[i].info.remote()
                if not ok(res) and attempts[key(u)] < max_attempts:
                    queue.append(u)
                    log(f"{key(u)}: {res.get('status')} {str(res.get('error', ''))[:200]} -> again")
                on_result(u, res)
            if time.time() - last > status_every_s:
                on_status(len(inflight), len(ready))
                last = time.time()
    finally:
        for node in nodes.values():
            try:
                ray.get(node.shutdown.remote(), timeout=180)
            except Exception:  # noqa: BLE001
                pass
        if request_gpus:
            from ray.autoscaler.sdk import request_resources

            request_resources(bundles=[])
    return {"nodes_replaced": replaced}

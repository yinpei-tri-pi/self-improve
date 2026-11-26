"""The scheduler's state machine end to end, with S3, the drivers and SageMaker faked."""

import json

import pytest

from selfimprove import s3, scheduler as S
from selfimprove.config import Config


class FakeS3:
    def __init__(self):
        self.objects = {}

    def read_json(self, uri, default=None):
        return json.loads(self.objects[uri]) if uri in self.objects else default

    def write_json(self, uri, obj):
        self.objects[uri] = json.dumps(obj)

    def write_bytes(self, uri, data, content_type=""):
        self.objects[uri] = data


@pytest.fixture()
def world(monkeypatch, tmp_path):
    fs = FakeS3()
    for name in ("read_json", "write_json", "write_bytes"):
        monkeypatch.setattr(s3, name, getattr(fs, name))
    cfg = Config.load(None, [f"work_dir={tmp_path}", "exp_id=t", "iterations=2", "cluster.mode=local"])
    log = {"drivers": [], "trains": []}

    def start_driver(self, name, module, k, ckpt, cluster):
        log["drivers"].append((name, k, ckpt))
        uri = self.cfg.rollout_uri(k) if name == "rollout" else self.cfg.eval_uri(k)
        fs.write_json(f"{uri}/_DONE", {"episodes_done": 3, "success_rate": 0.9, "overall": 0.55 + 0.01 * k})
        return {"mode": "local", "pid": 0}

    monkeypatch.setattr(S.Scheduler, "_start_driver", start_driver)
    monkeypatch.setattr(S.data_stage, "run", lambda cfg, k, log=print: (
        fs.write_json(f"{cfg.s1_shard_uri(k)}/_DONE", {"samples": 10}) or {"samples": 10, "shards": 1}))
    monkeypatch.setattr(S.train_launch, "launch", lambda cfg, k, ckpt, dry_run=False: (
        log["trains"].append((k, ckpt)) or {"job_name": f"job{k}", "exp_name": f"e{k}"}))
    monkeypatch.setattr(S.train_launch, "job_status", lambda cfg, job: {"status": "Completed"})
    monkeypatch.setattr(S.train_launch, "final_checkpoint", lambda cfg, k: f"s3://ckpt/iter{k}/10000")
    return cfg, fs, log


def test_loop_runs_every_stage_in_order(world):
    cfg, fs, log = world
    sched = S.Scheduler(cfg, wandb_on=False)
    for _ in range(40):
        if sched.tick():
            break
    assert sched.tick()
    # iteration 0 rolls out with the start checkpoint, iteration 1 with iteration 0's trained one
    rollouts = [d for d in log["drivers"] if d[0] == "rollout"]
    assert rollouts[0] == ("rollout", 0, cfg.start_s1_ckpt)
    assert ("rollout", 1, "s3://ckpt/iter0/10000") in log["drivers"]
    assert log["trains"] == [(0, cfg.start_s1_ckpt), (1, "s3://ckpt/iter0/10000")]
    # each trained checkpoint is evaluated
    assert ("eval", 0, "s3://ckpt/iter0/10000") in log["drivers"] and ("eval", 1, "s3://ckpt/iter1/10000") in log["drivers"]
    st = json.loads(fs.objects[cfg.state_uri])
    for k in ("0", "1"):
        assert {s: st["iterations"][k][s]["status"] for s in S.STAGES} == {s: "done" for s in S.STAGES}
    assert st["s2"]["status"] == "skipped"
    assert json.loads(fs.objects[f"{cfg.s1_ckpt_uri(0)}/_FINAL"])["uri"] == "s3://ckpt/iter0/10000"


def test_failed_training_is_retried_then_stops(world, monkeypatch):
    cfg, fs, log = world
    monkeypatch.setattr(S.train_launch, "job_status", lambda cfg, job: {"status": "Failed", "failure": "oom"})
    monkeypatch.setattr(S.train_launch, "final_checkpoint", lambda cfg, k: None)
    sched = S.Scheduler(cfg, wandb_on=False)
    for _ in range(20):
        sched.tick()
    assert len(log["trains"]) == 2  # first try + one retry (train.max_retries = 1)
    assert sched.it(0)["train"]["status"] == "failed"
    assert not any(d[0] == "rollout" and d[1] == 1 for d in log["drivers"])

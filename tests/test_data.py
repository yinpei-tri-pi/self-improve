"""Episode parsing, the PRM, and the S1 shard writer on synthetic records."""

import io
import json
import tarfile

import numpy as np
import yaml

from selfimprove.config import DEFAULT_CONFIG
from selfimprove.data import s1_shards
from selfimprove.data.episode import load_episode
from selfimprove.data.prm import assign, awr_weights
from selfimprove.data.trajectory import Trajectory

from fake_episode import write_episode

FCFG = yaml.safe_load(DEFAULT_CONFIG.read_text())["filter"]


def test_segments_follow_turns(tmp_path):
    rec = load_episode(write_episode(tmp_path / "ep"))
    segs = rec.segments
    assert [(s.start, s.end, s.by) for s in segs] == [(0, 39, "s1"), (40, 59, "s1"), (60, 89, "s1"),
                                                      (90, 114, "agent"), (115, 129, "s1")]
    # each segment is judged by the next turn
    assert [s.judge for s in segs] == ["subgoal_complete", "subgoal_complete", "subgoal_failed",
                                       "subgoal_complete", "task_done"]
    assert segs[2].next == "taken_over" and segs[0].next == "moved_on" and segs[-1].next == "task_done"
    assert segs[0].s1_est_length == 50


def test_prm_levels_and_use(tmp_path):
    rec = load_episode(write_episode(tmp_path / "ep"))
    segs = assign(rec, FCFG)
    assert [s.score for s in segs] == ["very_good", "very_good", "bad", "very_good", "very_good"]
    assert [s.use for s in segs] == ["positive", "positive", "negative", "positive", "positive"]
    rec = load_episode(write_episode(tmp_path / "ep2", success=False))
    segs = assign(rec, FCFG)
    assert segs[2].score == "very_bad" and segs[0].score == "good"
    w = awr_weights(segs, FCFG)
    frames = sum(s.steps for s in segs if s.use == "positive")
    assert abs(sum(w[id(s)] * s.steps for s in segs if s.use == "positive") / frames - 1.0) < 1e-6


def test_error_episode_is_dropped(tmp_path):
    folder = write_episode(tmp_path / "ep")
    r = json.loads((folder / "result.json").read_text())
    r["errors"] = ["gateway lost"]
    (folder / "result.json").write_text(json.dumps(r))
    assert all(s.use is None for s in assign(load_episode(folder), FCFG))


def test_trajectory_reader(tmp_path):
    t = Trajectory.load(write_episode(tmp_path / "ep"))
    assert len(t) == 130 and t.has_sim_state
    assert t.image(77, "wrist").shape == (32, 32, 3)
    spans = t.spans()
    el = t.s1_element(45, spans[1], est_length=50)
    assert el["actions"].shape == (20, 12) and el["executed_step"] == 5
    # past the span's end: settle pad (velocities 0, gripper + mode held), progress 1
    assert np.all(el["actions"][15:, 5:8] == 0) and np.all(el["actions"][15:, 11] == el["actions"][14, 11])
    assert np.all(el["progress_action"][15:] == 1.0)


def test_build_shards(tmp_path):
    eps = []
    for i, ok in enumerate((True, False, True)):
        f = write_episode(tmp_path / "rollout" / f"ep{i}", seed=100 + i, success=ok)
        eps.append((str(f), "OpenDrawer", 100 + i))
    out = tmp_path / "shards"
    meta = s1_shards.build(eps, out, exp_id="t", iteration=3,
                           fcfg={**FCFG, "samples_per_shard": 64, "planner": "prm", "slim": False}, workers=2)
    assert meta["shards"] == len(list((out / "shards").glob("*.tar")))
    assert meta["samples"] == meta["positive_samples"] + meta["negative_samples"] > 0
    total = 0
    labels = set()
    for shard in sorted((out / "shards").glob("*.tar")):
        with tarfile.open(shard) as tar:
            names = tar.getnames()
            keys = sorted({n.split(".", 1)[0] for n in names})
            for key in keys[:5]:
                members = {n.split(".", 1)[1] for n in names if n.split(".", 1)[0] == key}
                assert members == {"scene_left.jpg", "scene_right.jpg", "wrist.jpg", "anchor_scene_left.jpg",
                                   "anchor_scene_right.jpg", "anchor_wrist.jpg", "arrays.npz", "meta.json"}
                meta_s = json.loads(tar.extractfile(f"{key}.meta.json").read())
                arrays = np.load(io.BytesIO(tar.extractfile(f"{key}.arrays.npz").read()))
                assert arrays["lean_action_subgoal_pad"].shape == (20, 11) and arrays["state"].shape == (16,)
                assert meta_s["si_iter"] == 3 and meta_s["episode_id"].startswith("SelfImprove/t/iter_03/")
                labels.add(meta_s["si_label"])
            total += len(keys)
    assert total == meta["samples"]
    assert labels <= {1.0, -1.0}
    lines = (out / "manifest.jsonl").read_text().splitlines()
    assert sum(json.loads(x)["num_samples"] for x in lines) == total
    assert len((out / "segments.jsonl").read_text().splitlines()) == meta["segments"]


def test_sim_env_has_the_machine_hexashell_settings(tmp_path, monkeypatch):
    """The warm simulator servers get the machine file's HEXASHELL_* settings (split demo's recorded states)."""
    from types import SimpleNamespace

    from selfimprove.rollout import node

    monkeypatch.delenv("HEXASHELL_ROBOCASA_DEMOS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    paths = SimpleNamespace(robocasa_env=None, hexashell=tmp_path,
                            machine={"HEXASHELL_ROBOCASA_DEMOS": "/mnt/data/sim_states/v1.0/pretrain", "OTHER": "y"})
    env = node.sim_env(paths, "0")
    assert env["HEXASHELL_ROBOCASA_DEMOS"] == "/mnt/data/sim_states/v1.0/pretrain" and "OTHER" not in env
    assert "OPENAI_API_KEY" not in env and env["HEXA_EGL_DEVICE"] == "0"


def test_build_shards_v2_slim(tmp_path):
    """The agreed rules: successful episodes only, positives only, the slim layout the production loader reads."""
    eps = []
    for i, ok in enumerate((True, False, True)):
        f = write_episode(tmp_path / "rollout" / f"ep{i}", seed=100 + i, success=ok)
        eps.append((str(f), "OpenDrawer", 100 + i))
    out = tmp_path / "shards"
    meta = s1_shards.build(eps, out, exp_id="t", iteration=3, fcfg={"samples_per_shard": 64}, workers=2)
    assert meta["negative_samples"] == 0 and meta["samples"] == meta["positive_samples"]
    assert meta["episodes_used"] <= 2 and "dropped" in " ".join(meta["segment_use"])
    for shard in sorted((out / "shards").glob("*.tar")):
        with tarfile.open(shard) as tar:
            names = tar.getnames()
            for key in sorted({n.split(".", 1)[0] for n in names})[:5]:
                meta_s = json.loads(tar.extractfile(f"{key}.meta.json").read())
                arrays = np.load(io.BytesIO(tar.extractfile(f"{key}.arrays.npz").read()))
                assert set(arrays.files) == {"state", "state_anchor", "lean_state", "lean_state_anchor",
                                             "lean_action_subgoal_pad", "action_pad_mask_subgoal"}
                assert meta_s["quality"] == "Success" and meta_s["est_length"] >= 50 and "si_label" not in meta_s
                assert meta_s["si_source"] in ("s1", "harness") and meta_s["seed"] in (100, 102)


def test_s2_rows_v2(tmp_path):
    """New single-mode S2 rows: one per turn (+ a closing task_finish row on success), decision = who executed the
    turn's step, history grouped by milestone with movers and outcomes."""
    from selfimprove.s2 import rows_v2

    f = write_episode(tmp_path / "ep", seed=7, success=True)
    rows = rows_v2.episode_rows(f, tmp_path / "s2", make_media=False)
    assert len(rows) == 6 and rows[-1]["labels"]["judge"] == "task_finish" and rows[-1]["meta"]["kind"] == "closing"
    assert [r["meta"]["decision"] for r in rows[:5]] == ["s1", "s1", "s1", "harness", "s1"]
    assert rows[0]["labels"]["judge"] == "task_begin" and "<image>" in rows[0]["messages"][1]["content"]
    u = rows[4]["messages"][1]["content"]
    assert "<video>" in u and "pull the drawer open (failed)" in u and "'pull the drawer open again', executed by the expert" in u
    assert rows[4]["labels"]["judge"] == "subgoal_complete" and "<decision>model</decision>" in rows[4]["messages"][2]["content"]
    text = json.dumps([r["messages"] for r in rows])
    assert "harness" not in text and "System1" not in text
    # the task status line: begin, ongoing ..., success on the closing row
    assert "Current task status: begin." in rows[0]["messages"][1]["content"]
    assert all("Current task status: ongoing." in r["messages"][1]["content"] for r in rows[1:-1])
    assert "Current task status: success." in rows[-1]["messages"][1]["content"]
    assert rows[3]["labels"]["judge"] == "subgoal_failed"
    failed = rows_v2.episode_rows(write_episode(tmp_path / "ep2", seed=8, success=False), tmp_path / "s2b", make_media=False)
    # failed episodes are kept, with a closing row: the status failed -> task_failed
    assert len(failed) == 6 and failed[-1]["meta"]["kind"] == "closing" and failed[-1]["labels"]["judge"] == "task_failed"
    assert "Current task status: failed." in failed[-1]["messages"][1]["content"]


def test_s2_checklist():
    from selfimprove.s2.rows_v2 import checklist_text, parse_milestones

    ms = parse_milestones("M1: pick the mug from the cabinet\nM2: place the mug under the dispenser\nM3: press the start button")
    d = lambda m, sg, out: {"milestone": m, "subgoal": sg, "outcome": out}  # noqa: E731
    done = [d("M1: pick", "reach to the mug", "done"), d("M1: pick", "grasp the mug", "failed"),
            d("M1: pick", "grasp the mug again", "done"), d("M2: place", "carry the mug to the dispenser", "done"),
            d("M3: press", "reach to the start button", "failed")]
    # going back: M2 now again, the return marked, M3 emptied and "to do, tried before"
    assert checklist_text(done, "M2: place", ms).splitlines() == [
        "  M1: pick the mug from the cabinet (done)",
        "  M2: place the mug under the dispenser (ongoing, again)",
        "    - carry the mug to the dispenser (done)",
        "    (back from M3)",
        "  M3: press the start button (to do, tried before)"]
    # forward: M1 done, M2 now with its steps (the attempts at one step are one line), M3 to do
    assert checklist_text(done[:3], "M2: place", ms).splitlines() == [
        "  M1: pick the mug from the cabinet (done)", "  M2: place the mug under the dispenser (ongoing)",
        "  M3: press the start button (to do)"]
    lines = checklist_text(done[:3], "M1: pick", ms).splitlines()
    assert lines[1:3] == ["    - reach to the mug (done)", "    - grasp the mug (done)"]
    # a long milestone: only its last steps, no tally
    many = [d("M1: pick", f"step {i}", "failed" if i % 3 == 0 else "done") for i in range(9)]
    out = checklist_text(many, "M1: pick", ms).splitlines()
    assert out[1:6] == [f"    - step {i} ({'failed' if i % 3 == 0 else 'done'})" for i in range(4, 9)]
    assert out[6] == "  M2: place the mug under the dispenser (to do)" and not any("so far" in x for x in out)
    # the last step (the video): listed last under the ongoing milestone, as ongoing - after a failed try too
    assert checklist_text(done[:2], "M1: pick", ms, "grasp the mug again").splitlines()[:4] == [
        "  M1: pick the mug from the cabinet (ongoing)", "    - reach to the mug (done)", "    - grasp the mug (failed)",
        "    - grasp the mug again (ongoing)"]
    # a new milestone: the last step is its first line
    assert checklist_text(done[:3], "M2: place", ms, "carry the mug to the dispenser").splitlines()[1:3] == [
        "  M2: place the mug under the dispenser (ongoing)", "    - carry the mug to the dispenser (ongoing)"]


def test_s2_merge_fragments_stop_at_a_milestone():
    import numpy as np

    from selfimprove.s2.rows_v2 import merge_fragments

    t = lambda turn, a, b, sg, m: {"turn": turn, "start": a, "end": b, "subgoal": sg, "by": "agent",  # noqa: E731
                                   "judge_of_this": "subgoal_complete", "milestone": m}
    turns = [t(1, 0, 19, "reach to the mug", "M1: a"), t(2, 20, 39, "grasp the mug", "M1: a"),
             t(3, 40, 59, "reach to the button", "M2: b"), t(4, 60, 79, "press the button", "M2: b")]
    out = merge_fragments(turns, np.zeros((100, 11)))
    assert [(x["turn"], x["end"], x["subgoal"], x.get("merged")) for x in out] == [
        (1, 39, "grasp the mug", [1, 2]), (3, 79, "press the button", [3, 4])]


def test_s2_text_says_the_expert():
    from selfimprove.s2.rows_v2 import natural

    assert natural("The harness should grasp it; a harness grasp is safer than S1's. System1 should lift it.") == \
        "The expert should grasp it; an expert grasp is safer than the model's. The model should lift it."


def test_s2_short_video_rows_skipped(tmp_path):
    """A row whose video has under MIN_FRAMES frames is marked skipped and left out of rows.jsonl; its step stays in
    the later rows' history."""
    from selfimprove.s2 import rows_v2

    out = tmp_path / "s2"
    rows = rows_v2.episode_rows(write_episode(tmp_path / "ep", seed=7, success=True), out, make_media=False)
    for r in rows:
        n = r["meta"].get("clip_frames")
        assert bool(r["meta"].get("skipped")) == (n is not None and n < rows_v2.MIN_FRAMES)
    written = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert len(written) == len(rows_v2.trained(rows))
    assert all((r["meta"].get("clip_frames") or rows_v2.MIN_FRAMES) >= rows_v2.MIN_FRAMES for r in written)
    assert len(rows_v2.clip_steps(0, 12)) < rows_v2.MIN_FRAMES <= len(rows_v2.clip_steps(0, 13))

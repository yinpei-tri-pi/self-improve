"""The S1 shard math is RoboAnnotator's (build_system1_shards / preprocess_robocasa_to_tar / build_samples).
Skipped when the RoboAnnotator checkout is not on this machine."""

import os
import sys

import numpy as np
import pytest

from selfimprove.data import trajectory as T

RA = os.environ.get("ROBOANNOTATOR_DIR", os.path.expanduser("~/RoboAnnotator"))


@pytest.fixture(scope="module")
def ra():
    if not os.path.isdir(RA):
        pytest.skip("no RoboAnnotator checkout")
    sys.path.insert(0, RA)
    try:
        from producers import build_system1_shards as S
        from producers import preprocess_robocasa_to_tar as P
        from robo_annotator.system2 import build_samples as B
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RoboAnnotator not importable: {exc}")
    return S, P, B


def _raw_state(n, rng):
    s = rng.normal(size=(n, 16)).astype(np.float32)
    for sl in (slice(3, 7), slice(10, 14)):
        q = s[:, sl]
        s[:, sl] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return s


def test_lean_state_and_action(ra):
    S, P, B = ra
    rng = np.random.default_rng(0)
    raw = _raw_state(50, rng)
    ref_pos, ref_yaw = raw[0, 0:3], P._yaw_from_quat_xyzw(raw[0, 3:7])
    assert abs(ref_yaw - T.yaw_from_quat_xyzw(raw[0, 3:7])) < 1e-9
    np.testing.assert_array_equal(T.lean_state(raw, ref_pos[0:2], ref_yaw), P._lean_state(raw, ref_pos[0:2], ref_yaw))
    act = rng.normal(size=(50, 12)).astype(np.float32)
    np.testing.assert_array_equal(T.lean_action(act), P._lean_action(act))


def test_padded_and_halt_chunks(ra):
    S, P, B = ra
    rng = np.random.default_rng(1)
    act = rng.normal(size=(40, 12)).astype(np.float32)
    for t, end in ((0, 39), (5, 12), (30, 35), (39, 39)):
        ours = T.padded_chunk(act, t, end, T.HORIZON, T.RAW_HOLD_DIMS)
        theirs = S._ref_padded_chunk(act, t, end, T.HORIZON, S.RAW_HOLD_DIMS)
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_array_equal(ours[1], theirs[1])
    np.testing.assert_array_equal(T.halt_chunk(act[7], T.HORIZON, T.RAW_HOLD_DIMS),
                                  S._halt_chunk(act[7], T.HORIZON, S.RAW_HOLD_DIMS))
    assert T.RAW_HOLD_DIMS == tuple(S.RAW_HOLD_DIMS) and T.LEAN_HOLD_DIMS == tuple(S.LEAN_HOLD_DIMS)


def test_prompt_length_grid_progress(ra):
    S, P, B = ra
    for n in (1, 8, 49, 62, 63, 87, 88, 260, 900, 5000):
        assert T.estimated_length(n) == B._estimated_length(n)
    for f, s, e in ((0, 0, 5), (3, 0, 5), (10, 0, 100), (99, 0, 100), (150, 100, 160)):
        assert T.progress_class(f, s, e) == B.progress_class(f, s, e)
    assert T.system1_prompt("Open the fridge.", "Pull the door", "Success", 75, "Close") == \
        B.system1_prompt("Open the fridge.", "Pull the door", "Success", 75, gripper="Close")


def test_action_order_inverts_hexashell():
    """sim_to_lerobot is the inverse of HexaShell's lerobot_action_to_sim."""
    rng = np.random.default_rng(2)
    lr = rng.normal(size=12)
    sim = np.empty(12)
    sim[0:3], sim[3:6], sim[6], sim[7:11], sim[11] = lr[5:8], lr[8:11], lr[11], lr[0:4], lr[4]
    np.testing.assert_allclose(T.sim_to_lerobot(sim), lr, rtol=1e-6)

from types import SimpleNamespace

import numpy as np
import pytest

from ar_lava.calibration import BOARD, QUALITY_VERSION, CameraCalibration
from ar_lava.calibration_check import CalibrationPairCheck, validate_pair
from ar_lava.stereo import StereoRectifier


def camera_pair():
    cameras = {}
    matrix = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    for index, name in enumerate(("cam1", "cam2")):
        calibration = CameraCalibration(
            name, (640, 480), matrix.copy(), np.zeros(5), 0.1,
            np.zeros((3, 1)), np.array([[-0.1 - index * 0.15], [-0.14], [1.2]]),
            quality={"version": QUALITY_VERSION, "intrinsics": {"passed": True}},
        )
        cameras[name] = SimpleNamespace(
            generation=1, profile=name, connected=True, calibration=calibration,
            floor_validated=True, clock=SimpleNamespace(rtt_ms=10),
            guide=SimpleNamespace(board_observation=None), pair_quality={},
            background=np.ones((480, 640, 3), np.uint8),
            depth_composite=np.ones((480, 640, 3), np.uint8), landmarks=[1], mode="play",
        )
    return cameras


def observation(camera, stamp, shift=(0.0, 0.0)):
    world = BOARD.getChessboardCorners()
    return {
        "captured_ms": stamp, "profile": camera.profile, "size": camera.calibration.size,
        "ids": np.arange(len(world), dtype=np.int32).reshape(-1, 1),
        "corners": (camera.calibration.project(world) + shift).astype(np.float32).reshape(-1, 1, 2),
    }


def supply(check, cameras, stamp, skew=0, now=None, shift_left=(0.0, 0.0)):
    cameras["cam1"].guide.board_observation = observation(cameras["cam1"], stamp, shift_left)
    cameras["cam2"].guide.board_observation = observation(cameras["cam2"], stamp + skew)
    check.update(cameras, stamp + max(0, skew) if now is None else now)


def approve(check, cameras):
    for stamp in range(1000, 2200, 200):
        supply(check, cameras, stamp)
    assert all(camera.pair_quality["passed"] for camera in cameras.values())


def test_six_independent_synchronized_observations_validate_frozen_geometry():
    cameras = camera_pair()
    check = CalibrationPairCheck()
    for stamp in range(1000, 2000, 200):
        supply(check, cameras, stamp)
        assert not cameras["cam1"].pair_quality["passed"]
    supply(check, cameras, 2000)
    quality = cameras["cam1"].pair_quality
    assert quality["passed"]
    assert quality["samples"] == 6
    assert quality["common_points"] == len(BOARD.getChessboardCorners())
    assert quality["source_rms_px"] < 0.001
    assert quality["board_scale_p95_percent"] < 0.01
    assert cameras["cam2"].pair_quality["passed"]


def test_repeated_frame_or_six_frames_in_less_than_one_second_cannot_complete():
    cameras = camera_pair()
    check = CalibrationPairCheck()
    supply(check, cameras, 1000)
    for now in range(1000, 1200, 10):
        check.update(cameras, now)
    assert cameras["cam1"].pair_quality["samples"] == 1
    for stamp in range(1210, 1260, 10):
        supply(check, cameras, stamp)
    assert cameras["cam1"].pair_quality["samples"] == 6
    assert not cameras["cam1"].pair_quality["passed"]


@pytest.mark.parametrize("skew,age,rtt", [(41, 0, 10), (0, 351, 10), (0, 0, 101)])
def test_unsynchronized_stale_or_high_rtt_observations_are_not_counted(skew, age, rtt):
    cameras = camera_pair()
    cameras["cam1"].clock.rtt_ms = rtt
    check = CalibrationPairCheck()
    for stamp in range(1000, 2200, 200):
        supply(check, cameras, stamp, skew=skew, now=stamp + max(0, skew) + age)
    assert not cameras["cam1"].pair_quality["passed"]
    assert cameras["cam1"].pair_quality["samples"] == 0


def test_reprojection_against_frozen_pose_rejects_shift_even_if_refitting_would_fit():
    cameras = camera_pair()
    check = CalibrationPairCheck()
    for stamp in range(1000, 2200, 200):
        supply(check, cameras, stamp, shift_left=(1.0, 0.0))
    assert not cameras["cam1"].pair_quality["passed"]
    assert cameras["cam1"].pair_quality["samples"] == 0
    assert cameras["cam1"].pair_quality["monitoring"] == "failed"


def test_same_individual_pose_fits_can_still_fail_pair_geometry():
    cameras = camera_pair()
    rectifier = StereoRectifier(*(camera.calibration for camera in cameras.values()))
    # Detect stale relative geometry even when each camera's changed pose fits its own image.
    cameras["cam2"].calibration.tvec[1, 0] += 0.01
    views = tuple(observation(camera, 1000) for camera in cameras.values())
    with pytest.raises(ValueError, match="같은 높이"):
        validate_pair(rectifier, views)


def test_scale_reconstruction_can_fail_even_when_each_source_error_is_below_limit():
    cameras = camera_pair()
    cameras["cam2"].calibration.tvec[0, 0] = -0.151
    rectifier = StereoRectifier(*(camera.calibration for camera in cameras.values()))
    views = (
        observation(cameras["cam1"], 1000, (0.65, 0.0)),
        observation(cameras["cam2"], 1000, (-0.65, 0.0)),
    )
    with pytest.raises(ValueError, match="크기 오차"):
        validate_pair(rectifier, views)


def test_three_fresh_drift_observations_invalidate_pose_and_cached_depth():
    cameras = camera_pair()
    check = CalibrationPairCheck()
    approve(check, cameras)
    previous_generation = cameras["cam1"].generation
    for stamp in (2200, 2400):
        supply(check, cameras, stamp, shift_left=(3.0, 0.0))
        assert cameras["cam1"].floor_validated
    supply(check, cameras, 2600, shift_left=(3.0, 0.0))
    camera = cameras["cam1"]
    assert not camera.floor_validated
    assert not camera.calibration.ready
    assert camera.generation == previous_generation + 1
    assert camera.background is None and camera.depth_composite is None and camera.landmarks is None
    assert camera.mode == "calibration"
    assert not cameras["cam2"].pair_quality["passed"]


@pytest.mark.parametrize("failure", ["residual", "projection_exception"])
@pytest.mark.parametrize("partner", ["stale", "skewed"])
def test_bad_source_revokes_approval_without_a_synchronized_partner(failure, partner):
    cameras = camera_pair()
    check = CalibrationPairCheck()
    approve(check, cameras)
    camera = cameras["cam1"]
    camera.guide.board_observation = observation(camera, 2400, (1.0, 0.0))
    if failure == "projection_exception":
        def cannot_project(points):
            raise ValueError("invalid pose")
        camera.calibration.project = cannot_project
    # The other camera cannot supply a qualifying synchronized observation.
    now = 2400
    if partner == "skewed":
        cameras["cam2"].guide.board_observation = observation(cameras["cam2"], 2441)
        now = 2441
    check.update(cameras, now)
    assert camera.floor_validated
    assert not any(camera.pair_quality["passed"] for camera in cameras.values())


def test_fixed_board_visibility_lease_expires_and_requires_a_new_sequence():
    cameras = camera_pair()
    check = CalibrationPairCheck()
    approve(check, cameras)
    for camera in cameras.values():
        camera.guide.board_observation = None
    check.update(cameras, 12_000)
    assert cameras["cam1"].pair_quality["passed"]
    assert cameras["cam1"].pair_quality["monitoring"] == "unobserved"
    check.update(cameras, 12_001)
    assert not cameras["cam1"].pair_quality["passed"]
    assert cameras["cam1"].pair_quality["samples"] == 0
    supply(check, cameras, 12_100)
    assert not cameras["cam1"].pair_quality["passed"]
    assert cameras["cam1"].pair_quality["samples"] == 1


@pytest.mark.parametrize("change", ["disconnect", "generation", "profile", "lens_unverified"])
def test_pair_approval_does_not_survive_state_changes(change):
    cameras = camera_pair()
    check = CalibrationPairCheck()
    approve(check, cameras)
    camera = cameras["cam2"]
    if change == "disconnect":
        camera.connected = False
    elif change == "generation":
        camera.generation += 1
    elif change == "profile":
        camera.profile = "another-lens"
    else:
        camera.calibration.quality = {}
    check.update(cameras, 2100)
    assert not any(camera.pair_quality["passed"] for camera in cameras.values())

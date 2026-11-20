import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ar_lava.calibration import BOARD, QUALITY_VERSION, CameraCalibration
from ar_lava.guidance import FLOOR_OBSERVATIONS
from ar_lava.protocol import MAX_HEADER, pack, unpack
from ar_lava.server import Camera, create_app, now_ms

MATRIX = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])


def board_view(x=320, y=240, distance=0.65, rotation=(0.2, -0.15, 0)):
    """A known pinhole camera observing a physical 200x280mm board."""
    rvec = np.array(rotation, dtype=np.float64)
    center = cv2.Rodrigues(rvec)[0] @ np.array([0.1, 0.14, 0])
    tvec = np.array([(x - 320) * distance / 500, (y - 240) * distance / 500, distance])
    tvec -= center
    world = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.28, 0], [0, 0.28, 0]], np.float32)
    points = cv2.projectPoints(world, rvec, tvec, MATRIX, np.zeros(5))[0]
    source = np.array([[0, 0], [599, 0], [599, 839], [0, 839]], np.float32)
    transform = cv2.getPerspectiveTransform(source, points.reshape(4, 2))
    image = cv2.warpPerspective(
        BOARD.generateImage((600, 840)), transform, (640, 480), borderValue=255
    )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def observe_still(camera, frame, stamp):
    for index in range(3):
        camera.capture_ms = stamp + index * 400
        camera.frame = frame
        camera.guide.observe(camera, frame, camera.capture_ms)
    return stamp + 1200


def validated_lens(profile="synthetic"):
    return CameraCalibration(
        profile, (640, 480), MATRIX, np.zeros(5), 0.1,
        quality={"version": QUALITY_VERSION, "intrinsics": {"passed": True}},
    )


def test_guide_rejects_duplicates_missing_board_and_motion(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    stamp = observe_still(camera, board_view(), 1000)
    assert len(camera.calibrator.corners) == 1
    assert "center" in camera.guide.coverage
    stamp = observe_still(camera, board_view(), stamp)
    assert len(camera.calibrator.corners) == 1
    stamp = observe_still(camera, np.full((480, 640, 3), 255, np.uint8), stamp)
    assert camera.guide.visible is False
    assert camera.guide.stable == 0
    assert camera.guide.snapshot(camera, stamp)["outline"] == []
    for x in [200, 400, 240, 440]:
        camera.capture_ms = stamp
        camera.guide.observe(camera, board_view(x=x), stamp)
        stamp += 400
    assert len(camera.calibrator.corners) == 1
    assert camera.calibration is None
    assert camera.guide.snapshot(camera, stamp + 2000)["checks"][0]["done"] is False


def test_capture_count_is_visible_before_lens_validation(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    observe_still(camera, board_view(), 1000)
    state = camera.guide.snapshot(camera, camera.capture_ms)
    assert state["collection"]["accepted"] == 1
    assert state["collection"]["minimum"] == 16
    assert state["progress"] == 0
    assert state["stage"] == "collect" and camera.calibration is None


def test_last_quality_failure_survives_duplicate_capture_feedback(tmp_path, monkeypatch):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    corners, ids = camera.calibrator.detect(board_view())
    camera.calibrator.corners = [corners.copy() for _ in range(15)]
    camera.calibrator.ids = [ids.copy() for _ in range(15)]
    camera.calibrator.size, camera.calibrator.profile = (640, 480), "synthetic"

    def reject():
        raise ValueError("별도 검증 오차가 커요.")

    monkeypatch.setattr(camera.calibrator, "solve_intrinsics", reject)
    stamp = observe_still(camera, board_view(x=450), 1000)
    observe_still(camera, board_view(x=450), stamp)
    state = camera.guide.snapshot(camera, camera.capture_ms)
    assert state["collection"]["accepted"] == 16
    assert state["collection"]["last_error"] == "별도 검증 오차가 커요."
    assert state["progress"] == 0 and camera.calibration is None
    header = {
        "type": "frame", "camera_id": "cam1", "seq": 100, "captured_ms": camera.capture_ms,
        "processing_ms": 50, "guide": state, "composite_jpeg_bytes": 4, "reference_jpeg_bytes": 4,
    }
    packet = pack(header, b"jpegjpeg")
    assert len(packet) - 12 <= MAX_HEADER


def test_real_charuco_auto_capture_solves_intrinsics_then_waits_for_fixing(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    stamp = 1000
    placements = ((130, 170), (510, 310), (510, 170), (130, 310),
                  (320, 170), (320, 310), (160, 240), (480, 240))
    views = []
    for index, (distance, (x, y)) in enumerate(
        (distance, placement) for distance in (0.5, 0.6, 0.7) for placement in placements
    ):
        rotation = (0.3 if index % 2 else -0.3,
                    0.3 if index % 3 else -0.3, (index % 3 - 1) * 0.1)
        views.append((x, y, distance, rotation))
    for x, y, distance, rotation in views:
        stamp = observe_still(camera, board_view(x, y, distance, rotation), stamp)
    assert "tilt" in camera.guide.coverage
    assert len(camera.calibrator.corners) >= 16
    assert camera.calibration is not None, camera.guide.message
    assert camera.calibration.lens_validated
    np.testing.assert_allclose(camera.calibration.matrix[:2, :], MATRIX[:2, :], atol=20)
    assert camera.calibration.ready is False
    assert camera.guide.snapshot(camera, camera.capture_ms)["stage"] == "place"
    stamp = observe_still(camera, board_view(), stamp)
    assert camera.calibration.ready is False
    camera.guide.arm_floor(camera)
    stamp = observe_still(camera, board_view(), stamp)
    assert not camera.floor_validated
    assert not camera.calibration.ready
    for _ in range(3):
        stamp = observe_still(camera, board_view(), stamp)
    assert camera.calibration.ready
    assert camera.floor_validated
    assert camera.guide.snapshot(camera, camera.capture_ms)["stage"] == "verify"
    assert camera.guide.snapshot(camera, camera.capture_ms)["progress"] < 1
    camera.pair_quality = {
        "passed": True, "samples": 6, "required": 6,
        "generation": camera.generation, "expires_ms": stamp + 10_000,
    }
    assert camera.guide.snapshot(camera, camera.capture_ms)["stage"] == "background"
    packet = pack({"guide": camera.guide.snapshot(camera, camera.capture_ms)}, b"jpeg")
    assert len(packet) - 8 <= MAX_HEADER
    assert unpack(packet)[0]["guide"]["checks"][4]["done"]


def test_guided_actions_enforce_floor_and_background_preconditions(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    model_path = tmp_path / "pose.task"
    model_path.touch()
    app = create_app(data_dir=tmp_path, token="test-key", model_path=model_path)
    camera = app.state.hub.cameras["cam1"]
    headers = {"X-Session-Key": "test-key"}
    camera.connected, camera.profile = True, "synthetic"
    camera.capture_ms, camera.frame = now_ms(), board_view()
    with TestClient(app) as client:
        assert client.post("/api/cameras/cam1/guide-floor").status_code == 401
        assert client.post("/api/cameras/cam1/guide-floor", headers=headers).status_code == 400
        camera.calibration = validated_lens()
        assert client.post("/api/cameras/cam1/guide-background", headers=headers).status_code == 400
        result = client.post("/api/cameras/cam1/guide-floor", headers=headers)
        assert result.json()["guide"]["stage"] == "floor"
        camera.calibrator.solve_extrinsics(camera.frame, camera.calibration)
        camera.floor_validated = True
        other = app.state.hub.cameras["cam2"]
        other.connected = True
        assert client.post("/api/cameras/cam1/guide-background", headers=headers).status_code == 400
        other.connected = False
        camera.capture_ms = now_ms()
        result = client.post("/api/cameras/cam1/guide-background", headers=headers)
        assert result.status_code == 400
        assert client.post("/api/cameras/cam1/play", headers=headers).status_code == 400
        camera.calibration.save(camera.path)
        camera.capture_ms = now_ms() - 2000
        assert client.post("/api/cameras/cam1/guide-floor", headers=headers).status_code == 400
        client.post("/api/cameras/cam1/guide-restart", headers=headers)
        assert camera.calibration is None
        assert camera.background is None
        assert camera.mode == "calibration"
        assert camera.guide.enabled
        assert not camera.path.exists()


def test_saved_intrinsics_wait_for_live_profile_and_reconnect_resets_progress(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.calibration = CameraCalibration("rear", (640, 480), MATRIX, np.zeros(5), 0.1)
    assert camera.guide.snapshot(camera, time.monotonic() * 1000)["stage"] == "collect"
    camera.profile, camera.connected = "rear", True
    camera.frame = board_view()
    camera.capture_ms = now_ms()
    assert camera.guide.snapshot(camera, now_ms())["stage"] == "collect"
    camera.calibration = validated_lens("rear")
    assert camera.guide.snapshot(camera, now_ms())["stage"] == "place"
    camera.guide.coverage.add("center")
    camera.background = np.zeros((480, 640, 3), np.uint8)
    camera.reset_guidance()
    assert not camera.guide.coverage
    assert camera.background is None


def test_profile_change_returns_playing_camera_to_automatic_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    hub = create_app(data_dir=tmp_path, token="test-key").state.hub
    camera = hub.cameras["cam1"]
    camera.calibration = CameraCalibration("old", (640, 480), MATRIX, np.zeros(5), 0.1)
    camera.profile, camera.mode = "new", "play"
    camera.guide.enabled = False
    camera.background = board_view()
    camera.next_segmentation = np.ones((480, 640), np.float32)
    hub.analyze_frame(camera, board_view(), hub.game.snapshot())
    assert camera.calibration is None
    assert camera.background is None
    assert camera.mode == "calibration"
    assert camera.guide.enabled
    assert camera.next_segmentation is None


def test_floor_checks_new_frames_and_rejects_drift_without_saving_pose(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.calibration = validated_lens()
    camera.connected, camera.profile = True, "synthetic"
    camera.guide.arm_floor(camera)
    frame = board_view()
    stamp = observe_still(camera, frame, 1000)
    assert camera.guide.floor_samples == 1
    # Reusing a captured frame at later processing times cannot advance validation.
    for index in range(12):
        camera.guide.observe(camera, frame, stamp + index * 400)
    assert camera.guide.floor_samples == 1
    assert not camera.path.exists()
    stamp += 6000
    observe_still(camera, board_view(x=340), stamp)
    assert camera.guide.floor_samples < FLOOR_OBSERVATIONS
    assert camera.floor_validated is False
    assert camera.calibration.ready is False
    assert not camera.path.exists()


def test_completion_requires_live_pair_and_observed_fixed_board(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.calibration = validated_lens()
    camera.calibrator.solve_extrinsics(board_view(), camera.calibration)
    camera.connected, camera.profile, camera.mode = True, "synthetic", "play"
    camera.capture_ms, camera.frame = 1000, board_view()
    camera.floor_validated = True
    camera.background = camera.frame.copy()
    assert camera.guide.snapshot(camera, 1000)["progress"] < 1
    camera.pair_quality = {
        "passed": True, "monitoring": "observed", "generation": camera.generation,
        "expires_ms": 2000,
    }
    assert camera.guide.snapshot(camera, 1000)["progress"] == 1
    assert camera.guide.snapshot(camera, 3000)["progress"] < 1
    camera.capture_ms = 2100
    assert camera.guide.snapshot(camera, 2100)["progress"] < 1
    camera.capture_ms = 1000
    camera.pair_quality["monitoring"] = "unobserved"
    assert camera.guide.snapshot(camera, 1000)["progress"] < 1
    assert camera.guide.snapshot(camera, 1000)["stage"] == "verify"


def test_rolling_board_in_image_does_not_count_as_perspective_tilt(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    observe_still(camera, board_view(rotation=(0, 0, 0.5)), 1000)
    assert "tilt" not in camera.guide.coverage
    assert camera.guide.snapshot(camera, camera.capture_ms)["progress"] == 0

import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ar_lava.calibration import BOARD, CameraCalibration
from ar_lava.guidance import COVERAGE
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


def test_real_charuco_auto_capture_solves_intrinsics_then_waits_for_fixing(tmp_path):
    camera = Camera("cam1", tmp_path)
    camera.connected, camera.profile = True, "synthetic"
    stamp = 1000
    views = [
        (320, 240, 0.65, (0.2, -0.15, 0)),
        (190, 240, 0.75, (-0.2, 0.25, 0)),
        (450, 240, 0.75, (0.25, 0.1, 0.05)),
        (320, 135, 0.85, (-0.3, 0.1, -0.05)),
        (320, 345, 0.85, (0.3, -0.2, 0)),
        (320, 240, 0.43, (0.08, -0.1, 0.1)),
        (300, 220, 1.0, (-0.2, -0.25, 0.4)),
        (260, 240, 0.6, (0.35, -0.3, -0.4)),
        (380, 230, 0.6, (-0.3, 0.35, 0.3)),
        (290, 290, 0.65, (0.1, 0.4, 0.15)),
        (350, 190, 0.7, (-0.4, -0.1, -0.2)),
        (270, 250, 0.8, (0.4, 0.2, 0.5)),
        (370, 270, 0.55, (-0.25, -0.3, -0.3)),
    ]
    for x, y, distance, rotation in views:
        stamp = observe_still(camera, board_view(x, y, distance, rotation), stamp)
    assert camera.guide.coverage == {name for name, _, _ in COVERAGE}
    assert len(camera.calibrator.corners) >= 12
    assert camera.calibration is not None, camera.guide.message
    assert camera.calibration.rms <= 1.5
    np.testing.assert_allclose(camera.calibration.matrix[:2, :], MATRIX[:2, :], atol=20)
    assert camera.calibration.ready is False
    assert camera.guide.snapshot(camera, camera.capture_ms)["stage"] == "place"
    stamp = observe_still(camera, board_view(), stamp)
    assert camera.calibration.ready is False
    camera.guide.floor_armed = True
    camera.guide.stable, camera.guide.previous = 0, None
    observe_still(camera, board_view(), stamp)
    assert camera.calibration.ready
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
        camera.calibration = CameraCalibration("synthetic", (640, 480), MATRIX, np.zeros(5), 0.1)
        assert client.post("/api/cameras/cam1/guide-background", headers=headers).status_code == 400
        result = client.post("/api/cameras/cam1/guide-floor", headers=headers)
        assert result.json()["guide"]["stage"] == "floor"
        camera.calibrator.solve_extrinsics(camera.frame, camera.calibration)
        other = app.state.hub.cameras["cam2"]
        other.connected = True
        assert client.post("/api/cameras/cam1/guide-background", headers=headers).status_code == 400
        other.connected = False
        camera.capture_ms = now_ms()
        result = client.post("/api/cameras/cam1/guide-background", headers=headers)
        assert result.status_code == 200
        assert result.json()["guide"]["stage"] == "ready"
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

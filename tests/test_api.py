import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ar_lava.calibration import CameraCalibration
from ar_lava.server import Viewer, create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    return create_app(data_dir=tmp_path, token="test-key")


def test_http_auth_game_validation_and_no_depth_before_capture(app):
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/state").status_code == 401
        headers = {"X-Session-Key": "test-key"}
        assert client.get("/api/state", headers=headers).status_code == 200
        assert client.post("/api/game/start", headers=headers).status_code == 409
        assert (
            client.post("/api/game/config", json={"cell_m": -1}, headers=headers).status_code == 400
        )
        assert client.post("/api/cameras/cam1/sample", headers=headers).status_code == 400
        assert client.get("/api/depth/frame.npz", headers=headers).status_code == 409
        assert client.get("/").status_code == 200
        assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


def test_websocket_state_auth_and_invalid_camera(app):
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"role": "viewer", "token": "test-key"})
            assert websocket.receive_json()["type"] == "state"
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"role": "camera", "camera_id": "cam4", "token": "test-key"})
            assert websocket.receive()["code"] == 1008
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"role": "viewer", "token": "wrong"})
            assert websocket.receive()["code"] == 1008


def test_only_two_cameras_are_registered_and_third_camera_is_rejected(app):
    headers = {"X-Session-Key": "test-key"}
    with TestClient(app) as client:
        assert set(app.state.hub.cameras) == {"cam1", "cam2"}
        state = client.get("/api/state", headers=headers).json()
        assert {camera["id"] for camera in state["cameras"]} == {"cam1", "cam2"}
        assert client.post("/api/cameras/cam3/sample", headers=headers).status_code == 404
        assert (
            client.get("/api/pairing/cam3.png?address=127.0.0.1", headers=headers).status_code
            == 404
        )
        assert client.post("/api/rtc/camera/cam3", headers=headers).status_code == 404
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"role": "camera", "camera_id": "cam3", "token": "test-key"})
            assert websocket.receive()["code"] == 1008


def test_latest_only_viewer_queue_is_bounded():
    async def check():
        viewer = Viewer()
        for seq in range(100):
            viewer.push("cam1", seq)
            viewer.push("cam2", seq)
        assert viewer.pending == {"cam1": 99, "cam2": 99}

    asyncio.run(check())


@pytest.mark.parametrize(
    "action", ["play", "background", "guide-background", "guide-floor", "extrinsics"]
)
def test_legacy_low_rms_pose_cannot_bypass_quality_checks(app, action):
    from ar_lava.server import now_ms

    camera = app.state.hub.cameras["cam1"]
    camera.connected, camera.profile = True, "legacy"
    camera.frame = np.zeros((480, 640, 3), np.uint8)
    camera.capture_ms = now_ms()
    camera.calibration = CameraCalibration(
        "legacy", (640, 480), np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]]),
        np.zeros(5), 0.01, np.zeros((3, 1)), np.array([[0.], [0.], [1.]]),
    )
    with TestClient(app) as client:
        response = client.post(f"/api/cameras/cam1/{action}", headers={"X-Session-Key": "test-key"})
    assert response.status_code == 400
    assert not camera.depth_usable(now_ms())


def test_failed_new_intrinsics_invalidates_active_pose(app, monkeypatch):
    from ar_lava.server import now_ms

    camera = app.state.hub.cameras["cam1"]
    camera.connected, camera.profile, camera.mode = True, "test", "play"
    camera.frame = np.zeros((480, 640, 3), np.uint8)
    camera.capture_ms = now_ms()

    def reject():
        raise ValueError("holdout rejected")

    monkeypatch.setattr(camera.calibrator, "solve_intrinsics", reject)
    with TestClient(app) as client:
        response = client.post(
            "/api/cameras/cam1/intrinsics", headers={"X-Session-Key": "test-key"}
        )
    assert response.status_code == 400
    assert camera.mode == "calibration" and camera.calibration is None
    assert not camera.floor_validated and not camera.pair_quality

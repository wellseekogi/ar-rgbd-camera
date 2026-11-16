import base64
import json
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ar_lava.calibration import CameraCalibration
from ar_lava.rgbd import encode_rgbd
from ar_lava.server import create_app
from ar_lava.stereo import StereoRectifier
from scripts.fetch_rgbd import decode_frame


def frame_result(hub, stamp=1000, seq=7):
    matrix = np.array([[500.0, 0, 3], [0, 500, 1], [0, 0, 1]])
    for index, name in enumerate(("cam1", "cam2")):
        camera = hub.cameras[name]
        camera.calibration = CameraCalibration(
            name,
            (6, 2),
            matrix,
            np.zeros(5),
            0.1,
            np.zeros((3, 1)),
            np.array([[-index * 0.2], [0.0], [2.0]]),
        )
        camera.connected, camera.mode = True, "play"
        camera.capture_ms, camera.clock.rtt_ms = stamp, 1
        camera.sample_frame = np.full((2, 6, 3), 120, np.uint8)
    left, right = hub.cameras["cam1"], hub.cameras["cam2"]
    metadata = StereoRectifier(left.calibration, right.calibration).metadata()
    metadata.update(
        {
            "captured_ms": stamp,
            "right_captured_ms": stamp - 1,
            "source_seq": seq,
            "pair_skew_ms": 1,
            "calibration_generation": [left.generation, right.generation],
            "timestamp_kind": "browser_draw_monotonic_estimate",
            "processing_ms": 10,
        }
    )
    _, reference = cv2.imencode(".jpg", left.sample_frame)
    return {
        "metadata": metadata,
        "depth_m": np.array(
            [
                [np.nan, np.inf, -1, 0.1, 0.2, 1],
                [2, 4, 8, 14, 15, 15.1],
            ],
            dtype=np.float32,
        ),
        "reference": reference.tobytes(),
        "tracking": {
            "capture_ms": stamp,
            "valid": False,
            "reason": "양발 미확인",
            "joints": {"29": [0.3, 0.4, 0.15]},
            "feet": [],
        },
    }


@pytest.fixture
def rgbd_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    monkeypatch.setattr("ar_lava.server.now_ms", lambda: 1050)
    app = create_app(data_dir=tmp_path, token="test-key")
    frame = frame_result(app.state.hub)
    app.state.hub.stereo.frame_result = frame
    app.state.hub.stereo.result = frame["tracking"]
    return app


def decode_depth(body):
    return np.frombuffer(base64.b64decode(body["depth"]["data_base64"]), dtype="<f4").reshape(
        body["depth"]["shape"]
    )


def test_rest_rgbd_auth_normalization_restore_mask_and_matching_reference(rgbd_app):
    headers = {"X-Session-Key": "test-key"}
    with TestClient(rgbd_app) as client:
        assert client.get("/api/rgbd/frame").status_code == 401
        response = client.get("/api/rgbd/frame", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["schema"] == "ar-lava.rgbd.v1"
        assert body["camera_id"] == "cam1"
        assert body["source_camera_ids"] == ["cam1", "cam2"]
        assert body["depth"]["near_m"] == 0.1 and body["depth"]["far_m"] == 15
        normalized = decode_depth(body)
        mask = np.frombuffer(base64.b64decode(body["valid_mask"]["data_base64"]), np.uint8)
        mask = mask.reshape(normalized.shape).astype(bool)
        assert normalized.dtype == np.dtype("<f4")
        assert np.isfinite(normalized).all()
        assert (0 <= normalized).all() and (normalized <= 1).all()
        assert not mask[0, :4].any() and not mask[1, 5]
        assert (normalized[~mask] == 0).all() and normalized[1, 4] == 1
        restored = 0.1 + normalized * (15 - 0.1)
        original = rgbd_app.state.hub.stereo.frame_result
        np.testing.assert_allclose(restored[mask], original["depth_m"][mask], atol=2e-6)
        assert base64.b64decode(body["image"]["data_base64"]) == original["reference"]
        assert body["captured_ms"] == 1000 and body["age_ms"] == 50
        assert body["tracking"]["valid"] is False  # RGB-D does not require visible feet.
        assert body["tracking"]["joints_world_m"]["29"] == [0.3, 0.4, -0.15]
        assert body["camera"]["image_space"] == "original_left_distorted"
        assert client.get("/api/rgbd/frame", headers=headers).json()["frame_id"] == body["frame_id"]


@pytest.mark.parametrize("invalidation", ["expired", "disconnect", "recalibrate", "empty"])
def test_rest_rgbd_rejects_stale_or_invalidated_results(rgbd_app, monkeypatch, invalidation):
    hub = rgbd_app.state.hub
    if invalidation == "expired":
        monkeypatch.setattr("ar_lava.server.now_ms", lambda: 1351)
    elif invalidation == "disconnect":
        hub.cameras["cam2"].connected = False
    elif invalidation == "recalibrate":
        hub.cameras["cam1"].generation += 1
    else:
        hub.stereo.frame_result = None
    with TestClient(rgbd_app) as client:
        assert (
            client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"}).status_code == 409
        )


def test_rest_rgbd_holds_same_snapshot_when_a_new_frame_publishes(rgbd_app, monkeypatch):
    hub = rgbd_app.state.hub
    original = hub.stereo.frame_result
    original["person_mask"] = np.full((2, 6), 0.25, np.float32)

    def publish_during_encoding(result, server_ms):
        hub.stereo.frame_result = frame_result(hub, stamp=1025, seq=8)
        hub.stereo.frame_result["person_mask"] = np.full((2, 6), 0.75, np.float32)
        return encode_rgbd(result, server_ms)

    monkeypatch.setattr("ar_lava.server.encode_rgbd", publish_during_encoding)
    with TestClient(rgbd_app) as client:
        response = client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"})
        assert response.status_code == 200
        assert response.json()["captured_ms"] == original["metadata"]["captured_ms"]
        assert base64.b64decode(response.json()["image"]["data_base64"]) == original["reference"]
        mask = response.json()["segmentation"]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(mask["data_base64"]), np.uint8), 0)
        assert (decoded == 64).all()
        assert mask["frame_id"] == response.json()["frame_id"]


def test_rest_rgbd_rechecks_expiry_after_encoding(rgbd_app, monkeypatch):
    def expire_during_encoding(result, server_ms):
        content = encode_rgbd(result, server_ms)
        monkeypatch.setattr("ar_lava.server.now_ms", lambda: 1351)
        return content

    monkeypatch.setattr("ar_lava.server.encode_rgbd", expire_during_encoding)
    with TestClient(rgbd_app) as client:
        assert (
            client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"}).status_code == 409
        )


def test_normalization_does_not_change_when_scene_range_changes(rgbd_app):
    frame = rgbd_app.state.hub.stereo.frame_result
    before = decode_depth(json.loads(encode_rgbd(frame, 1050)))
    frame["depth_m"][1, 4] = 8
    after = decode_depth(json.loads(encode_rgbd(frame, 1050)))
    assert before[1, 0] == after[1, 0]  # An unchanged 2m point keeps its normalized value.


def test_consumer_example_decodes_actual_rest_response(rgbd_app):
    with TestClient(rgbd_app) as client:
        body = client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"}).json()
    rgb, normalized, valid, meters = decode_frame(body)
    original = rgbd_app.state.hub.stereo.frame_result
    assert rgb == original["reference"]
    assert normalized.dtype == np.float32 and valid.dtype == bool
    np.testing.assert_allclose(meters[valid], original["depth_m"][valid], atol=2e-6)
    assert np.isnan(meters[~valid]).all()
    body["depth"]["shape"] = [6, 2]
    with pytest.raises(ValueError, match="형식·크기"):
        decode_frame(body)


def test_existing_npz_keeps_metric_depth_and_matching_rgb(rgbd_app):
    with TestClient(rgbd_app) as client:
        response = client.get("/api/depth/frame.npz", headers={"X-Session-Key": "test-key"})
    assert response.status_code == 200
    with np.load(BytesIO(response.content), allow_pickle=False) as frame:
        original = rgbd_app.state.hub.stereo.frame_result
        np.testing.assert_array_equal(frame["depth_m"], original["depth_m"])
        assert frame["reference_jpeg"].tobytes() == original["reference"]


def test_rest_segmentation_is_same_size_probability_png_and_named_provider(rgbd_app):
    probabilities = np.array([[0, 0.1, 0.3, 0.5, 0.8, 1]] * 2, np.float32)
    rgbd_app.state.hub.stereo.frame_result["person_mask"] = probabilities
    with TestClient(rgbd_app) as client:
        body = client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"}).json()
    mask = body["segmentation"]
    assert mask["available"] is True
    assert mask["model"] == "mediapipe_pose_landmarker_lite"
    assert mask["shape"] == body["depth"]["shape"]
    assert mask["semantics"] == "person_foreground_probability"
    png = np.frombuffer(base64.b64decode(mask["data_base64"]), np.uint8)
    decoded = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint8 and decoded.shape == probabilities.shape
    np.testing.assert_allclose(decoded * mask["probability_scale"], probabilities, atol=1 / 255)


@pytest.mark.parametrize(
    "mask",
    [None, np.zeros((1, 6)), np.full((2, 6), np.nan), np.full((2, 6), -0.1), np.full((2, 6), 1.1)],
)
def test_missing_or_invalid_segmentation_does_not_replace_rgbd(rgbd_app, mask):
    rgbd_app.state.hub.stereo.frame_result["person_mask"] = mask
    with TestClient(rgbd_app) as client:
        response = client.get("/api/rgbd/frame", headers={"X-Session-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["segmentation"]["available"] is False
    assert "data_base64" not in body["segmentation"]
    assert body["image"]["data_base64"] and decode_depth(body).shape == (2, 6)

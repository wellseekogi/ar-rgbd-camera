from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ar_lava.pairing import phone_url
from ar_lava.protocol import pack, unpack
from ar_lava.server import Camera, create_app


def test_pairing_qr_requires_auth_and_contains_fragment_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    app = create_app(tmp_path, token="pair-test-key")
    with TestClient(app) as client:
        assert client.get("/api/pairing").status_code == 401
        assert client.get("/api/pairing/cam1.png?address=127.0.0.1").status_code == 401
        headers = {"X-Session-Key": "pair-test-key"}
        response = client.get("/api/pairing/cam1.png?address=127.0.0.1", headers=headers)
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        qr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        url = cv2.QRCodeDetector().detectAndDecode(qr)[0]
        parsed = urlsplit(url)
        assert parsed.hostname == "127.0.0.1" and not parsed.query
        assert parse_qs(parsed.fragment) == {"key": ["pair-test-key"], "role": ["cam1"]}
        assert (
            client.get("/api/pairing/cam1.png?address=8.8.8.8", headers=headers).status_code == 400
        )


def test_invitation_preserves_special_key_characters():
    url = phone_url("https", "127.0.0.1", 8443, "a&b+c", "cam2")
    assert parse_qs(urlsplit(url).fragment) == {"key": ["a&b+c"], "role": ["cam2"]}


def test_connection_status_requires_received_frames(tmp_path):
    camera = Camera("cam1", tmp_path)
    assert camera.info(100)["connection_status"] == "disconnected"
    camera.connected = True
    assert camera.info(100)["connection_status"] == "waiting"
    camera.frame_times.extend([100, 200, 300])
    info = camera.info(400)
    assert info["connection_status"] == "receiving" and info["fps"] == 10
    assert camera.info(2500)["connection_status"] == "stalled"
    camera.connected = False
    assert camera.info(400)["fps"] == 0


def test_websocket_frame_ack_and_detected_camera_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    app = create_app(tmp_path, token="pair-test-key")
    _, jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"role": "camera", "camera_id": "cam1", "token": "pair-test-key"})
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "ping", "client_ms": 100})
            assert websocket.receive_json()["type"] == "pong"
            websocket.send_json({"type": "sync", "client_send": 100, "client_receive": 102})
            websocket.send_bytes(
                pack({"seq": 0, "captured_ms": 101, "profile": "rear"}, jpeg.tobytes())
            )
            header, _ = unpack(websocket.receive_bytes())
            assert header["seq"] == 0 and header["camera_id"] == "cam1"
            camera = app.state.hub.cameras["cam1"]
            assert camera.received_frames == 1
            assert camera.info(camera.frame_times[-1])["connection_status"] == "receiving"
        assert not app.state.hub.cameras["cam1"].connected

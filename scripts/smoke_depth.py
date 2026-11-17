"""Exercise CPU or CUDA depth + HTTP/WebSocket output with official rectified sample images.

This is a software integration check, not physical phone calibration or accuracy validation.
"""

import argparse
import json
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ar_lava.calibration import QUALITY_VERSION, CameraCalibration  # noqa: E402
from ar_lava.protocol import unpack  # noqa: E402
from ar_lava.server import create_app, now_ms  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    os.environ["LAVA_DEPTH_DEVICE"] = args.depth_device
    left = cv2.imdecode(
        np.frombuffer((ROOT / "data/benchmark/left.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    right = cv2.imdecode(
        np.frombuffer((ROOT / "data/benchmark/right.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    if left is None or right is None:
        raise SystemExit("Download the official demo_data/{left.png,right.png,K.txt} first")
    lines = (ROOT / "data/benchmark/K.txt").read_text().splitlines()
    matrix = np.array([float(value) for value in lines[0].split()]).reshape(3, 3)
    baseline = float(lines[1])
    app = create_app(ROOT / "data/smoke", token="local-smoke-key")
    hub = app.state.hub
    # Official rectified input has known geometry; physical calibration is tested separately.
    hub.calibration_check = SimpleNamespace(update=lambda cameras, now: None)
    headers = {"X-Session-Key": "local-smoke-key"}
    stopped = threading.Event()

    def supply_frames():
        while not stopped.wait(0.05):
            for i, frame in enumerate((left, right)):
                camera = hub.cameras[f"cam{i + 1}"]
                camera.capture_ms = now_ms()
                camera.seq += 1

    with TestClient(app) as client:
        for i, frame in enumerate((left, right)):
            camera = hub.cameras[f"cam{i + 1}"]
            camera.calibration = CameraCalibration(
                f"demo{i}",
                (frame.shape[1], frame.shape[0]),
                matrix,
                np.zeros(5),
                0,
                np.zeros((3, 1)),
                np.array([[-baseline * i], [0.0], [0.0]]),
                quality={"version": QUALITY_VERSION, "intrinsics": {"passed": True}},
            )
            camera.connected, camera.mode = True, "play"
            camera.profile = camera.calibration.profile
            camera.floor_validated = True
            camera.pair_quality = {
                "passed": True, "generation": camera.generation, "expires_ms": now_ms() + 60_000,
            }
            camera.clock.rtt_ms = 1
            camera.sample_frame = frame
        supplier = threading.Thread(target=supply_frames, daemon=True)
        supplier.start()
        try:
            deadline = time.monotonic() + 25
            response = None
            while time.monotonic() < deadline:
                response = client.get("/api/depth/frame.npz", headers=headers)
                if response.status_code == 200:
                    break
                time.sleep(0.05)
            assert response is not None and response.status_code == 200, hub.stereo.info()
            with np.load(BytesIO(response.content), allow_pickle=False) as result:
                depth = result["depth_m"]
                assert depth.shape == left.shape[:2]
                assert np.isfinite(depth).mean() > 0.2
                assert np.nanmin(depth) > 0.1 and np.nanmax(depth) <= 15
                metadata = json.loads(str(result["metadata"]))
                assert metadata["units"] == "meters"
                assert metadata["camera_id"] == "cam1"
                assert metadata["depth_device"] == hub.stereo.model.device
            rgbd = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                rgbd = client.get("/api/rgbd/frame", headers=headers)
                if rgbd.status_code == 200:
                    break
                time.sleep(0.02)
            assert rgbd is not None and rgbd.status_code == 200, hub.stereo.info()
            assert rgbd.json()["depth_device"] == hub.stereo.model.device
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"role": "depth", "token": "local-smoke-key"})
                header, payload = unpack(websocket.receive_bytes())
                size = header["depth_png_bytes"]
                png = cv2.imdecode(
                    np.frombuffer(payload[:size], dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                assert png.shape == left.shape[:2] and png.dtype == np.uint16
            assert hub.cameras["cam1"].landmarks is None
            assert hub.cameras["cam2"].landmarks is None
            report = {
                "backend": hub.stereo.model.backend,
                "device": hub.stereo.model.device,
                "providers": (
                    hub.stereo.model.session.get_providers()
                    if hasattr(hub.stereo.model, "session")
                    else ["OpenCV CPU"]
                ),
                "model_ms": hub.stereo.processing_ms,
                "valid_depth_fraction": float(np.isfinite(depth).mean()),
                "median_depth_m": float(np.nanmedian(depth)),
                "output_shape": list(depth.shape),
                "people_required": False,
                "physical_calibration_validation": "not_exercised_known_geometry_fixture",
                "http_npz": "passed",
                "websocket_png16": "passed",
                "http_rgbd_json": "passed",
            }
            (ROOT / f"data/depth-smoke-{hub.stereo.model.device}.json").write_text(
                json.dumps(report, indent=2)
            )
            (ROOT / "data/sample-depth-preview.jpg").write_bytes(hub.stereo.visualization)
            print(json.dumps(report), flush=True)
        finally:
            stopped.set()
            supplier.join(timeout=1)


if __name__ == "__main__":
    main()

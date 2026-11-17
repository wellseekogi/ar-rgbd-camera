import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ar_lava import stereo
from ar_lava.server import create_app


def test_cpu_recovers_known_disparity_and_rejects_blank_images():
    random = np.random.default_rng(123)
    left = random.integers(0, 256, (192, 320, 3), np.uint8)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    model = stereo.create_depth_model("not-required.onnx", "cpu")
    disparity = model.disparity(left, right)
    assert model.device == "cpu"
    assert np.nanmedian(disparity[20:-20, 110:-20]) == pytest.approx(12, abs=0.15)
    blank = np.zeros_like(left)
    assert np.isnan(model.disparity(blank, blank)).all()


def test_auto_falls_back_and_forced_cuda_reports_failure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(stereo, "FastFoundationStereo", unavailable)
    assert stereo.create_depth_model("missing.onnx", "auto").device == "cpu"
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        stereo.create_depth_model("missing.onnx", "cuda")
    with pytest.raises(ValueError):
        stereo.create_depth_model("missing.onnx", "invalid")


def test_auto_preserves_cuda_when_available(monkeypatch):
    expected = object()
    monkeypatch.setattr(stereo, "FastFoundationStereo", lambda path: expected)
    assert stereo.create_depth_model("model.onnx", "auto") is expected


def test_auto_works_without_onnxruntime_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert stereo.create_depth_model("missing.onnx", "auto").device == "cpu"


def test_cpu_preparation_does_not_download_neural_depth(tmp_path, monkeypatch):
    from scripts import prepare

    (tmp_path / "models").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "models/pose_landmarker_lite.task").write_bytes(b"existing-pose")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "local_ips", lambda: ["127.0.0.1"])
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--depth-device", "cpu"])

    def no_download(*args, **kwargs):
        raise AssertionError("CPU preparation must not download the neural depth model")

    monkeypatch.setattr(prepare.urllib.request, "urlretrieve", no_download)
    prepare.main()
    assert (tmp_path / "web/board.png").is_file()
    assert not (tmp_path / "models/c_fast_foundationstereo.onnx").exists()


def test_cpu_service_starts_without_onnx_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DEPTH_DEVICE", "cpu")
    monkeypatch.delenv("LAVA_DISABLE_STEREO", raising=False)
    app = create_app(data_dir=tmp_path, token="test-key")
    app.state.hub.stereo.model_path = tmp_path / "missing.onnx"
    with TestClient(app) as client:
        deadline = time.monotonic() + 2
        while app.state.hub.stereo.model is None and time.monotonic() < deadline:
            time.sleep(0.01)
        state = client.get("/api/state", headers={"X-Session-Key": "test-key"}).json()
        assert state["depth"]["device"] == "cpu"
        assert state["depth"]["ready"] is True
        assert state["depth"]["backend"] == "OpenCV StereoSGBM / CPU"

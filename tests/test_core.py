import json
import random
import struct

import cv2
import numpy as np
import pytest

from ar_lava.calibration import BOARD, Calibrator, CameraCalibration
from ar_lava.game import Game, GameConfig
from ar_lava.protocol import ClockSync, frame_header, pack, unpack
from ar_lava.render import render_with_depth
from ar_lava.rtc import read_stamp
from ar_lava.stereo import StereoDepth, StereoRectifier, colorize_depth, tracking_from_disparity


def calibrated_pair():
    k = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    left = CameraCalibration(
        "left", (640, 480), k, np.zeros(5), 0.1, np.zeros((3, 1)), np.array([[0.0], [0.0], [2.0]])
    )
    right = CameraCalibration(
        "right", (640, 480), k, np.zeros(5), 0.1, np.zeros((3, 1)), np.array([[-0.2], [0.0], [2.0]])
    )
    return left, right


def test_packet_and_malformed_inputs():
    packet = pack({"seq": 1, "captured_ms": 123, "profile": "rear"}, b"jpeg")
    header, payload = unpack(packet)
    assert payload == b"jpeg"
    assert frame_header(header) == (1, 123.0, "rear")
    for bad in (b"", b"0000", struct.pack("!I", 5000) + b"{}x", struct.pack("!I", 2) + b"[]x"):
        with pytest.raises(ValueError):
            unpack(bad)
    for bad in (float("nan"), "42", -1):
        with pytest.raises(ValueError):
            frame_header({"seq": 1, "captured_ms": bad, "profile": "rear"})


def test_clock_uses_lowest_rtt_and_rejects_invalid_samples():
    clock = ClockSync()
    with pytest.raises(ValueError):
        clock.to_server(100)
    clock.update(100, 1010, 120)
    assert clock.to_server(200) == 1100
    clock.update(100, 2000, 180)
    assert clock.to_server(200) == 1100
    with pytest.raises(ValueError):
        clock.update(100, 200, float("nan"))


def test_calibration_roundtrip_and_charuco_detection(tmp_path):
    camera, _ = calibrated_pair()
    path = tmp_path / "cam.json"
    camera.save(path)
    restored = CameraCalibration.load(path)
    np.testing.assert_allclose(restored.project([[0, 0, 0]]), [[320, 240]])
    image = cv2.copyMakeBorder(
        BOARD.generateImage((500, 700)), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
    )
    calibrator = Calibrator()
    assert calibrator.add(image, "test") == 1
    with pytest.raises(ValueError, match="비슷"):
        calibrator.add(image, "test")
    with pytest.raises(ValueError, match="12장"):
        calibrator.solve_intrinsics()


def test_rectification_depth_and_original_alignment():
    left, right = calibrated_pair()
    rectifier = StereoRectifier(left, right)
    disparity = np.full((192, 320), rectifier.p1[0, 0] * 0.2 / 2, dtype=np.float32)
    np.testing.assert_allclose(
        rectifier.world_point(160, 96, disparity[96, 160]), [0, 0, 0], atol=0.01
    )
    aligned = rectifier.align_to_original(disparity)
    assert aligned.shape == (480, 640)
    # alpha=0 rectification crops the vertical FOV when the output aspect ratio changes.
    assert 0.79 < np.isfinite(aligned).mean() < 0.81
    assert np.isfinite(aligned[60:420, 2:638]).all()
    np.testing.assert_allclose(aligned[240, 320], 2, atol=1e-5)
    disparity[:] = np.nan
    assert np.isnan(rectifier.align_to_original(disparity)).all()
    with pytest.raises(ValueError, match="왼쪽"):
        StereoRectifier(right, left)


def test_depth_does_not_require_a_person():
    left, right = calibrated_pair()
    rectifier = StereoRectifier(left, right)
    tracking = tracking_from_disparity(rectifier, np.ones((192, 320)), None, None, 123)
    assert tracking["capture_ms"] == 123
    assert tracking["valid"] is False


def test_depth_colors_use_fixed_metric_stops_and_black_for_invalid_depth():
    depth = np.array([[0.1, 0.2, 2, 4, 8, 15, 15.1, np.nan, np.inf]], np.float32)
    colors = colorize_depth(depth)
    assert colors.dtype == np.uint8
    np.testing.assert_array_equal(colors[0, [0, 6, 7, 8]], np.zeros((4, 3), np.uint8))
    np.testing.assert_array_equal(colors[0, 2], [92, 202, 255])
    np.testing.assert_array_equal(colors[0, 3], [177, 217, 87])
    np.testing.assert_array_equal(colors[0, 4], [255, 146, 93])
    np.testing.assert_array_equal(colors[0, 5], colors[0, 4])
    assert colors[0, 1, 2] > colors[0, 1, 0]


def test_depth_preview_and_reference_share_original_coordinates_and_metric_metadata():
    left, right = calibrated_pair()
    rectifier = StereoRectifier(left, right)

    class KnownDepth:
        def disparity(self, left_image, right_image):
            return np.full((192, 320), rectifier.p1[0, 0] * 0.2 / 2, np.float32)

    class Scene:
        game = Game()

    stereo = StereoDepth(Scene(), "unused.onnx")
    stereo.rectifier, stereo.model = rectifier, KnownDepth()
    frame = np.full((480, 640, 3), 120, np.uint8)
    person_mask = np.full(frame.shape[:2], 0.25, np.float32)
    result = stereo.compute(frame, frame, None, None, 123, {}, person_mask)
    np.testing.assert_array_equal(result["person_mask"], person_mask)
    preview = cv2.imdecode(np.frombuffer(result["preview"], np.uint8), cv2.IMREAD_COLOR)
    assert preview.shape == frame.shape
    assert result["metadata"]["preview_space"] == "original_left"
    assert result["metadata"]["center_depth_m"] == 2
    assert 70 < result["metadata"]["valid_percent"] < 80
    assert result["metadata"]["preview_range_m"] == [0.1, 8]
    json.dumps(result["metadata"], allow_nan=False)


def test_video_stamp_survives_jpeg_codec():
    frame = np.zeros((496, 640, 3), dtype=np.uint8)
    stamp = (0xA5C3 << 48) | (12345678 << 16) | 542
    for bit in range(64):
        if stamp & (1 << (63 - bit)):
            frame[-16:, bit * 8 : bit * 8 + 8] = 255
    ok, compressed = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
    assert ok
    cropped, captured, seq = read_stamp(cv2.imdecode(compressed, cv2.IMREAD_COLOR))
    assert (captured, seq, cropped.shape) == (12345678, 542, (480, 640, 3))


def valid_feet(game):
    cfg = game.config
    return {
        "valid": True,
        "feet": [
            [cfg.origin_x + 0.1, cfg.origin_y + 0.1, 0],
            [cfg.origin_x + 0.2, cfg.origin_y + 0.1, 0],
        ],
    }


def test_game_pauses_on_missing_or_outside_feet_and_resumes():
    game = Game(rng=random.Random(1))
    game.start()
    game.step(0.05, {"valid": False, "feet": []})
    assert game.status == "paused" and game.remaining == 60
    game.step(0.05, valid_feet(game))
    assert game.status == "playing" and game.remaining == 59.95
    game.step(0.05, {"valid": True, "feet": [[100, 100, 0], [100, 100, 0]]})
    assert game.status == "paused" and game.remaining == 59.95


def test_game_hazards_can_force_movement_and_damage_has_cooldown():
    game = Game(rng=random.Random(4))
    game.start()
    tracking = valid_feet(game)
    game.phase_left = 0
    game.step(0.05, tracking)
    assert game.phase == "warning"
    assert len(game.cells) < game.config.rows * game.config.cols - 1
    cell = game.cells[0]
    row, col = divmod(cell, game.config.cols)
    point = [
        game.config.origin_x + (col + 0.5) * game.config.cell_m,
        game.config.origin_y + (row + 0.5) * game.config.cell_m,
        0,
    ]
    game.phase_left = 0
    game.step(0.05, {"valid": True, "feet": [point, point]})
    assert game.phase == "lava" and game.lives == 2
    game.step(0.05, {"valid": True, "feet": [point, point]})
    assert game.lives == 2
    game.remaining = 0.01
    game.step(0.05, tracking)
    assert game.status == "won"


@pytest.mark.parametrize(
    "settings",
    [
        {"rows": 1},
        {"rows": True},
        {"cell_m": float("nan")},
        {"lives": 0},
        {"duration_s": 9},
        {"warning_s": 0.1},
    ],
)
def test_invalid_game_settings(settings):
    with pytest.raises(ValueError):
        Game(GameConfig(**settings))


def test_depth_occlusion_is_finite_and_correct_size():
    camera, _ = calibrated_pair()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    output = render_with_depth(frame, depth, camera, Game().snapshot())
    assert output.shape == frame.shape and output.dtype == np.uint8
    assert np.isfinite(output).all()


def test_json_metadata_does_not_serialize_nan():
    camera, right = calibrated_pair()
    json.dumps(StereoRectifier(camera, right).metadata(), allow_nan=False)


def test_camera_without_person_can_supply_depth_and_reconnect_invalidates(tmp_path):
    from ar_lava.server import Camera

    camera = Camera("cam1", tmp_path)
    camera.calibration = calibrated_pair()[0]
    camera.connected, camera.mode = True, "play"
    camera.sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera.capture_ms = 100
    camera.clock.rtt_ms = 10
    assert camera.landmarks is None and camera.depth_usable(150)
    assert not camera.depth_usable(500)
    camera.calibration.save(camera.path)
    reconnected = Camera("cam1", tmp_path)
    assert reconnected.calibration is not None and not reconnected.calibration.ready


def test_https_certificates_include_chain_identifiers_and_reuse_ca(tmp_path, monkeypatch):
    from cryptography import x509

    from scripts import prepare

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    ca_path = prepare.certificates(["127.0.0.1"])
    original_ca = ca_path.read_bytes()
    ca = x509.load_pem_x509_certificate(original_ca)
    leaf = x509.load_pem_x509_certificate((tmp_path / "certs/server.pem").read_bytes())
    subject = ca.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
    authority = leaf.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value
    assert authority.key_identifier == subject
    ca.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
    prepare.certificates(["127.0.0.1", "192.168.0.10"])
    assert ca_path.read_bytes() == original_ca

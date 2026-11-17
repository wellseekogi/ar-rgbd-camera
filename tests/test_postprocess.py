import asyncio
import copy
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ar_lava.calibration import CameraCalibration
from ar_lava.game import Game
from ar_lava.render import render, render_with_depth
from ar_lava.stereo import StereoDepth, StereoRectifier


def camera_pair(size, tilted):
    w, h = size
    matrix = np.array([[w * 0.8, 0, w / 2], [0, w * 0.8, h / 2], [0, 0, 1.0]])
    left = CameraCalibration(
        "left",
        size,
        matrix,
        np.array([0.02, -0.005, 0.001, -0.001, 0]),
        0.1,
        np.array([[0.08], [0.12], [0.02]]) if tilted else np.zeros((3, 1)),
        np.array([[0.0], [0.0], [2.0]]),
    )
    rotation = cv2.Rodrigues(np.array([0.01, -0.08, 0.01]))[0] if tilted else np.eye(3)
    right = copy.deepcopy(left)
    right.profile = "right"
    right.rvec = cv2.Rodrigues(rotation @ cv2.Rodrigues(left.rvec)[0])[0]
    right.tvec = rotation @ left.tvec + np.array([[-0.22], [0], [0]])
    return left, right


def original_alignment(rectifier, disparity):
    # Keep the previous full XYZ transform as an independent correctness reference.
    w, h = rectifier.left.size
    pixels = np.indices((h, w), dtype=np.float32)[::-1].transpose(1, 2, 0)
    points = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        rectifier.left.matrix,
        rectifier.left.distortion,
        R=rectifier.r1,
        P=rectifier.p1,
    ).reshape(h, w, 2)
    sampled = cv2.remap(
        disparity,
        points[..., 0],
        points[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    rays = np.c_[points.reshape(-1, 2), sampled.ravel(), np.ones(h * w)]
    with np.errstate(divide="ignore", invalid="ignore"):
        homogeneous = rays @ rectifier.q.T
        camera_points = (homogeneous[:, :3] / homogeneous[:, 3:]) @ rectifier.r1
    output = camera_points[:, 2].reshape(h, w).astype(np.float32)
    output[~np.isfinite(output) | (output <= 0.1) | (output > 15) | (sampled <= 0.5)] = np.nan
    return output


def original_render(frame, depth, calibration, game):
    output = render(frame, None, calibration, game)
    h, w = depth.shape
    pixels = np.indices((h, w), dtype=np.float32)[::-1].transpose(1, 2, 0).reshape(-1, 1, 2)
    rays_xy = cv2.undistortPoints(pixels, calibration.matrix, calibration.distortion)
    rays = np.c_[rays_xy.reshape(-1, 2), np.ones(h * w)]
    normal = cv2.Rodrigues(calibration.rvec)[0][:, 2]
    denominator = rays @ normal
    floor_z = np.full(h * w, np.nan)
    np.divide(
        float(normal @ calibration.tvec.reshape(3)),
        denominator,
        out=floor_z,
        where=np.abs(denominator) > 1e-8,
    )
    occluder = (
        np.isfinite(depth) & (floor_z.reshape(h, w) > 0) & (depth < floor_z.reshape(h, w) - 0.05)
    )
    output[occluder] = (frame[occluder] * 0.45 + output[occluder] * 0.55).astype(np.uint8)
    return output


@pytest.mark.parametrize("size", [(640, 480), (1280, 720)])
@pytest.mark.parametrize("tilted", [False, True])
def test_cached_geometry_preserves_depth_and_render_without_repeating_undistortion(
    size,
    tilted,
    monkeypatch,
):
    left, right = camera_pair(size, tilted)
    rectifier = StereoRectifier(left, right)
    random = np.random.default_rng(13)
    disparity = random.uniform(0.1, 80, (192, 320)).astype(np.float32)
    disparity[:, :6] = [np.nan, np.inf, -np.inf, 0, 0.5, -1]
    cases = [
        (values, original_alignment(rectifier, values)) for values in (disparity, disparity * 0.9)
    ]
    frame = random.integers(0, 256, (size[1], size[0], 3), np.uint8)
    # Include both sides of the occlusion threshold and invalid depth.
    depth = rectifier.floor_z.astype(np.float32) + random.choice([-0.2, 0, 0.2], frame.shape[:2])
    depth[::11] = np.nan
    game = Game().snapshot()
    expected_render = original_render(frame, depth, left, game)

    def repeated_undistortion(*args, **kwargs):
        pytest.fail("fixed pixel geometry must not be recalculated for each frame")

    monkeypatch.setattr(cv2, "undistortPoints", repeated_undistortion)
    for values, expected in cases:
        actual = rectifier.align_to_original(values)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(
            render_with_depth(frame, depth, left, game, floor_z=rectifier.floor_z),
            expected_render,
        )


@pytest.mark.parametrize("change", ["generation", "replacement"])
def test_depth_loop_replaces_geometry_cache_after_recalibration(change, monkeypatch):
    from ar_lava import stereo

    left, right = camera_pair((640, 480), True)
    frame = np.full((480, 640, 3), 120, np.uint8)
    cameras = {}
    for name, calibration in [("cam1", left), ("cam2", right)]:
        cameras[name] = SimpleNamespace(
            calibration=calibration,
            generation=0,
            capture_ms=time.monotonic() * 1000,
            seq=0,
            sample_frame=frame,
            sample_segmentation=None,
            landmarks=None,
            depth_usable=lambda now: True,
        )
    engine = StereoDepth(
        SimpleNamespace(cameras=cameras, game=Game(), depth_viewers=set()), "unused"
    )
    engine.enabled = True
    model = SimpleNamespace(disparity=lambda a, b: np.full((192, 320), 20, np.float32))
    monkeypatch.setattr(stereo, "create_depth_model", lambda path: model)
    rectifiers = []

    async def next_frame(delay):
        if engine.rectifier is not None:
            rectifiers.append(engine.rectifier)
            if len(rectifiers) == 2:
                raise asyncio.CancelledError
            if change == "replacement":
                cameras["cam1"].calibration = copy.deepcopy(left)
            else:
                cameras["cam1"].generation += 1
            cameras["cam1"].calibration.rvec[0, 0] += 0.05
            for camera in cameras.values():
                camera.capture_ms = time.monotonic() * 1000

    monkeypatch.setattr(stereo.asyncio, "sleep", next_frame)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.run())
    assert rectifiers[0] is not rectifiers[1]
    assert not np.allclose(rectifiers[0].floor_z, rectifiers[1].floor_z, equal_nan=True)
    fresh = StereoRectifier(cameras["cam1"].calibration, right)
    np.testing.assert_array_equal(rectifiers[1].floor_z, fresh.floor_z)
    for actual, expected in zip(rectifiers[1].original_maps, fresh.original_maps, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_inference_result_is_not_published_after_quality_is_revoked(monkeypatch):
    from ar_lava import stereo

    left, right = camera_pair((640, 480), False)
    frame = np.full((480, 640, 3), 120, np.uint8)
    approved = {"value": True, "computed": False}
    cameras = {
        name: SimpleNamespace(
            calibration=calibration, generation=0, capture_ms=time.monotonic() * 1000,
            seq=0, sample_frame=frame, sample_segmentation=None, landmarks=None,
            depth_usable=lambda now: approved["value"],
        )
        for name, calibration in (("cam1", left), ("cam2", right))
    }

    def disparity(a, b):
        approved["value"], approved["computed"] = False, True
        return np.full((192, 320), 20, np.float32)

    engine = StereoDepth(
        SimpleNamespace(cameras=cameras, game=Game(), depth_viewers=set()), "unused"
    )
    engine.enabled = True
    monkeypatch.setattr(
        stereo, "create_depth_model", lambda path: SimpleNamespace(disparity=disparity)
    )

    async def next_frame(delay):
        if approved["computed"]:
            raise asyncio.CancelledError

    monkeypatch.setattr(stereo.asyncio, "sleep", next_frame)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.run())
    assert approved["computed"]
    assert engine.frame_result is None and engine.result is None

import json

import cv2
import numpy as np
import pytest

from ar_lava.calibration import (
    BOARD,
    Calibrator,
    CameraCalibration,
    check_lens_model,
    observation_points,
)


def projected_views(*, flat=False, holdout_noise=0):
    calibrator = Calibrator()
    calibrator.size, calibrator.profile = (640, 480), "synthetic"
    matrix = np.array([[510.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    distortion = np.array([0.025, -0.015, 0.0005, -0.0004, 0.002])
    points = BOARD.getChessboardCorners()
    random = np.random.default_rng(902)
    for index in range(24):
        x = [150, 320, 490, 210, 430, 320][index % 6]
        y = [130, 350, 130, 340][index % 4]
        distance = [0.65, 0.8, 0.9][index % 3]
        rvec = np.array([
            0 if flat else [0.4, -0.4, 0.3, -0.35][index % 4],
            0 if flat else [-0.4, 0.25, 0.4][index % 3],
            (index % 5 - 2) * 0.1,
        ], dtype=np.float64)
        center = cv2.Rodrigues(rvec)[0] @ np.array([0.1, 0.14, 0])
        tvec = np.array([(x - 320) * distance / 510, (y - 240) * distance / 500, distance])
        tvec -= center
        corners = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0]
        corners += random.normal(0, 0.04, corners.shape)
        if index % 4 == 3:
            corners += random.normal(0, holdout_noise, corners.shape)
        calibrator.corners.append(corners.astype(np.float32))
        calibrator.ids.append(np.arange(len(points), dtype=np.int32).reshape(-1, 1))
    return calibrator, matrix


def test_independent_validation_accepts_diverse_views_and_persists_quality(tmp_path):
    calibrator, matrix = projected_views()
    result = calibrator.solve_intrinsics()
    assert result.lens_validated
    assert result.quality["intrinsics"]["validation_views"] >= 4
    assert result.quality["intrinsics"]["holdout_rms_px"] < 0.2
    np.testing.assert_allclose(result.matrix, matrix, atol=2)
    result.save(tmp_path / "quality.json")
    restored = CameraCalibration.load(tmp_path / "quality.json")
    assert restored.lens_validated
    assert restored.quality == result.quality


def test_frontal_translation_and_in_plane_rotation_do_not_count_as_tilt_diversity():
    calibrator, _ = projected_views(flat=True)
    with pytest.raises(ValueError, match="기울"):
        calibrator.solve_intrinsics()


def test_bad_unseen_points_fail_even_when_training_fit_is_good():
    calibrator, _ = projected_views(holdout_noise=2)
    with pytest.raises(ValueError, match="검증"):
        calibrator.solve_intrinsics()


def test_legacy_calibration_is_loadable_but_not_quality_validated(tmp_path):
    data = {
        "profile": "old", "size": [640, 480], "matrix": np.eye(3).tolist(),
        "distortion": [0] * 5, "rms": 0.1, "rvec": None, "tvec": None,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not CameraCalibration.load(path).lens_validated


def test_duplicate_views_cannot_pass_despite_low_training_error():
    calibrator, _ = projected_views()
    calibrator.corners = [calibrator.corners[0].copy() for _ in range(24)]
    with pytest.raises(ValueError):
        calibrator.solve_intrinsics()


def test_validation_views_are_never_passed_to_intrinsic_fit(monkeypatch):
    calibrator, _ = projected_views()
    heldout_ids = {id(corners) for corners in calibrator.corners[3::4]}
    original = cv2.aruco.calibrateCameraCharuco
    fitted_counts = []

    def fit(corners, *args, **kwargs):
        assert not any(id(view) in heldout_ids for view in corners)
        fitted_counts.append(len(corners))
        return original(corners, *args, **kwargs)

    monkeypatch.setattr(cv2.aruco, "calibrateCameraCharuco", fit)
    assert calibrator.solve_intrinsics().lens_validated
    assert len(fitted_counts) == 2


def test_unstable_fit_fails_even_if_independent_reprojection_error_is_low(monkeypatch):
    calibrator, _ = projected_views()
    original = cv2.aruco.calibrateCameraCharuco
    count = 0

    def fit(*args, **kwargs):
        nonlocal count
        count += 1
        result = list(original(*args, **kwargs))
        if count == 2:
            result[1] = result[1].copy()
            result[1][0, 0] *= 1.06
            result[1][1, 1] *= 1.06
        return tuple(result)

    monkeypatch.setattr(cv2.aruco, "calibrateCameraCharuco", fit)
    with pytest.raises(ValueError, match="일부"):
        calibrator.solve_intrinsics()


@pytest.mark.parametrize("distortion", [
    [-2.0, 0, 0, 0, 0],
    [0.37, -4.18, -0.004, 0.023, 18.85],
    [np.nan, 0, 0, 0, 0],
])
def test_invalid_or_noninvertible_distortion_is_not_accepted(distortion):
    matrix = np.array([[650.0, 0, 345], [0, 487, 313], [0, 0, 1]])
    with pytest.raises(ValueError):
        check_lens_model(matrix, np.array(distortion), (640, 640))


def test_many_points_in_only_two_columns_are_not_a_valid_observation():
    indices = np.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21], dtype=np.int32)
    pixels = np.array([[100 + (i % 4) * 50, 100 + (i // 4) * 40] for i in indices], np.float32)
    # All pixels are finite and have area; world points must also span both board axes.
    with pytest.raises(ValueError):
        observation_points(pixels, indices, (640, 480))

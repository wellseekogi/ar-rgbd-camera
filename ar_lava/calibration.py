import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# The same physical board stays still for both extrinsic captures.
SQUARE_M = 0.04
MARKER_M = 0.03
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), SQUARE_M, MARKER_M, DICTIONARY)
QUALITY_VERSION = 1
MIN_INTRINSIC_VIEWS = 16
MIN_TRAIN_VIEWS = 12
MIN_VALIDATION_VIEWS = 4
MIN_CORNERS = 12
MAX_TRAIN_RMS_PX = 0.7
MAX_TRAIN_VIEW_RMS_PX = 1.0
MAX_HOLDOUT_RMS_PX = 0.8
MAX_HOLDOUT_P95_PX = 1.5
MAX_HOLDOUT_VIEW_RMS_PX = 1.0
MAX_FOCAL_CHANGE_PERCENT = 3.0
MAX_MAPPING_CHANGE_P95_PX = 2.0
MAX_DISTORTION_ROUNDTRIP_PX = 0.5
MAX_POSE_RMS_PX = 1.0


@dataclass
class CameraCalibration:
    profile: str
    size: tuple[int, int]
    matrix: np.ndarray
    distortion: np.ndarray
    rms: float
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    quality: dict = field(default_factory=dict)

    @property
    def lens_validated(self):
        return (
            self.quality.get("version") == QUALITY_VERSION
            and self.quality.get("intrinsics", {}).get("passed") is True
        )

    @property
    def ready(self):
        return self.rvec is not None and self.tvec is not None

    @property
    def projection(self):
        rotation = cv2.Rodrigues(self.rvec)[0]
        return np.column_stack((rotation, self.tvec.reshape(3)))

    def project(self, points):
        return cv2.projectPoints(
            np.array(points, dtype=np.float64),
            self.rvec,
            self.tvec,
            self.matrix,
            self.distortion,
        )[0].reshape(-1, 2)

    def save(self, path: Path):
        result = {
            "profile": self.profile, "size": self.size, "rms": self.rms, "quality": self.quality,
        }
        for name in ("matrix", "distortion", "rvec", "tvec"):
            value = getattr(self, name)
            result[name] = value.tolist() if value is not None else None
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        for name in ("matrix", "distortion", "rvec", "tvec"):
            if data[name] is not None:
                data[name] = np.array(data[name], dtype=np.float64)
        data["size"] = tuple(data["size"])
        return cls(**data)


def observation_points(corners, ids, size):
    pixels = np.asarray(corners).reshape(-1, 2)
    indices = np.asarray(ids).ravel()
    board = BOARD.getChessboardCorners()
    if (
        len(indices) < MIN_CORNERS
        or len(pixels) != len(indices)
        or len(np.unique(indices)) != len(indices)
        or (indices < 0).any()
        or (indices >= len(board)).any()
        or not np.isfinite(pixels).all()
        or (pixels < 0).any()
        or (pixels >= size).any()
    ):
        raise ValueError("선명한 기준판 교차점이 12개 이상 필요해요. 기준판 전체를 보여 주세요.")
    world = board[indices]
    if (
        len(np.unique(world[:, 0])) < 3
        or len(np.unique(world[:, 1])) < 3
        or cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) < np.prod(size) * 0.015
    ):
        raise ValueError("기준판의 가로·세로 교차점이 넓게 보여야 해요. 조금 가까이 가 주세요.")
    return world, pixels


def check_lens_model(matrix, distortion, size):
    width, height = size
    focal = np.diag(matrix)[:2]
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(distortion).all()
        or (focal < max(size) * 0.2).any()
        or (focal > max(size) * 5).any()
        or not 0.5 <= focal[0] / focal[1] <= 2
        or not width * 0.25 <= matrix[0, 2] <= width * 0.75
        or not height * 0.25 <= matrix[1, 2] <= height * 0.75
    ):
        raise ValueError(
            "렌즈 계산값이 불안정해요. 화면 가장자리와 다른 기울기의 사진을 더 모아 주세요."
        )
    pixels = np.array(
        [[x, y] for y in np.linspace(0, height - 1, 9) for x in np.linspace(0, width - 1, 9)],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    normalized = (pixels.reshape(-1, 2) - matrix[:2, 2]) / focal
    radii = np.linspace(0, np.linalg.norm(normalized, axis=1).max(), 100)
    k1, k2, p1, p2, k3 = distortion.ravel()
    derivative = 1 + 3 * k1 * radii**2 + 5 * k2 * radii**4 + 7 * k3 * radii**6
    rays = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 2)
    recovered = cv2.projectPoints(
        np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3), matrix, distortion
    )[0].reshape(-1, 2)
    error = float(np.linalg.norm(recovered - pixels.reshape(-1, 2), axis=1).max())
    if (
        derivative.min() <= 0.2 or max(abs(p1), abs(p2)) > 0.1
        or not np.isfinite(error) or error > MAX_DISTORTION_ROUNDTRIP_PX
    ):
        raise ValueError(
            "렌즈 왜곡이 화면 가장자리에서 되돌려지지 않아요. 가장자리 사진을 다시 모아 주세요."
        )
    return error


class Calibrator:
    def __init__(self):
        self.corners = []
        self.ids = []
        self.size = None
        self.profile = None
        self.detector = cv2.aruco.CharucoDetector(BOARD)

    def detect(self, image):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if corners is None or ids is None:
            raise ValueError("기준판 내부 교차점이 12개 이상 보여야 해요.")
        observation_points(corners, ids, (image.shape[1], image.shape[0]))
        return corners, ids

    def add(self, image, profile):
        size = (image.shape[1], image.shape[0])
        if self.size is not None and (self.size != size or self.profile != profile):
            raise ValueError("카메라 설정이 바뀌었어요. 보정 사진을 초기화해 주세요.")
        corners, ids = self.detect(image)
        # Reject near-identical photographs instead of reporting false calibration quality.
        descriptor = np.r_[corners.mean(axis=0).ravel(), np.ptp(corners[:, 0], axis=0)]
        for previous in self.corners:
            other = np.r_[previous.mean(axis=0).ravel(), np.ptp(previous[:, 0], axis=0)]
            if np.linalg.norm(descriptor - other) < min(size) * 0.04:
                raise ValueError("직전 사진과 비슷해요. 기준판의 위치·크기·기울기를 바꿔 주세요.")
        self.corners.append(corners)
        self.ids.append(ids)
        self.size, self.profile = size, profile
        return len(self.corners)

    def solve_intrinsics(self):
        if len(self.corners) < MIN_INTRINSIC_VIEWS:
            raise ValueError("학습 12장과 별도 검증 4장, 서로 다른 보정 사진 16장 이상 필요해요.")
        observations = [
            observation_points(corners, ids, self.size)
            for corners, ids in zip(self.corners, self.ids, strict=True)
        ]
        pixels = np.vstack([pixels for _, pixels in observations]) / self.size
        cells = np.minimum((pixels * [4, 3]).astype(int), [3, 2])
        coverage_cells = len(np.unique(cells, axis=0))
        coverage_hull = cv2.contourArea(cv2.convexHull(pixels.astype(np.float32)))
        if coverage_cells < 9 or coverage_hull < 0.35:
            raise ValueError(
                "화면 전체의 정보가 부족해요. 가장자리와 위·아래에서 기준판을 촬영해 주세요."
            )
        training = [i for i in range(len(observations)) if i % 4 != 3]
        validation = [i for i in range(len(observations)) if i % 4 == 3]
        if len(training) < MIN_TRAIN_VIEWS or len(validation) < MIN_VALIDATION_VIEWS:
            raise ValueError("학습 12장과 별도 검증 4장이 필요해요.")
        rms, matrix, distortion, rotations, translations = cv2.aruco.calibrateCameraCharuco(
            [self.corners[i] for i in training], [self.ids[i] for i in training],
            BOARD,
            self.size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
        )
        roundtrip = check_lens_model(matrix, distortion, self.size)
        normals = np.array([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
        tilted = int((np.degrees(np.arccos(np.clip(abs(normals[:, 2]), 0, 1))) >= 15).sum())
        spread = float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
        if tilted < 4 or spread < 25 or np.ptp(normals[:, 0]) < 0.3 or np.ptp(normals[:, 1]) < 0.3:
            raise ValueError("원근 기울기가 부족해요. 기준판을 좌우·앞뒤로 기울여 촬영해 주세요.")
        train_errors = []
        for i, rvec, tvec in zip(training, rotations, translations, strict=True):
            world, observed = observations[i]
            projected = cv2.projectPoints(world, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
            train_errors.append(float(np.sqrt(np.mean(np.sum((projected - observed)**2, axis=1)))))
        if (
            not np.isfinite(rms) or rms > MAX_TRAIN_RMS_PX
            or max(train_errors) > MAX_TRAIN_VIEW_RMS_PX
        ):
            raise ValueError(
                "보정 사진의 재투영 오차가 커요. 흐리거나 흔들린 사진을 다시 촬영해 주세요."
            )
        holdout_errors, holdout_views = [], []
        for i in validation:
            world, observed = observations[i]
            # Neither this view nor the scored points were used to fit the tested parameters/pose.
            fit = np.arange(len(world)) % 3 != 0
            success, rvec, tvec = cv2.solvePnP(world[fit], observed[fit], matrix, distortion)
            if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                raise ValueError(
                    "별도 검증 영상의 위치를 확인하지 못했어요. 사진을 더 모아 주세요."
                )
            if ((cv2.Rodrigues(rvec)[0] @ world.T + tvec)[2] <= 0).any():
                raise ValueError(
                    "별도 검증 기준점의 깊이가 카메라 뒤에 있어요. 사진을 다시 모아 주세요."
                )
            projected = cv2.projectPoints(
                world[~fit], rvec, tvec, matrix, distortion
            )[0].reshape(-1, 2)
            errors = np.linalg.norm(projected - observed[~fit], axis=1)
            holdout_errors.extend(errors.tolist())
            holdout_views.append(float(np.sqrt(np.mean(errors**2))))
        holdout_rms = float(np.sqrt(np.mean(np.square(holdout_errors))))
        holdout_p95 = float(np.percentile(holdout_errors, 95))
        if (
            not np.isfinite(holdout_rms) or holdout_rms > MAX_HOLDOUT_RMS_PX
            or holdout_p95 > MAX_HOLDOUT_P95_PX or max(holdout_views) > MAX_HOLDOUT_VIEW_RMS_PX
        ):
            raise ValueError("별도 검증 오차가 커요. 새로운 위치·기울기의 사진을 더 모아 주세요.")
        subset = [i for position, i in enumerate(training) if position % 4 != 3]
        _, check_matrix, check_distortion, _, _ = cv2.aruco.calibrateCameraCharuco(
            [self.corners[i] for i in subset], [self.ids[i] for i in subset],
            BOARD, self.size, None, None, flags=cv2.CALIB_FIX_K3,
        )
        check_lens_model(check_matrix, check_distortion, self.size)
        focal_change = float((abs(np.diag(check_matrix)[:2] / np.diag(matrix)[:2] - 1) * 100).max())
        principal_change = float(np.linalg.norm(check_matrix[:2, 2] - matrix[:2, 2]))
        grid = np.array(
            [[x, y] for y in np.linspace(0, self.size[1] - 1, 7)
             for x in np.linspace(0, self.size[0] - 1, 7)], dtype=np.float64,
        )
        rays = cv2.undistortPoints(grid.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
        mapped = cv2.projectPoints(
            np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3),
            check_matrix, check_distortion,
        )[0].reshape(-1, 2)
        mapping_change = float(np.percentile(np.linalg.norm(mapped - grid, axis=1), 95))
        if (
            focal_change > MAX_FOCAL_CHANGE_PERCENT
            or principal_change > np.linalg.norm(self.size) * 0.02
            or mapping_change > MAX_MAPPING_CHANGE_P95_PX
        ):
            raise ValueError(
                "사진 일부를 빼면 보정값이 크게 바뀌어요. 다른 위치·기울기의 사진이 더 필요해요."
            )
        quality = {
            "version": QUALITY_VERSION,
            "intrinsics": {
                "passed": True, "training_views": len(training),
                "validation_views": len(validation),
                "training_rms_px": float(rms), "training_max_view_rms_px": max(train_errors),
                "holdout_rms_px": holdout_rms, "holdout_p95_px": holdout_p95,
                "holdout_max_view_rms_px": max(holdout_views), "coverage_cells": coverage_cells,
                "coverage_hull_fraction": coverage_hull, "tilted_views": tilted,
                "normal_spread_deg": spread, "focal_change_percent": focal_change,
                "principal_change_px": principal_change, "mapping_change_p95_px": mapping_change,
                "distortion_roundtrip_max_px": roundtrip,
            },
        }
        return CameraCalibration(
            self.profile, self.size, matrix, distortion, float(rms), quality=quality
        )

    def solve_extrinsics(self, image, calibration):
        corners, ids = self.detect(image)
        points = BOARD.getChessboardCorners()[ids.ravel()]
        success, rvec, tvec = cv2.solvePnP(
            points,
            corners,
            calibration.matrix,
            calibration.distortion,
        )
        if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError("기준판의 위치를 구하지 못했어요.")
        projected = cv2.projectPoints(
            points,
            rvec,
            tvec,
            calibration.matrix,
            calibration.distortion,
        )[0]
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
        depths = (cv2.Rodrigues(rvec)[0] @ points.T + tvec)[2]
        if not np.isfinite(error) or error > MAX_POSE_RMS_PX or (depths <= 0).any():
            raise ValueError("기준판 위치 오차가 커요. 더 선명하게 촬영해 주세요.")
        calibration.rvec, calibration.tvec = rvec, tvec
        return error

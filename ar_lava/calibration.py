import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The same physical board stays still for all three extrinsic captures.
SQUARE_M = 0.04
MARKER_M = 0.03
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), SQUARE_M, MARKER_M, DICTIONARY)


@dataclass
class CameraCalibration:
    profile: str
    size: tuple[int, int]
    matrix: np.ndarray
    distortion: np.ndarray
    rms: float
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None

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
        result = {"profile": self.profile, "size": self.size, "rms": self.rms}
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


class Calibrator:
    def __init__(self):
        self.corners = []
        self.ids = []
        self.size = None
        self.profile = None
        self.detector = cv2.aruco.CharucoDetector(BOARD)

    def detect(self, image):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 8:
            raise ValueError("기준판 내부 교차점이 8개 이상 보여야 해요.")
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
        if len(self.corners) < 12:
            raise ValueError("다른 위치와 각도의 보정 사진이 12장 이상 필요해요.")
        rms, matrix, distortion, _, _ = cv2.aruco.calibrateCameraCharuco(
            self.corners,
            self.ids,
            BOARD,
            self.size,
            None,
            None,
        )
        if not np.isfinite(rms) or rms > 1.5 or not np.isfinite(matrix).all():
            raise ValueError("보정 오차가 1.5px를 넘어요. 사진을 다시 촬영해 주세요.")
        return CameraCalibration(self.profile, self.size, matrix, distortion, float(rms))

    def solve_extrinsics(self, image, calibration):
        corners, ids = self.detect(image)
        points = BOARD.getChessboardCorners()[ids.ravel()]
        success, rvec, tvec = cv2.solvePnP(
            points,
            corners,
            calibration.matrix,
            calibration.distortion,
        )
        if not success:
            raise ValueError("기준판의 위치를 구하지 못했어요.")
        projected = cv2.projectPoints(
            points,
            rvec,
            tvec,
            calibration.matrix,
            calibration.distortion,
        )[0]
        error = float(np.sqrt(np.mean((projected - corners) ** 2)))
        if error > 1.5 or tvec[2, 0] <= 0:
            raise ValueError("기준판 위치 오차가 커요. 더 선명하게 촬영해 주세요.")
        calibration.rvec, calibration.tvec = rvec, tvec
        return error

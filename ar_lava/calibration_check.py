"""Validate frozen camera poses on later synchronized observations, never refit here."""

from collections import deque

import cv2
import numpy as np

from ar_lava.calibration import BOARD
from ar_lava.stereo import StereoRectifier

PAIR_SAMPLES = 6
PAIR_SPAN_MS = 1000
QUALITY_LEASE_MS = 10_000
SOURCE_RMS_PX = 0.75
SOURCE_P95_PX = 1.5
EPIPOLAR_P95_PX = 1.0


def pose_errors(calibration, observation):
    world = BOARD.getChessboardCorners()[observation["ids"].ravel()]
    actual = observation["corners"].reshape(-1, 2)
    errors = np.linalg.norm(calibration.project(world) - actual, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("기준판 위치 오차를 계산하지 못했어요. 바닥 위치를 다시 보정해 주세요.")
    return float(np.sqrt(np.mean(errors**2))), float(np.percentile(errors, 95))


def validate_pair(rectifier, observations):
    errors = [
        pose_errors(camera, view)
        for camera, view in zip((rectifier.left, rectifier.right), observations, strict=True)
    ]
    if any(rms > SOURCE_RMS_PX or p95 > SOURCE_P95_PX for rms, p95 in errors):
        raise ValueError(
            "저장된 바닥 위치와 현재 기준판이 맞지 않아요. 두 폰을 고정하고 다시 보정해 주세요."
        )
    points = [
        dict(zip(view["ids"].ravel().tolist(), view["corners"].reshape(-1, 2)))
        for view in observations
    ]
    common = sorted(points[0].keys() & points[1].keys())
    world = BOARD.getChessboardCorners()[common]
    if len(common) < 12 or np.linalg.matrix_rank(world[:, :2] - world[:, :2].mean(0)) < 2:
        raise ValueError("두 카메라에서 같은 기준점 12개 이상이 넓게 보여야 해요.")
    left, right = [
        np.array([rectifier.rectified_point(side[i], index) for i in common])
        for index, side in enumerate(points)
    ]
    epipolar = float(np.percentile(np.abs(left[:, 1] - right[:, 1]), 95))
    if epipolar > EPIPOLAR_P95_PX or not np.all(left[:, 0] - right[:, 0] > 0.5):
        raise ValueError(
            "좌우 기준점이 같은 높이에 맞지 않아요. 바닥 보정과 폰의 좌우 배치를 확인해 주세요."
        )
    homogeneous = cv2.triangulatePoints(rectifier.p1, rectifier.p2, left.T, right.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        xyz = (homogeneous[:3] / homogeneous[3]).T
    if not np.isfinite(xyz).all() or not np.all(xyz[:, 2] > 0):
        raise ValueError("두 영상으로 기준판의 양수 깊이를 복원하지 못했어요.")
    a, b = np.triu_indices(len(common), 1)
    known = np.linalg.norm(world[a] - world[b], axis=1)
    reconstructed = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    relative = np.abs(reconstructed - known) / known
    median, p95 = float(np.median(relative)), float(np.percentile(relative, 95))
    if median > 0.03 or p95 > 0.05:
        raise ValueError(
            "좌우 영상으로 복원한 기준판 크기 오차가 커요. 같은 고정판으로 다시 보정해 주세요."
        )
    x, y = rectifier.original_maps
    width, height = rectifier.size
    visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if visible.mean() < 0.5 or not visible[visible.shape[0] // 2, visible.shape[1] // 2]:
        raise ValueError(
            "깊이 계산 시야가 좁거나 중앙이 잘려요. 두 렌즈의 높이·앞뒤 위치를 맞춰 주세요."
        )
    return {
        "source_rms_px": round(max(error[0] for error in errors), 3),
        "source_p95_px": round(max(error[1] for error in errors), 3),
        "epipolar_p95_px": round(epipolar, 3),
        "board_scale_median_percent": round(median * 100, 2),
        "board_scale_p95_percent": round(p95 * 100, 2),
        "reference_coverage_percent": round(float(visible.mean()) * 100, 1),
        "common_points": len(common),
    }


class CalibrationPairCheck:
    def __init__(self):
        self.signature = None
        self.rectifier = None
        self.buffers = [deque(maxlen=12), deque(maxlen=12)]
        self.seen = [None, None]
        self.drift = [0, 0]
        self.used = [set(), set()]
        self.samples = 0
        self.first_ms = None
        self.validated_ms = None
        self.last_sample_ms = None
        self.metrics = {}

    def reset(self):
        self.__init__()

    def publish(self, cameras, now, message, monitoring=None):
        if monitoring is None:
            monitoring = (
                "observed"
                if all(
                    c.guide.board_observation is not None
                    and 0 <= now - c.guide.board_observation["captured_ms"] <= 1500
                    for c in cameras
                )
                else "unobserved"
            )
        passed = self.validated_ms is not None and 0 <= now - self.validated_ms <= QUALITY_LEASE_MS
        for camera in cameras:
            camera.pair_quality = {
                "passed": passed,
                "samples": min(self.samples, PAIR_SAMPLES),
                "required": PAIR_SAMPLES,
                "generation": camera.generation,
                "validated_ms": self.validated_ms,
                "expires_ms": self.validated_ms + QUALITY_LEASE_MS if passed else None,
                "monitoring": monitoring,
                "message": message,
                **self.metrics,
            }

    def update(self, cameras, now):
        cameras = [cameras[name] for name in ("cam1", "cam2")]
        signature = tuple((c.generation, id(c.calibration), c.profile) for c in cameras)
        if signature != self.signature:
            self.reset()
            self.signature = signature
        if not all(
            c.connected
            and c.calibration
            and c.calibration.lens_validated
            and c.calibration.ready
            and c.floor_validated
            for c in cameras
        ):
            self.reset()
            self.publish(cameras, now, "두 카메라의 렌즈·바닥 검증을 먼저 완료해 주세요.")
            return
        source_failure = False
        for index, camera in enumerate(cameras):
            view = camera.guide.board_observation
            if (
                view is None
                or view["captured_ms"] == self.seen[index]
                or not 0 <= now - view["captured_ms"] <= 350
                or view["profile"] != camera.profile
                or tuple(view["size"]) != camera.calibration.size
            ):
                continue
            self.seen[index] = view["captured_ms"]
            try:
                rms, p95 = pose_errors(camera.calibration, view)
            except (ValueError, cv2.error):
                rms = p95 = float("inf")
            source_failure |= rms > SOURCE_RMS_PX or p95 > SOURCE_P95_PX
            self.drift[index] = self.drift[index] + 1 if rms > 2 else 0
            if self.drift[index] >= 3:
                camera.floor_validated = False
                camera.calibration.rvec = camera.calibration.tvec = None
                camera.generation += 1
                camera.background = camera.depth_composite = camera.landmarks = None
                camera.mode = "calibration"
                self.reset()
                self.publish(
                    cameras,
                    now,
                    "고정판과 카메라 위치가 달라졌어요. 바닥 위치를 다시 보정해 주세요.",
                )
                return
            self.buffers[index].append(view)
        if source_failure:
            self.samples, self.first_ms, self.validated_ms = 0, None, None
            self.metrics = {}
            for buffer in self.buffers:
                buffer.clear()
            self.publish(
                cameras, now, "현재 기준판이 보정 위치와 맞지 않아 깊이를 중단했어요.", "failed"
            )
            return
        candidates = [
            (abs(a["captured_ms"] - b["captured_ms"]), a, b)
            for a in self.buffers[0]
            for b in self.buffers[1]
            if a["captured_ms"] not in self.used[0]
            and b["captured_ms"] not in self.used[1]
            and 0 <= now - a["captured_ms"] <= 350
            and 0 <= now - b["captured_ms"] <= 350
        ]
        if (
            not candidates
            or min(item[0] for item in candidates) > 40
            or any(c.clock.rtt_ms > 100 for c in cameras)
        ):
            if self.validated_ms is not None and now - self.validated_ms > QUALITY_LEASE_MS:
                self.samples, self.first_ms, self.validated_ms = 0, None, None
                self.metrics = {}
            self.publish(
                cameras, now, "동시 기준판 검증 대기 · 기준판이 두 화면에 보이도록 유지해 주세요."
            )
            return
        _, left, right = min(candidates, key=lambda item: item[0])
        self.used[0].add(left["captured_ms"])
        self.used[1].add(right["captured_ms"])
        # Buffers retain only recent observations; old timestamps cannot be selected again.
        self.used = [{stamp for stamp in values if now - stamp <= 1500} for values in self.used]
        try:
            if self.rectifier is None:
                self.rectifier = StereoRectifier(cameras[0].calibration, cameras[1].calibration)
            self.metrics = validate_pair(self.rectifier, (left, right))
        except (ValueError, cv2.error) as error:
            self.samples, self.first_ms, self.validated_ms = 0, None, None
            self.metrics = {}
            self.publish(cameras, now, str(error), "failed")
            return
        stamp = min(left["captured_ms"], right["captured_ms"])
        if self.last_sample_ms is not None and stamp - self.last_sample_ms > 1500:
            self.samples, self.first_ms, self.validated_ms = 0, None, None
        self.last_sample_ms = stamp
        if self.first_ms is None:
            self.first_ms = stamp
        self.samples += 1
        if self.samples >= PAIR_SAMPLES and stamp - self.first_ms >= PAIR_SPAN_MS:
            self.validated_ms = stamp
        self.publish(
            cameras,
            now,
            f"동시 영상 {min(self.samples, PAIR_SAMPLES)}/{PAIR_SAMPLES}쌍 검증",
            "observed",
        )

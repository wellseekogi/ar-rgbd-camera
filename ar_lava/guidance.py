"""Capture guidance whose completion follows measured calibration quality."""

import copy

import cv2
import numpy as np

from ar_lava.calibration import BOARD, MIN_INTRINSIC_VIEWS

FLOOR_OBSERVATIONS = 8
FLOOR_SPAN_MS = 2000
FLOOR_RMS_PX = 0.75
FLOOR_P95_PX = 1.5

COVERAGE = (
    ("center", "가운데", "기준판이 화면 가운데에 오도록 움직여 주세요."),
    ("left", "왼쪽", "기준판이 화면 왼쪽에 오도록 카메라를 움직여 주세요."),
    ("right", "오른쪽", "기준판이 화면 오른쪽에 오도록 카메라를 움직여 주세요."),
    ("top", "위쪽", "기준판이 화면 위쪽에 오도록 카메라를 움직여 주세요."),
    ("bottom", "아래쪽", "기준판이 화면 아래쪽에 오도록 카메라를 움직여 주세요."),
    ("near", "가까이", "기준판 전체가 보이도록 조금 가까이 가 주세요."),
    ("far", "멀리", "기준판 전체가 보이도록 조금 멀리 가 주세요."),
    ("tilt", "앞뒤 기울기", "기준판의 위아래·좌우 면이 기울어 보이도록 촬영해 주세요."),
)


class CalibrationGuide:
    def __init__(self):
        self.enabled = True
        self.coverage = set()
        self.visible = False
        self.sharp = False
        self.stable = 0
        self.previous = None
        self.outline = []
        self.last_ms = -1000
        self.floor_armed = False
        self.floor_candidate = None
        self.floor_samples = 0
        self.floor_start_ms = None
        self.floor_last_ms = None
        self.floor_rms = None
        self.floor_p95 = None
        self.board_observation = None
        self.attempted_samples = 0
        self.message = "출력한 기준판 전체를 카메라에 보여 주세요."

    def arm_floor(self, camera):
        self.floor_armed = True
        self._reset_floor()
        self.stable, self.previous = 0, None
        camera.floor_validated = False
        camera.pair_quality = {"passed": False, "samples": 0, "required": 6}
        camera.background = None
        camera.calibration.rvec = camera.calibration.tvec = None
        self.message = "폰과 기준판을 고정해 주세요. 여러 새 영상에서 바닥 위치를 검증해요."

    def _reset_floor(self):
        self.floor_candidate = None
        self.floor_samples = 0
        self.floor_start_ms = self.floor_last_ms = None
        self.floor_rms = self.floor_p95 = None

    def snapshot(self, camera, current_ms):
        live = (
            camera.connected and camera.frame is not None
            and 0 <= current_ms - camera.capture_ms < 1500
        )
        calibration = camera.calibration
        intrinsics = (
            calibration is not None and calibration.lens_validated
            and calibration.profile == camera.profile and camera.frame is not None
            and calibration.size == (camera.frame.shape[1], camera.frame.shape[0])
        )
        floor = intrinsics and calibration.ready and camera.floor_validated
        pair = camera.pair_quality
        paired = (
            floor and pair.get("passed", False)
            and pair.get("generation") == camera.generation
            and pair.get("expires_ms") is not None and current_ms <= pair["expires_ms"]
        )
        observed = pair.get("monitoring", "observed") == "observed"
        background = paired and camera.background is not None
        ready = live and background and camera.mode == "play" and observed
        stage, message = "collect", self.message
        action, label = None, None
        if intrinsics:
            stage = "place"
            message = (
                "같은 기준판을 바닥에 두고 두 폰을 최종 위치에 고정해 주세요. "
                "판은 움직이지 마세요."
            )
            action, label = "guide-floor", "카메라 고정 완료 · 바닥 맞추기"
            if self.floor_armed:
                stage, message = "floor", self.message
                action, label = None, None
        if floor:
            stage, message = "verify", pair.get("message", "두 카메라의 좌우 정렬을 검증해요.")
            action, label = None, None
        if paired and observed:
            stage = "background"
            message = "좌우 정렬 검증을 통과했어요. 판과 폰을 고정하고 빈 배경을 저장해 주세요."
            action, label = "guide-background", "사람 없는 빈 배경 저장 · 깊이 시작"
        if ready:
            stage, message = "ready", "검증을 통과했어요. 폰과 기준판을 고정한 채 유지해 주세요."
            action, label = None, None
        if paired and not observed:
            stage = "verify"
            message = "기준판이 가려져 고정 상태 확인을 기다려요. 기준판을 보여 주세요."
            action, label = None, None
        if not self.enabled:
            stage, message = "manual", "수동 보정에도 같은 품질 검사가 적용돼요."
            action, label = "guide-resume", "자동 안내 켜기"
        elif not intrinsics and len(camera.calibrator.corners) >= 40:
            message = (
                "40장을 검사했지만 품질 기준을 통과하지 못했어요. "
                "촬영 조건을 바꾸고 다시 시작해 주세요."
            )
            action, label = "guide-restart", "사진 초기화 · 다시 보정"
        if not live:
            message = "카메라의 새 영상을 기다리고 있어요. 연결과 카메라 권한을 확인해 주세요."
        checks = [
            ("video", "새 영상 수신", live),
            ("board", "기준판 인식 · 선명도", live and self.visible and self.sharp),
            (
                "samples",
                f"유효 사진 {len(camera.calibrator.corners)}/{MIN_INTRINSIC_VIEWS}장 이상",
                intrinsics or len(camera.calibrator.corners) >= MIN_INTRINSIC_VIEWS,
            ),
            ("intrinsics", "별도 영상 · 렌즈 품질 검사", intrinsics),
            ("floor", "8개 새 영상 · 바닥 고정 검증", floor),
            ("stereo", "좌우 정렬 · 기준판 거리 검증", paired and observed and live),
            ("background", "빈 배경 저장 · 깊이 시작", ready),
        ]
        metrics = calibration.quality.get("intrinsics", {}) if calibration else {}
        quality = [
            {
                "label": "별도 영상 오차",
                "value": (
                    f"{metrics['holdout_rms_px']:.2f}px"
                    if "holdout_rms_px" in metrics else "검사 전"
                ),
                "passed": intrinsics,
            },
            {
                "label": "바닥 고정 검사", "value": f"{self.floor_samples}/{FLOOR_OBSERVATIONS}개",
                "passed": floor,
            },
            {
                "label": "좌우 동시 영상",
                "value": f"{pair.get('samples', 0)}/{pair.get('required', 6)}쌍",
                "passed": paired and observed,
            },
        ]
        floor_metrics = calibration.quality.get("floor", {}) if calibration else {}
        floor_rms = (
            self.floor_rms if self.floor_rms is not None else floor_metrics.get("max_rms_px")
        )
        floor_p95 = (
            self.floor_p95 if self.floor_p95 is not None else floor_metrics.get("max_p95_px")
        )
        quality.append({
            "label": "바닥 RMS / 상위 95% · 기준 0.75 / 1.5px",
            "value": (
                f"{floor_rms:.2f} / {floor_p95:.2f}px"
                if floor_rms is not None and floor_p95 is not None else "검사 전"
            ),
            "passed": floor,
        })
        for key, label, unit in (
            ("source_rms_px", "기준판 위치 오차 · 기준 0.75px", "px"),
            ("epipolar_p95_px", "좌우 높이 차이 상위 95% · 기준 1px", "px"),
            ("board_scale_p95_percent", "기준판 크기 오차 상위 95% · 기준 5%", "%"),
        ):
            value = pair.get(key)
            quality.append({
                "label": label,
                "value": f"{value:.2f}{unit}" if value is not None else "검사 전",
                "passed": paired and observed,
            })
        milestones = (intrinsics, floor, paired and observed and live, ready)
        return {
            "enabled": self.enabled, "stage": stage, "message": message,
            "checks": [
                {"id": name, "label": text, "done": bool(done)} for name, text, done in checks
            ],
            "quality": quality,
            "coverage": [
                {"id": name, "label": text, "done": name in self.coverage}
                for name, text, _ in COVERAGE
            ],
            "progress": sum(bool(done) for done in milestones) / 4,
            "outline": self.outline if live and self.visible else [],
            "stable": live and self.stable >= 2,
            "action": action, "action_label": label,
            "can_act": live and bool(action) and (
                intrinsics or action in ("guide-resume", "guide-restart")
            ),
        }

    def observe(self, camera, frame, current_ms):
        if current_ms - self.last_ms < 350:
            return
        self.last_ms = current_ms
        self.board_observation = None
        try:
            corners, ids = camera.calibrator.detect(frame)
        except ValueError:
            self.visible = self.sharp = False
            self.previous, self.stable, self.outline = None, 0, []
            if self.floor_armed:
                self._reset_floor()
            self.message = "기준판 전체가 보이도록 움직여 주세요. 유효 교차점이 12개 이상 필요해요."
            return
        height, width = frame.shape[:2]
        homography, _ = cv2.findHomography(BOARD.getChessboardCorners()[ids.ravel(), :2], corners)
        if homography is None or not np.isfinite(homography).all():
            self.visible = self.sharp = False
            self.previous, self.stable, self.outline = None, 0, []
            if self.floor_armed:
                self._reset_floor()
            return
        board_width, board_height = BOARD.getRightBottomCorner()[:2]
        board = np.array(
            [[[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]]],
            np.float32,
        )
        outline = cv2.perspectiveTransform(board, homography)[0] / [width, height]
        self.outline = outline.tolist()
        self.visible = bool(
            np.isfinite(outline).all() and (outline >= 0.01).all() and (outline <= 0.99).all()
        )
        if not self.visible:
            self.stable, self.previous = 0, None
            self.sharp = False
            if self.floor_armed:
                self._reset_floor()
            self.message = "기준판 가장자리가 잘리지 않도록 조금 물러나 주세요."
            return
        low = np.maximum(0, corners.min(axis=0).ravel().astype(int))
        high = np.minimum([width, height], corners.max(axis=0).ravel().astype(int))
        crop = frame[low[1]:high[1], low[0]:high[0]]
        self.sharp = crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() >= 50
        descriptor = outline.ravel()
        motion = np.max(np.abs(descriptor - self.previous)) if self.previous is not None else 1
        self.previous = descriptor
        self.stable = self.stable + 1 if motion < 0.018 and self.sharp else 0
        if not self.sharp:
            if self.floor_armed:
                self._reset_floor()
            self.message = "화면이 흐려요. 렌즈를 닦고 밝은 곳에서 잠시 멈춰 주세요."
            return
        stamp = getattr(camera, "observation_capture_ms", None)
        stamp = camera.capture_ms if stamp is None else stamp
        self.board_observation = {
            "corners": corners.copy(), "ids": ids.copy(), "captured_ms": stamp,
            "profile": camera.profile, "size": (width, height),
        }
        if self.floor_armed:
            if self.stable < 2:
                self._reset_floor()
                self.message = "폰과 판을 움직이지 말고 잠시 기다려 주세요."
            else:
                self._observe_floor(camera, frame, corners, ids)
            return
        if not self.enabled or camera.mode != "calibration":
            return
        if camera.calibration is not None and camera.calibration.lens_validated:
            return
        if self.stable < 2:
            self.message = "잠시 멈추면 선명하고 서로 다른 사진만 저장해요."
            return
        if len(camera.calibrator.corners) >= 40:
            return
        try:
            camera.calibrator.add(frame, camera.profile)
        except ValueError as error:
            self.message = str(error)
            return
        self._update_coverage(outline, corners, ids, width, height)
        count = len(camera.calibrator.corners)
        if count >= MIN_INTRINSIC_VIEWS and count > self.attempted_samples:
            self.attempted_samples = count
            try:
                calibration = camera.calibrator.solve_intrinsics()
                calibration.save(camera.path)
                camera.calibration = calibration
                camera.floor_validated = False
                camera.generation += 1
                camera.error = ""
            except (ValueError, cv2.error) as error:
                self.message = f"{error} 다른 각도의 사진을 더 모아 주세요."
                return
        self.message = self.next_capture_message(camera)

    def _observe_floor(self, camera, frame, corners, ids):
        stamp = self.board_observation["captured_ms"]
        if self.floor_last_ms is not None and stamp <= self.floor_last_ms:
            return
        if self.floor_last_ms is not None and stamp - self.floor_last_ms > 1500:
            self._reset_floor()
        try:
            if self.floor_candidate is None:
                self.floor_candidate = copy.copy(camera.calibration)
                self.floor_candidate.quality = dict(camera.calibration.quality)
                camera.calibrator.solve_extrinsics(frame, self.floor_candidate)
                self.floor_start_ms = stamp
            points = BOARD.getChessboardCorners()[ids.ravel()]
            errors = np.linalg.norm(self.floor_candidate.project(points) - corners[:, 0], axis=1)
            rms = float(np.sqrt(np.mean(errors ** 2)))
            p95 = float(np.percentile(errors, 95))
            if not np.isfinite(errors).all() or rms > FLOOR_RMS_PX or p95 > FLOOR_P95_PX:
                raise ValueError("바닥 위치가 흔들리거나 오차가 커요. 폰과 판을 고정해 주세요.")
        except (ValueError, cv2.error) as error:
            self._reset_floor()
            self.message = str(error)
            return
        self.floor_samples += 1
        self.floor_last_ms = stamp
        self.floor_rms = max(self.floor_rms or 0, rms)
        self.floor_p95 = max(self.floor_p95 or 0, p95)
        self.message = (
            f"바닥 고정 검사 {self.floor_samples}/{FLOOR_OBSERVATIONS}개 · 오차 {rms:.2f}px"
        )
        if self.floor_samples < FLOOR_OBSERVATIONS or stamp - self.floor_start_ms < FLOOR_SPAN_MS:
            return
        self.floor_candidate.quality["floor"] = {
            "passed": True, "observations": self.floor_samples,
            "span_ms": stamp - self.floor_start_ms,
            "max_rms_px": self.floor_rms, "max_p95_px": self.floor_p95,
        }
        self.floor_candidate.save(camera.path)
        camera.calibration = self.floor_candidate
        camera.floor_validated = True
        camera.generation += 1
        self.floor_armed = False
        self.message = "바닥 고정 검사를 통과했어요. 두 카메라의 좌우 정렬을 확인해요."

    def _update_coverage(self, outline, corners, ids, width, height):
        x, y = outline.mean(axis=0)
        area = abs(cv2.contourArea(outline.astype(np.float32)))
        approximate = np.array([
            [max(width, height), 0, width / 2], [0, max(width, height), height / 2], [0, 0, 1.0]
        ])
        success, rvec, _ = cv2.solvePnP(
            BOARD.getChessboardCorners()[ids.ravel()], corners, approximate, np.zeros(5)
        )
        tilt = 0.0
        if success:
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            tilt = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
        for name, passed in (
            ("center", 0.38 <= x <= 0.62 and 0.38 <= y <= 0.62),
            ("left", x < 0.38), ("right", x > 0.62), ("top", y < 0.38), ("bottom", y > 0.62),
            ("near", area >= 0.18), ("far", area <= 0.1), ("tilt", tilt >= 15),
        ):
            if passed:
                self.coverage.add(name)

    def next_capture_message(self, camera):
        for name, _, instruction in COVERAGE:
            if name not in self.coverage:
                return instruction
        return f"검증용 사진을 포함해 서로 다른 유효 사진 {MIN_INTRINSIC_VIEWS}장 이상이 필요해요."

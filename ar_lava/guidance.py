"""Live ChArUco capture guidance. Final camera placement needs a human confirmation."""

import cv2
import numpy as np

from ar_lava.calibration import BOARD

COVERAGE = (
    ("center", "가운데", "기준판이 화면 가운데에 오도록 움직여 주세요."),
    ("left", "왼쪽", "기준판이 화면 왼쪽에 오도록 카메라를 움직여 주세요."),
    ("right", "오른쪽", "기준판이 화면 오른쪽에 오도록 카메라를 움직여 주세요."),
    ("top", "위쪽", "기준판이 화면 위쪽에 오도록 카메라를 움직여 주세요."),
    ("bottom", "아래쪽", "기준판이 화면 아래쪽에 오도록 카메라를 움직여 주세요."),
    ("near", "가까이", "기준판 전체가 보이도록 조금 가까이 가 주세요."),
    ("far", "멀리", "기준판 전체가 보이도록 조금 멀리 가 주세요."),
    ("tilt", "기울이기", "기준판을 보면서 카메라를 15도 이상 살짝 돌려 주세요."),
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
        self.attempted_samples = 0
        self.message = "출력한 기준판 전체를 카메라에 보여 주세요."

    def snapshot(self, camera, current_ms):
        live = (
            camera.connected
            and camera.frame is not None
            and 0 <= current_ms - camera.capture_ms < 1500
        )
        intrinsics = (
            camera.calibration is not None
            and camera.calibration.profile == camera.profile
            and camera.frame is not None
            and camera.calibration.size == (camera.frame.shape[1], camera.frame.shape[0])
        )
        floor = intrinsics and camera.calibration.ready
        background = floor and camera.background is not None
        ready = background and camera.mode == "play"
        stage = "collect"
        action, label = None, None
        message = self.message
        if intrinsics:
            stage = "place"
            message = (
                "같은 기준판을 바닥에 두고 모든 카메라를 최종 위치에 고정해 주세요. "
                "판은 움직이지 마세요."
            )
            action, label = "guide-floor", "카메라 고정 완료 · 바닥 맞추기"
            if self.floor_armed:
                stage, message = "floor", self.message
                action, label = None, None
        if floor:
            stage = "background"
            message = (
                "모든 카메라의 바닥 보정이 끝날 때까지 판을 그대로 두세요. "
                "이후 사람이 화면에서 나간 뒤 빈 배경을 저장해 주세요."
            )
            action, label = "guide-background", "사람 없는 빈 배경 저장 · 깊이 시작"
        if ready:
            stage = "ready"
            message = (
                "보정 완료. 카메라를 고정한 채 유지해 주세요. "
                "카메라 1·2가 준비되면 깊이맵이 나와요."
            )
            action, label = None, None
        if not self.enabled:
            stage, message = "manual", "수동 보정 중이에요. 자동 안내를 다시 켤 수 있어요."
            action, label = "guide-resume", "자동 안내 켜기"
        elif not intrinsics and len(camera.calibrator.corners) >= 40:
            message = (
                "사진이 40장 모였지만 보정하지 못했어요. "
                "기준판 크기·조명을 확인하고 다시 시작해 주세요."
            )
            action, label = "guide-restart", "사진 초기화 · 다시 보정"
        if not live:
            message = "카메라의 새 영상을 기다리고 있어요. 연결과 카메라 권한을 확인해 주세요."
        checks = [
            ("video", "새 영상 수신", live),
            (
                "board",
                "기준판 인식 · 선명도",
                floor
                or (live and current_ms - self.last_ms < 1500 and self.visible and self.sharp),
            ),
            (
                "samples",
                "저장된 렌즈 보정 재사용"
                if intrinsics and not camera.calibrator.corners
                else f"다양한 보정 사진 {len(camera.calibrator.corners)}/12장 이상",
                intrinsics
                or (len(camera.calibrator.corners) >= 12 and len(self.coverage) == len(COVERAGE)),
            ),
            ("intrinsics", "렌즈 보정 · 오차 1.5px 이하", intrinsics),
            ("floor", "고정된 카메라의 바닥 위치 보정", floor),
            ("background", "빈 배경 저장 · 추적 시작", ready),
        ]
        return {
            "enabled": self.enabled,
            "stage": stage,
            "message": message,
            "checks": [
                {"id": name, "label": text, "done": bool(done)} for name, text, done in checks
            ],
            "coverage": [
                {"id": name, "label": text, "done": name in self.coverage}
                for name, text, _ in COVERAGE
            ],
            "progress": round(
                (
                    1
                    if intrinsics
                    else min(len(camera.calibrator.corners) / 12, 1)
                    * len(self.coverage)
                    / len(COVERAGE)
                )
                / 4
                + sum(bool(done) for _, _, done in checks[3:]) / 4,
                2,
            ),
            "outline": self.outline if live and self.visible else [],
            "stable": live and self.stable >= 2,
            "action": action,
            "action_label": label,
            "can_act": live and (intrinsics or action in ("guide-resume", "guide-restart")),
        }

    def observe(self, camera, frame, current_ms):
        if not self.enabled or camera.mode != "calibration" or current_ms - self.last_ms < 350:
            return
        self.last_ms = current_ms
        if camera.calibration and not self.floor_armed:
            return
        if not camera.calibration and len(camera.calibrator.corners) >= 40:
            return
        try:
            corners, ids = camera.calibrator.detect(frame)
        except ValueError:
            self.visible, self.sharp = False, False
            self.previous, self.stable, self.outline = None, 0, []
            self.message = "기준판 전체가 보이도록 움직여 주세요. 교차점이 8개 이상 필요해요."
            return
        height, width = frame.shape[:2]
        homography, _ = cv2.findHomography(BOARD.getChessboardCorners()[ids.ravel(), :2], corners)
        if homography is None or not np.isfinite(homography).all():
            self.visible, self.sharp = False, False
            self.previous, self.stable, self.outline = None, 0, []
            self.message = "기준판 전체를 더 선명하게 보여 주세요."
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
            self.message = "기준판 가장자리가 잘리지 않도록 조금 물러나 주세요."
            return
        low = np.maximum(0, corners.min(axis=0).ravel().astype(int))
        high = np.minimum([width, height], corners.max(axis=0).ravel().astype(int))
        crop = frame[low[1] : high[1], low[0] : high[0]]
        self.sharp = crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() >= 50
        descriptor = outline.ravel()
        motion = np.max(np.abs(descriptor - self.previous)) if self.previous is not None else 1
        self.previous = descriptor
        self.stable = self.stable + 1 if motion < 0.018 and self.sharp else 0
        if not self.sharp:
            self.message = "화면이 흐려요. 렌즈를 닦고 밝은 곳에서 잠시 멈춰 주세요."
            return
        if self.stable < 2:
            self.message = "좋아요. 잠시 멈추면 자동으로 확인해요."
            return
        if self.floor_armed:
            try:
                camera.calibrator.solve_extrinsics(frame, camera.calibration)
                camera.calibration.save(camera.path)
                camera.generation += 1
                self.floor_armed = False
            except (ValueError, cv2.error) as error:
                self.message = str(error)
            return
        try:
            camera.calibrator.add(frame, camera.profile)
        except ValueError:
            self.message = self.next_capture_message(camera)
            return
        x, y = outline.mean(axis=0)
        area = abs(cv2.contourArea(outline.astype(np.float32)))
        angle = abs(np.degrees(np.arctan2(*(outline[1] - outline[0])[::-1])))
        angle = min(angle, 180 - angle)
        for name, passed in (
            ("center", 0.38 <= x <= 0.62 and 0.38 <= y <= 0.62),
            ("left", x < 0.38),
            ("right", x > 0.62),
            ("top", y < 0.38),
            ("bottom", y > 0.62),
            ("near", area >= 0.18),
            ("far", area <= 0.1),
            ("tilt", angle >= 15),
        ):
            if passed:
                self.coverage.add(name)
        count = len(camera.calibrator.corners)
        if count >= 12 and len(self.coverage) == len(COVERAGE) and count > self.attempted_samples:
            self.attempted_samples = count
            try:
                calibration = camera.calibrator.solve_intrinsics()
                calibration.save(camera.path)
                camera.calibration = calibration
                camera.generation += 1
                camera.error = ""
            except (ValueError, cv2.error) as error:
                self.message = f"{error} 다른 각도의 사진을 더 모아 주세요."
                return
        self.message = self.next_capture_message(camera)

    def next_capture_message(self, camera):
        for name, _, instruction in COVERAGE:
            if name not in self.coverage:
                return instruction
        return (
            "여러 거리와 각도로 천천히 움직여 주세요. 서로 다른 사진 12장 이상을 자동으로 모아요."
        )

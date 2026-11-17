"""Rectification and dense stereo depth. Camera 1 is left, camera 2 is right."""

import asyncio
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from ar_lava.protocol import pack
from ar_lava.render import floor_depth_map, render_with_depth

MODEL_SHA256 = "2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3"
MAX_PAIR_SKEW_MS = 40
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 15.0
LOG = logging.getLogger("ar_lava")


class CpuStereo:
    name = "OpenCV StereoSGBM"
    device = "cpu"
    backend = "OpenCV StereoSGBM / CPU"

    def __init__(self):
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=5,
            P1=8 * 25,
            P2=32 * 25,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def disparity(self, left, right):
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16
        disparity[disparity <= 0] = np.nan
        return disparity


def create_depth_model(model_path, device=None):
    device = device or os.environ.get("LAVA_DEPTH_DEVICE", "auto")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("깊이 장치는 auto, cpu, cuda 중 하나여야 해요.")
    if device == "cpu":
        return CpuStereo()
    try:
        return FastFoundationStereo(model_path)
    except Exception as error:
        if device == "cuda":
            raise
        LOG.info("CUDA를 사용할 수 없어 CPU StereoSGBM으로 전환합니다: %s", error)
        return CpuStereo()


def colorize_depth(depth_m):
    """Fixed metric scale shared by the live preview and its legend, in BGR order."""
    stops = [0.1, 2.0, 4.0, 8.0]
    colors = np.array([[81, 89, 255], [92, 202, 255], [177, 217, 87], [255, 146, 93]])
    safe = np.nan_to_num(depth_m, nan=0, posinf=0, neginf=0)
    preview = np.stack(
        [np.interp(safe, stops, colors[:, channel]) for channel in range(3)], axis=-1
    )
    preview[~np.isfinite(depth_m) | (depth_m <= DEPTH_MIN_M) | (depth_m > DEPTH_MAX_M)] = 0
    return preview.astype(np.uint8)


class FastFoundationStereo:
    name = "C-Fast-FoundationStereo"

    def __init__(self, model_path, device="cuda"):
        import onnxruntime as ort

        if device not in ("cpu", "cuda"):
            raise ValueError("모델 실행 장치는 cpu 또는 cuda여야 해요.")
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA 실행 공급자가 없어요.")
        if device == "cuda":
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {"gpu_mem_limit": 2 * 1024**3, "cudnn_conv_algo_search": "HEURISTIC"},
                ),
            )
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("Fast-FoundationStereo의 CUDA 실행을 준비하지 못했어요.")
        self.device = device
        self.backend = f"{self.name} / ONNX Runtime {device.upper()}"
        inputs = [item.name for item in self.session.get_inputs()]
        if inputs != ["left_image", "right_image"]:
            raise ValueError("공식 C-Fast-FoundationStereo ONNX 입력 형식과 달라요.")

    def disparity(self, left, right):
        # The official single-ONNX export strips normalization from the graph.
        inputs = {}
        for name, frame in (("left_image", left), ("right_image", right)):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = ((image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]).astype(
                np.float32
            )
            inputs[name] = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        result = self.session.run(["disparity"], inputs)[0].squeeze()
        if result.shape != left.shape[:2] or not np.isfinite(result).any():
            raise ValueError("깊이 모델 출력이 잘못됐어요.")
        return result


class StereoRectifier:
    def __init__(self, left, right, output_size=(320, 192)):
        self.left, self.right = left, right
        self.size = output_size
        left_r = cv2.Rodrigues(left.rvec)[0]
        right_r = cv2.Rodrigues(right.rvec)[0]
        rotation = right_r @ left_r.T
        translation = right.tvec - rotation @ left.tvec
        self.baseline_m = float(np.linalg.norm(translation))
        if not 0.05 <= self.baseline_m <= 3:
            raise ValueError("좌우 카메라 간 거리를 확인해 주세요. 현재 허용 범위는 0.05~3m예요.")
        left_k, right_k = self.scaled_matrix(left), self.scaled_matrix(right)
        result = cv2.stereoRectify(
            left_k,
            left.distortion,
            right_k,
            right.distortion,
            output_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            # Forced cropping can zoom almost the entire reference view out of depth coverage.
            alpha=-1,
        )
        self.r1, self.r2, self.p1, self.p2, self.q = result[:5]
        if abs(self.p2[0, 3]) < 1e-5 or self.p2[0, 3] >= 0:
            raise ValueError(
                "카메라 1은 왼쪽, 2는 오른쪽에 배치해 주세요. 세로 스테레오는 지원하지 않아요."
            )
        self.maps = []
        for camera, matrix, rectification, projection in (
            (left, left_k, self.r1, self.p1),
            (right, right_k, self.r2, self.p2),
        ):
            self.maps.append(
                cv2.initUndistortRectifyMap(
                    matrix,
                    camera.distortion,
                    rectification,
                    projection,
                    output_size,
                    cv2.CV_32FC1,
                )
            )
        w, h = output_size
        self.valid_pixels = [
            ((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)).astype(np.uint8)
            for x, y in self.maps
        ]
        # This rectifier is replaced whenever the calibration signature changes.
        w, h = left.size
        pixels = np.indices((h, w), dtype=np.float32)[::-1].transpose(1, 2, 0)
        rectified = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), left.matrix, left.distortion, R=self.r1, P=self.p1
        ).reshape(h, w, 2)
        self.original_maps = tuple(np.ascontiguousarray(rectified[..., i]) for i in (0, 1))
        x, y = rectified.astype(np.float64).transpose(2, 0, 1)
        # Combine Q and R1.T so each frame only evaluates the original optical Z.
        z_projection = self.r1[:, 2] @ self.q[:3]
        self.z_offset = x * z_projection[0] + y * z_projection[1] + z_projection[3]
        self.z_disparity = z_projection[2]
        self.w_offset = x * self.q[3, 0] + y * self.q[3, 1] + self.q[3, 3]
        self.w_disparity = self.q[3, 2]
        self.floor_z = floor_depth_map(left, left.size)

    def scaled_matrix(self, calibration):
        matrix = calibration.matrix.copy()
        matrix[0] *= self.size[0] / calibration.size[0]
        matrix[1] *= self.size[1] / calibration.size[1]
        return matrix

    def images(self, left, right):
        output = []
        for image, maps in zip((left, right), self.maps, strict=True):
            resized = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            output.append(cv2.remap(resized, *maps, interpolation=cv2.INTER_LINEAR))
        return output

    def rectified_point(self, pixel, side=0):
        camera = self.left if side == 0 else self.right
        rotation = self.r1 if side == 0 else self.r2
        projection = self.p1 if side == 0 else self.p2
        return cv2.undistortPoints(
            np.array(pixel, dtype=np.float64).reshape(1, 1, 2),
            camera.matrix,
            camera.distortion,
            R=rotation,
            P=projection,
        )[0, 0]

    def world_point(self, x, y, disparity):
        homogeneous = self.q @ [x, y, disparity, 1]
        if abs(homogeneous[3]) < 1e-8:
            return None
        rectified = homogeneous[:3] / homogeneous[3]
        if not 0.1 <= rectified[2] <= 15:
            return None
        camera_point = self.r1.T @ rectified
        rotation = cv2.Rodrigues(self.left.rvec)[0]
        return rotation.T @ (camera_point - self.left.tvec.reshape(3))

    def align_to_original(self, disparity):
        # Inverse mapping avoids holes when a low-resolution depth map feeds a larger camera.
        sampled = cv2.remap(
            disparity,
            *self.original_maps,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        sampled64 = sampled.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            output = (
                (self.z_offset + self.z_disparity * sampled64)
                / (self.w_offset + self.w_disparity * sampled64)
            ).astype(np.float32)
        invalid = (
            ~np.isfinite(output)
            | (output <= DEPTH_MIN_M)
            | (output > DEPTH_MAX_M)
            | (sampled <= 0.5)
        )
        output[invalid] = np.nan
        return output

    def metadata(self):
        return {
            "camera_id": "cam1",
            "width": self.left.size[0],
            "height": self.left.size[1],
            "units": "meters",
            "depth_definition": "original_left_camera_optical_z",
            "invalid_float": "NaN",
            "invalid_png": 0,
            "png_scale_m": 0.001,
            "K": self.left.matrix.tolist(),
            "distortion": self.left.distortion.tolist(),
            "world_to_camera": self.left.projection.tolist(),
            "rectified_K": self.p1[:, :3].tolist(),
            "rectification_R": self.r1.tolist(),
            "rectified_size": list(self.size),
            "baseline_m": self.baseline_m,
        }


def tracking_from_disparity(rectifier, disparity, left_points, right_points, capture_ms):
    joints = {}
    epipolar_errors = []
    for index in (29, 30, 31, 32) if left_points and right_points else ():
        if left_points[index][2] < 0.65 or right_points[index][2] < 0.65:
            continue
        left = rectifier.rectified_point(left_points[index][:2])
        right = rectifier.rectified_point(right_points[index][:2], 1)
        epipolar_error = abs(left[1] - right[1])
        epipolar_errors.append(float(epipolar_error))
        if epipolar_error > 5:
            continue
        x, y = np.rint(left).astype(int)
        if not 3 <= x < disparity.shape[1] - 3 or not 3 <= y < disparity.shape[0] - 3:
            continue
        patch = disparity[y - 2 : y + 3, x - 2 : x + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.5) & (patch < x)]
        if len(valid) < 15:
            continue
        measured = float(np.median(valid))
        expected = left[0] - right[0]
        if expected <= 0 or abs(measured - expected) > max(5, expected * 0.15):
            continue
        point = rectifier.world_point(left[0], left[1], measured)
        if point is None:
            continue
        joints[str(index)] = [float(point[0]), float(point[1]), float(-point[2])]
    feet = []
    for heel, toe in ((29, 31), (30, 32)):
        if str(heel) in joints and str(toe) in joints:
            points = np.array([joints[str(heel)], joints[str(toe)]])
            if -0.12 <= points[:, 2].min() <= 0.2:
                feet.append(points.mean(axis=0).tolist())
    return {
        "valid": len(feet) == 2,
        "feet": feet,
        "joints": joints,
        "reason": "" if len(feet) == 2 else "좌우 영상에서 바닥의 양발을 확인해 주세요.",
        "capture_ms": capture_ms,
        "source": "C-Fast-FoundationStereo",
        "epipolar_error_px": round(max(epipolar_errors, default=0), 2),
    }


class StereoDepth:
    def __init__(self, hub, model_path):
        self.hub = hub
        self.model_path = Path(model_path)
        self.model = None
        self.rectifier = None
        self.signature = None
        self.last_pair = None
        self.result = None
        self.visualization = None
        self.depth_m = None
        self.frame_result = None
        self.error = "깊이 모델 준비 중"
        self.processing_ms = 0.0
        self.enabled = os.environ.get("LAVA_DISABLE_STEREO") != "1"

    def info(self):
        return {
            "backend": getattr(self.model, "backend", "깊이 계산 준비 중"),
            "device": getattr(self.model, "device", None),
            "ready": self.model is not None,
            "error": self.error,
            "processing_ms": round(self.processing_ms, 1),
            "resolution": [320, 192],
            "max_pair_skew_ms": MAX_PAIR_SKEW_MS,
            "baseline_m": self.rectifier.baseline_m if self.rectifier else None,
        }

    def frame_is_current(self, frame, now):
        if frame is None:
            return False
        cameras = [self.hub.cameras[name] for name in ("cam1", "cam2")]
        current_signature = tuple(camera.generation for camera in cameras)
        return (
            0 <= now - frame["metadata"]["captured_ms"] <= 350
            and all(camera.depth_usable(now) for camera in cameras)
            and current_signature == tuple(frame["metadata"]["calibration_generation"])
        )

    def current(self, now):
        if self.result and self.frame_is_current(self.frame_result, now):
            return self.result
        return {
            "valid": False,
            "feet": [],
            "joints": {},
            "capture_ms": None,
            "source": getattr(self.model, "name", "깊이 계산 준비 중"),
            "reason": self.error or "새 좌우 깊이 영상을 기다리고 있어요.",
        }

    def compute(
        self,
        left_frame,
        right_frame,
        left_points,
        right_points,
        captured,
        details,
        person_mask=None,
    ):
        left, right = self.rectifier.images(left_frame, right_frame)
        started = time.perf_counter()
        disparity = self.model.disparity(left, right)
        self.processing_ms = (time.perf_counter() - started) * 1000
        tracking = tracking_from_disparity(
            self.rectifier,
            disparity,
            left_points,
            right_points,
            captured,
        )
        tracking["source"] = getattr(self.model, "name", "C-Fast-FoundationStereo")
        xyz = cv2.reprojectImageTo3D(disparity, self.rectifier.q)
        depth = xyz[..., 2]
        invalid = (
            ~np.isfinite(depth)
            | (depth <= DEPTH_MIN_M)
            | (depth > DEPTH_MAX_M)
            | (disparity <= 0.5)
        )
        y, x = np.indices(disparity.shape, dtype=np.float32)
        right_valid = cv2.remap(
            self.rectifier.valid_pixels[1], x - disparity, y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        invalid |= (self.rectifier.valid_pixels[0] == 0) | (right_valid == 0)
        depth[invalid] = np.nan
        clean_disparity = disparity.copy()
        clean_disparity[invalid] = np.nan
        aligned = self.rectifier.align_to_original(clean_disparity)
        visualization = colorize_depth(aligned)
        _, jpeg = cv2.imencode(".jpg", visualization)
        _, reference = cv2.imencode(".jpg", left_frame)
        depth_mm = np.rint(np.nan_to_num(aligned, nan=0) * 1000).astype(np.uint16)
        _, png = cv2.imencode(".png", depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        _, valid_mask = cv2.imencode(
            ".png", np.isfinite(aligned).astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        metadata = self.rectifier.metadata()
        valid = aligned[np.isfinite(aligned)]
        center = aligned[aligned.shape[0] // 2, aligned.shape[1] // 2]
        metadata.update(
            {
                "type": "depth",
                "captured_ms": captured,
                "processing_ms": self.processing_ms,
                "depth_backend": getattr(self.model, "backend", "C-Fast-FoundationStereo"),
                "depth_device": getattr(self.model, "device", "cuda"),
                "depth_png_bytes": len(png),
                "reference_jpeg_bytes": len(reference),
                "preview_jpeg_bytes": len(jpeg),
                "valid_mask_png_bytes": len(valid_mask),
                "preview_space": "original_left",
                "preview_range_m": [0.1, 8.0],
                "center_depth_m": round(float(center), 2) if np.isfinite(center) else None,
                "valid_percent": round(100 * valid.size / aligned.size, 1),
                "depth_range_m": [round(float(valid.min()), 2), round(float(valid.max()), 2)]
                if valid.size
                else None,
            }
        )
        metadata.update(details)
        packet = pack(
            metadata, png.tobytes() + reference.tobytes() + jpeg.tobytes() + valid_mask.tobytes()
        )
        composite = render_with_depth(
            left_frame,
            aligned,
            self.rectifier.left,
            self.hub.game.snapshot(),
            floor_z=self.rectifier.floor_z,
        )
        return {
            "tracking": tracking,
            "depth_m": aligned,
            "rectified_depth_m": depth,
            "preview": jpeg.tobytes(),
            "reference": reference.tobytes(),
            "packet": packet,
            "metadata": metadata,
            "composite": composite,
            "person_mask": person_mask,
        }

    async def run(self):
        if not self.enabled:
            self.error = "검사용 실행: 깊이 모델 비활성"
            return
        try:
            self.model = await asyncio.to_thread(create_depth_model, self.model_path)
            self.error = "좌우 카메라를 보정하고 추적을 시작해 주세요."
        except Exception as error:
            self.error = str(error)
            return
        while True:
            await asyncio.sleep(0.02)
            left, right = self.hub.cameras["cam1"], self.hub.cameras["cam2"]
            if not left.depth_usable(time.monotonic() * 1000) or not right.depth_usable(
                time.monotonic() * 1000
            ):
                self.result = None
                continue
            pair = (left.capture_ms, right.capture_ms)
            skew = abs(pair[0] - pair[1])
            if pair == self.last_pair or skew > MAX_PAIR_SKEW_MS:
                if skew > MAX_PAIR_SKEW_MS:
                    self.error = (
                        "좌우 촬영 시각 차이가 40ms를 넘어요. 네트워크·촬영 상태를 확인해 주세요."
                    )
                continue
            self.last_pair = pair
            signature = (
                left.generation,
                right.generation,
                id(left.calibration),
                id(right.calibration),
            )
            try:
                if signature != self.signature:
                    self.rectifier = await asyncio.to_thread(
                        StereoRectifier, left.calibration, right.calibration
                    )
                    self.signature = signature
                result = await asyncio.to_thread(
                    self.compute,
                    left.sample_frame,
                    right.sample_frame,
                    left.landmarks,
                    right.landmarks,
                    pair[0],
                    {
                        "source_seq": left.seq,
                        "right_captured_ms": pair[1],
                        "pair_skew_ms": skew,
                        "calibration_generation": [left.generation, right.generation],
                        "timestamp_kind": "browser_draw_monotonic_estimate",
                    },
                    left.sample_segmentation,
                )
                if signature != (
                    left.generation,
                    right.generation,
                    id(left.calibration),
                    id(right.calibration),
                ):
                    continue
                published_ms = time.monotonic() * 1000
                if not all(camera.depth_usable(published_ms) for camera in (left, right)):
                    self.result = self.frame_result = None
                    self.error = (
                        "처리 중 보정 품질 검증이 취소됐어요. 기준판과 카메라 상태를 확인해 주세요."
                    )
                    continue
                if published_ms - pair[0] > 350:
                    self.result = None
                    self.error = (
                        "깊이 처리 결과가 350ms를 넘었어요. 장치 부하와 촬영 상태를 확인해 주세요."
                    )
                    continue
                self.frame_result = result
                self.result = result["tracking"]
                self.depth_m, self.visualization = result["depth_m"], result["preview"]
                self.hub.cameras["cam1"].depth_composite = result["composite"]
                for viewer in self.hub.depth_viewers.copy():
                    viewer.push("depth", result["packet"])
                self.result["skew_ms"] = skew
                self.error = ""
            except Exception as error:
                self.error, self.result = str(error), None

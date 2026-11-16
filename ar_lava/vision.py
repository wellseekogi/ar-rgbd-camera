import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class PoseTracker:
    """One model per camera; calls are serialized by that camera's asyncio lock."""

    def __init__(self, model_path):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_segmentation_masks=True,
        )
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_ms = 0

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.last_ms = max(self.last_ms + 1, int(time.monotonic() * 1000))
        result = self.model.detect_for_video(image, self.last_ms)
        if len(result.pose_landmarks) != 1:
            return None, None, len(result.pose_landmarks)
        h, w = frame.shape[:2]
        points = [
            [point.x * w, point.y * h, min(point.visibility, point.presence)]
            for point in result.pose_landmarks[0]
        ]
        mask = result.segmentation_masks[0].numpy_view().copy().squeeze()
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return points, mask, 1

    def close(self):
        self.model.close()


def decode_jpeg(payload):
    # Read dimensions before native decoding to bound allocations from untrusted JPEGs.
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        w, h = image.size
        if image.format != "JPEG" or w < 160 or h < 120 or w > 1280 or h > 1280:
            raise ValueError("JPEG 해상도는 160×120~1280×1280 범위여야 해요.")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG 프레임을 읽을 수 없어요.")
    return frame

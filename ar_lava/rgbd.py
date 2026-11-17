"""One REST snapshot: reference RGB, linear normalized Z and matching floor tracking."""

import base64
import json

import cv2
import numpy as np

from ar_lava.stereo import DEPTH_MAX_M, DEPTH_MIN_M


def encode_rgbd(result, server_ms):
    metadata = result["metadata"]
    depth = result["depth_m"]
    valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)
    normalized = np.zeros(depth.shape, dtype="<f4")
    normalized[valid] = (depth[valid] - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    tracking = result["tracking"]
    generations = ":".join(str(value) for value in metadata["calibration_generation"])
    frame_id = (
        f"cam1:{generations}:{metadata['source_seq']}:"
        f"{metadata['captured_ms']}:{metadata['right_captured_ms']}"
    )
    response = {
        "schema": "ar-lava.rgbd.v1",
        "frame_id": frame_id,
        "camera_id": "cam1",
        "depth_backend": metadata.get("depth_backend", "C-Fast-FoundationStereo"),
        "depth_device": metadata.get("depth_device", "cuda"),
        "source_camera_ids": ["cam1", "cam2"],
        "captured_ms": metadata["captured_ms"],
        "server_ms": server_ms,
        "age_ms": round(server_ms - metadata["captured_ms"], 2),
        "timestamp_kind": metadata["timestamp_kind"],
        "pair_skew_ms": metadata["pair_skew_ms"],
        "calibration_generation": metadata["calibration_generation"],
        "image": {
            "encoding": "jpeg",
            "width": metadata["width"],
            "height": metadata["height"],
            "origin": "top_left",
            "data_base64": base64.b64encode(result["reference"]).decode("ascii"),
        },
        "depth": {
            "encoding": "base64",
            "dtype": "float32",
            "byte_order": "little",
            "shape": list(depth.shape),
            "layout": "row_major",
            "definition": metadata["depth_definition"],
            "normalization": "linear",
            "near_m": DEPTH_MIN_M,
            "far_m": DEPTH_MAX_M,
            "invalid_value": 0,
            "data_base64": base64.b64encode(normalized.tobytes(order="C")).decode("ascii"),
        },
        "valid_mask": {
            "encoding": "base64",
            "dtype": "uint8",
            "shape": list(depth.shape),
            "layout": "row_major",
            "valid_value": 1,
            "invalid_value": 0,
            "data_base64": base64.b64encode(valid.astype(np.uint8).tobytes()).decode("ascii"),
        },
        "camera": {
            "K": metadata["K"],
            "distortion": metadata["distortion"],
            "world_to_camera": metadata["world_to_camera"],
            "camera_axes": "x_right_y_down_z_forward",
            "world_axes": "board_x_right_y_down_z_into_floor",
            "image_space": "original_left_distorted",
            "units": "meters",
        },
        "tracking": {
            "kind": "feet_3d",
            "valid": tracking["valid"],
            "reason": tracking["reason"],
            "coordinate_system": "board_world",
            "units": "meters",
            "joints_world_m": {
                name: [point[0], point[1], -point[2]] for name, point in tracking["joints"].items()
            },
            "feet_world_m": [[point[0], point[1], -point[2]] for point in tracking["feet"]],
        },
        "model_processing_ms": metadata["processing_ms"],
        "segmentation": {"available": False, "model": None},
    }
    mask = result.get("person_mask")
    if (
        mask is not None
        and mask.shape == depth.shape
        and np.isfinite(mask).all()
        and (mask >= 0).all()
        and (mask <= 1).all()
    ):
        # A probability mask is not an optical alpha matte or a depth-validity mask.
        success, png = cv2.imencode(".png", np.rint(mask * 255).astype(np.uint8))
        if success:
            response["segmentation"] = {
                "available": True,
                "model": "mediapipe_pose_landmarker_lite",
                "frame_id": frame_id,
                "encoding": "png",
                "dtype": "uint8",
                "shape": list(mask.shape),
                "semantics": "person_foreground_probability",
                "probability_scale": 1 / 255,
                "data_base64": base64.b64encode(png).decode("ascii"),
            }
    return json.dumps(response, separators=(",", ":"), allow_nan=False).encode("utf-8")

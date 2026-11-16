"""Fetch one matching RGB/normalized-depth REST frame using only stdlib + NumPy."""

import argparse
import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def decode_frame(body):
    if body.get("schema") != "ar-lava.rgbd.v1":
        raise ValueError("지원하는 RGB-D 응답 규약이 아니에요.")
    info = body["depth"]
    shape = (body["image"]["height"], body["image"]["width"])
    if (
        info["dtype"] != "float32"
        or info["byte_order"] != "little"
        or info["normalization"] != "linear"
        or tuple(info["shape"]) != shape
        or tuple(body["valid_mask"]["shape"]) != shape
    ):
        raise ValueError("영상과 깊이의 형식·크기가 맞지 않아요.")
    depth = np.frombuffer(base64.b64decode(info["data_base64"], validate=True), "<f4")
    depth = depth.reshape(shape).copy()
    mask = np.frombuffer(
        base64.b64decode(body["valid_mask"]["data_base64"], validate=True), np.uint8
    ).reshape(shape)
    if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 1).any():
        raise ValueError("정규화 깊이가 0~1 범위를 벗어났어요.")
    near, far = info["near_m"], info["far_m"]
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("미터 복원 범위가 잘못됐어요.")
    valid = mask == 1
    meters = (near + depth * (far - near)).astype(np.float32)
    meters[~valid] = np.nan
    rgb = base64.b64decode(body["image"]["data_base64"], validate=True)
    return rgb, depth, valid, meters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://localhost:8443/api/rgbd/frame")
    parser.add_argument("--ca", type=Path, default=Path("certs/lava-local-ca.crt"))
    parser.add_argument("--output", type=Path, default=Path("data/rgbd-capture"))
    args = parser.parse_args()
    key = os.environ.get("LAVA_SESSION_KEY", "")
    if not key:
        raise SystemExit("LAVA_SESSION_KEY 환경 변수에 접속 키를 설정해 주세요.")
    request = urllib.request.Request(args.url, headers={"X-Session-Key": key})
    context = ssl.create_default_context(cafile=str(args.ca))
    try:
        with urllib.request.urlopen(request, context=context, timeout=5) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read()).get("detail", str(error))
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    rgb, depth, valid, meters = decode_frame(body)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference.jpg").write_bytes(rgb)
    np.save(args.output / "depth_normalized.npy", depth, allow_pickle=False)
    np.save(args.output / "valid_mask.npy", valid, allow_pickle=False)
    np.save(args.output / "depth_m.npy", meters, allow_pickle=False)
    segmentation = body.get("segmentation", {})
    if segmentation.get("available"):
        (args.output / "person_mask.png").write_bytes(
            base64.b64decode(segmentation.pop("data_base64"), validate=True)
        )
    for name in ("image", "depth", "valid_mask"):
        del body[name]["data_base64"]
    (args.output / "metadata.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"frame_id={body['frame_id']} shape={depth.shape} output={args.output.resolve()}")


if __name__ == "__main__":
    main()

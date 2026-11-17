import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ar_lava.stereo import create_depth_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="data/depth-benchmark.json")
    parser.add_argument("--depth-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    model = create_depth_model(ROOT / "models/c_fast_foundationstereo.onnx", args.depth_device)
    left = cv2.imdecode(np.frombuffer(Path(args.left).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    right = cv2.imdecode(np.frombuffer(Path(args.right).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise SystemExit("Cannot open stereo sample images")
    result = {
        "backend": model.backend,
        "device": model.device,
        "providers": model.session.get_providers() if hasattr(model, "session") else ["OpenCV CPU"],
        "measurements": [],
    }
    for width, height in ((320, 192), (640, 384)):
        a, b = cv2.resize(left, (width, height)), cv2.resize(right, (width, height))
        model.disparity(a, b)
        times = []
        for _ in range(5):
            started = time.perf_counter()
            disparity = model.disparity(a, b)
            times.append((time.perf_counter() - started) * 1000)
        measurement = {
            "width": width,
            "height": height,
            "runs": len(times),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "finite_fraction": float(np.isfinite(disparity).mean()),
            "median_disparity_px": (
                float(np.nanmedian(disparity)) if np.isfinite(disparity).any() else None
            ),
        }
        print(json.dumps(measurement), flush=True)
        result["measurements"].append(measurement)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Keep virtual-camera checks separate from saved phone calibration."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ar_lava.server import create_app  # noqa: E402

if __name__ == "__main__":
    # Chrome's built-in fake device can disappear after track.stop(); file capture stays reusable.
    target = ROOT / "data/browser-camera.y4m"
    target.parent.mkdir(exist_ok=True)
    frame = bytes([120]) * (640 * 480) + bytes([128]) * (640 * 480 // 2)
    target.write_bytes(
        b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n" + (b"FRAME\n" + frame) * 3
    )
    with TemporaryDirectory(prefix="lava-browser-") as directory:
        uvicorn.run(create_app(data_dir=Path(directory)), host="127.0.0.1", port=8000)

import argparse
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from ar_lava.server import create_app  # noqa: E402


def open_browser_when_ready(url, ca_file=None, token=None):
    handlers = [urllib.request.ProxyHandler({})]
    if ca_file:
        handlers.append(
            urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca_file))
        )
    opener = urllib.request.build_opener(*handlers)
    for _ in range(40):
        try:
            with opener.open(f"{url}/api/health", timeout=1) as response:
                if response.status == 200:
                    destination = (
                        f"{url}/#{urlencode({'key': token, 'role': 'viewer'})}" if token else url
                    )
                    webbrowser.open(destination)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    print(f"브라우저에서 직접 접속해 주세요: {url}", flush=True)


def main(open_browser=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--http-local", action="store_true", help="Local browser tests only")
    parser.add_argument(
        "--depth-device",
        choices=("auto", "cpu", "cuda"),
        default=os.environ.get("LAVA_DEPTH_DEVICE", "cuda"),
        help="auto: CUDA if available, otherwise CPU StereoSGBM",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=open_browser,
        help="Open the page after startup",
    )
    args = parser.parse_args()
    os.environ["LAVA_DEPTH_DEVICE"] = args.depth_device
    app = create_app()
    token = app.state.hub.token
    print(f"접속 키: {token}", flush=True)
    print("노트북: https://localhost:8443", flush=True)
    print("핸드폰 3대: 같은 Wi-Fi에서 https://노트북의IP:8443, 카메라 1·2·3 선택", flush=True)
    options = {}
    if not args.http_local:
        for path in (ROOT / "certs/server.pem", ROOT / "certs/server-key.pem"):
            if not path.is_file():
                raise SystemExit(
                    "먼저 .venv\\Scripts\\python scripts\\prepare.py 를 실행해 주세요."
                )
        options = {
            "ssl_certfile": str(ROOT / "certs/server.pem"),
            "ssl_keyfile": str(ROOT / "certs/server-key.pem"),
        }
    if args.open_browser:
        url = "http://127.0.0.1:8000" if args.http_local else "https://localhost:8443"
        ca_file = None if args.http_local else str(ROOT / "certs/lava-local-ca.crt")
        threading.Thread(
            target=open_browser_when_ready, args=(url, ca_file, token), daemon=True
        ).start()
    uvicorn.run(
        app,
        host="127.0.0.1" if args.http_local else "0.0.0.0",
        port=8000 if args.http_local else 8443,
        ws_max_size=1_500_000,
        ws_max_queue=1,
        ws_per_message_deflate=False,
        workers=1,
        **options,
    )


if __name__ == "__main__":
    main()

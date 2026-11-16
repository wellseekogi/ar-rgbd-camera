import asyncio
import socket
import ssl
import threading
import time

import cv2
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from ar_lava.protocol import pack, unpack
from ar_lava.server import create_app
from scripts import prepare


def test_secure_websocket_receives_video_without_webrtc(tmp_path, monkeypatch):
    monkeypatch.setenv("LAVA_DISABLE_STEREO", "1")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    ca = prepare.certificates(["127.0.0.1"])
    app = create_app(tmp_path / "data", token="tls-test-key")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            ssl_certfile=str(tmp_path / "certs/server.pem"),
            ssl_keyfile=str(tmp_path / "certs/server-key.pem"),
            log_level="error",
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()

    async def exchange():
        context = ssl.create_default_context(cafile=str(ca))
        async with connect(f"wss://localhost:{port}/ws", ssl=context) as camera:
            await camera.send('{"role":"camera","camera_id":"cam1","token":"tls-test-key"}')
            import json

            assert json.loads(await camera.recv())["type"] == "ready"
            await camera.send('{"type":"ping","client_ms":100}')
            assert json.loads(await camera.recv())["type"] == "pong"
            await camera.send('{"type":"sync","client_send":100,"client_receive":102}')
            _, jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
            await camera.send(
                pack({"seq": 0, "captured_ms": 101, "profile": "rear"}, jpeg.tobytes())
            )
            header, _ = unpack(await asyncio.wait_for(camera.recv(), 3))
            assert header["seq"] == 0
            source = app.state.hub.cameras["cam1"]
            assert source.transport == "HTTPS/WSS" and source.received_frames == 1
            assert source.info(source.frame_times[-1])["connection_status"] == "receiving"

    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        asyncio.run(exchange())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()

import asyncio
import contextlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from ar_lava.calibration import Calibrator, CameraCalibration
from ar_lava.calibration_check import CalibrationPairCheck
from ar_lava.game import Game, GameConfig
from ar_lava.guidance import CalibrationGuide
from ar_lava.pairing import install_pairing
from ar_lava.protocol import CAMERA_IDS, MAX_AGE_MS, ClockSync, frame_header, pack, unpack
from ar_lava.render import render
from ar_lava.rgbd import encode_rgbd
from ar_lava.stereo import StereoDepth
from ar_lava.vision import PoseTracker, decode_jpeg

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("ar_lava")


def now_ms():
    return time.monotonic() * 1000


class Camera:
    def __init__(self, camera_id, data_dir):
        self.camera_id = camera_id
        self.path = data_dir / f"{camera_id}.json"
        self.calibration = CameraCalibration.load(self.path) if self.path.exists() else None
        # A phone can move between sessions. Reuse intrinsics, never silently reuse its pose.
        if self.calibration:
            self.calibration.rvec = self.calibration.tvec = None
        self.calibrator = Calibrator()
        self.guide = CalibrationGuide()
        self.clock = ClockSync()
        self.connected = False
        self.profile = None
        self.frame = None
        self.background = None
        self.landmarks = None
        self.capture_ms = 0.0
        self.processing_ms = 0.0
        self.age_ms = 0.0
        self.people = 0
        self.seq = -1
        self.dropped = 0
        self.error = ""
        self.mode = "calibration"
        self.pending = None
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.tracker = None
        self.inflight = False
        self.generation = 0
        self.composite = None
        self.transport = ""
        self.next_landmarks = None
        self.next_segmentation = None
        self.sample_segmentation = None
        self.sample_frame = None
        self.depth_composite = None
        self.frame_times = deque(maxlen=30)
        self.received_frames = 0
        self.floor_validated = False
        self.pair_quality = {}
        self.observation_capture_ms = None

    def received(self):
        self.received_frames += 1
        self.frame_times.append(now_ms())

    def depth_usable(self, current_ms):
        return (
            self.connected
            and self.mode == "play"
            and self.calibration is not None
            and self.calibration.ready
            and self.calibration.lens_validated
            and self.calibration.profile == self.profile
            and self.floor_validated
            and self.pair_quality.get("passed") is True
            and self.pair_quality.get("generation") == self.generation
            and current_ms <= self.pair_quality.get("expires_ms", 0)
            and self.sample_frame is not None
            and self.calibration.size == (self.sample_frame.shape[1], self.sample_frame.shape[0])
            and 0 <= current_ms - self.capture_ms <= MAX_AGE_MS
            and self.clock.rtt_ms <= 100
        )

    def info(self, current_ms):
        calibration = self.calibration
        recent = [stamp for stamp in self.frame_times if current_ms - stamp < 2000]
        fps = (len(recent) - 1) * 1000 / (recent[-1] - recent[0]) if len(recent) > 1 else 0
        if not self.connected:
            fps = 0
        connection_status = "disconnected"
        if self.connected:
            connection_status = "waiting"
            if self.frame_times:
                connection_status = "receiving" if recent else "stalled"
        return {
            "id": self.camera_id,
            "connected": self.connected,
            "mode": self.mode,
            "intrinsics": calibration is not None and calibration.lens_validated,
            "calibrated": (
                calibration is not None and calibration.lens_validated
                and calibration.ready and self.floor_validated
            ),
            "samples": len(self.calibrator.corners),
            "people": self.people,
            "processing_ms": round(self.processing_ms, 1),
            "age_ms": round(max(0, current_ms - self.capture_ms), 1) if self.capture_ms else None,
            "rtt_ms": round(self.clock.rtt_ms, 1) if self.clock.offset_ms is not None else None,
            "dropped": self.dropped,
            "error": self.error,
            "background": self.background is not None,
            "rms_px": round(calibration.rms, 3) if calibration else None,
            "transport": self.transport,
            "connection_status": connection_status,
            "received_frames": self.received_frames,
            "fps": round(fps, 1),
            "guide": self.guide.snapshot(self, current_ms),
            "quality": self.pair_quality,
        }

    def reset_guidance(self):
        self.guide = CalibrationGuide()
        self.calibrator = Calibrator()
        self.background = self.frame = self.sample_frame = self.depth_composite = None
        self.next_segmentation = self.sample_segmentation = None
        self.error = ""
        self.floor_validated = False
        self.pair_quality = {}
        self.observation_capture_ms = None


class Viewer:
    def __init__(self):
        self.pending = {}
        self.event = asyncio.Event()

    def push(self, key, message):
        self.pending[key] = message
        self.event.set()

    async def send(self, websocket):
        while True:
            await self.event.wait()
            self.event.clear()
            pending, self.pending = self.pending, {}
            for message in pending.values():
                if isinstance(message, bytes):
                    await asyncio.wait_for(websocket.send_bytes(message), 2)
                else:
                    await asyncio.wait_for(websocket.send_json(message), 2)

    async def run(self, websocket):
        sender = asyncio.create_task(self.send(websocket))
        receiver = asyncio.create_task(websocket.receive())
        try:
            done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)


class Hub:
    def __init__(self, data_dir, model_path, token=None):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("LAVA_SESSION_KEY")
        key_path = data_dir / "session.key"
        if not self.token:
            if not key_path.exists():
                key_path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
            self.token = key_path.read_text(encoding="utf-8").strip()
        self.cameras = {camera_id: Camera(camera_id, data_dir) for camera_id in sorted(CAMERA_IDS)}
        self.model_path = model_path
        self.viewers = set()
        self.depth_viewers = set()
        self.game = Game()
        self.tracking = {
            "valid": False,
            "feet": [],
            "joints": {},
            "reason": "카메라를 연결해 주세요.",
        }
        self.stereo = StereoDepth(self, ROOT / "models/c_fast_foundationstereo.onnx")
        self.calibration_check = CalibrationPairCheck()

    def snapshot(self):
        current = now_ms()
        return {
            "type": "state",
            "server_ms": current,
            "cameras": [camera.info(current) for camera in self.cameras.values()],
            "tracking": self.tracking,
            "game": self.game.snapshot(),
            "depth": self.stereo.info(),
        }

    async def tick(self):
        last = now_ms()
        while True:
            current = now_ms()
            self.calibration_check.update(self.cameras, current)
            self.tracking = self.stereo.current(current)
            self.game.step((current - last) / 1000, self.tracking)
            last = current
            state = self.snapshot()
            for viewer in self.viewers.copy():
                viewer.push("state", state)
            await asyncio.sleep(0.05)

    def analyze(self, camera, payload, game):
        frame = decode_jpeg(payload)
        composite = self.analyze_frame(camera, frame, game)
        success, encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            raise ValueError("게임 화면을 만들지 못했어요.")
        return encoded.tobytes()

    def analyze_frame(self, camera, frame, game):
        started = now_ms()
        size = (frame.shape[1], frame.shape[0])
        if camera.calibration and (
            camera.calibration.profile != camera.profile or camera.calibration.size != size
        ):
            camera.calibration = None
            camera.floor_validated = False
            camera.pair_quality = {}
            camera.generation += 1
            camera.background = None
            camera.calibrator = Calibrator()
            camera.guide = CalibrationGuide()
            camera.mode = "calibration"
            camera.error = "카메라·렌즈·해상도가 바뀌었어요. 다시 보정해 주세요."
        camera.frame = frame
        camera.guide.observe(camera, frame, started)
        mask = None
        if (
            camera.mode == "play"
            and camera.calibration
            and camera.calibration.ready
            and self.model_path.is_file()
        ):
            if camera.tracker is None:
                camera.tracker = PoseTracker(self.model_path)
            camera.next_landmarks, mask, camera.people = camera.tracker.process(frame)
        else:
            camera.next_landmarks, camera.people = None, 0
        camera.next_segmentation = mask
        composite = render(frame, mask, camera.calibration, game, camera.background)
        camera.processing_ms = now_ms() - started
        camera.composite = composite
        return composite

    async def process(self, camera, websocket, send_lock):
        while True:
            await camera.event.wait()
            camera.event.clear()
            item, camera.pending = camera.pending, None
            if item is None:
                continue
            header, payload = item
            seq, client_ms, profile = frame_header(header)
            capture_ms = camera.clock.to_server(client_ms)
            age = now_ms() - capture_ms
            if seq <= camera.seq or not -25 <= age <= MAX_AGE_MS:
                camera.dropped += 1
                async with send_lock:
                    await websocket.send_json({"type": "ack", "seq": seq, "dropped": True})
                continue
            async with camera.lock:
                camera.inflight = True
                try:
                    camera.observation_capture_ms = capture_ms
                    if camera.profile != profile:
                        camera.background = None
                        camera.landmarks = None
                    camera.profile = profile
                    # to_thread keeps websocket receiving, game clocks and other cameras responsive.
                    encoded = await asyncio.to_thread(
                        self.analyze,
                        camera,
                        payload,
                        self.game.snapshot(),
                    )
                    camera.capture_ms = capture_ms
                    camera.landmarks = camera.next_landmarks
                    camera.sample_frame = camera.frame
                    camera.sample_segmentation = camera.next_segmentation
                    camera.seq = seq
                    camera.received()
                    packet = pack(
                        {
                            "type": "frame",
                            "camera_id": camera.camera_id,
                            "seq": seq,
                            "captured_ms": capture_ms,
                            "processing_ms": camera.processing_ms,
                            "guide": camera.guide.snapshot(camera, now_ms()),
                            "composite_jpeg_bytes": len(encoded),
                            "reference_jpeg_bytes": len(payload),
                        },
                        encoded + payload,
                    )
                    for viewer in self.viewers.copy():
                        viewer.push(camera.camera_id, packet)
                    async with send_lock:
                        await asyncio.wait_for(websocket.send_bytes(packet), 2)
                except (ValueError, OSError, RuntimeError) as error:
                    camera.landmarks = None
                    camera.error = str(error)
                    async with send_lock:
                        await websocket.send_json(
                            {"type": "error", "message": str(error), "seq": seq}
                        )
                finally:
                    camera.inflight = False


def create_app(data_dir=None, model_path=None, token=None):
    hub = Hub(
        data_dir or ROOT / "data", model_path or ROOT / "models/pose_landmarker_lite.task", token
    )

    @asynccontextmanager
    async def lifespan(app):
        cv2.setNumThreads(1)
        task = asyncio.create_task(hub.tick())
        stereo_task = asyncio.create_task(hub.stereo.run())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await rtc()
        stereo_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stereo_task
        for camera in hub.cameras.values():
            if camera.tracker:
                camera.tracker.close()

    app = FastAPI(title="AR Floor Is Lava", lifespan=lifespan)
    app.state.hub = hub

    def authorize(x_session_key: str = Header(default="")):
        if not hmac.compare_digest(x_session_key, hub.token):
            raise HTTPException(401, "접속 키가 맞지 않아요.")

    from ar_lava.rtc import install_rtc

    rtc = install_rtc(app, hub, authorize)
    install_pairing(app, hub, authorize)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/api/health")
    async def health():
        return {"ok": True, "model_available": hub.model_path.is_file()}

    @app.get("/api/state", dependencies=[Depends(authorize)])
    async def state():
        return hub.snapshot()

    @app.get("/api/rgbd/frame", dependencies=[Depends(authorize)])
    async def rgbd_frame():
        result = hub.stereo.frame_result
        if not hub.stereo.frame_is_current(result, now_ms()):
            raise HTTPException(
                409, "정합할 새 RGB·깊이 결과가 없어요. 카메라 보정과 지연을 확인해 주세요."
            )
        content = await asyncio.to_thread(encode_rgbd, result, now_ms())
        # Encoding can overlap a new frame, camera replacement or a disconnect.
        if not hub.stereo.frame_is_current(result, now_ms()):
            raise HTTPException(409, "응답 준비 중 결과가 만료됐어요. 새 프레임을 요청해 주세요.")
        return Response(content, media_type="application/json")

    @app.get("/api/depth/{format}", dependencies=[Depends(authorize)])
    async def depth_output(format: str):
        if hub.stereo.result is None or not hub.stereo.current(now_ms()).get("capture_ms"):
            raise HTTPException(409, "새 좌우 깊이 결과를 기다리고 있어요.")
        timestamp = str(hub.stereo.result["capture_ms"])
        if format == "metadata":
            return hub.stereo.frame_result["metadata"]
        if format == "reference.jpg":
            return Response(
                hub.stereo.frame_result["reference"],
                media_type="image/jpeg",
                headers={"X-Capture-Ms": timestamp},
            )
        if format == "preview.jpg":
            return Response(
                hub.stereo.visualization,
                media_type="image/jpeg",
                headers={"X-Capture-Ms": timestamp},
            )
        if format in ("depth.npy", "frame.npz"):
            stream = BytesIO()
            if format == "frame.npz":
                result = hub.stereo.frame_result
                np.savez_compressed(
                    stream,
                    depth_m=result["depth_m"],
                    reference_jpeg=np.frombuffer(result["reference"], dtype=np.uint8),
                    metadata=json.dumps(result["metadata"]),
                )
            else:
                np.save(stream, hub.stereo.depth_m, allow_pickle=False)
            return Response(
                stream.getvalue(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{format}"',
                    "X-Capture-Ms": timestamp,
                },
            )
        raise HTTPException(404)

    @app.post("/api/game/{action}", dependencies=[Depends(authorize)])
    async def game_action(action: str, request: Request):
        if action == "start":
            if not hub.model_path.is_file():
                raise HTTPException(409, "추적 모델을 먼저 설치해 주세요.")
            if not hub.tracking.get("valid") or any(
                hub.game.cell(foot) is None for foot in hub.tracking["feet"]
            ):
                raise HTTPException(
                    409, "보정된 카메라 2대에서 게임판 안의 양발을 확인해 주세요."
                )
            hub.game.start()
        elif action == "stop":
            hub.game.reset()
        elif action == "config":
            if hub.game.status in ("playing", "paused"):
                raise HTTPException(409, "게임을 종료한 뒤 설정해 주세요.")
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("설정은 객체여야 해요.")
                hub.game = Game(GameConfig(**body))
            except (TypeError, ValueError) as error:
                raise HTTPException(400, str(error)) from error
        else:
            raise HTTPException(404)
        return hub.snapshot()

    @app.post("/api/cameras/{camera_id}/{action}", dependencies=[Depends(authorize)])
    async def camera_action(camera_id: str, action: str):
        if camera_id not in hub.cameras:
            raise HTTPException(404)
        camera = hub.cameras[camera_id]
        async with camera.lock:
            try:
                if action in ("guide-resume", "guide-restart"):
                    if action == "guide-restart":
                        camera.path.unlink(missing_ok=True)
                        camera.calibration = None
                    elif camera.calibration:
                        camera.calibration.rvec = camera.calibration.tvec = None
                    camera.reset_guidance()
                    camera.generation += 1
                    camera.landmarks = None
                    camera.mode = "calibration"
                    camera.floor_validated = False
                    camera.pair_quality = {}
                    hub.calibration_check.reset()
                elif action in ("guide-floor", "guide-background"):
                    if not camera.guide.enabled:
                        raise ValueError("자동 안내를 먼저 켜 주세요.")
                    if (
                        not camera.connected
                        or camera.frame is None
                        or now_ms() - camera.capture_ms > 1500
                    ):
                        raise ValueError("카메라의 새 영상을 먼저 연결해 주세요.")
                    if (camera.calibration is None or not camera.calibration.lens_validated
                            or camera.calibration.profile != camera.profile):
                        raise ValueError("렌즈 보정을 먼저 완료해 주세요.")
                    if action == "guide-floor":
                        if camera.calibration.ready:
                            raise ValueError(
                                "바닥 위치가 이미 보정됐어요. 이동했다면 다시 보정해 주세요."
                            )
                        camera.guide.arm_floor(camera)
                        camera.generation += 1
                        hub.calibration_check.reset()
                    else:
                        hub.calibration_check.update(hub.cameras, now_ms())
                        if not all(other.connected and other.pair_quality.get("passed")
                                   and now_ms() <= other.pair_quality.get("expires_ms", 0)
                                   for other in hub.cameras.values()):
                            raise ValueError("두 카메라의 동시 기준판 검증을 먼저 통과해 주세요.")
                        if not hub.model_path.is_file():
                            raise ValueError("추적 모델을 먼저 설치해 주세요.")
                        camera.background = camera.frame.copy()
                        camera.mode = "play"
                        camera.landmarks = None
                elif action == "reset-samples":
                    camera.calibrator = Calibrator()
                elif action == "invalidate":
                    camera.generation += 1
                    if camera.calibration:
                        camera.calibration.rvec = camera.calibration.tvec = None
                    camera.landmarks, camera.background = None, None
                    camera.mode = "calibration"
                    camera.floor_validated = False
                    camera.pair_quality = {}
                    hub.calibration_check.reset()
                elif action in ("calibration", "play"):
                    if action == "play":
                        hub.calibration_check.update(hub.cameras, now_ms())
                        if not all(c.connected and c.pair_quality.get("passed")
                                   and now_ms() <= c.pair_quality.get("expires_ms", 0)
                                   for c in hub.cameras.values()) or camera.background is None:
                            raise ValueError(
                                "두 카메라의 품질 검증과 빈 배경 저장을 먼저 완료해 주세요."
                            )
                        if not hub.model_path.is_file():
                            raise ValueError("추적 모델을 먼저 설치해 주세요.")
                    camera.mode = action
                    camera.landmarks = None
                else:
                    if camera.frame is None or now_ms() - camera.capture_ms > 1500:
                        raise ValueError("카메라의 새 영상을 먼저 연결해 주세요.")
                    if action == "sample":
                        await asyncio.to_thread(camera.calibrator.add, camera.frame, camera.profile)
                    elif action == "intrinsics":
                        camera.generation += 1
                        camera.landmarks = None
                        camera.calibration = None
                        camera.floor_validated = False
                        camera.pair_quality = {}
                        camera.background = None
                        camera.mode = "calibration"
                        hub.calibration_check.reset()
                        camera.calibration = await asyncio.to_thread(
                            camera.calibrator.solve_intrinsics
                        )
                        camera.calibration.save(camera.path)
                    elif action == "extrinsics":
                        camera.generation += 1
                        camera.landmarks = None
                        if camera.calibration is None or not camera.calibration.lens_validated:
                            raise ValueError("검증된 렌즈 보정을 먼저 완료해 주세요.")
                        camera.guide.arm_floor(camera)
                        camera.guide.enabled = True
                        camera.mode = "calibration"
                        hub.calibration_check.reset()
                    elif action == "background":
                        hub.calibration_check.update(hub.cameras, now_ms())
                        if not all(c.connected and c.pair_quality.get("passed")
                                   and now_ms() <= c.pair_quality.get("expires_ms", 0)
                                   for c in hub.cameras.values()):
                            raise ValueError("두 카메라의 동시 기준판 검증을 먼저 통과해 주세요.")
                        camera.background = camera.frame.copy()
                    else:
                        raise HTTPException(404)
                if not action.startswith("guide-") and action != "extrinsics":
                    camera.guide.enabled = False
                camera.error = ""
            except (ValueError, cv2.error) as error:
                raise HTTPException(400, str(error)) from error
        return camera.info(now_ms())

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        camera, viewer, worker = None, None, None
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), 5)
            if not isinstance(hello, dict) or not isinstance(hello.get("token"), str):
                await websocket.close(code=1008)
                return
            if not hmac.compare_digest(hello["token"], hub.token):
                await websocket.close(code=1008)
                return
            if hello.get("role") in ("viewer", "depth"):
                viewer = Viewer()
                if hello["role"] == "depth":
                    hub.depth_viewers.add(viewer)
                    if hub.stereo.frame_result and hub.stereo.current(now_ms()).get("capture_ms"):
                        viewer.push("depth", hub.stereo.frame_result["packet"])
                else:
                    hub.viewers.add(viewer)
                    viewer.push("state", hub.snapshot())
                await viewer.run(websocket)
                return
            camera_id = hello.get("camera_id")
            if hello.get("role") != "camera" or camera_id not in CAMERA_IDS:
                await websocket.close(code=1008)
                return
            candidate = hub.cameras[camera_id]
            if candidate.connected:
                await websocket.close(code=1008, reason="이 카메라 번호는 이미 연결됐어요.")
                return
            camera = candidate
            camera.connected = True
            camera.transport = "HTTPS/WSS" if websocket.url.scheme == "wss" else "WS (local test)"
            camera.generation += 1
            camera.capture_ms = 0
            camera.frame_times.clear()
            camera.received_frames = 0
            camera.clock = ClockSync()
            camera.seq = -1
            camera.pending = None
            camera.event.clear()
            camera.landmarks = None
            camera.mode = "calibration"
            camera.reset_guidance()
            if camera.calibration:
                camera.calibration.rvec = camera.calibration.tvec = None
            send_lock = asyncio.Lock()
            worker = asyncio.create_task(hub.process(camera, websocket, send_lock))
            await websocket.send_json({"type": "ready", "camera_id": camera_id})
            probes = {}
            while True:
                receiver = asyncio.create_task(websocket.receive())
                finished, _ = await asyncio.wait(
                    (receiver, worker), return_when=asyncio.FIRST_COMPLETED
                )
                if worker in finished:
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver
                    await worker
                message = receiver.result()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    header, payload = unpack(message["bytes"])
                    frame_header(header)
                    if camera.pending is not None:
                        camera.dropped += 1
                    camera.pending = (header, payload)
                    camera.event.set()
                elif message.get("text") is not None:
                    if len(message["text"]) > 4096:
                        raise ValueError("메시지가 너무 커요.")
                    command = json.loads(message["text"])
                    if not isinstance(command, dict):
                        raise ValueError("메시지는 객체여야 해요.")
                    if command.get("type") == "ping":
                        value = command.get("client_ms")
                        if type(value) not in (int, float):
                            raise ValueError("시계 값이 잘못됐어요.")
                        server_ms = now_ms()
                        if len(probes) >= 16:
                            probes.clear()
                        probes[value] = server_ms
                        async with send_lock:
                            await websocket.send_json(
                                {"type": "pong", "client_ms": value, "server_ms": server_ms}
                            )
                    elif command.get("type") == "sync":
                        sent = command.get("client_send")
                        if type(sent) not in (int, float) or sent not in probes:
                            raise ValueError("시계 확인 요청이 만료됐어요.")
                        camera.clock.update(sent, probes.pop(sent), command.get("client_receive"))
                    else:
                        raise ValueError("지원하지 않는 메시지예요.")
        except (WebSocketDisconnect, asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except (ValueError, TypeError, KeyError) as error:
            LOG.info("Rejected websocket message: %s", error)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=1008)
        except Exception:
            LOG.exception("Camera/viewer connection failed")
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=1011)
        finally:
            if viewer:
                hub.viewers.discard(viewer)
                hub.depth_viewers.discard(viewer)
            if worker:
                # Native inference cannot be cancelled. Let it finish before camera reuse.
                if camera.inflight:
                    async with camera.lock:
                        pass
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker
            if camera:
                camera.connected = False
                camera.generation += 1
                camera.landmarks, camera.pending = None, None
                if camera.calibration:
                    camera.calibration.rvec = camera.calibration.tvec = None

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app

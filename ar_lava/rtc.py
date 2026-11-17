import asyncio
import json
import time
from fractions import Fraction

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fastapi import Depends, HTTPException, Request

from ar_lava.protocol import CAMERA_IDS, ClockSync


def millis():
    return time.monotonic() * 1000


def read_stamp(frame):
    """Decode the browser's 16px footer after video compression; crop it before CV."""
    height, width = frame.shape[:2]
    if width != 640 or height not in (376, 496, 656):
        raise ValueError(f"전송 영상 크기 {width}×{height}를 지원하지 않아요. 다시 연결해 주세요.")
    value = 0
    for bit in range(64):
        block = frame[-12:-4, bit * 8 + 2 : bit * 8 + 6]
        value = (value << 1) | int(np.mean(block) >= 128)
    if value >> 48 != 0xA5C3:
        raise ValueError("프레임 시간 표식을 읽지 못했어요.")
    captured_ms = (value >> 16) & 0xFFFFFFFF
    seq = value & 0xFFFF
    return frame[:-16].copy(), captured_ms, seq


class CompositeTrack(VideoStreamTrack):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.started = time.monotonic()
        self.index = 0

    async def recv(self):
        self.index += 1
        await asyncio.sleep(max(0, self.started + self.index / 15 - time.monotonic()))
        image = self.camera.composite
        if self.camera.camera_id == "cam1" and self.camera.depth_composite is not None:
            image = self.camera.depth_composite
        if image is None or not self.camera.connected or millis() - self.camera.capture_ms > 1500:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self.index * 6000
        frame.time_base = Fraction(1, 90000)
        return frame


class RtcSession:
    def __init__(self, hub, camera=None):
        self.hub, self.camera = hub, camera
        self.peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.tasks = set()
        self.closed = False
        self.probes = {}

    def launch(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def close(self):
        if self.closed:
            return
        self.closed = True
        other_tasks = self.tasks - {asyncio.current_task()}
        for task in other_tasks:
            task.cancel()
        await asyncio.gather(*other_tasks, return_exceptions=True)
        await self.peer.close()
        if self.camera:
            camera = self.camera
            async with camera.lock:
                camera.connected = False
                camera.generation += 1
                camera.pending, camera.landmarks = None, None
                camera.mode = "calibration"
                if camera.calibration:
                    camera.calibration.rvec = camera.calibration.tvec = None

    async def expire(self):
        await asyncio.sleep(20)
        if self.peer.connectionState != "connected":
            await self.close()

    def configure(self, profile):
        camera = self.camera

        @self.peer.on("datachannel")
        def datachannel(channel):
            camera.channel = channel

            @channel.on("message")
            def message(raw):
                try:
                    if not isinstance(raw, str) or len(raw) > 4096:
                        raise ValueError("잘못된 제어 메시지")
                    data = json.loads(raw)
                    if data["type"] == "ping":
                        client = data["client_ms"]
                        if type(client) not in (int, float):
                            raise ValueError("잘못된 시계 값")
                        server = millis()
                        if len(self.probes) > 16:
                            self.probes.clear()
                        self.probes[client] = server
                        channel.send(
                            json.dumps({"type": "pong", "client_ms": client, "server_ms": server})
                        )
                    elif data["type"] == "sync":
                        client = data["client_send"]
                        camera.clock.update(client, self.probes.pop(client), data["client_receive"])
                except (ValueError, TypeError, KeyError):
                    self.launch(self.close())

        @self.peer.on("track")
        def track(received):
            if received.kind == "video":
                self.launch(self.receive(received))
                self.launch(self.process(profile))

    async def receive(self, track):
        camera = self.camera
        try:
            while True:
                frame = await track.recv()
                if camera.pending is not None:
                    camera.dropped += 1
                camera.pending = frame
                camera.event.set()
        except MediaStreamError:
            self.launch(self.close())

    async def process(self, profile):
        camera = self.camera
        while True:
            await camera.event.wait()
            camera.event.clear()
            video, camera.pending = camera.pending, None
            if video is None or camera.clock.offset_ms is None:
                continue
            async with camera.lock:
                camera.inflight = True
                try:
                    frame, client_ms, seq = read_stamp(video.to_ndarray(format="bgr24"))
                    captured_ms = camera.clock.to_server(client_ms)
                    age = millis() - captured_ms
                    if not -25 <= age <= 350:
                        camera.dropped += 1
                        continue
                    camera.profile = profile
                    camera.observation_capture_ms = captured_ms
                    # Shield native inference so disconnect cannot release a camera mid-write.
                    native = asyncio.create_task(
                        asyncio.to_thread(
                            self.hub.analyze_frame,
                            camera,
                            frame,
                            self.hub.game.snapshot(),
                        )
                    )
                    try:
                        await asyncio.shield(native)
                    except asyncio.CancelledError:
                        await native
                        raise
                    camera.capture_ms, camera.seq = captured_ms, seq
                    camera.landmarks = camera.next_landmarks
                    camera.sample_frame = camera.frame
                    camera.sample_segmentation = camera.next_segmentation
                    camera.received()
                    if getattr(camera, "channel", None) and camera.channel.readyState == "open":
                        camera.channel.send(
                            json.dumps(
                                {
                                    "type": "result",
                                    "seq": seq,
                                    "processing_ms": camera.processing_ms,
                                    "age_ms": millis() - captured_ms,
                                    "people": camera.people,
                                    "error": camera.error,
                                }
                            )
                        )
                except (ValueError, RuntimeError) as error:
                    camera.error, camera.landmarks = str(error), None
                finally:
                    camera.inflight = False


def install_rtc(app, hub, authorize):
    sessions = set()

    async def negotiate(session, body):
        sessions.add(session)

        @session.peer.on("connectionstatechange")
        async def state_changed():
            if session.peer.connectionState in ("failed", "closed"):
                await session.close()
                sessions.discard(session)

        try:
            await session.peer.setRemoteDescription(RTCSessionDescription(body["sdp"], "offer"))
            if session.camera is None:
                for camera in hub.cameras.values():
                    session.peer.addTrack(CompositeTrack(camera))
            answer = await session.peer.createAnswer()
            await session.peer.setLocalDescription(answer)
            session.launch(session.expire())
            return {"sdp": session.peer.localDescription.sdp, "type": "answer"}
        except Exception:
            await session.close()
            sessions.discard(session)
            raise

    async def offer_body(request):
        if int(request.headers.get("content-length", "0")) > 100_000:
            raise HTTPException(413)
        try:
            raw = await request.body()
            if len(raw) > 100_000:
                raise ValueError("연결 정보가 너무 커요.")
            body = json.loads(raw)
            if not isinstance(body.get("sdp"), str) or body.get("type") != "offer":
                raise ValueError("연결 정보가 잘못됐어요.")
            return body
        except (ValueError, TypeError, AttributeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/rtc/viewer", dependencies=[Depends(authorize)])
    async def viewer_offer(request: Request):
        if sum(session.camera is None for session in sessions) >= 3:
            raise HTTPException(409, "노트북 화면 연결은 최대 3개예요.")
        body = await offer_body(request)
        return await negotiate(RtcSession(hub), body)

    @app.post("/api/rtc/camera/{camera_id}", dependencies=[Depends(authorize)])
    async def camera_offer(camera_id: str, request: Request):
        if camera_id not in CAMERA_IDS:
            raise HTTPException(404)
        camera = hub.cameras[camera_id]
        if camera.connected:
            raise HTTPException(409, "이 카메라 번호는 이미 연결됐어요.")
        body = await offer_body(request)
        profile = body.get("profile")
        if not isinstance(profile, str) or not 1 <= len(profile) <= 240:
            raise HTTPException(400, "카메라 설정을 확인해 주세요.")
        camera.connected, camera.transport = True, "WebRTC"
        camera.generation += 1
        camera.capture_ms = 0
        camera.frame_times.clear()
        camera.received_frames = 0
        camera.clock = ClockSync()
        camera.landmarks = camera.pending = camera.background = None
        camera.event.clear()
        camera.mode = "calibration"
        camera.reset_guidance()
        if camera.calibration:
            camera.calibration.rvec = camera.calibration.tvec = None
        session = RtcSession(hub, camera)
        session.configure(profile)
        return await negotiate(session, body)

    async def close_all():
        await asyncio.gather(*(session.close() for session in sessions.copy()))
        sessions.clear()

    return close_all

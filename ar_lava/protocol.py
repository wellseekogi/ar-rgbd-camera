import json
import math
import struct

MAX_PACKET = 1_500_000
MAX_HEADER = 4096
CAMERA_IDS = {"cam1", "cam2"}
MAX_AGE_MS = 350


def pack(header: dict, payload: bytes) -> bytes:
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("!I", len(encoded)) + encoded + payload


def unpack(packet: bytes) -> tuple[dict, bytes]:
    if len(packet) < 5 or len(packet) > MAX_PACKET:
        raise ValueError("프레임 크기가 허용 범위를 벗어났어요.")
    size = struct.unpack("!I", packet[:4])[0]
    if not 0 < size <= MAX_HEADER or 4 + size >= len(packet):
        raise ValueError("프레임 헤더가 잘못됐어요.")
    header = json.loads(packet[4 : 4 + size])
    if not isinstance(header, dict):
        raise ValueError("헤더는 객체여야 해요.")
    return header, packet[4 + size :]


def frame_header(header: dict) -> tuple[int, float, str]:
    seq, captured, profile = header.get("seq"), header.get("captured_ms"), header.get("profile")
    if type(seq) is not int or not 0 <= seq <= 2**53 - 1:
        raise ValueError("프레임 번호가 잘못됐어요.")
    if type(captured) not in (int, float) or not math.isfinite(captured) or captured < 0:
        raise ValueError("촬영 시간이 잘못됐어요.")
    if not isinstance(profile, str) or not 1 <= len(profile) <= 240:
        raise ValueError("카메라 프로필이 잘못됐어요.")
    return seq, float(captured), profile


class ClockSync:
    """Use the lowest RTT sample; never compare wall clocks across devices."""

    def __init__(self):
        self.offset_ms = None
        self.rtt_ms = float("inf")
        self.updated_ms = 0.0

    def update(self, client_send: float, server_ms: float, client_receive: float):
        values = (client_send, server_ms, client_receive)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("시계 동기화 값이 잘못됐어요.")
        rtt = client_receive - client_send
        if not 0 <= rtt <= 2000:
            raise ValueError("시계 동기화 응답이 너무 늦어요.")
        if rtt <= self.rtt_ms or server_ms - self.updated_ms > 30_000:
            self.rtt_ms = rtt
            self.offset_ms = server_ms - (client_send + client_receive) / 2
            self.updated_ms = server_ms

    def to_server(self, client_ms: float) -> float:
        if self.offset_ms is None:
            raise ValueError("시계 동기화가 필요해요.")
        return client_ms + self.offset_ms

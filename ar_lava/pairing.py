import ipaddress
import socket
from urllib.parse import urlencode

import cv2
from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from ar_lava.protocol import CAMERA_IDS


def local_addresses():
    addresses = {"127.0.0.1"}
    preferred = None
    try:
        # UDP connect selects a local route without sending a packet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(("192.0.2.1", 9))
            preferred = route.getsockname()[0]
            addresses.add(preferred)
    except OSError:
        pass
    try:
        addresses.update(
            item[4][0]
            for item in socket.getaddrinfo(
                socket.gethostname(),
                None,
                socket.AF_INET,
            )
        )
    except OSError:
        pass
    available = sorted(addresses, key=lambda item: (item != preferred, item == "127.0.0.1", item))
    return available


def phone_url(scheme, address, port, token, camera_id):
    ipaddress.ip_address(address)
    return f"{scheme}://{address}:{port}/#{urlencode({'key': token, 'role': camera_id})}"


def install_pairing(app, hub, authorize):
    @app.get("/api/pairing", dependencies=[Depends(authorize)])
    async def pairing_info():
        return {"addresses": local_addresses()}

    @app.get("/api/pairing/{camera_id}.png", dependencies=[Depends(authorize)])
    async def pairing_qr(camera_id: str, address: str, request: Request):
        if camera_id not in CAMERA_IDS:
            raise HTTPException(404)
        if address not in local_addresses():
            raise HTTPException(400, "노트북의 네트워크 주소를 선택해 주세요.")
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        url = phone_url(request.url.scheme, address, port, hub.token, camera_id)
        qr = cv2.QRCodeEncoder_create().encode(url)
        qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        success, image = cv2.imencode(".png", qr)
        if not success:
            raise HTTPException(500, "QR을 만들지 못했어요.")
        return Response(image.tobytes(), media_type="image/png")

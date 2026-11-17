"""Download the model and generate a printable board and a private local HTTPS CA."""

import argparse
import datetime
import hashlib
import ipaddress
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

from ar_lava.calibration import BOARD  # noqa: E402
from ar_lava.pairing import local_addresses  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
STEREO_URL = (
    "https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/53a84e2/"
    "onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx"
)
STEREO_SHA256 = "2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3"


def local_ips():
    return local_addresses()


def certificates(ips):
    directory = ROOT / "certs"
    directory.mkdir(exist_ok=True)
    ca_key_path, ca_path = directory / "ca-key.pem", directory / "lava-local-ca.crt"
    today = datetime.datetime.now(datetime.UTC)
    ca = None
    if ca_path.exists() and ca_key_path.exists():
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), None)
        try:
            ca.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
            ca.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
        except x509.ExtensionNotFound:
            ca = None
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    if ca is None:
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Lava Lab Local CA")])
        ca = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(today - datetime.timedelta(days=1))
            .not_valid_after(today + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        ca_key_path.write_bytes(
            ca_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Lava Laptop")]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(today - datetime.timedelta(days=1))
        .not_valid_after(today + datetime.timedelta(days=90))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost")]
                + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips],
            ),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    (directory / "server.pem").write_bytes(server.public_bytes(serialization.Encoding.PEM))
    (directory / "server-key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", action="append", default=[])
    parser.add_argument(
        "--depth-device",
        choices=("auto", "cpu", "cuda"),
        default=os.environ.get("LAVA_DEPTH_DEVICE", "cuda"),
    )
    args = parser.parse_args()
    model = ROOT / "models/pose_landmarker_lite.task"
    model.parent.mkdir(exist_ok=True)
    if not model.exists():
        temporary = model.with_suffix(".download")
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Model download is incomplete")
        temporary.replace(model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    stereo_path = ROOT / "models/c_fast_foundationstereo.onnx"
    use_neural = args.depth_device == "cuda"
    if args.depth_device == "auto":
        try:
            import onnxruntime as ort

            use_neural = "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            use_neural = False
    if use_neural and not stereo_path.exists():
        temporary = stereo_path.with_suffix(".download")
        urllib.request.urlretrieve(STEREO_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != STEREO_SHA256:
            raise RuntimeError("Stereo model checksum mismatch")
        temporary.replace(stereo_path)
    if use_neural and hashlib.sha256(stereo_path.read_bytes()).hexdigest() != STEREO_SHA256:
        raise RuntimeError("Stereo model checksum mismatch")
    (ROOT / "models/PROVENANCE.json").write_text(
        json.dumps(
            {
                "pose": {"url": MODEL_URL, "sha256": digest},
                "stereo": {
                    "url": STEREO_URL,
                    "sha256": STEREO_SHA256,
                    "license": "NVIDIA Open Model Agreement",
                    "required_for_cpu": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Pixel dimensions correspond to 40mm squares at 100 pixels per square.
    image = BOARD.generateImage((500, 700), marginSize=0, borderBits=1)
    success, board_png = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Cannot create calibration board")
    (ROOT / "web/board.png").write_bytes(board_png.tobytes())
    ips = sorted(set(local_ips() + args.ip))
    ca = certificates(ips)
    print(f"Model SHA256: {digest}")
    print(f"Install ONLY this public CA certificate on phones: {ca}")
    print("Print board at exactly 200mm x 280mm, no fit-to-page; each square must be 40mm.")
    print("Laptop URLs (same Wi-Fi):")
    for ip in ips:
        print(f"  https://{ip}:8443")


if __name__ == "__main__":
    main()

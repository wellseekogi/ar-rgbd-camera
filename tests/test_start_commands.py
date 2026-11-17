import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows CMD launchers")


@pytest.fixture
def launcher_folder(tmp_path):
    # Exercise real CMD routing without installing packages or opening a server/browser.
    folder = tmp_path / "launcher with spaces"
    scripts = folder / "scripts"
    scripts.mkdir(parents=True)
    for name in ("start.cmd", "start_cpu.cmd", "start_gpu.cmd"):
        shutil.copyfile(ROOT / name, folder / name)
    python_dir = folder / ".venv/Scripts"
    python_dir.mkdir(parents=True)
    shutil.copyfile(sys.executable, python_dir / "python.exe")
    (folder / ".venv/pyvenv.cfg").write_text(
        f"home = {sys.base_prefix}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    modules = folder / "stub_modules"
    modules.mkdir()
    for name in ("fastapi", "uvicorn", "aiortc", "cv2", "mediapipe", "cryptography", "onnxruntime"):
        (modules / f"{name}.py").write_text(
            'import os\nif os.environ.get("LAVA_TEST_MISSING") == __name__:\n'
            '    raise ImportError(__name__)\n',
            encoding="utf-8",
        )
    record = """import json, os, sys
from pathlib import Path
with open(os.environ["LAVA_LAUNCH_LOG"], "a", encoding="utf-8") as log:
    event = {"stage": Path(__file__).stem, "args": sys.argv[1:],
             "device": os.environ.get("LAVA_DEPTH_DEVICE")}
    log.write(json.dumps(event) + "\\n")
if Path(__file__).stem == "prepare":
    sys.exit(int(os.environ.get("LAVA_TEST_PREPARE_EXIT", "0")))
"""
    for name in ("prepare.py", "run.py"):
        (scripts / name).write_text(record, encoding="utf-8")
    (folder / "setup.ps1").write_text(
        "param([string]$DepthDevice)\n"
        '$entry = @{stage="setup"; device=$DepthDevice} | ConvertTo-Json -Compress\n'
        "Add-Content -LiteralPath $env:LAVA_LAUNCH_LOG -Value $entry\n",
        encoding="utf-8",
    )
    return folder, modules


@pytest.mark.parametrize(
    "device,missing,setup",
    [
        ("cpu", "", False),
        ("cuda", "", False),
        ("cpu", "onnxruntime", False),
        ("cuda", "onnxruntime", True),
        ("cpu", "fastapi", True),
        ("cuda", "fastapi", True),
    ],
)
def test_dedicated_cmd_selects_device_for_setup_prepare_and_run(
    launcher_folder,
    device,
    missing,
    setup,
):
    folder, modules = launcher_folder
    log = folder / "events.jsonl"
    opposite = "cuda" if device == "cpu" else "cpu"
    env = dict(
        os.environ,
        PYTHONPATH=str(modules),
        LAVA_LAUNCH_LOG=str(log),
        LAVA_DEPTH_DEVICE=opposite,
        LAVA_TEST_MISSING=missing,
    )
    launcher = "start_cpu.cmd" if device == "cpu" else "start_gpu.cmd"
    completed = subprocess.run(
        [
            os.environ["COMSPEC"],
            "/d",
            "/c",
            "call",
            str(folder / launcher),
            "--http-local",
            "--depth-device",
            opposite,
        ],
        cwd=ROOT,
        env=env,
        input="\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    events = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
    assert [event["stage"] for event in events] == (["setup"] if setup else []) + ["prepare", "run"]
    assert all(event["device"] == device for event in events)
    prepare, run = events[-2:]
    assert prepare["args"] == ["--depth-device", device]
    assert run["args"][-2:] == ["--depth-device", device]
    assert "--open-browser" in run["args"] and "--http-local" in run["args"]


def test_failed_preparation_does_not_start_server(launcher_folder):
    folder, modules = launcher_folder
    log = folder / "events.jsonl"
    env = dict(
        os.environ,
        PYTHONPATH=str(modules),
        LAVA_LAUNCH_LOG=str(log),
        LAVA_TEST_PREPARE_EXIT="7",
    )
    completed = subprocess.run(
        [os.environ["COMSPEC"], "/d", "/c", "call", str(folder / "start_cpu.cmd")],
        cwd=ROOT,
        env=env,
        input="\n",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 1
    events = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
    assert [event["stage"] for event in events] == ["prepare"]

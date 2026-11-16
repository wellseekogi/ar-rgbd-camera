"""Create a source-only handoff, excluding credentials, recorded frames and environments."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = []
    for name in ("ar_lava", "web", "docs", "scripts", "tests", "tools"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.append(path)
    for name in (
        "README.md",
        "run_ui.py",
        "run_api.py",
        "setup.ps1",
        "start.ps1",
        "start.cmd",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-lock.txt",
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        ".oxlintrc.json",
        ".gitignore",
        "playwright.config.js",
        "models/PROVENANCE.json",
        "data/.gitkeep",
    ):
        paths.append(ROOT / name)
    output_dir = ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    archive = output_dir / "ar-floor-is-lava.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(paths):
            package.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        names = package.namelist()
        assert "README.md" in names and "web/board.png" in names
        assert not any("session.key" in name or name.startswith("certs/") for name in names)
    report = {
        "files": len(paths),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()

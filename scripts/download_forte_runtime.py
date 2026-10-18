"""Restore the pinned Windows Vulkan runtime and GGUF weights, checking SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return (
        path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
    )


def safe_extract(archive: Path, destination: Path) -> None:
    """Validate all member paths before extracting the authenticated upstream ZIP."""
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or ":" in name:
                raise ValueError("Unsafe archive member")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink archive member")
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive member outside runtime directory")
        zipped.extractall(destination)


def restore(root: Path, cache: Path | None = None) -> list[str]:
    manifest_path = root / "configs/forte_model_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = json.loads(
        (root / "configs/forte_deployment/config.json").read_text(encoding="utf-8")
    )
    data_root = root / experiment["data_root"]
    destination = data_root / "compact"
    destination.mkdir(parents=True, exist_ok=True)
    checked = []
    for entry in manifest["files"]:
        name = entry["name"]
        if not name or Path(name).name != name or "\\" in name or ":" in name:
            raise ValueError("Invalid download filename")
        target = destination / name
        if not verified(target, entry):
            partial = target.with_name(name + ".download")
            if cache is not None and verified(cache / name, entry):
                shutil.copyfile(cache / name, partial)
            else:
                if not entry["url"].startswith(
                    ("https://github.com/ggml-org/llama.cpp/", "https://huggingface.co/Qwen/")
                ):
                    raise ValueError("Unexpected upstream URL")
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "AI-GYRLAR-runtime-setup"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    with partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
            if not verified(partial, entry):
                raise ValueError(f"Download integrity mismatch: {name}")
            partial.replace(target)
        checked.append(name)
        print(f"SHA256 OK: {name}", flush=True)
        if name.endswith(".zip"):
            safe_extract(target, destination / "runtime")
    runtime = root / experiment["server_command"][0]
    if sha256(runtime) != experiment["runtime_executable_sha256"]:
        raise ValueError("Runtime executable integrity mismatch")
    shutil.copyfile(manifest_path, data_root / "compact_sources.json")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Optional local verified download cache")
    args = parser.parse_args()
    restore(ROOT, args.cache_dir)
    print("Runtime restored. Run: python -m src.forte.service --check")


if __name__ == "__main__":
    main()

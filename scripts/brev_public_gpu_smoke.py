"""Synthetic-only CUDA smoke; no project data, keys, prompts or videos are read.

Run on a Linux GPU VM in a fresh venv with Pillow==12.1.0 installed.
Use --check on any machine to validate the synthetic request without downloads/GPU.
The owned model process always stops on completion; the VM must be stopped separately.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import secrets
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

REVISION = "f982a07559d4a2f6c8744d840bf6fccab30eea96"
MODEL = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
PROJECTOR = "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11669/"
MODEL_BASE = f"https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/{REVISION}/"
ASSETS = [
    (
        RELEASE + "llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "4703ed1a3cd6e8504affb6cb9866f501e96d634d68a002dc9a8251c9e2819b49",
    ),
    (
        RELEASE + "cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "ab3eda89a82096bcba00b6448218c3c11eec2016b0f38d64884dc29736549f26",
    ),
    (MODEL_BASE + MODEL, "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
    (
        MODEL_BASE + PROJECTOR,
        "c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd",
    ),
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, expected: str, directory: Path) -> Path:
    destination = directory / url.rsplit("/", 1)[-1]
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Existing artifact hash mismatch: {destination.name}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(json.dumps({"download": destination.name}), flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"User-Agent": "public-gpu-smoke/1"})
    with opener.open(request, timeout=90) as response, temporary.open("wb") as target:
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
            target.write(chunk)
    if digest(temporary) != expected:
        raise ValueError(f"Downloaded artifact hash mismatch: {destination.name}")
    temporary.replace(destination)
    return destination


def extract_verified_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        root = destination.resolve()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes destination")
            if member.issym() or member.islnk():
                base = target.parent if member.issym() else root
                if not (base / member.linkname).resolve().is_relative_to(root):
                    raise ValueError("Archive link escapes destination")
            if member.isdev() or member.isfifo():
                raise ValueError("Unexpected special archive entry")
        if hasattr(tarfile, "data_filter"):
            bundle.extractall(destination, filter="data")
        else:
            bundle.extractall(destination)


def synthetic_payload() -> dict:
    content = [
        {
            "type": "text",
            "text": (
                "These are 16 synthetic drawings, not real video. Describe the color "
                "and direction of motion of the single rectangle. Return a short JSON "
                'object with keys "color", "direction", "synthetic". '
                "Do not infer any people or real-world events."
            ),
        }
    ]
    for index in range(16):
        frame = Image.new("RGB", (464, 848), "white")
        drawing = ImageDraw.Draw(frame)
        x = 20 + 20 * index
        drawing.rectangle((x, 320, x + 80, 480), fill=(220, 0, 0))
        drawing.text((20, 20), f"SYNTHETIC FRAME {index + 1:02d}", fill="black")
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}
        )
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_k": 1,
        "seed": 2026,
        "max_tokens": 128,
    }


def gpu_info() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-size", choices=("4B", "8B"), default="8B")
    parser.add_argument("--workspace", type=Path, default=Path("qwen-public-smoke"))
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--server-sha256")
    args = parser.parse_args()
    if bool(args.server_binary) != bool(args.server_sha256):
        parser.error("--server-binary and --server-sha256 must be supplied together")
    model_name, projector_name, revision, assets = MODEL, PROJECTOR, REVISION, ASSETS
    if args.model_size == "4B":
        model_name = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
        projector_name = "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
        revision = "1cd86afb9a95c410a6038ab3b40d8b578c892266"
        model_base = f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/{revision}/"
        assets = ASSETS[:2] + [
            (
                model_base + model_name,
                "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
            ),
            (
                model_base + projector_name,
                "30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
            ),
        ]
    payload = synthetic_payload()
    payload["model"] = model_name
    if args.check:
        print(
            json.dumps(
                {
                    "synthetic_frames": 16,
                    "model": model_name,
                    "frame_size": [464, 848],
                    "assets": len(assets),
                    "request_bytes": len(json.dumps(payload).encode()),
                }
            )
        )
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This smoke requires a Linux x86_64 CUDA VM")
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"gpu_before": gpu_info()}), flush=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for url, expected in assets[2:] if args.server_binary else assets:
        artifact = download(url, expected, artifacts)
        if artifact.name.endswith(".tar.gz"):
            extract_verified_archive(artifact, root / "runtime")
    binaries = (
        [args.server_binary.resolve()]
        if args.server_binary
        else list((root / "runtime").rglob("llama-server"))
    )
    if len(binaries) != 1:
        raise RuntimeError("Expected exactly one llama-server executable")
    runtime_sha = digest(binaries[0])
    if args.server_sha256 and runtime_sha != args.server_sha256.lower():
        raise ValueError("Built server executable hash mismatch")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8777))
    run = root / ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run.mkdir(exist_ok=False)
    token = secrets.token_urlsafe(32)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LLAMA_", "MTMD_")) and not key.lower().endswith("_proxy")
    }
    library_root = binaries[0].parent if args.server_binary else root / "runtime"
    library_dirs = sorted({str(p.parent) for p in library_root.rglob("*.so*")})
    environment.update(
        LLAMA_API_KEY=token,
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        LD_LIBRARY_PATH=":".join(library_dirs),
    )
    command = [
        str(binaries[0]),
        "--model",
        str(artifacts / model_name),
        "--mmproj",
        str(artifacts / projector_name),
        "--device",
        "CUDA0",
        "--mmproj-device",
        "CUDA0",
        "--gpu-layers",
        "all",
        "--ctx-size",
        "12288",
        "--parallel",
        "1",
        "--threads",
        "4",
        "--threads-batch",
        "4",
        "--cache-ram",
        "0",
        "--fit",
        "off",
        "--image-min-tokens",
        "64",
        "--image-max-tokens",
        "384",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        "8777",
        "--no-webui",
        "--no-agent",
        "--no-context-shift",
    ]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path: str, value: dict | None = None) -> dict:
        headers = {"Authorization": "Bearer " + token}
        data = None
        if value is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(value).encode()
        req = urllib.request.Request("http://127.0.0.1:8777" + path, data=data, headers=headers)
        with opener.open(req, timeout=180) as response:
            return json.loads(response.read(4 * 1024 * 1024))

    with (run / "model.log").open("wb") as logfile:
        process = subprocess.Popen(command, env=environment, stdout=logfile, stderr=logfile)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Model startup failed; see model.log")
                try:
                    if request("/health").get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Model readiness timed out")
            payload["model"] = request("/v1/models")["data"][0]["id"]
            started = time.monotonic()
            response = request("/v1/chat/completions", payload)
            result = {
                "scope": "Synthetic-only smoke; not competition quality evaluation",
                "runtime_tag": "b11669",
                "runtime_executable_sha256": runtime_sha,
                "runtime_mode": "built_override" if args.server_binary else "official_archive",
                "model_revision": revision,
                "model_filename": model_name,
                "synthetic_frames": 16,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "gpu_after": gpu_info(),
                "response": response,
            }
            (run / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    print(
        json.dumps(
            {
                "owned_model_stopped": True,
                "vm_requires_separate_stop": True,
                "summary": str(run / "summary.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Check the Forte service using synthetic bytes only; never upload competition data.

Example from the repository root (the key and report must be Git-ignored paths):
    python scripts/check_forte_endpoint.py --base-url http://127.0.0.1:8000 \
        --key-file outputs/forte-service/operator-key.txt \
        --output outputs/forte-service-checks/local-check.json

After permission to test an external endpoint, use its HTTPS origin and a NEW
report path. Add ``--upload-mib 29`` to test transport with 29 MiB of generated
invalid bytes; the default is 1 KiB. This checks transport/auth/error handling,
not model quality. A synthetic upload MUST finish as ``failed``, never as a
successful negative prediction. A busy/unready endpoint fails this check.

The script sends a key only in X-API-Key, disables redirects and environment
proxies, enforces an overall deadline of at most 300 seconds, and saves only
aggregate statuses/timings. It prints no response bodies, credentials, or URLs.
Exit codes: 0 checks passed; 1 endpoint check failed; 2 local setup/output error.
Requires the already-used httpx dependency. No model or public tunnel is started.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

REPO = Path(__file__).resolve().parents[1]
MAX_RESPONSE_BYTES = 128 * 1024
PUBLIC_ERROR_CODES = {
    "model_unavailable",
    "deadline_exceeded",
    "input_changed",
    "invalid_result",
    "invalid_video_or_evidence",
    "analysis_failed",
    "server_stopping",
    "result_too_large",
}


class CheckFailure(Exception):
    """Only a fixed, non-sensitive machine code may leave the checking routine."""


def validate_base_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
            or "?" in value
            or "#" in value
            or "\\" in value
            or any(character.isspace() for character in value)
        ):
            raise ValueError
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError
        if url.scheme == "http" and not ipaddress.ip_address(url.hostname).is_loopback:
            raise ValueError
    except ValueError:
        raise ValueError("Use an HTTPS origin or an HTTP literal loopback IP") from None
    return value.rstrip("/")


def ignored_path(value: str) -> Path:
    path = Path(value).resolve()
    try:
        relative = path.relative_to(REPO)
    except ValueError:
        raise ValueError(
            "Key and output paths must be ignored paths inside the repository"
        ) from None
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if checked.returncode != 0:
        raise ValueError("Key and output paths must be ignored by Git")
    return path


def read_key(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(1025)
    if len(data) > 1024:
        raise ValueError("Invalid API key file")
    key = data.decode("utf-8").strip()
    if not 16 <= len(key) <= 256 or any(character.isspace() for character in key):
        raise ValueError("Invalid API key file")
    return key


def synthetic_payload(stream, size: int) -> None:
    header = b"SYNTHETIC INVALID MOV - TRANSPORT CHECK ONLY\n"
    stream.write(header)
    remaining = size - len(header)
    block = bytes(64 * 1024)
    while remaining:
        count = min(remaining, len(block))
        stream.write(block[:count])
        remaining -= count
    stream.seek(0)


async def request(client, method, url, *, want_json=True, **kwargs):
    async with client.stream(method, url, **kwargs) as response:
        status = response.status_code
        if not want_json:
            return status, None
        if status not in {200, 202}:
            return status, {}
        payload = bytearray()
        async for chunk in response.aiter_bytes():
            if len(payload) + len(chunk) > MAX_RESPONSE_BYTES:
                raise CheckFailure("response_too_large")
            payload.extend(chunk)
        try:
            body = json.loads(payload)
        except (ValueError, UnicodeError):
            raise CheckFailure("invalid_response_json") from None
        if not isinstance(body, dict):
            raise CheckFailure("invalid_response_object")
        return status, body


async def check_endpoint(base_url, key, payload, size, *, deadline_seconds=300, transport=None):
    """Return only numeric/status fields; transport injection supports synthetic tests."""
    base_url = validate_base_url(base_url)
    if not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 300:
        raise ValueError("Deadline must be in (0, 300] seconds")
    summary = {
        "ok": False,
        "payload_bytes": size,
        "deadline_seconds": deadline_seconds,
        "health_http_status": None,
        "model_ready": False,
        "ui_http_status": None,
        "unauthenticated_http_status": None,
        "upload_http_status": None,
        "job_http_status": None,
        "job_status": None,
        "error_code": None,
        "verdict_issued": False,
        "poll_count": 0,
        "elapsed_sec": None,
        "job_elapsed_sec": None,
        "failure_code": None,
    }
    started = time.monotonic()
    job_started = None
    try:
        async with asyncio.timeout(deadline_seconds):
            async with httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=min(30, deadline_seconds),
                transport=transport,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            ) as client:
                status, body = await request(client, "GET", base_url + "/healthz")
                summary["health_http_status"] = status
                summary["model_ready"] = body.get("model_ready") is True
                if status != 200 or body.get("status") != "ok" or not summary["model_ready"]:
                    raise CheckFailure("health_not_ready")
                status, _ = await request(client, "GET", base_url + "/", want_json=False)
                summary["ui_http_status"] = status
                if status != 200:
                    raise CheckFailure("ui_unavailable")
                status, _ = await request(
                    client,
                    "GET",
                    base_url + "/api/v1/jobs/" + "0" * 32,
                    want_json=False,
                )
                summary["unauthenticated_http_status"] = status
                if status != 401:
                    raise CheckFailure("authentication_not_enforced")
                job_started = time.monotonic()
                status, body = await request(
                    client,
                    "POST",
                    base_url + "/api/v1/analyze",
                    headers={"X-API-Key": key},
                    files={"file": ("synthetic.mov", payload, "video/quicktime")},
                )
                summary["upload_http_status"] = status
                if status != 202:
                    raise CheckFailure("upload_not_accepted")
                job_id = body.get("job_id")
                if not isinstance(job_id, str) or re.fullmatch("[0-9a-f]{32}", job_id) is None:
                    raise CheckFailure("invalid_job_identifier")
                while True:
                    summary["poll_count"] += 1
                    status, body = await request(
                        client,
                        "GET",
                        base_url + "/api/v1/jobs/" + job_id,
                        headers={"X-API-Key": key},
                    )
                    summary["job_http_status"] = status
                    if status != 200:
                        raise CheckFailure("job_poll_http_error")
                    state = body.get("status")
                    summary["job_status"] = (
                        state if state in {"processing", "failed", "done"} else "unknown"
                    )
                    summary["verdict_issued"] = "verdict" in body
                    if state == "failed":
                        error = body.get("error")
                        code = error.get("code") if isinstance(error, dict) else None
                        summary["error_code"] = (
                            code if code in PUBLIC_ERROR_CODES else "unrecognized_error"
                        )
                        if summary["verdict_issued"]:
                            raise CheckFailure("verdict_on_failed_job")
                        if code not in PUBLIC_ERROR_CODES:
                            raise CheckFailure("invalid_failure_contract")
                        summary["ok"] = True
                        break
                    if state != "processing":
                        raise CheckFailure("synthetic_upload_did_not_fail")
                    if summary["verdict_issued"]:
                        raise CheckFailure("verdict_before_completion")
                    await asyncio.sleep(0.5)
    except CheckFailure as exc:
        summary["failure_code"] = str(exc)
    except TimeoutError:
        summary["failure_code"] = "overall_deadline_exceeded"
    except httpx.HTTPError:
        summary["failure_code"] = "http_transport_error"
    except Exception:
        # Do not print exception text: third-party errors may include requests.
        summary["failure_code"] = "check_internal_error"
    finally:
        summary["elapsed_sec"] = round(time.monotonic() - started, 3)
        if job_started is not None:
            summary["job_elapsed_sec"] = round(time.monotonic() - job_started, 3)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output", required=True, help="New Git-ignored JSON report path")
    parser.add_argument(
        "--upload-mib",
        type=int,
        default=None,
        help="Generated size from 1 to 120 MiB; default 1 KiB",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=300,
        help="Overall check deadline, at most 300 seconds",
    )
    args = parser.parse_args(argv)
    try:
        base_url = validate_base_url(args.base_url)
        key_path, output = ignored_path(args.key_file), ignored_path(args.output)
        if output.exists() or args.upload_mib is not None and not 1 <= args.upload_mib <= 120:
            raise ValueError("Use a new output path and a bounded synthetic size")
        if not math.isfinite(args.deadline_seconds) or not 0 < args.deadline_seconds <= 300:
            raise ValueError("Deadline must be in (0, 300]")
        key = read_key(key_path)
        size = 1024 if args.upload_mib is None else args.upload_mib * 1024 * 1024
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(mode="w+b", dir=output.parent) as payload:
            synthetic_payload(payload, size)
            summary = asyncio.run(
                check_endpoint(
                    base_url,
                    key,
                    payload,
                    size,
                    deadline_seconds=args.deadline_seconds,
                )
            )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=True, indent=2, allow_nan=False)
        print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
        return 0 if summary["ok"] else 1
    except Exception:
        print(json.dumps({"ok": False, "failure_code": "local_setup_or_output_error"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

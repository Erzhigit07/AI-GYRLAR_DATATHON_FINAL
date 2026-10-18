"""Local-only ForteBank API entry point: ``python -m src.forte.api``.

Run exactly one server process; the in-memory admission limit is not shared by
multiple Uvicorn workers. No recognition model is configured by default.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

from src.forte.jobs import AdmissionFull, JobStore


def create_app(
    *,
    api_keys: Mapping[str, str] | None = None,
    adapter: str | None = None,
    directory: str | Path = "outputs/forte-api/uploads",
    max_upload_bytes: int = 128 * 1024 * 1024,
    capacity: int = 2,
    deadline_seconds: float = 300,
    max_result_bytes: int = 64 * 1024 * 1024,
    manual_examples_manifest: str | Path | None = None,
) -> FastAPI:
    """Create an API using a mapping of principal IDs to separate secret keys.

    If omitted, a single operator key is read from FORTE_API_KEY. Configure the
    adapter via this function or FORTE_ANALYZER=trusted_module:function. Missing
    models fail explicitly. No uploaded filename is passed to the analyzer.
    """
    if api_keys is None:
        secret = os.environ.get("FORTE_API_KEY", "")
        api_keys = {"operator": secret}
    if not api_keys or any(not name or len(secret) < 16 for name, secret in api_keys.items()):
        raise ValueError("Configure separate API keys of at least 16 characters")
    if len(set(api_keys.values())) != len(api_keys):
        raise ValueError("Each principal requires a distinct API key")
    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")
    key_hashes = [
        (name, hashlib.sha256(secret.encode()).digest()) for name, secret in api_keys.items()
    ]
    store = JobStore(
        Path(directory),
        adapter=adapter if adapter is not None else os.environ.get("FORTE_ANALYZER") or None,
        capacity=capacity,
        deadline_seconds=deadline_seconds,
        max_result_bytes=max_result_bytes,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.start()
        try:
            yield
        finally:
            store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.jobs = store

    def principal(request: Request) -> str:
        values = request.headers.getlist("x-api-key")
        supplied = values[0] if len(values) == 1 else ""
        supplied_hash = hashlib.sha256(supplied.encode()).digest()
        match = None
        # Compare fixed-size hashes and do not return early for a matching key.
        for name, expected in key_hashes:
            if hmac.compare_digest(supplied_hash, expected):
                match = name
        if match is None:
            raise HTTPException(
                status_code=401,
                detail="invalid_api_key",
                headers={"Cache-Control": "no-store"},
            )
        return match

    @app.get("/api/v1/auth/check")
    async def check_access(request: Request):
        # No model, upload, job reservation, cookie, or echoed credential is needed.
        principal(request)
        return JSONResponse({"authenticated": True}, headers={"Cache-Control": "no-store"})

    if manual_examples_manifest is not None:
        from src.forte.manual_examples import install_example_routes

        install_example_routes(app, principal, manual_examples_manifest)

    async def save_upload(request: Request, destination: Path) -> None:
        count = 0
        body_limit = max_upload_bytes + 64 * 1024  # bounded multipart envelope
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise HTTPException(400, "invalid_content_length") from exc
            if declared < 0:
                raise HTTPException(400, "invalid_content_length")
            if declared > body_limit:
                raise HTTPException(413, "upload_too_large")
        media_type, options = parse_options_header(request.headers.get("content-type", ""))
        boundary = options.get(b"boundary", b"")
        if media_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
            raise HTTPException(400, "multipart_file_required")
        # Stream directly to our private file. Unlike automatic form parsing,
        # cancellation/oversize cannot leave parser-owned temporary uploads open.
        parts = 0
        total = 0
        header_bytes = 0
        header_name = bytearray()
        header_value = bytearray()
        headers: dict[bytes, bytes] = {}
        ready = False
        completed = False

        def part_begin():
            nonlocal parts
            parts += 1
            if parts != 1:
                raise HTTPException(400, "one_file_field_required")

        def header_field(data, start, end):
            nonlocal header_bytes
            header_bytes += end - start
            if header_bytes > 64 * 1024:
                raise HTTPException(413, "multipart_headers_too_large")
            header_name.extend(data[start:end])

        def header_data(data, start, end):
            nonlocal header_bytes
            header_bytes += end - start
            if header_bytes > 64 * 1024:
                raise HTTPException(413, "multipart_headers_too_large")
            header_value.extend(data[start:end])

        def header_end():
            name = bytes(header_name).lower()
            if name in headers:
                raise HTTPException(400, "duplicate_part_header")
            headers[name] = bytes(header_value)
            header_name.clear()
            header_value.clear()

        def headers_finished():
            nonlocal ready
            disposition, fields = parse_options_header(headers.get(b"content-disposition", b""))
            if disposition != b"form-data" or fields.get(b"name") != b"file":
                raise HTTPException(400, "one_file_field_required")
            filename = fields.get(b"filename", b"")
            if not filename.lower().endswith(b".mov"):
                raise HTTPException(415, "mov_file_required")
            ready = True

        def end():
            nonlocal completed
            completed = True

        with destination.open("xb") as output:

            def part_data(data, start, end):
                nonlocal total
                if not ready:
                    raise HTTPException(400, "invalid_multipart")
                total += end - start
                if total > max_upload_bytes:
                    raise HTTPException(413, "upload_too_large")
                output.write(data[start:end])

            parser = MultipartParser(
                boundary,
                {
                    "on_part_begin": part_begin,
                    "on_header_field": header_field,
                    "on_header_value": header_data,
                    "on_header_end": header_end,
                    "on_headers_finished": headers_finished,
                    "on_part_data": part_data,
                    "on_end": end,
                },
            )
            try:
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > body_limit:
                        raise HTTPException(413, "upload_too_large")
                    parser.write(chunk)
                parser.finalize()
            except MultipartParseError as exc:
                raise HTTPException(400, "invalid_multipart") from exc
        if parts != 1 or not completed:
            raise HTTPException(400, "incomplete_multipart")
        if total == 0:
            raise HTTPException(400, "empty_upload")

    @app.post("/api/v1/analyze", status_code=202)
    async def analyze(request: Request):
        received_at = time.monotonic()
        owner = principal(request)  # Authenticate before reading/spooling any body.
        try:
            key, destination, deadline = store.reserve(owner, received_at=received_at)
        except AdmissionFull as exc:
            raise HTTPException(429, "queue_full", headers={"Retry-After": "1"}) from exc
        submitted = False
        try:
            await asyncio.wait_for(
                save_upload(request, destination), timeout=deadline - time.monotonic()
            )
            store.submit(key)
            submitted = True
            return JSONResponse(
                {"job_id": key}, status_code=202, headers={"Cache-Control": "no-store"}
            )
        except TimeoutError as exc:
            raise HTTPException(408, "upload_deadline_exceeded") from exc
        except ClientDisconnect as exc:
            raise HTTPException(400, "upload_disconnected") from exc
        finally:
            if not submitted:
                store.discard(key)

    @app.get("/api/v1/jobs/{job_id}")
    async def status(job_id: str, request: Request):
        owner = principal(request)
        result = store.get(job_id, owner)
        if result is None:
            raise HTTPException(404, "job_not_found")
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/jobs/{job_id}/explanation")
    async def explanation(job_id: str, request: Request):
        owner = principal(request)
        result = store.get(job_id, owner, explanation=True)
        if result is None:
            raise HTTPException(404, "job_not_found", headers={"Cache-Control": "no-store"})
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000, workers=1, access_log=False)


if __name__ == "__main__":
    main()

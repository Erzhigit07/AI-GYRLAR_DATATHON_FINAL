"""Only generated synthetic video fixtures; no local competition files are read."""

import asyncio
import base64
import io
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
httpx = pytest.importorskip("httpx")
av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from src.forte.api import create_app  # noqa: E402
from src.forte.jobs import JobStore  # noqa: E402

# Deliberately public, synthetic fixture credentials, never real API secrets.
KEYS = {"alice": "synthetic-alice-key-only", "bob": "synthetic-bob-key-only"}
AUTH = {"X-API-Key": KEYS["alice"]}


def synthetic_positive(video_path: str, *, deadline: float) -> dict:
    """Test-only adapter, never a recognition fallback."""
    assert Path(video_path).name == "input.mov"
    return {
        "verdict": "violation",
        "analysis_complete": True,
        "uncertain": False,
        "violations": [
            {"type": "synthetic_event", "timecode_sec": 0.25, "description": "Synthetic fixture"}
        ],
    }


def synthetic_uncertain(video_path: str, *, deadline: float) -> dict:
    return {
        "verdict": "no_violation",
        "violations": [],
        "analysis_complete": False,
        "uncertain": True,
    }


def synthetic_negative(video_path: str, *, deadline: float) -> dict:
    return {
        "verdict": "no_violation",
        "violations": [],
        "analysis_complete": True,
        "uncertain": False,
    }


def synthetic_crash(video_path: str, *, deadline: float) -> dict:
    raise RuntimeError("Do not disclose this private exception text")


def synthetic_hang(video_path: str, *, deadline: float) -> dict:
    while True:
        time.sleep(10)


def synthetic_mutation(video_path: str, *, deadline: float) -> dict:
    candidate = synthetic_positive(video_path, deadline=deadline)
    Path(video_path).write_bytes(b"synthetic adapter changed the upload copy")
    return candidate


@pytest.fixture
def synthetic_mov(tmp_path):
    """Generate one second of solid-color frames, explicitly synthetic."""
    path = tmp_path / "synthetic.mov"
    with av.open(str(path), mode="w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (index * 60, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


def make_app(tmp_path, adapter=None, **kwargs):
    return create_app(
        api_keys=KEYS,
        adapter=f"{__name__}:{adapter}" if adapter else None,
        directory=tmp_path / "uploads",
        **kwargs,
    )


def upload(client, data, filename="do-not-expose-original.mov"):
    return client.post(
        "/api/v1/analyze", headers=AUTH, files={"file": (filename, data, "video/quicktime")}
    )


def completed(client, job_id, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.025)
    pytest.fail("Synthetic job did not finish")


def test_api_requires_configuration_and_distinct_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("FORTE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API keys"):
        create_app(directory=tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        create_app(api_keys={"a": KEYS["alice"], "b": KEYS["alice"]}, directory=tmp_path)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, 401),
        ({"X-API-Key": "wrong"}, 401),
        ({"X-API-Key": ""}, 401),
        ([("X-API-Key", KEYS["alice"]), ("X-API-Key", KEYS["alice"])], 401),
        (AUTH, 200),
        ({"X-API-Key": KEYS["bob"]}, 200),
    ],
)
def test_access_check_is_authenticated_private_and_has_no_job_side_effects(
    tmp_path, headers, expected, monkeypatch
):
    monkeypatch.delenv("FORTE_ANALYZER", raising=False)
    app = make_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/v1/auth/check", headers=headers)
        assert response.status_code == expected
        assert response.headers["cache-control"] == "no-store"
        assert "set-cookie" not in response.headers
        assert response.json() == (
            {"authenticated": True} if expected == 200 else {"detail": "invalid_api_key"}
        )
        assert not app.state.jobs._jobs
        assert not list(app.state.jobs.directory.iterdir())
        for secret in KEYS.values():
            assert secret not in response.text


def test_authentication_isolation_and_missing_model(tmp_path, synthetic_mov, monkeypatch):
    monkeypatch.delenv("FORTE_ANALYZER", raising=False)
    with TestClient(make_app(tmp_path)) as client:
        assert client.post("/api/v1/analyze").status_code == 401
        assert client.post("/api/v1/analyze", headers={"X-API-Key": "wrong"}).status_code == 401
        response = upload(client, synthetic_mov)
        assert response.status_code == 202
        assert "set-cookie" not in response.headers
        assert set(response.json()) == {"job_id"}
        key = response.json()["job_id"]
        assert len(key) == 32
        assert client.get(f"/api/v1/jobs/{key}").status_code == 401
        assert (
            client.get(f"/api/v1/jobs/{key}", headers={"X-API-Key": KEYS["bob"]}).status_code == 404
        )
        assert client.get("/api/v1/jobs/unknown", headers=AUTH).status_code == 404
        for hidden in ("/", "/api/v1/jobs", "/media/input.mov", "/docs", "/openapi.json"):
            assert client.get(hidden, headers=AUTH).status_code == 404
        result = completed(client, key)
        assert result == {
            "job_id": key,
            "status": "failed",
            "error": {"code": "model_unavailable"},
        }
        assert not list((tmp_path / "uploads").iterdir())


def test_success_uses_real_server_extracted_jpeg(tmp_path, synthetic_mov):
    with TestClient(make_app(tmp_path, "synthetic_positive")) as client:
        response = upload(client, synthetic_mov)
        assert response.status_code == 202
        result = completed(client, response.json()["job_id"])
        assert result["status"] == "done"
        assert result["verdict"] == "violation"
        event = result["violations"][0]
        assert event["timecode_sec"] == 0.25
        assert 1 <= len(event["evidence_frames"]) <= 3
        frame = event["evidence_frames"][0]
        assert frame["timestamp_sec"] == event["timecode_sec"]
        jpeg = base64.b64decode(frame["image_base64"], validate=True)
        with Image.open(io.BytesIO(jpeg)) as image:
            assert image.format == "JPEG"
            image.load()
        assert not list((tmp_path / "uploads").iterdir())


def test_complete_negative_result_still_requires_decodable_video(tmp_path, synthetic_mov):
    with TestClient(make_app(tmp_path, "synthetic_negative")) as client:
        result = completed(client, upload(client, synthetic_mov).json()["job_id"])
        assert result["status"] == "done"
        assert result["verdict"] == "no_violation"
        assert result["violations"] == []
        corrupt = completed(client, upload(client, b"not a video").json()["job_id"])
        assert corrupt["status"] == "failed"
        assert "verdict" not in corrupt


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        ("synthetic_uncertain", "invalid_result"),
        ("synthetic_crash", "analysis_failed"),
        ("synthetic_mutation", "input_changed"),
    ],
)
def test_errors_do_not_become_negative_verdicts(tmp_path, synthetic_mov, adapter, expected):
    with TestClient(make_app(tmp_path, adapter)) as client:
        result = completed(client, upload(client, synthetic_mov).json()["job_id"])
        assert result["status"] == "failed"
        assert result["error"] == {"code": expected}
        assert "verdict" not in result
        assert "private exception" not in str(result)


def test_corrupt_mov_fails_without_fake_verdict(tmp_path):
    with TestClient(make_app(tmp_path, "synthetic_positive")) as client:
        result = completed(client, upload(client, b"not a video").json()["job_id"])
        assert result["error"] == {"code": "invalid_video_or_evidence"}
        assert "verdict" not in result


@pytest.mark.parametrize(
    ("filename", "data", "expected"),
    [
        ("synthetic.mp4", b"123", 415),
        ("synthetic.mov", b"", 400),
        ("synthetic.mov", b"x" * 65, 413),
    ],
)
def test_upload_validation_and_cleanup(tmp_path, filename, data, expected):
    with TestClient(make_app(tmp_path, max_upload_bytes=64)) as client:
        assert upload(client, data, filename).status_code == expected
        assert not list((tmp_path / "uploads").iterdir())
        # Rejections must release admission, even when multipart parsing failed.
        assert upload(client, b"small").status_code == 202


def test_multiple_parts_and_incomplete_multipart_are_rejected(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=AUTH,
            files=[("file", ("a.mov", b"a")), ("file", ("b.mov", b"b"))],
        )
        assert response.status_code == 400
        response = client.post(
            "/api/v1/analyze",
            headers={**AUTH, "Content-Type": "multipart/form-data; boundary=fixture"},
            content=(
                b'--fixture\r\nContent-Disposition: form-data; name="file"; '
                b'filename="a.mov"\r\n\r\na'
            ),
        )
        assert response.status_code == 400
        assert not list((tmp_path / "uploads").iterdir())


def test_chunked_upload_cannot_bypass_body_limit(tmp_path):
    with TestClient(make_app(tmp_path, max_upload_bytes=64)) as client:
        response = client.post(
            "/api/v1/analyze",
            headers={**AUTH, "Content-Type": "multipart/form-data; boundary=fixture"},
            content=iter([b"x" * 70_000]),
        )
        assert response.status_code == 413
        assert not list((tmp_path / "uploads").iterdir())


def test_admission_includes_uploads(tmp_path):
    app = make_app(tmp_path, capacity=1)
    with TestClient(app) as client:
        key, _, _ = app.state.jobs.reserve("alice", received_at=time.monotonic())
        response = upload(client, b"synthetic")
        assert response.status_code == 429
        app.state.jobs.discard(key)
        assert upload(client, b"synthetic").status_code == 202


def test_upload_time_counts_toward_deadline(tmp_path):
    app = make_app(tmp_path, deadline_seconds=0.1)

    async def slow_request():
        async def body():
            yield (
                b'--fixture\r\nContent-Disposition: form-data; name="file"; '
                b'filename="a.mov"\r\n\r\n'
            )
            await asyncio.sleep(0.3)
            yield b"synthetic\r\n--fixture--\r\n"

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/analyze",
                headers={**AUTH, "Content-Type": "multipart/form-data; boundary=fixture"},
                content=body(),
            )

    with TestClient(app):
        response = asyncio.run(slow_request())
        assert response.status_code == 408
        assert not list((tmp_path / "uploads").iterdir())


def test_queue_deadline_is_measured_from_admission(tmp_path):
    directory = tmp_path / "uploads"
    directory.mkdir()
    store = JobStore(directory, deadline_seconds=0.1)
    key, path, _ = store.reserve("alice", received_at=time.monotonic())
    path.write_bytes(b"synthetic")
    store.submit(key)
    # No worker starts: all elapsed time is queue time.
    time.sleep(0.12)
    assert store.get(key, "alice")["error"] == {"code": "deadline_exceeded"}
    assert not list(directory.iterdir())
    store.close()


def test_hung_adapter_is_terminated_and_worker_recovers(tmp_path, synthetic_mov):
    app = make_app(tmp_path, "synthetic_hang", deadline_seconds=2)
    with TestClient(app) as client:
        started = time.monotonic()
        result = completed(client, upload(client, synthetic_mov).json()["job_id"])
        assert result["error"] == {"code": "deadline_exceeded"}
        assert time.monotonic() - started < 5
        # Change trusted test configuration, not any client-supplied model choice.
        app.state.jobs.adapter = f"{__name__}:synthetic_positive"
        app.state.jobs.deadline_seconds = 15
        result = completed(client, upload(client, synthetic_mov).json()["job_id"])
        assert result["status"] == "done"


def test_result_budget_rejects_oversize_payload(tmp_path, synthetic_mov):
    with pytest.raises(ValueError, match="max_result_bytes"):
        JobStore(tmp_path, max_result_bytes=0)
    with TestClient(make_app(tmp_path, "synthetic_positive", max_result_bytes=1024)) as client:
        result = completed(client, upload(client, synthetic_mov).json()["job_id"])
        assert result["status"] == "failed"
        assert result["error"] == {"code": "result_too_large"}


def test_retention_budget_evicts_oldest_completed_jobs(tmp_path):
    store = JobStore(tmp_path / "uploads", max_results=100, max_result_bytes=1024)
    store.start()
    keys = []
    try:
        for _ in range(25):
            key, path, _ = store.reserve("alice", received_at=time.monotonic())
            path.write_bytes(b"synthetic fixture; missing model should fail")
            store.submit(key)
            until = time.monotonic() + 2
            while store.get(key, "alice")["status"] == "processing":
                assert time.monotonic() < until
                time.sleep(0.005)
            keys.append(key)
        assert store.get(keys[0], "alice") is None
        assert store.get(keys[-1], "alice")["error"] == {"code": "model_unavailable"}
        assert sum(job.result_bytes for job in store._jobs.values()) <= 1024
    finally:
        store.close()

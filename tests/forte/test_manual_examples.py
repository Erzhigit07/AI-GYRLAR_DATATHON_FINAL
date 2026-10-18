"""Manual demos cannot masquerade as model output or expose unauthenticated media."""

import json

import pytest
from fastapi.testclient import TestClient

from src.forte.api import create_app

PUBLIC_TEST_KEY = "synthetic-manual-example-test-key"
AUTH = {"X-API-Key": PUBLIC_TEST_KEY}


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "examples"
    root.mkdir()
    common = {"example_id": "X-20", "result_source": "human_annotation", "model_generated": False}
    (root / "result.json").write_text(json.dumps({**common, "verdict": "violation"}))
    (root / "explanation.json").write_text(
        json.dumps(
            {
                **common,
                "explanation": {
                    "kind": "human_annotation_with_video_evidence",
                    "windows": [],
                    "confidence": None,
                },
            }
        )
    )
    (root / "video.mov").write_bytes(b"synthetic fixture, not a real video")
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "examples": [
                    {
                        "example_id": "X-20",
                        "video_id": "X-20",
                        "title": "Synthetic annotation",
                        "timecode_sec": 5,
                        "result_file": "result.json",
                        "explanation_file": "explanation.json",
                        "video_file": "video.mov",
                        "private_note": "must_not_leak",
                    }
                ],
            }
        )
    )
    return path


def client_for(manifest):
    return TestClient(
        create_app(
            api_keys={"operator": PUBLIC_TEST_KEY},
            directory=manifest.parent / "uploads",
            manual_examples_manifest=manifest,
        )
    )


@pytest.mark.parametrize("suffix", ["", "/X-20", "/X-20/explanation", "/X-20/video"])
def test_examples_require_authentication(manifest, suffix):
    with client_for(manifest) as client:
        response = client.get("/api/v1/examples" + suffix)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"


def test_manual_provenance_and_media_are_available_without_model(manifest):
    with client_for(manifest) as client:
        listing = client.get("/api/v1/examples", headers=AUTH)
        assert listing.status_code == 200
        assert listing.json()["model_generated"] is False
        assert "private_note" not in listing.text and "result_file" not in listing.text
        entry = listing.json()["examples"][0]
        result = client.get(entry["result_url"], headers=AUTH).json()
        assert result["result_source"] == "human_annotation"
        assert result["model_generated"] is False
        xai = client.get(entry["explanation_url"], headers=AUTH).json()["explanation"]
        assert xai["windows"] == [] and xai["confidence"] is None
        media = client.get(entry["video_url"], headers=AUTH)
        assert media.content == b"synthetic fixture, not a real video"
        assert media.headers["cache-control"] == "no-store"
        assert media.headers["content-type"] == "video/quicktime"
        assert client.get("/api/v1/examples/X-99", headers=AUTH).status_code == 404


def test_example_files_cannot_escape_private_directory(manifest):
    (manifest.parent.parent / "outside.mov").write_bytes(b"outside")
    data = json.loads(manifest.read_text())
    data["examples"][0]["video_file"] = "../outside.mov"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside its private directory"):
        client_for(manifest)


def test_manual_examples_must_not_be_identified_as_model_generated(manifest):
    path = manifest.parent / "result.json"
    data = json.loads(path.read_text())
    data["model_generated"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="human provenance"):
        client_for(manifest)


def test_manual_only_launcher_does_not_start_or_require_gpu(tmp_path, monkeypatch):
    import sys

    import uvicorn

    from src.forte import local_analyzer, service

    settings = tmp_path / "settings.json"
    config = {
        "runtime_root": tmp_path,
        "api_port": 8000,
        "host": "127.0.0.1",
        "api_key_file": tmp_path / "key.txt",
        "ui_path": tmp_path / "index.html",
        "deadline_seconds": 300,
        "start_model": False,
    }
    prepared = {
        "settings_path": settings,
        "rules": [],
        "scope_note": "Manual demo",
        "ruleset_id": None,
    }
    monkeypatch.setattr(sys, "argv", ["service"])
    monkeypatch.setattr(service, "load_service_config", lambda _path: config)
    monkeypatch.setattr(service, "prepare_runtime", lambda *_args: prepared)
    monkeypatch.setattr(local_analyzer, "load_settings", lambda _path: {})
    monkeypatch.setattr(service, "get_api_key", lambda _path: PUBLIC_TEST_KEY)
    ports = []
    monkeypatch.setattr(service, "ensure_port_free", lambda port, *_args: ports.append(port))

    def unexpected(*_args):
        pytest.fail("Manual-only mode must not access GPU or construct ModelRuntime")

    monkeypatch.setattr(service, "ensure_gpu_available", unexpected)
    monkeypatch.setattr(service, "ModelRuntime", unexpected)
    created = []
    monkeypatch.setattr(service, "create_service_app", lambda **kwargs: created.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    assert service.main() == 0
    assert ports == [8000]
    assert created[0]["runtime"] is None
    assert (
        json.loads((tmp_path / "active_service.json").read_text())["mode"] == "manual_examples_only"
    )

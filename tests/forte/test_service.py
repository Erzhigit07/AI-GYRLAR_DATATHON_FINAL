"""Synthetic service checks: no competition data, model weights, GPU or network API."""

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("psutil")
pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from src.forte import jobs, service  # noqa: E402

# Intentionally public test values, unrelated to the operator's credentials.
PUBLIC_TEST_KEY = "synthetic-public-service-test-key"
MODEL_TEST_TOKEN = "synthetic-internal-model-test-token"
AUTH = {"X-API-Key": PUBLIC_TEST_KEY}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_configuration(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-repository"
    root.mkdir()
    monkeypatch.setattr(service, "REPO", root)
    # Keep the production resolver itself; substitute its repository default.
    monkeypatch.setattr(service.resolve_path, "__defaults__", (root,))
    prompt = root / "prompt.txt"
    prompt.write_text("SYNTHETIC OBSERVATION INSTRUCTION", encoding="utf-8")
    rules = write_json(
        root / "rules.json",
        {
            "verified_detection_rules": [
                {
                    "rule_id": "synthetic_rule",
                    "enabled": True,
                    "title": "Synthetic visible action",
                    "source": "https://example.invalid/synthetic-rule",
                    "clause": "Synthetic clause",
                    "private_note": "DO_NOT_PUBLISH_THIS_SYNTHETIC_FIELD",
                }
            ]
        },
    )
    hashes = {}
    for relative in ("experiments/forte_qwen_baseline.py", "src/forte/qwen_baseline.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic source fixture\n", encoding="utf-8")
        hashes[relative] = digest(path)
    for name in ("runtime.exe", "model.gguf", "projector.gguf"):
        (root / name).write_bytes(b"Synthetic bytes, never an executable or real model")
    write_json(
        root / "model-data/compact_sources.json",
        {
            "files": [
                {"name": name, "sha256": digest(root / name)}
                for name in ("model.gguf", "projector.gguf")
            ]
        },
    )
    experiment = {
        "output_contract_version": 2,
        "prompt_file": "prompt.txt",
        "prompt_sha256": digest(prompt),
        "rules_file": "rules.json",
        "rules_sha256": digest(rules),
        "rule_ids": ["synthetic_rule"],
        "code_sha256": hashes,
        "runtime_executable_sha256": digest(root / "runtime.exe"),
        "data_root": "model-data",
        "sampling": {"sample_step_sec": 0.5},
        "generation": {"temperature": 0},
        "request_timeout_sec": 10,
        "server_command": [
            "runtime.exe",
            "--model",
            "model.gguf",
            "--mmproj",
            "projector.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            "8776",
            "--offline",
            "--no-webui",
        ],
    }
    experiment_path = write_json(root / "experiment/config.json", experiment)
    ui = root / "index.html"
    ui.write_text("<!doctype html><title>Synthetic service UI</title>", encoding="utf-8")
    config_path = write_json(
        root / "service.json",
        {
            "experiment_config": "experiment/config.json",
            "ui_path": "index.html",
            "runtime_root": "generated",
            "api_key_file": "generated/key.txt",
        },
    )
    return SimpleNamespace(root=root, path=config_path, experiment_path=experiment_path)


def test_paths_remain_repository_relative_after_cwd_changes(synthetic_configuration, monkeypatch):
    fixture = synthetic_configuration
    monkeypatch.chdir(fixture.root.parent)
    config = service.load_service_config(fixture.path)
    assert config["experiment_config"] == fixture.experiment_path
    assert config["runtime_root"] == fixture.root / "generated"
    assert config["api_key_file"] == fixture.root / "generated/key.txt"
    assert config["ui_path"] == fixture.root / "index.html"
    assert all(
        config[key].is_absolute()
        for key in ("experiment_config", "runtime_root", "api_key_file", "ui_path")
    )


@pytest.mark.parametrize(
    "override",
    [
        {"model_port": True},
        {"model_port": 80},
        {"api_port": 65536},
        {"api_port": 8777},
        {"host": "example.invalid"},
        {"deadline_seconds": 301},
        {"deadline_seconds": float("nan")},
        {"minimum_start_ram_gib": False},
        {"api_key": "DO_NOT_ACCEPT_INLINE_CREDENTIALS"},
        {"weights_manifest": "missing.json"},
        {"weights_manifest_sha256": "a" * 64},
        {"weights_manifest": "model-data/compact_sources.json", "weights_manifest_sha256": "bad"},
    ],
)
def test_invalid_configuration_fails_closed(synthetic_configuration, override):
    fixture = synthetic_configuration
    source = json.loads(fixture.path.read_text(encoding="utf-8"))
    write_json(fixture.path, {**source, **override})
    with pytest.raises(ValueError):
        service.load_service_config(fixture.path)


def test_preparation_freezes_sources_and_keeps_secrets_out(synthetic_configuration, monkeypatch):
    fixture = synthetic_configuration
    config = service.load_service_config(fixture.path)
    runtime_dir = fixture.root / "runtime-snapshot"
    monkeypatch.setenv("FORTE_API_KEY", PUBLIC_TEST_KEY)
    monkeypatch.setenv("FORTE_LOCAL_TOKEN", MODEL_TEST_TOKEN)
    before = {path: digest(path) for path in fixture.root.rglob("*") if path.is_file()}
    prepared = service.prepare_runtime(config, runtime_dir)
    settings = json.loads(prepared["settings_path"].read_text(encoding="utf-8"))
    assert prepared["model_command"][prepared["model_command"].index("--host") + 1] == "127.0.0.1"
    assert prepared["model_command"][prepared["model_command"].index("--port") + 1] == "8777"
    assert prepared["rules"][0]["rule_id"] == "synthetic_rule"
    assert "private_note" not in prepared["rules"][0]
    assert settings["prompt"] == "SYNTHETIC OBSERVATION INSTRUCTION"
    for name in ("runner", "parser"):
        frozen = Path(settings[f"{name}_path"])
        assert frozen.is_relative_to(runtime_dir)
        assert digest(frozen) == settings[f"{name}_sha256"]
    assert all(digest(path) == original for path, original in before.items())
    for path in runtime_dir.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            assert PUBLIC_TEST_KEY not in content
            assert MODEL_TEST_TOKEN not in content
    with pytest.raises(FileExistsError):
        service.prepare_runtime(config, runtime_dir)


@pytest.mark.parametrize(
    "relative",
    [
        "prompt.txt",
        "rules.json",
        "experiments/forte_qwen_baseline.py",
        "src/forte/qwen_baseline.py",
        "runtime.exe",
        "model.gguf",
        "projector.gguf",
    ],
)
def test_changed_frozen_input_is_rejected(synthetic_configuration, relative):
    fixture = synthetic_configuration
    config = service.load_service_config(fixture.path)
    (fixture.root / relative).write_bytes(b"Synthetic tampered fixture")
    with pytest.raises(ValueError):
        service.prepare_runtime(config, fixture.root / "rejected-snapshot")


@pytest.mark.parametrize("damage", [None, "manifest", "weights", "revision", "duplicate"])
def test_pinned_candidate_weights_manifest(synthetic_configuration, damage):
    """A new model can replace both weights without altering the baseline manifest."""
    fixture = synthetic_configuration
    baseline_manifest = fixture.root / "model-data/compact_sources.json"
    baseline_digest = digest(baseline_manifest)
    experiment = json.loads(fixture.experiment_path.read_text(encoding="utf-8"))
    experiment["model_revision"] = "public-test-revision"
    names = ("candidate-model.gguf", "candidate-projector.gguf")
    for option, name in zip(("--model", "--mmproj"), names, strict=True):
        (fixture.root / name).write_bytes(b"Different synthetic candidate weight bytes")
        experiment["server_command"][experiment["server_command"].index(option) + 1] = name
    write_json(fixture.experiment_path, experiment)
    manifest = {
        "revision": "wrong-revision" if damage == "revision" else experiment["model_revision"],
        "files": [{"name": name, "sha256": digest(fixture.root / name)} for name in names],
    }
    if damage == "duplicate":
        manifest["files"].append(manifest["files"][0])
    manifest_path = write_json(fixture.root / "candidate-sources.json", manifest)
    source = json.loads(fixture.path.read_text(encoding="utf-8"))
    write_json(
        fixture.path,
        {
            **source,
            "weights_manifest": "candidate-sources.json",
            "weights_manifest_sha256": digest(manifest_path),
        },
    )
    config = service.load_service_config(fixture.path)
    if damage == "manifest":
        manifest_path.write_text("{}", encoding="utf-8")
    if damage == "weights":
        (fixture.root / names[0]).write_bytes(b"Tampered candidate")
    runtime_dir = fixture.root / "candidate-runtime"
    if damage:
        with pytest.raises(ValueError):
            service.prepare_runtime(config, runtime_dir)
    else:
        service.prepare_runtime(config, runtime_dir)
        metadata = json.loads((runtime_dir / "service_metadata.json").read_text(encoding="utf-8"))
        assert metadata["weights_manifest_sha256"] == digest(manifest_path)
        assert metadata["verified_weights_sha256"] == {
            name: digest(fixture.root / name) for name in names
        }
    assert digest(baseline_manifest) == baseline_digest


def test_service_lock_never_replaces_an_existing_owner(tmp_path):
    path = tmp_path / "service.lock"
    with service.service_lock(path):
        owner = path.read_bytes()
        with pytest.raises(RuntimeError, match="lock already exists"):
            with service.service_lock(path):
                pytest.fail("A second owner acquired the service lock")
        assert path.read_bytes() == owner
    assert not path.exists()


def test_occupied_port_is_preserved():
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen(1)
        port = owner.getsockname()[1]
        with pytest.raises(RuntimeError, match="existing service preserved"):
            service.ensure_port_free(port)
        assert owner.getsockname()[1] == port
    service.ensure_port_free(port)


def test_existing_gpu_server_is_preserved(monkeypatch):
    process = SimpleNamespace(info={"name": "llama-server.exe"})
    monkeypatch.setattr(service.psutil, "process_iter", lambda *_args: [process])
    launch = Mock()
    monkeypatch.setattr(service.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Another llama-server"):
        service.ensure_gpu_available(2.5)
    launch.assert_not_called()


def test_operator_key_reused_and_environment_key_never_written(tmp_path, monkeypatch):
    path = tmp_path / "key.txt"
    monkeypatch.delenv("FORTE_API_KEY", raising=False)
    original = service.get_api_key(path)
    assert len(original) >= 16
    assert service.get_api_key(path) == original
    monkeypatch.setenv("FORTE_API_KEY", PUBLIC_TEST_KEY)
    assert service.get_api_key(path) == PUBLIC_TEST_KEY
    assert path.read_text(encoding="utf-8").strip() == original
    monkeypatch.setenv("FORTE_API_KEY", "synthetic key with whitespace")
    with pytest.raises(ValueError):
        service.get_api_key(path)


@pytest.fixture
def service_app(tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_SERVICE_SETTINGS", "synthetic-previous-settings")
    monkeypatch.setenv("FORTE_LOCAL_TOKEN", MODEL_TEST_TOKEN)
    settings = write_json(tmp_path / "settings.json", {})
    ui = tmp_path / "index.html"
    ui.write_text("<!doctype html><title>Synthetic service UI</title>", encoding="utf-8")
    runtime = SimpleNamespace(ready=threading.Event(), close=Mock())
    rules = [{"rule_id": "synthetic_rule", "title": "Synthetic rule", "source": "Synthetic"}]
    app = service.create_service_app(
        api_keys={"synthetic-user": PUBLIC_TEST_KEY},
        settings_path=settings,
        rules=rules,
        ui_path=ui,
        directory=tmp_path / "uploads",
        runtime=runtime,
    )
    return app, runtime


def test_public_routes_expose_neither_credentials_nor_paths(service_app, tmp_path):
    app, runtime = service_app
    with TestClient(app) as client:
        for path in ("/", "/healthz", "/api/v1/rules"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert PUBLIC_TEST_KEY not in response.text
            assert MODEL_TEST_TOKEN not in response.text
            assert str(tmp_path) not in response.text
        assert client.get("/").headers["content-security-policy"].endswith("frame-ancestors 'none'")
        assert client.get("/healthz").json() == {"status": "ok", "model_ready": False}
        assert client.get("/api/v1/rules").json()["scope_note"]
        runtime.ready.set()
        assert client.get("/healthz").json()["model_ready"] is True
        for path in ("/docs", "/openapi.json", "/api/v1/jobs", "/settings.json"):
            assert client.get(path).status_code == 404
    runtime.close.assert_called_once_with()


def test_unready_model_rejects_upload_without_storing_or_analyzing(service_app, monkeypatch):
    app, _runtime = service_app
    analyze = Mock()
    monkeypatch.setattr(jobs, "run_isolated", analyze)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=AUTH,
            files={"file": ("synthetic.mov", b"NOT A REAL VIDEO", "video/quicktime")},
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "model_unavailable"}
        assert response.headers["retry-after"]
        assert not list(app.state.jobs.directory.iterdir())
        assert not app.state.jobs._jobs
        analyze.assert_not_called()


def test_access_check_works_before_model_is_ready(service_app):
    app, runtime = service_app
    runtime.ready.clear()
    with TestClient(app) as client:
        assert client.get("/api/v1/auth/check").status_code == 401
        response = client.get("/api/v1/auth/check", headers=AUTH)
        assert response.status_code == 200
        assert response.json() == {"authenticated": True}
        assert response.headers["cache-control"] == "no-store"
        assert not app.state.jobs._jobs
        assert not list(app.state.jobs.directory.iterdir())


def test_ready_wrapper_preserves_auth_single_admission_and_worker_lifecycle(service_app):
    app, runtime = service_app
    runtime.ready.set()
    with TestClient(app) as client:
        assert app.state.jobs.capacity == 1
        assert app.state.jobs.adapter == "src.forte.local_analyzer:analyze"
        assert app.state.jobs._thread.is_alive()
        assert client.post("/api/v1/analyze").status_code == 401
        assert client.get("/api/v1/jobs/unknown").status_code == 401
        key, _path, _deadline = app.state.jobs.reserve("synthetic-user", received_at=0)
        response = client.post(
            "/api/v1/analyze",
            headers=AUTH,
            files={"file": ("synthetic.mov", b"NOT A REAL VIDEO", "video/quicktime")},
        )
        assert response.status_code == 429
        app.state.jobs.discard(key)
    assert not app.state.jobs._thread.is_alive()
    assert not list(app.state.jobs.directory.iterdir())
    runtime.close.assert_called_once_with()


def test_runtime_start_and_close_use_only_the_owned_process(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    settings = write_json(tmp_path / "settings.json", {"model_id": "before-start"})
    prepared = {
        "control_dir": control,
        "settings_path": settings,
        "config": {"model_port": 8777},
        "model_command": ["SYNTHETIC-NOT-EXECUTABLE"],
    }
    runtime = service.ModelRuntime(prepared, tmp_path, MODEL_TEST_TOKEN)
    owned = Mock()
    owned.poll.return_value = None
    owned.terminate.side_effect = lambda: setattr(owned.poll, "return_value", 0)
    launch = Mock(return_value=owned)
    monkeypatch.setattr(service.subprocess, "Popen", launch)
    port_check = Mock()
    monkeypatch.setattr(service, "ensure_port_free", port_check)
    monkeypatch.setattr(
        runtime,
        "request",
        Mock(
            side_effect=[
                {"status": "ok"},
                {"data": [{"id": "synthetic-loaded-model"}]},
            ]
        ),
    )
    monkeypatch.setenv("HTTPS_PROXY", "http://synthetic.invalid")
    monkeypatch.setenv("LLAMA_UNTRUSTED_OPTION", "synthetic")
    runtime._start_process()
    assert runtime.ready.is_set()
    assert (control / "ready").is_file()
    assert json.loads(settings.read_text(encoding="utf-8"))["model_id"] == "synthetic-loaded-model"
    port_check.assert_called_once_with(8777)
    environment = launch.call_args.kwargs["env"]
    assert environment["LLAMA_API_KEY"] == MODEL_TEST_TOKEN
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert "HTTPS_PROXY" not in environment
    assert "LLAMA_UNTRUSTED_OPTION" not in environment
    runtime.close()
    runtime.close()
    owned.terminate.assert_called_once_with()
    owned.kill.assert_not_called()
    assert not runtime.ready.is_set()
    assert not (control / "ready").exists()


def test_restart_preserves_newly_admitted_request_watchdog(tmp_path, monkeypatch):
    """A request arriving as restart becomes ready must keep its own deadline marker."""
    control = tmp_path / "control"
    control.mkdir()
    old_marker = write_json(control / "active-old.json", {"deadline": 0})
    new_marker = control / "active-new.json"
    runtime = service.ModelRuntime({"control_dir": control}, tmp_path, MODEL_TEST_TOKEN)
    runtime.process = Mock(poll=Mock(return_value=None))
    runtime.stop_event = Mock(wait=Mock(side_effect=[False, True]))
    monkeypatch.setattr(service.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**30))
    monkeypatch.setattr(runtime, "_stop_process", runtime.ready.clear)

    def restart_and_admit_new_request():
        runtime.ready.set()
        write_json(new_marker, {"deadline": 10**12})

    monkeypatch.setattr(runtime, "_start_process", restart_and_admit_new_request)
    runtime._watch()
    assert runtime.restart_count == 1
    assert runtime.ready.is_set()
    assert not old_marker.exists()
    assert new_marker.exists()


def test_only_new_success_resets_consecutive_restart_budget(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(service.time, "monotonic", lambda: clock[0])
    control = tmp_path / "control"
    control.mkdir()
    runtime = service.ModelRuntime({"control_dir": control}, tmp_path, MODEL_TEST_TOKEN)
    runtime.process = Mock(poll=Mock(return_value=None))
    runtime.restart_count = 3
    monkeypatch.setattr(service.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**30))
    monkeypatch.setattr(runtime, "_stop_process", Mock())
    monkeypatch.setattr(runtime, "_start_process", Mock())

    def watchdog_tick():
        runtime.stop_event = Mock(wait=Mock(side_effect=[False, True]))
        runtime._watch()

    first_success = time.monotonic()
    write_json(control / "last_success.json", {"completed_at": first_success})
    watchdog_tick()
    assert runtime.restart_count == 0
    assert runtime.last_success_seen == first_success

    for expected_count in (1, 2):
        write_json(control / "active-expired.json", {"deadline": 0})
        watchdog_tick()
        assert runtime.restart_count == expected_count
        assert runtime.last_success_seen == first_success

    clock[0] += 1.0
    next_success = time.monotonic()
    assert next_success > first_success
    write_json(control / "last_success.json", {"completed_at": next_success})
    watchdog_tick()
    assert runtime.restart_count == 0
    assert runtime.last_success_seen == next_success


def test_old_success_cannot_bypass_consecutive_restart_limit(tmp_path, monkeypatch):
    control = tmp_path / "control"
    control.mkdir()
    runtime = service.ModelRuntime({"control_dir": control}, tmp_path, MODEL_TEST_TOKEN)
    runtime.process = Mock(poll=Mock(return_value=None))
    runtime.last_success_seen = time.monotonic()
    write_json(control / "last_success.json", {"completed_at": runtime.last_success_seen})
    monkeypatch.setattr(service.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**30))
    stop = Mock()
    start = Mock()
    monkeypatch.setattr(runtime, "_stop_process", stop)
    monkeypatch.setattr(runtime, "_start_process", start)
    for expected_count in (1, 2, 3, 4):
        write_json(control / "active-expired.json", {"deadline": 0})
        runtime.stop_event = Mock(wait=Mock(side_effect=[False, True]))
        runtime._watch()
        assert runtime.restart_count == expected_count
    assert start.call_count == 3
    assert stop.call_count == 4
    assert json.loads((tmp_path / "runtime_error.json").read_text(encoding="utf-8")) == {
        "code": "restart_limit"
    }

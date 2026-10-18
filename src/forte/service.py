"""One local Qwen process, the competition API, and a small upload interface.

Run from the repository: python -m src.forte.service --config configs/forte_service.example.json
The model is owned by this launcher; processes belonging to other chats are never stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi.responses import FileResponse, JSONResponse

from src.forte.api import create_app
from src.forte.rule_catalog import catalog_rules

REPO = Path(__file__).resolve().parents[2]
SCOPE_NOTE = (
    "Проверяется ограниченный набор наблюдаемых признаков. "
    "Результат предназначен для проверки человеком и не является окончательным заключением."
)
RULE_TITLES = {
    "prior_customer_cash_exposed": "Деньги предыдущего клиента остаются на рабочем столе",
    "cashier_leaves_active_service": "Кассир покидает рабочее место во время обслуживания",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(value: str, root: Path = REPO) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty path is required")
    return (root / value).resolve()


def load_service_config(path: str | Path) -> dict:
    """Paths are repository-relative, independent of the shell working directory."""
    source = read_json(Path(path))
    allowed = {
        "experiment_config",
        "model_port",
        "api_port",
        "host",
        "runtime_root",
        "api_key_file",
        "ui_path",
        "minimum_start_ram_gib",
        "deadline_seconds",
        "weights_manifest",
        "weights_manifest_sha256",
        "manual_examples_manifest",
        "start_model",
    }
    if set(source) - allowed:
        raise ValueError("Unknown service configuration fields")
    result = {
        "model_port": 8777,
        "api_port": 8000,
        "host": "127.0.0.1",
        "runtime_root": "outputs/forte-service",
        "api_key_file": "outputs/forte-service/operator-key.txt",
        "ui_path": "web/forte_service/index.html",
        "minimum_start_ram_gib": 2.5,
        "deadline_seconds": 300,
        "start_model": True,
        **source,
    }
    if type(result["start_model"]) is not bool:
        raise ValueError("start_model must be boolean")
    for key in ("model_port", "api_port"):
        if type(result[key]) is not int or not 1024 <= result[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if result["model_port"] == result["api_port"]:
        raise ValueError("Model and API need different ports")
    if result["host"] not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError("API host must be 127.0.0.1 or 0.0.0.0")
    if type(result["deadline_seconds"]) not in (int, float) or not (
        30 <= result["deadline_seconds"] <= 300
    ):
        raise ValueError("Deadline must be between 30 and 300 seconds")
    if type(result["minimum_start_ram_gib"]) not in (int, float) or not (
        1 <= result["minimum_start_ram_gib"] <= 128
    ):
        raise ValueError("Invalid minimum_start_ram_gib")
    for key in ("experiment_config", "runtime_root", "api_key_file", "ui_path"):
        if key not in result:
            raise ValueError(f"Missing {key}")
        result[key] = resolve_path(result[key])
    if not result["experiment_config"].is_file() or not result["ui_path"].is_file():
        raise ValueError("Experiment configuration and UI must exist")
    if "manual_examples_manifest" in result:
        result["manual_examples_manifest"] = resolve_path(result["manual_examples_manifest"])
        if not result["manual_examples_manifest"].is_file():
            raise ValueError("Manual example manifest must exist")
    if ("weights_manifest" in source) != ("weights_manifest_sha256" in source):
        raise ValueError("Weights manifest and its SHA256 must be supplied together")
    if "weights_manifest" in source:
        expected = source["weights_manifest_sha256"]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("Invalid weights manifest SHA256")
        result["weights_manifest"] = resolve_path(source["weights_manifest"])
        if not result["weights_manifest"].is_file():
            raise ValueError("Weights manifest must exist")
    return result


def ensure_port_free(port: int, host: str = "127.0.0.1") -> None:
    with socket.socket() as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied; existing service preserved") from exc


def ensure_gpu_available(minimum_ram: float) -> None:
    """Do not compete with an existing local llama-server experiment."""
    for process in psutil.process_iter(["name"]):
        if (process.info.get("name") or "").lower() in {"llama-server", "llama-server.exe"}:
            raise RuntimeError("Another llama-server owns the GPU; wait for its owner")
    if psutil.virtual_memory().available / 2**30 < minimum_ram:
        raise MemoryError("Not enough free RAM; close unused applications before starting")


@contextmanager
def service_lock(path: Path):
    """An existing lock is never silently deleted, even after an unclean exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}, stream
            )
    except FileExistsError as exc:
        raise RuntimeError("Service lock already exists; verify its owning process first") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def get_api_key(path: Path) -> str:
    """Keep the operator secret in the environment or an ignored local file."""
    value = os.environ.get("FORTE_API_KEY", "")
    if not value:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32) + "\n")
            path.chmod(0o600)
        except FileExistsError:
            pass
        value = path.read_text(encoding="utf-8").strip()
    if len(value) < 16 or len(value) > 256 or any(c.isspace() for c in value):
        raise ValueError("API key must contain 16–256 characters without whitespace")
    return value


def prepare_runtime(config: dict, runtime_dir: Path) -> dict:
    """Freeze the evaluated prompt, rules and parser without modifying the experiment."""
    runtime_dir.mkdir(parents=True, exist_ok=False)
    experiment_path = config["experiment_config"]
    experiment = read_json(experiment_path)
    if experiment.get("output_contract_version") != 2:
        raise ValueError("Select a schema-v2 experiment")
    prompt_path = resolve_path(experiment["prompt_file"])
    rules_path = resolve_path(experiment["rules_file"])
    for path, expected in (
        (prompt_path, experiment["prompt_sha256"]),
        (rules_path, experiment["rules_sha256"]),
    ):
        if file_sha(path) != expected:
            raise ValueError(
                "Prompt/rules changed after experiment; select a new frozen configuration"
            )
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    raw_rules = read_json(rules_path)
    catalog = dict(raw_rules)
    if "verified_detection_rules" in catalog:
        catalog["verified_detection_rules"] = [
            {**rule, "title": RULE_TITLES.get(rule["rule_id"], rule.get("title", rule["rule_id"]))}
            for rule in catalog["verified_detection_rules"]
        ]
    rules = catalog_rules(catalog, experiment["rule_ids"])
    scope_note = raw_rules.get("scope_note", SCOPE_NOTE)
    if not isinstance(scope_note, str) or not scope_note.strip() or len(scope_note) > 2000:
        raise ValueError("Invalid rule scope note")
    frozen = {}
    for name, relative in (
        ("runner", "experiments/forte_qwen_baseline.py"),
        ("parser", "src/forte/qwen_baseline.py"),
    ):
        origin = experiment_path.parent / "frozen_sources" / relative
        if not origin.is_file():
            origin = REPO / relative
        payload = origin.read_bytes()
        expected = experiment["code_sha256"][relative]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Frozen {name} integrity mismatch")
        destination = runtime_dir / "snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        frozen[f"{name}_path"] = str(destination.resolve())
        frozen[f"{name}_sha256"] = expected
    command = list(experiment["server_command"])
    command[0] = str(resolve_path(command[0]))
    if file_sha(Path(command[0])) != experiment["runtime_executable_sha256"]:
        raise ValueError("Model executable integrity mismatch")
    for option in ("--model", "--mmproj"):
        index = command.index(option) + 1
        command[index] = str(resolve_path(command[index]))
        if not Path(command[index]).is_file():
            raise ValueError("Local model weights are missing")
    source_path = config.get(
        "weights_manifest", resolve_path(experiment["data_root"]) / "compact_sources.json"
    )
    if "weights_manifest" in config and file_sha(source_path) != config["weights_manifest_sha256"]:
        raise ValueError("Weights manifest integrity mismatch")
    source_info = read_json(source_path)
    if "weights_manifest" in config and (
        not experiment.get("model_revision")
        or source_info.get("revision") != experiment["model_revision"]
    ):
        raise ValueError("Weights manifest revision differs from evaluated experiment")
    entries = source_info.get("files")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("name"), str)
        or not isinstance(entry.get("sha256"), str)
        for entry in entries
    ):
        raise ValueError("Invalid weights manifest entries")
    expected_weights = {entry["name"]: entry["sha256"] for entry in entries}
    if len(expected_weights) != len(entries):
        raise ValueError("Duplicate weights manifest entries")
    verified_weights = {}
    for option in ("--model", "--mmproj"):
        weight = Path(command[command.index(option) + 1])
        if file_sha(weight) != expected_weights.get(weight.name):
            raise ValueError("Model weights integrity mismatch")
        verified_weights[weight.name] = expected_weights[weight.name]
    command[command.index("--port") + 1] = str(config["model_port"])
    command[command.index("--host") + 1] = "127.0.0.1"
    if "--offline" not in command or "--no-webui" not in command:
        raise ValueError("Model must be offline and expose no web UI")
    control = runtime_dir / "control"
    control.mkdir()
    settings = {
        **experiment["sampling"],
        "generation": experiment["generation"],
        "port": config["model_port"],
        "prompt": prompt,
        "model_id": Path(command[command.index("--model") + 1]).name,
        "rule_ids": experiment["rule_ids"],
        "request_timeout_sec": experiment["request_timeout_sec"],
        "evidence_reserve_sec": experiment.get("evidence_reserve_sec", 20),
        "control_dir": str(control.resolve()),
        **frozen,
    }
    settings_path = runtime_dir / "worker_settings.json"
    write_json(settings_path, settings)
    write_json(runtime_dir / "experiment_snapshot.json", experiment)
    write_json(runtime_dir / "rules_snapshot.json", raw_rules)
    write_json(
        runtime_dir / "service_metadata.json",
        {
            "experiment_config_sha256": file_sha(experiment_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_port": config["model_port"],
            "api_port": config["api_port"],
            "capacity": 1,
            "deadline_seconds": config["deadline_seconds"],
            "scope": scope_note,
            "code": frozen,
            "weights_manifest_sha256": file_sha(source_path),
            "verified_weights_sha256": verified_weights,
        },
    )
    return {
        "settings_path": settings_path,
        "model_command": command,
        "rules": rules,
        "scope_note": scope_note,
        "ruleset_id": raw_rules.get("ruleset_id"),
        "control_dir": control,
        "config": config,
    }


class ModelRuntime:
    """Own only one Popen handle; recover stuck requests without touching other processes."""

    def __init__(self, prepared: dict, directory: Path, token: str):
        self.prepared, self.directory, self.token = prepared, directory, token
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.process = None
        self.log = None
        self.monitor = None
        self.generation = 0
        self.restart_count = 0
        self.last_success_seen = 0.0
        self.control = prepared["control_dir"]
        self._process_lock = threading.RLock()

    def request(self, endpoint: str) -> dict:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_args, **_kwargs):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        url = f"http://127.0.0.1:{self.prepared['config']['model_port']}{endpoint}"
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.token})
        with opener.open(req, timeout=2) as response:
            return json.loads(response.read(1024 * 1024))

    def _unready(self):
        self.ready.clear()
        (self.control / "ready").unlink(missing_ok=True)

    def _stop_process(self):
        with self._process_lock:
            self._stop_owned_process()

    def _stop_owned_process(self):
        self._unready()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
            self.log = None

    def _start_process(self):
        with self._process_lock:
            self._start_owned_process()

    def _start_owned_process(self):
        ensure_port_free(self.prepared["config"]["model_port"])
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LLAMA_", "MTMD_")) and not key.lower().endswith("_proxy")
        }
        environment.update(
            LLAMA_API_KEY=self.token,
            HF_HUB_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
        )
        self.generation += 1
        self.log = (self.directory / f"model_{self.generation:03d}.log").open("xb")
        self.process = subprocess.Popen(
            self.prepared["model_command"],
            cwd=REPO,
            env=environment,
            stdout=self.log,
            stderr=self.log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self.process.poll() is not None:
                raise RuntimeError("Local model exited during startup")
            try:
                if self.request("/health").get("status") == "ok":
                    model_id = self.request("/v1/models")["data"][0]["id"]
                    settings = read_json(self.prepared["settings_path"])
                    settings["model_id"] = model_id
                    write_json(self.prepared["settings_path"], settings)
                    (self.control / "ready").write_text("ready\n", encoding="ascii")
                    self.ready.set()
                    return
            except (OSError, ValueError, KeyError, IndexError):
                pass
            self.stop_event.wait(0.25)
        raise TimeoutError("Model startup deadline exceeded")

    def _watch(self):
        low_memory = 0
        while not self.stop_event.wait(0.5):
            try:
                completed_at = read_json(self.control / "last_success.json")["completed_at"]
                if (
                    type(completed_at) in (int, float)
                    and self.last_success_seen < completed_at <= time.monotonic()
                ):
                    self.last_success_seen = completed_at
                    self.restart_count = 0
            except (OSError, ValueError, KeyError, TypeError):
                pass
            low_memory = low_memory + 1 if psutil.virtual_memory().available < 0.25 * 2**30 else 0
            if low_memory >= 6:
                self._stop_process()
                write_json(self.directory / "runtime_error.json", {"code": "low_memory"})
                return
            reset = self.process is None or self.process.poll() is not None
            for path in self.control.glob("active-*.json"):
                try:
                    marker = read_json(path)
                    reset = (
                        reset
                        or marker.get("restart_required") is True
                        or (time.monotonic() >= marker["deadline"])
                    )
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    reset = True
            if reset:
                old_markers = list(self.control.glob("active-*.json"))
                self._stop_process()
                self.restart_count += 1
                if self.restart_count > 3:
                    write_json(self.directory / "runtime_error.json", {"code": "restart_limit"})
                    return
                try:
                    self._start_process()
                    for path in old_markers:
                        path.unlink(missing_ok=True)
                except Exception as exc:
                    self._stop_process()
                    write_json(
                        self.directory / "runtime_error.json",
                        {"code": "restart_failed", "error_type": type(exc).__name__},
                    )
                    return

    def start(self):
        try:
            self._start_process()
        except BaseException:
            self._stop_process()
            raise
        self.monitor = threading.Thread(target=self._watch, daemon=True, name="forte-model")
        self.monitor.start()

    def close(self):
        self.stop_event.set()
        self._unready()
        if self.monitor is not None:
            self.monitor.join(timeout=15)
        self._stop_process()


def create_service_app(
    *,
    api_keys,
    settings_path: Path,
    rules: list[dict],
    ui_path: Path,
    directory: Path,
    runtime=None,
    deadline_seconds=300,
    scope_note=SCOPE_NOTE,
    ruleset_id=None,
    manual_examples_manifest=None,
):
    os.environ["FORTE_SERVICE_SETTINGS"] = str(settings_path.resolve())
    app = create_app(
        api_keys=api_keys,
        adapter="src.forte.local_analyzer:analyze",
        directory=directory,
        capacity=1,
        deadline_seconds=deadline_seconds,
        manual_examples_manifest=manual_examples_manifest,
    )

    def model_ready():
        return runtime is not None and runtime.ready.is_set()

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application):
                yield
        finally:
            if runtime is not None:
                runtime.close()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def model_admission(request, call_next):
        if request.method == "POST" and request.url.path == "/api/v1/analyze" and not model_ready():
            return JSONResponse(
                {"detail": "model_unavailable"},
                status_code=503,
                headers={"Retry-After": "5", "Cache-Control": "no-store"},
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    def index():
        return FileResponse(
            ui_path,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            },
        )

    @app.get("/healthz")
    def health():
        result = {"status": "ok", "model_ready": model_ready()}
        if manual_examples_manifest is not None:
            result["manual_examples_available"] = True
        return result

    @app.get("/api/v1/rules")
    def public_rules():
        return {"rules": rules, "scope_note": scope_note, "ruleset_id": ruleset_id}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/forte_service.example.json"))
    parser.add_argument(
        "--check", action="store_true", help="Verify/snapshot files without starting GPU"
    )
    args = parser.parse_args()
    try:
        config = load_service_config(args.config)
        runtime_root = config["runtime_root"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        with service_lock(runtime_root / "service.lock"):
            prepared = prepare_runtime(config, runtime_root / stamp)
            from src.forte.local_analyzer import load_settings

            load_settings(prepared["settings_path"])
            if args.check:
                print(
                    json.dumps(
                        {"status": "VERIFIED_NO_GPU_RUN", "runtime_dir": str(runtime_root / stamp)}
                    )
                )
                return 0
            ensure_port_free(config["api_port"], config["host"])
            key = get_api_key(config["api_key_file"])
            runtime = None
            try:
                if config["start_model"]:
                    ensure_port_free(config["model_port"])
                    ensure_gpu_available(config["minimum_start_ram_gib"])
                    token = secrets.token_urlsafe(32)
                    os.environ["FORTE_LOCAL_TOKEN"] = token
                    runtime = ModelRuntime(prepared, runtime_root / stamp, token)
                    runtime.start()
                app = create_service_app(
                    api_keys={"operator": key},
                    settings_path=prepared["settings_path"],
                    rules=prepared["rules"],
                    scope_note=prepared["scope_note"],
                    ruleset_id=prepared["ruleset_id"],
                    ui_path=config["ui_path"],
                    directory=runtime_root / stamp / "uploads",
                    runtime=runtime,
                    deadline_seconds=config["deadline_seconds"],
                    manual_examples_manifest=config.get("manual_examples_manifest"),
                )
                write_json(
                    runtime_root / "active_service.json",
                    {
                        "pid": os.getpid(),
                        "runtime_dir": str(runtime_root / stamp),
                        "url": f"http://127.0.0.1:{config['api_port']}",
                        "api_key_file": str(config["api_key_file"]),
                        "mode": "model_and_examples" if runtime else "manual_examples_only",
                    },
                )
                print(f"Service: http://127.0.0.1:{config['api_port']}", flush=True)
                print(f"Operator key file: {config['api_key_file']}", flush=True)
                import uvicorn

                uvicorn.run(
                    app, host=config["host"], port=config["api_port"], workers=1, access_log=False
                )
            finally:
                if runtime is not None:
                    runtime.close()
        return 0
    except (OSError, ValueError, RuntimeError, MemoryError) as exc:
        print(f"Service did not start: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

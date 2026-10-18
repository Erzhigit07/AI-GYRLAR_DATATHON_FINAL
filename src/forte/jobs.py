"""Bounded, local job execution with a deadline that includes queueing.

Adapters are trusted local Python callables, configured by the operator rather than
by API clients. Each runs in a fresh process so a hung adapter cannot keep the
single worker occupied indefinitely. This is process isolation, not a sandbox for
untrusted code. Results are transient and disappear after restart or retention.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class _InputChanged(RuntimeError):
    """The private upload copy must remain byte-identical throughout analysis."""


def _video_hash(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_video_hash(path: str, expected: str) -> None:
    try:
        actual = _video_hash(path)
    except OSError as exc:
        raise _InputChanged from exc
    if actual != expected:
        raise _InputChanged


def _analyze_child(adapter: str | None, video: str, result_file: str, deadline: float) -> None:
    # The adapter must not print images, descriptions, credentials, or raw errors
    # into the HTTP server's logs. Only safe machine-readable status leaves here.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            from src.forte.contracts import ContractError
            from src.forte.evidence import EvidenceError, build_result_with_context
            from src.forte.explanations import build_explanation

            if adapter is None:
                payload = {"status": "failed", "error": {"code": "model_unavailable"}}
            else:
                try:
                    module, separator, function = adapter.partition(":")
                    if not separator or not module or not function:
                        raise ValueError("Invalid adapter configuration")
                    analyze = getattr(importlib.import_module(module), function)
                    original_hash = _video_hash(video)
                    candidate = analyze(video, deadline=deadline)
                    _verify_video_hash(video, original_hash)
                    result, contexts = build_result_with_context(
                        video, candidate, deadline=deadline
                    )
                    explanation = build_explanation(candidate, result, contexts)
                    _verify_video_hash(video, original_hash)
                    if time.monotonic() >= deadline:
                        raise TimeoutError
                    payload = {"status": "done", **result, "_explanation": explanation}
                except TimeoutError:
                    payload = {"status": "failed", "error": {"code": "deadline_exceeded"}}
                except _InputChanged:
                    payload = {"status": "failed", "error": {"code": "input_changed"}}
                except ContractError:
                    payload = {"status": "failed", "error": {"code": "invalid_result"}}
                except EvidenceError:
                    payload = {"status": "failed", "error": {"code": "invalid_video_or_evidence"}}
                except Exception:
                    payload = {"status": "failed", "error": {"code": "analysis_failed"}}
            Path(result_file).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def run_isolated(
    adapter: str | None,
    video_path: str | Path,
    *,
    deadline: float,
    stop: threading.Event | None = None,
) -> dict[str, Any]:
    """Run a trusted adapter, terminate it on deadline, return a safe public status.

    ``deadline`` is absolute ``time.monotonic()`` time, set before upload. A fresh
    spawned process also keeps this behavior consistent on Windows and Linux.
    The adapter must execute in this process, not start detached model workers.
    """
    if adapter is None:
        return {"status": "failed", "error": {"code": "model_unavailable"}}
    if time.monotonic() >= deadline:
        return {"status": "failed", "error": {"code": "deadline_exceeded"}}
    with tempfile.TemporaryDirectory(prefix="result-", dir=Path(video_path).parent) as folder:
        result_file = Path(folder) / "result.json"
        process = multiprocessing.get_context("spawn").Process(
            target=_analyze_child,
            args=(adapter, str(video_path), str(result_file), deadline),
            daemon=True,
        )
        try:
            process.start()
            while process.is_alive():
                if stop is not None and stop.is_set():
                    return {"status": "failed", "error": {"code": "server_stopping"}}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"status": "failed", "error": {"code": "deadline_exceeded"}}
                process.join(timeout=min(0.025, remaining))
            if time.monotonic() >= deadline:
                return {"status": "failed", "error": {"code": "deadline_exceeded"}}
            if process.exitcode != 0 or not result_file.is_file():
                return {"status": "failed", "error": {"code": "analysis_failed"}}
            if result_file.stat().st_size > 32 * 1024 * 1024:
                return {"status": "failed", "error": {"code": "invalid_result"}}
            return json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "failed", "error": {"code": "analysis_failed"}}
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
                process.close()


class AdmissionFull(RuntimeError):
    """All upload/queue/worker slots are occupied."""


@dataclass
class _Job:
    principal: str
    deadline: float
    directory: Path
    state: str = "uploading"
    payload: dict[str, Any] | None = None
    finished: float | None = None
    result_bytes: int = 0


class JobStore:
    """One worker, bounded admission and result retention; no public job listing."""

    def __init__(
        self,
        directory: Path,
        *,
        adapter: str | None = None,
        capacity: int = 2,
        deadline_seconds: float = 300,
        retention_seconds: float = 3600,
        max_results: int = 100,
        max_result_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if capacity < 1 or not 0 < deadline_seconds <= 300:
            raise ValueError("capacity must be positive; deadline must be in (0, 300]")
        if retention_seconds <= 0 or max_results < 1:
            raise ValueError("Result retention limits must be positive")
        if (
            isinstance(max_result_bytes, bool)
            or not isinstance(max_result_bytes, int)
            or max_result_bytes < 1024
        ):
            raise ValueError("max_result_bytes must be an integer of at least 1024")
        self.directory = directory
        self.adapter = adapter
        self.capacity = capacity
        self.deadline_seconds = deadline_seconds
        self.retention_seconds = retention_seconds
        self.max_results = max_results
        self.max_result_bytes = max_result_bytes
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[str] = deque()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._work, name="forte-worker", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError(
                    "The analysis worker did not stop; private uploads were retained"
                )
        with self._condition:
            for job in self._jobs.values():
                self._clean(job)
            self._jobs.clear()
            self._queue.clear()

    def _clean(self, job: _Job) -> None:
        # Every directory here was created by mkdtemp beneath the configured
        # private upload directory. Never delete operator-provided source paths.
        target = job.directory.resolve()
        root = self.directory.resolve()
        if target.parent != root or not target.name.startswith("job-"):
            raise RuntimeError("Refusing cleanup outside this service's private job directories")
        shutil.rmtree(target, ignore_errors=True)

    def _finish(self, job: _Job, payload: dict[str, Any]) -> None:
        # Count serialized bytes once, not on every status poll. Python container
        # overhead is additional; this bounds the dominant base64 JPEG payloads.
        size = len(json.dumps(payload, allow_nan=False).encode("utf-8"))
        if size > self.max_result_bytes:
            payload = {"status": "failed", "error": {"code": "result_too_large"}}
            size = len(json.dumps(payload).encode("utf-8"))
        if payload.get("status") == "done" and time.monotonic() >= job.deadline:
            payload = {"status": "failed", "error": {"code": "deadline_exceeded"}}
            size = len(json.dumps(payload).encode("utf-8"))
        job.payload = payload
        job.result_bytes = size
        job.state = "finished"
        job.finished = time.monotonic()

    def _prune(self) -> None:
        now = time.monotonic()
        for job in self._jobs.values():
            if job.state == "queued" and now >= job.deadline:
                self._finish(job, {"status": "failed", "error": {"code": "deadline_exceeded"}})
                self._clean(job)
        self._queue = deque(key for key in self._queue if self._jobs[key].state == "queued")
        completed = sorted(
            (key for key, job in self._jobs.items() if job.finished is not None),
            key=lambda key: self._jobs[key].finished,
        )
        retained_bytes = sum(self._jobs[key].result_bytes for key in completed)
        for index, key in enumerate(completed):
            job = self._jobs[key]
            if (
                now - job.finished > self.retention_seconds
                or index < len(completed) - self.max_results
                or retained_bytes > self.max_result_bytes
            ):
                retained_bytes -= job.result_bytes
                self._clean(job)
                del self._jobs[key]

    def reserve(self, principal: str, *, received_at: float) -> tuple[str, Path, float]:
        with self._condition:
            self._prune()
            active = sum(job.state != "finished" for job in self._jobs.values())
            if self._stop.is_set() or active >= self.capacity:
                raise AdmissionFull
            key = uuid.uuid4().hex
            folder = Path(tempfile.mkdtemp(prefix="job-", dir=self.directory))
            deadline = received_at + self.deadline_seconds
            self._jobs[key] = _Job(principal=principal, deadline=deadline, directory=folder)
            return key, folder / "input.mov", deadline

    def discard(self, key: str) -> None:
        with self._condition:
            job = self._jobs.pop(key, None)
            if job is not None:
                self._clean(job)

    def submit(self, key: str) -> None:
        with self._condition:
            job = self._jobs[key]
            if time.monotonic() >= job.deadline:
                raise TimeoutError
            job.state = "queued"
            self._queue.append(key)
            self._condition.notify()

    def get(self, key: str, principal: str, *, explanation: bool = False) -> dict[str, Any] | None:
        with self._condition:
            self._prune()
            job = self._jobs.get(key)
            if job is None or job.principal != principal or job.state == "uploading":
                return None
            if job.payload is not None:
                if explanation and job.payload.get("status") == "done":
                    return {
                        "job_id": key,
                        "status": "done",
                        "explanation": job.payload.get("_explanation"),
                    }
                return {
                    "job_id": key,
                    **{
                        name: value for name, value in job.payload.items() if name != "_explanation"
                    },
                }
            if time.monotonic() >= job.deadline:
                return {"job_id": key, "status": "failed", "error": {"code": "deadline_exceeded"}}
            return {"job_id": key, "status": "processing"}

    def _work(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._prune()
                if not self._queue:
                    self._condition.wait(timeout=0.1)
                    continue
                key = self._queue.popleft()
                job = self._jobs[key]
                job.state = "running"
            payload = run_isolated(
                self.adapter, job.directory / "input.mov", deadline=job.deadline, stop=self._stop
            )
            with self._condition:
                self._clean(job)
                self._finish(job, payload)
                self._prune()
                self._condition.notify_all()

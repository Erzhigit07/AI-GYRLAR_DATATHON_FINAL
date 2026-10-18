"""Authenticated, explicitly human-annotated examples, separate from inference."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def install_example_routes(app, principal, manifest_path: str | Path) -> None:
    """Serve only operator-selected files inside a private example directory."""
    path = Path(manifest_path).resolve(strict=True)
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("examples")
    if manifest.get("schema_version") != 1 or not isinstance(rows, list) or len(rows) > 16:
        raise ValueError("Invalid manual example manifest")
    examples = {}
    for row in rows:
        key = row.get("example_id")
        if not isinstance(key, str) or not re.fullmatch(r"X-\d{2}", key) or key in examples:
            raise ValueError("Invalid or duplicate manual example ID")
        files = {}
        for kind in ("result", "explanation", "video"):
            file = (root / row[kind + "_file"]).resolve(strict=True)
            if not file.is_relative_to(root) or not file.is_file():
                raise ValueError("Manual example file must stay inside its private directory")
            files[kind] = file
        for kind in ("result", "explanation"):
            payload = json.loads(files[kind].read_text(encoding="utf-8"))
            if (
                payload.get("example_id") != key
                or payload.get("result_source") != "human_annotation"
                or payload.get("model_generated") is not False
            ):
                raise ValueError("Manual examples must explicitly identify human provenance")
        summary = {
            field: row[field] for field in ("example_id", "title", "video_id", "timecode_sec")
        }
        for kind in ("result", "explanation", "video"):
            suffix = "" if kind == "result" else "/" + kind
            summary[kind + "_url"] = "/api/v1/examples/" + key + suffix
        examples[key] = {"summary": summary, **files}

    def selected(key):
        if key not in examples:
            raise HTTPException(404, "example_not_found", headers=HEADERS)
        return examples[key]

    @app.get("/api/v1/examples")
    def list_examples(request: Request):
        principal(request)
        return JSONResponse(
            {
                "source": "human_annotation",
                "model_generated": False,
                "examples": [item["summary"] for item in examples.values()],
            },
            headers=HEADERS,
        )

    @app.get("/api/v1/examples/{example_id}")
    def example_result(example_id: str, request: Request):
        principal(request)
        return FileResponse(
            selected(example_id)["result"], media_type="application/json", headers=HEADERS
        )

    @app.get("/api/v1/examples/{example_id}/explanation")
    def example_explanation(example_id: str, request: Request):
        principal(request)
        return FileResponse(
            selected(example_id)["explanation"], media_type="application/json", headers=HEADERS
        )

    @app.get("/api/v1/examples/{example_id}/video")
    def example_video(example_id: str, request: Request):
        principal(request)
        return FileResponse(
            selected(example_id)["video"], media_type="video/quicktime", headers=HEADERS
        )

#!/usr/bin/env python3
"""Simple update API server for Astro Ai Plus.

Endpoints:
- GET  /health
- POST /updates
- GET  /updates/latest?app_id=astro-ai-plus&target_os=linux
- POST /updates/decision

Optional auth:
- Set env UPDATE_API_TOKEN=your_token
- Then send header: Authorization: Bearer your_token (required for POST endpoints)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DB_PATH = os.getenv("UPDATE_API_DB", "update_api.db")
HOST = os.getenv("UPDATE_API_HOST", "0.0.0.0")
PORT = int(os.getenv("UPDATE_API_PORT", "8787"))
API_TOKEN = os.getenv("UPDATE_API_TOKEN", "").strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS updates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              app_id TEXT NOT NULL,
              target_os TEXT NOT NULL,
              version TEXT NOT NULL,
              changes_json TEXT NOT NULL,
              update_url TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              app_id TEXT NOT NULL,
              target_os TEXT NOT NULL,
              current_version TEXT NOT NULL,
              offered_version TEXT NOT NULL,
              decision TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "UpdateApi/1.0"

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

    def _require_token_for_post(self) -> bool:
        if not API_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {API_TOKEN}"
        if auth == expected:
            return True
        self._json_response(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._json_response(HTTPStatus.OK, {"ok": True, "time": utc_now()})
            return

        if parsed.path == "/updates/latest":
            query = parse_qs(parsed.query)
            app_id = (query.get("app_id") or [""])[0].strip()
            target_os = (query.get("target_os") or [""])[0].strip().lower()

            if not app_id or not target_os:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "app_id and target_os are required"})
                return

            with db_conn() as conn:
                row = conn.execute(
                    """
                    SELECT version, changes_json, update_url, created_at
                    FROM updates
                    WHERE app_id = ? AND target_os = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (app_id, target_os),
                ).fetchone()

            if row is None:
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "No update found"})
                return

            self._json_response(
                HTTPStatus.OK,
                {
                    "version": row["version"],
                    "changes": json.loads(row["changes_json"]),
                    "update_url": row["update_url"],
                    "created_at": row["created_at"],
                },
            )
            return

        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_token_for_post():
            return

        parsed = urlparse(self.path)

        if parsed.path == "/updates":
            try:
                data = self._read_json_body()
                app_id = str(data["app_id"]).strip()
                target_os = str(data["target_os"]).strip().lower()
                version = str(data["version"]).strip()
                update_url = str(data["update_url"]).strip()
                changes = data.get("changes", [])
                if not isinstance(changes, list):
                    raise ValueError("changes must be a list")
                if not app_id or not target_os or not version or not update_url:
                    raise ValueError("app_id, target_os, version, update_url are required")

                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO updates(app_id, target_os, version, changes_json, update_url, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (app_id, target_os, version, json.dumps(changes, ensure_ascii=True), update_url, utc_now()),
                    )

                self._json_response(HTTPStatus.CREATED, {"ok": True, "message": "Update saved"})
                return
            except (KeyError, ValueError) as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

        if parsed.path == "/updates/decision":
            try:
                data = self._read_json_body()
                app_id = str(data["app_id"]).strip()
                target_os = str(data["target_os"]).strip().lower()
                current_version = str(data["current_version"]).strip()
                offered_version = str(data["offered_version"]).strip()
                decision = str(data["decision"]).strip().lower()
                if decision not in {"update", "later"}:
                    raise ValueError("decision must be 'update' or 'later'")
                if not app_id or not target_os or not current_version or not offered_version:
                    raise ValueError("app_id, target_os, current_version, offered_version are required")

                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO decisions(app_id, target_os, current_version, offered_version, decision, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (app_id, target_os, current_version, offered_version, decision, utc_now()),
                    )

                self._json_response(HTTPStatus.CREATED, {"ok": True, "message": "Decision saved"})
                return
            except (KeyError, ValueError) as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, fmt: str, *args) -> None:
        return


def run() -> None:
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Update API listening on http://{HOST}:{PORT}")
    print("Health: /health")
    print("Latest: /updates/latest?app_id=astro-ai-plus&target_os=linux")
    server.serve_forever()


if __name__ == "__main__":
    run()

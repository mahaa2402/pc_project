#!/usr/bin/env python3
"""Serve visualization static files and POST /upload(/_batch) -> pipeline (PPM + MPI)."""
from __future__ import annotations

import json
import os
import posixpath
import re
import threading
import time
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from upload_pipeline import run_parallel_ppm, write_ppm_p6_rgb


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload.bin")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "upload.bin"


_BATCH_JOBS_LOCK = threading.Lock()
_BATCH_JOBS: dict[str, dict] = {}


def _job_now_ms() -> int:
    return int(time.time() * 1000)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.endswith("/batch_status"):
            job_id = self._get_query_param("job")
            if not job_id:
                self._send_json({"ok": False, "error": "Missing ?job=<id>"}, status=HTTPStatus.BAD_REQUEST)
                return
            with _BATCH_JOBS_LOCK:
                job = _BATCH_JOBS.get(job_id)
            if not job:
                self._send_json({"ok": False, "error": "Unknown job id"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, "job": job})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if not (path.endswith("/upload") or path.endswith("/upload_batch")):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        is_batch = path.endswith("/upload_batch")

        ctype = self.headers.get("Content-Type", "")
        m = re.match(r"multipart/form-data;\s*boundary=(.+)", ctype)
        if not m:
            self._send_json({"error": "Expected multipart/form-data"}, status=HTTPStatus.BAD_REQUEST)
            return

        boundary = m.group(1)
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self._send_json({"error": "Empty request body"}, status=HTTPStatus.BAD_REQUEST)
            return

        body = self.rfile.read(length)
        file_parts = self._extract_file_parts(body, boundary.encode("utf-8"))
        if not file_parts:
            self._send_json(
                {"error": "No file field found (expected multipart with name='image')"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if not is_batch:
            filename, file_bytes = file_parts[0]
            payload, status = self._process_one(filename, file_bytes)
            self._send_json(payload, status=status)
            return

        # Async job: return immediately, poll /batch_status?job=<id>
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "created_ms": _job_now_ms(),
            "status": "running",
            "count_total": len(file_parts),
            "count_done": 0,
            "ok_all": True,
            "results": [],
        }
        with _BATCH_JOBS_LOCK:
            _BATCH_JOBS[job_id] = job

        def worker(parts: list[tuple[str, bytes]]) -> None:
            for filename, file_bytes in parts:
                item, _status = self._process_one(filename, file_bytes)
                with _BATCH_JOBS_LOCK:
                    j = _BATCH_JOBS.get(job_id)
                    if not j:
                        return
                    if not item.get("ok"):
                        j["ok_all"] = False
                    j["results"].append(item)
                    j["count_done"] = len(j["results"])
                    j["updated_ms"] = _job_now_ms()
            with _BATCH_JOBS_LOCK:
                j = _BATCH_JOBS.get(job_id)
                if j:
                    j["status"] = "done"
                    j["updated_ms"] = _job_now_ms()

        t = threading.Thread(target=worker, args=(file_parts,), daemon=True)
        t.start()
        self._send_json({"ok": True, "job_id": job_id, "count": len(file_parts)})

    def _process_one(self, filename: str, file_bytes: bytes) -> tuple[dict, HTTPStatus]:
        safe = _safe_filename(filename)

        here = Path(__file__).resolve().parent
        root = here.parent
        target_dir = root / "images_original"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        rel_saved_as = posixpath.join("images_original", safe)
        stem = Path(safe).stem
        ppm_dir = root / "images_ppm"
        ppm_name = stem + ".ppm"
        ppm_path = ppm_dir / ppm_name
        ppm_rel_web = posixpath.join("images_ppm", ppm_name)
        ppm_rel_mpi = posixpath.join("..", "images_ppm", ppm_name).replace("\\", "/")

        payload: dict = {
            "ok": True,
            "saved_as": rel_saved_as,
            "filename": safe,
            "stem": stem,
            "ppm_saved_as": ppm_rel_web,
            "pipeline_ok": False,
        }

        try:
            write_ppm_p6_rgb(str(target_path), str(ppm_path))
        except Exception as e:
            payload["ok"] = False
            payload["error"] = "PPM conversion failed: " + str(e)
            return payload, HTTPStatus.OK

        ok_mpi, mpi_msg = run_parallel_ppm(here, ppm_rel_mpi)
        payload["pipeline_ok"] = ok_mpi
        if not ok_mpi:
            payload["ok"] = False
            payload["pipeline_error"] = mpi_msg
            return payload, HTTPStatus.OK

        return payload, HTTPStatus.OK

    def _extract_file_parts(self, body: bytes, boundary: bytes) -> list[tuple[str, bytes]]:
        delim = b"--" + boundary
        parts = body.split(delim)
        out: list[tuple[str, bytes]] = []
        for part in parts:
            part = part.strip(b"\r\n")
            if not part or part == b"--":
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            header_bytes = part[:header_end]
            data = part[header_end + 4 :]
            data = data.rstrip(b"\r\n")

            headers = header_bytes.decode("utf-8", errors="replace").split("\r\n")
            cd = ""
            for h in headers:
                if h.lower().startswith("content-disposition:"):
                    cd = h
                    break
            # We accept repeated name="image" parts.
            if 'name="image"' not in cd:
                continue

            m = re.search(r'filename="([^"]*)"', cd)
            filename = m.group(1) if m else "upload.bin"
            out.append((filename, data))
        return out

    def _get_query_param(self, key: str) -> str:
        try:
            from urllib.parse import parse_qs, urlsplit

            qs = parse_qs(urlsplit(self.path).query)
            v = qs.get(key, [""])[0]
            return str(v or "")
        except Exception:
            return ""

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://127.0.0.1:{port}/visualization.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
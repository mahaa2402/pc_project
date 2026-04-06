#!/usr/bin/env python3
"""Serve visualization static files and POST /upload -> pipeline (PPM + MPI)."""
from __future__ import annotations

import json
import os
import posixpath
import re
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from upload_pipeline import run_parallel_ppm, write_ppm_p6_rgb


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload.bin")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "upload.bin"


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if not path.endswith("/upload"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

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
        file_part = self._extract_file_part(body, boundary.encode("utf-8"))
        if not file_part:
            self._send_json({"error": "No file field named 'image' found"}, status=HTTPStatus.BAD_REQUEST)
            return

        filename, file_bytes = file_part
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
            "stem": stem,
            "ppm_saved_as": ppm_rel_web,
            "pipeline_ok": False,
        }

        try:
            write_ppm_p6_rgb(str(target_path), str(ppm_path))
        except Exception as e:
            payload["ok"] = False
            payload["error"] = "PPM conversion failed: " + str(e)
            self._send_json(payload, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        ok_mpi, mpi_msg = run_parallel_ppm(here, ppm_rel_mpi)
        payload["pipeline_ok"] = ok_mpi
        if not ok_mpi:
            payload["pipeline_error"] = mpi_msg
            self._send_json(payload, status=HTTPStatus.OK)
            return

        self._send_json(payload)

    def _extract_file_part(self, body: bytes, boundary: bytes):
        delim = b"--" + boundary
        parts = body.split(delim)
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
            if 'name="image"' not in cd:
                continue

            m = re.search(r'filename="([^"]*)"', cd)
            filename = m.group(1) if m else "upload.bin"
            return filename, data
        return None

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

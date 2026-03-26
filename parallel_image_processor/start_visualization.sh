#!/usr/bin/env bash
# Serves visualization.html and handles POST /upload -> ../images_original/
# Do NOT use: python3 -m http.server (returns 501 on upload)
cd "$(dirname "$0")"
export PORT="${PORT:-8080}"
echo "Open: http://127.0.0.1:${PORT}/visualization.html"
exec python3 server.py

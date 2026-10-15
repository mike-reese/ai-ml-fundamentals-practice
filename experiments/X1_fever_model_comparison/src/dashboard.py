"""Live view of training runs for experiment X1.

Serves one page that polls the results directory and redraws loss and macro-F1 per epoch
for every <run_name>.jsonl that train.py is appending to. Reads the files fresh on every
request; nothing is cached. No dependencies beyond the standard library.

    uv run python src/dashboard.py                 # results/ beside src/, port 8765
    uv run python src/dashboard.py --results DIR --port 9000

Routes
    GET /      the page (dashboard.html, next to this file)
    GET /runs  JSON: [{"run_name", "system", "rows": [record rows in file order]}, ...]
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "dashboard.html"
DEFAULT_RESULTS = HERE.parent / "results"


def read_run(path: Path) -> dict:
    """One run from one .jsonl file. A partial last line (mid-write) is skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # train.py is still writing this line; it will be whole on the next poll
    system = rows[0].get("system") if rows else None
    return {"run_name": path.stem, "system": system, "rows": rows}


def read_runs(results_dir: Path) -> list[dict]:
    if not results_dir.is_dir():
        return []
    return [read_run(p) for p in sorted(results_dir.glob("*.jsonl"))]


def make_handler(results_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/runs":
                body = json.dumps(read_runs(results_dir)).encode()
                self._send(200, "application/json", body)
            elif self.path.split("?")[0] == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # keep the terminal quiet; polls every few seconds
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.results.resolve()))
    print(f"dashboard: http://localhost:{args.port}  (reading {args.results.resolve()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

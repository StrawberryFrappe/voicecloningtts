"""Desktop launcher: runs the API server in-process and opens a native window.

Uses pywebview (Edge WebView2 on Windows). Falls back to the default browser
if pywebview isn't installed or ``--browser`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

from . import APP_NAME, __version__
from .config import get_paths


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port")


def _setup_logging() -> None:
    log_file = get_paths().logs / "vctts.log"
    if sys.stdout is None or sys.stderr is None:
        # pythonw.exe has no console; libraries that print progress bars (model
        # downloads) would crash writing to None, so send them to a file.
        stream = open(get_paths().logs / "console.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=os.environ.get("VCTTS_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vctts", description=f"{APP_NAME} {__version__}")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VCTTS_PORT", "8765")))
    parser.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    parser.add_argument("--server-only", action="store_true", help="run the API server without a UI")
    args = parser.parse_args(argv)

    _setup_logging()
    import uvicorn

    from .main import create_app

    port = _free_port(args.port)
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="info", log_config=None)
    server = uvicorn.Server(config)
    url = f"http://127.0.0.1:{port}/"

    if args.server_only:
        server.run()
        return

    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "api/status", timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    try:
        if args.browser:
            raise ImportError
        import webview
    except ImportError:
        webbrowser.open(url)
        print(f"{APP_NAME} running at {url} - press Ctrl+C to quit")
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            pass
    else:
        webview.create_window(APP_NAME, url, width=1280, height=860, min_size=(900, 600))
        webview.start()
    server.should_exit = True
    thread.join(timeout=10)


if __name__ == "__main__":
    main()

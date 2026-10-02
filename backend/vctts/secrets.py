"""API key storage.

Keys live in the OS credential store (Windows Credential Manager via
``keyring``). Environment variables override stored keys, which is handy for
development. If no keyring backend is available (e.g. a headless Linux box) we
fall back to a JSON file in the data dir with restrictive permissions and log a
warning.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from . import APP_NAME

log = logging.getLogger(__name__)

ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "discord": "DISCORD_BOT_TOKEN",
}


class SecretStore:
    def __init__(self, fallback_file: Path, use_keyring: bool = True):
        self._fallback_file = fallback_file
        self._lock = threading.Lock()
        self._keyring = None
        if use_keyring:
            try:
                import keyring
                from keyring.backends import fail

                if not isinstance(keyring.get_keyring(), fail.Keyring):
                    self._keyring = keyring
            except Exception:  # pragma: no cover - depends on platform
                self._keyring = None
        if self._keyring is None:
            log.warning(
                "No OS keyring available; API keys will be stored in %s", self._fallback_file
            )

    @property
    def backend_name(self) -> str:
        if self._keyring is not None:
            return type(self._keyring.get_keyring()).__name__
        return "file"

    # -- file fallback ---------------------------------------------------
    def _read_file(self) -> dict[str, str]:
        try:
            return json.loads(self._fallback_file.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_file(self, data: dict[str, str]) -> None:
        self._fallback_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._fallback_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), "utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self._fallback_file)

    # -- public API ------------------------------------------------------
    def get(self, name: str) -> str | None:
        env = ENV_VARS.get(name)
        if env and os.environ.get(env):
            return os.environ[env]
        with self._lock:
            if self._keyring is not None:
                try:
                    return self._keyring.get_password(APP_NAME, name)
                except Exception as e:  # pragma: no cover
                    log.error("keyring read failed: %s", e)
                    return None
            return self._read_file().get(name)

    def set(self, name: str, value: str) -> None:
        with self._lock:
            if self._keyring is not None:
                self._keyring.set_password(APP_NAME, name, value)
                return
            data = self._read_file()
            data[name] = value
            self._write_file(data)

    def delete(self, name: str) -> None:
        with self._lock:
            if self._keyring is not None:
                try:
                    self._keyring.delete_password(APP_NAME, name)
                except Exception:
                    pass
                return
            data = self._read_file()
            data.pop(name, None)
            self._write_file(data)

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def source(self, name: str) -> str | None:
        env = ENV_VARS.get(name)
        if env and os.environ.get(env):
            return "env"
        if self.get(name):
            return "stored"
        return None

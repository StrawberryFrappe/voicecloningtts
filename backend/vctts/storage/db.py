"""SQLite persistence for conversations, messages, personas and settings.

All access goes through a single connection guarded by a lock; the workload is
tiny (one local user) so this keeps things simple and safe across the asyncio
loop and worker threads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    persona_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT,
    name TEXT,
    meta TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, seq);
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- helpers ---------------------------------------------------------
    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # -- settings --------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._one("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        self._exec(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    # -- conversations ---------------------------------------------------
    def create_conversation(self, title: str = "New chat", persona_id: str | None = None) -> dict:
        now = time.time()
        cid = new_id()
        self._exec(
            "INSERT INTO conversations(id, title, persona_id, created_at, updated_at) VALUES(?,?,?,?,?)",
            (cid, title, persona_id, now, now),
        )
        return self.get_conversation(cid)  # type: ignore[return-value]

    def get_conversation(self, cid: str) -> dict | None:
        row = self._one("SELECT * FROM conversations WHERE id = ?", (cid,))
        return dict(row) if row else None

    def list_conversations(self) -> list[dict]:
        return [dict(r) for r in self._all("SELECT * FROM conversations ORDER BY updated_at DESC")]

    def update_conversation(self, cid: str, **fields: Any) -> dict | None:
        allowed = {k: v for k, v in fields.items() if k in ("title", "persona_id")}
        if allowed:
            sets = ", ".join(f"{k} = ?" for k in allowed)
            self._exec(
                f"UPDATE conversations SET {sets}, updated_at = ? WHERE id = ?",
                (*allowed.values(), time.time(), cid),
            )
        return self.get_conversation(cid)

    def touch_conversation(self, cid: str) -> None:
        self._exec("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), cid))

    def delete_conversation(self, cid: str) -> None:
        self._exec("DELETE FROM conversations WHERE id = ?", (cid,))

    # -- messages --------------------------------------------------------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        with self._lock:
            row = self._one(
                "SELECT COALESCE(MAX(seq), 0) AS s FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            seq = (row["s"] if row else 0) + 1
            mid = new_id()
            now = time.time()
            self._exec(
                "INSERT INTO messages(id, conversation_id, seq, role, content, tool_calls, tool_call_id, name, meta, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    conversation_id,
                    seq,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls else None,
                    tool_call_id,
                    name,
                    json.dumps(meta) if meta else None,
                    now,
                ),
            )
            self.touch_conversation(conversation_id)
        return self.get_message(mid)  # type: ignore[return-value]

    def get_message(self, mid: str) -> dict | None:
        row = self._one("SELECT * FROM messages WHERE id = ?", (mid,))
        return self._msg(row) if row else None

    def list_messages(self, conversation_id: str) -> list[dict]:
        rows = self._all(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq", (conversation_id,)
        )
        return [self._msg(r) for r in rows]

    def delete_messages_from(self, conversation_id: str, seq: int) -> None:
        """Delete message `seq` and everything after it (used for regenerate/edit)."""
        self._exec(
            "DELETE FROM messages WHERE conversation_id = ? AND seq >= ?", (conversation_id, seq)
        )

    @staticmethod
    def _msg(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["tool_calls"] = json.loads(d["tool_calls"]) if d["tool_calls"] else []
        d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
        return d

    # -- personas --------------------------------------------------------
    def upsert_persona(self, pid: str, data: dict) -> None:
        now = time.time()
        self._exec(
            "INSERT INTO personas(id, data, created_at, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (pid, json.dumps(data), now, now),
        )

    def get_persona(self, pid: str) -> dict | None:
        row = self._one("SELECT data FROM personas WHERE id = ?", (pid,))
        return json.loads(row["data"]) if row else None

    def list_personas(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._all("SELECT data FROM personas ORDER BY created_at")]

    def delete_persona(self, pid: str) -> None:
        self._exec("DELETE FROM personas WHERE id = ?", (pid,))
        self._exec("UPDATE conversations SET persona_id = NULL WHERE persona_id = ?", (pid,))

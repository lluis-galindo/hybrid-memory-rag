import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4.0


def count_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    return text[: int(max_tokens * CHARS_PER_TOKEN)]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqliteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
        self._conn.commit()

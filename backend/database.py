"""SQLite database layer using stdlib sqlite3.

Connections are opened inside dependencies, never at import time.
"""
import sqlite3
from pathlib import Path
from typing import Generator


def get_db_path() -> str:
    from backend.config import DB_PATH  # late import so tests can patch
    return DB_PATH


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables() -> None:
    """Create all tables if they do not exist. Called from lifespan."""
    conn = _connect(get_db_path())
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id),
                filename    TEXT    NOT NULL,
                path        TEXT    NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                status      TEXT    NOT NULL DEFAULT 'processing',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS queries (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id            INTEGER NOT NULL REFERENCES users(id),
                document_id        INTEGER REFERENCES documents(id),
                question           TEXT    NOT NULL,
                question_norm      TEXT    NOT NULL,
                answer             TEXT    NOT NULL,
                citations_json     TEXT    NOT NULL DEFAULT '[]',
                provider           TEXT,
                model              TEXT,
                top_k              INTEGER NOT NULL DEFAULT 4,
                prompt_tokens      INTEGER NOT NULL DEFAULT 0,
                completion_tokens  INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL    NOT NULL DEFAULT 0.0,
                cache_hit          INTEGER NOT NULL DEFAULT 0,
                created_at         TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
    finally:
        conn.close()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency: yield a DB connection, close on exit."""
    conn = _connect(get_db_path())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

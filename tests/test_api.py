"""
Pytest test suite for DocuQuery FastAPI backend.
Covers: 401 without token, 400 on bad input, 413 on oversized upload,
        404 on unknown document, and cache-hit behavior.
LLM calls are mocked with unittest.mock.patch.
"""
import io
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Test-level environment setup (must happen before backend imports) ─────────
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789abcdef")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")

# Use a temporary SQLite file per test session
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"

# Patch DB_PATH before any backend module reads it
import backend.config as _cfg
_cfg.DB_PATH = _tmp_db.name
_cfg.SECRET_KEY = "test-secret-key-for-pytest-only-0123456789abcdef"

from backend.database import create_tables
from backend.main import app

create_tables()

client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _register_and_login(username: str = "testuser", password: str = "testpass123") -> str:
    """Register (or ignore 400 if already exists) then login; return JWT."""
    client.post("/api/auth/register", json={"username": username, "password": password})
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Step 1 / Health ───────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Step 4 / Auth: 401 without token ─────────────────────────────────────────

def test_upload_no_token_returns_401():
    data = {"file": ("doc.txt", b"hello", "text/plain")}
    resp = client.post("/api/documents", files=data)
    assert resp.status_code == 401
    assert "error" in resp.json()


def test_list_documents_no_token_returns_401():
    resp = client.get("/api/documents")
    assert resp.status_code == 401


def test_chat_no_token_returns_401():
    resp = client.post("/api/chat", json={"question": "hello", "top_k": 4})
    assert resp.status_code == 401


# ── Step 4 / Auth: register + login ──────────────────────────────────────────

def test_register_and_login():
    resp = client.post(
        "/api/auth/register",
        json={"username": "newuser99", "password": "securepass"},
    )
    assert resp.status_code == 201
    assert "access_token" in resp.json()

    resp2 = client.post(
        "/api/auth/login",
        json={"username": "newuser99", "password": "securepass"},
    )
    assert resp2.status_code == 200
    assert "access_token" in resp2.json()


def test_login_wrong_password_returns_401():
    client.post("/api/auth/register", json={"username": "wrongpw", "password": "goodpass"})
    resp = client.post("/api/auth/login", json={"username": "wrongpw", "password": "badpass"})
    assert resp.status_code == 401


def test_register_duplicate_returns_400():
    client.post("/api/auth/register", json={"username": "dupuser", "password": "pass12345"})
    resp = client.post("/api/auth/register", json={"username": "dupuser", "password": "pass12345"})
    assert resp.status_code == 400


# ── Step 2 / Validation: bad top_k returns 400 ───────────────────────────────

def test_chat_bad_top_k_zero_returns_400():
    token = _register_and_login("topkuser", "topkpass1")
    resp = client.post(
        "/api/chat",
        json={"question": "test", "top_k": 0},
        headers=_auth(token),
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_chat_bad_top_k_too_large_returns_400():
    token = _register_and_login("topkuser2", "topkpass2")
    resp = client.post(
        "/api/chat",
        json={"question": "test", "top_k": 99},
        headers=_auth(token),
    )
    assert resp.status_code == 400


def test_chat_empty_question_returns_400():
    token = _register_and_login("emptyq", "emptyqpass")
    resp = client.post(
        "/api/chat",
        json={"question": "", "top_k": 4},
        headers=_auth(token),
    )
    assert resp.status_code == 400


# ── Step 2 / Upload: 413 on oversized file ───────────────────────────────────

def test_upload_oversized_file_returns_413():
    token = _register_and_login("bigfile_user", "bigfilepass")
    big_content = b"x" * (10 * 1024 * 1024 + 1)  # 10 MB + 1 byte
    resp = client.post(
        "/api/documents",
        files={"file": ("big.txt", big_content, "text/plain")},
        headers=_auth(token),
    )
    assert resp.status_code == 413
    assert "error" in resp.json()


# ── Step 2 / Upload: bad extension returns 400 ────────────────────────────────

def test_upload_bad_extension_returns_400():
    token = _register_and_login("extuser", "extpassword")
    resp = client.post(
        "/api/documents",
        files={"file": ("script.exe", b"MZ", "application/octet-stream")},
        headers=_auth(token),
    )
    assert resp.status_code == 400


# ── Step 5 / Chat: 404 on unknown document_id ─────────────────────────────────

def test_chat_unknown_document_id_returns_404():
    token = _register_and_login("docuser", "docpassword")
    resp = client.post(
        "/api/chat",
        json={"question": "what is ML?", "document_id": 999999, "top_k": 4},
        headers=_auth(token),
    )
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── Step 6 / Cache hit ────────────────────────────────────────────────────────

_MOCK_RAG_RESULT = {
    "answer": "Machine learning is a subset of AI.",
    "citations": [{"source": "ml_basics.txt", "page": 1, "chunk": 1, "text_snippet": "ML is..."}],
    "prompt_tokens": 120,
    "completion_tokens": 40,
    "estimated_cost_usd": 0.000016,
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
}


def test_cache_hit_on_second_identical_question():
    """Second identical question must return cache_hit=True without calling LLM."""
    token = _register_and_login("cacheuser", "cachepass1")

    with patch("backend.main.run_rag_query", return_value=_MOCK_RAG_RESULT) as mock_rag:
        # First call — LLM is invoked
        resp1 = client.post(
            "/api/chat",
            json={"question": "What is machine learning?", "top_k": 4},
            headers=_auth(token),
        )
        assert resp1.status_code == 200, resp1.text
        data1 = resp1.json()
        assert data1["cache_hit"] is False
        assert mock_rag.call_count == 1

        # Second call — same question, same params
        resp2 = client.post(
            "/api/chat",
            json={"question": "What is machine learning?", "top_k": 4},
            headers=_auth(token),
        )
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()
        assert data2["cache_hit"] is True, "Expected cache hit on second identical question"
        # LLM must NOT have been called again
        assert mock_rag.call_count == 1, "run_rag_query should not be called on a cache hit"
        assert data2["answer"] == _MOCK_RAG_RESULT["answer"]


def test_cache_normalizes_question_whitespace():
    """'  what is ML?  ' and 'what is ml?' hit the same cache entry."""
    token = _register_and_login("normuser", "normpass12")

    with patch("backend.main.run_rag_query", return_value=_MOCK_RAG_RESULT) as mock_rag:
        client.post(
            "/api/chat",
            json={"question": "What is ML?", "top_k": 4},
            headers=_auth(token),
        )
        resp = client.post(
            "/api/chat",
            json={"question": "  what is ml?  ", "top_k": 4},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True
        assert mock_rag.call_count == 1

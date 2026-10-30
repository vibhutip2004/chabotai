"""DocuQuery FastAPI backend — main application.

Implements: API endpoints, SQLite database, JWT authentication,
in-database caching, LLM integration, and background ingest.
"""
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.auth import create_access_token, get_current_user, hash_password, verify_password
from backend.database import create_tables, get_db
from backend.rag import run_rag_query
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and upload directory on startup."""
    create_tables()
    from backend.config import UPLOAD_DIR
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuQuery API",
    description="RAG chatbot backend — capstone project",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI 422 validation errors into consistent 400 JSON."""
    err = exc.errors()[0]
    field = err.get("loc", ["request"])[-1]
    msg = err.get("msg", "Validation error")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": f"Invalid field '{field}': {msg}"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure consistent {"error": "..."} format for all HTTPExceptions."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail},
        headers=exc.headers,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, status_code=200)
def health_check():
    return HealthResponse(status="ok")


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post(
    "/api/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
def register(payload: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (payload.username,)
    ).fetchone()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken.",
        )
    cur = db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (payload.username, hash_password(payload.password)),
    )
    db.commit()
    user_id = cur.lastrowid
    token = create_access_token(user_id, payload.username)
    return TokenResponse(access_token=token)


@app.post(
    "/api/auth/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
)
def login(payload: LoginRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (payload.username,),
    ).fetchone()
    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(row["id"], row["username"])
    return TokenResponse(access_token=token)


# ── Background ingest (Step 8) ────────────────────────────────────────────────

def _ingest_document_bg(doc_id: int, file_path: str, db_path: str) -> None:
    """Run in background: chunk + embed the uploaded file, update status."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        from src.ingest import load_documents, split_chunks, build_vectorstore
        from backend.config import DB_PATH

        chroma_dir = Path(DB_PATH).parent / "chroma_db" / str(doc_id)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        folder = Path(file_path).parent
        docs = load_documents(folder)
        chunks = split_chunks(docs)
        build_vectorstore(chunks, persist_dir=chroma_dir)

        conn.execute(
            "UPDATE documents SET chunk_count = ?, status = ? WHERE id = ?",
            (len(chunks), "ready", doc_id),
        )
        conn.commit()
    except Exception as exc:
        conn.execute(
            "UPDATE documents SET status = ? WHERE id = ?",
            ("failed", doc_id),
        )
        conn.commit()
        print(f"[ingest_bg] doc_id={doc_id} failed: {exc}")
    finally:
        conn.close()


# ── Document endpoints ────────────────────────────────────────────────────────

@app.post(
    "/api/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    from backend.config import UPLOAD_DIR, DB_PATH

    # Validate extension
    filename = file.filename or ""
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .txt, .md",
        )

    # Validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds 10 MB limit ({len(content)} bytes).",
        )

    # Persist file to disk
    user_dir = Path(UPLOAD_DIR) / str(current_user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / filename
    dest.write_bytes(content)

    # Insert DB row (status = processing)
    cur = db.execute(
        "INSERT INTO documents (user_id, filename, path, chunk_count, status) VALUES (?, ?, ?, ?, ?)",
        (current_user["id"], filename, str(dest), 0, "processing"),
    )
    db.commit()
    doc_id = cur.lastrowid

    # Schedule background ingest
    background_tasks.add_task(_ingest_document_bg, doc_id, str(dest), DB_PATH)

    return DocumentUploadResponse(
        id=doc_id,
        filename=filename,
        chunk_count=0,
        status="processing",
        message="Document uploaded. Ingestion started in the background.",
    )


@app.get(
    "/api/documents",
    response_model=List[DocumentResponse],
    status_code=200,
)
def list_documents(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        "SELECT id, filename, chunk_count, status, created_at FROM documents WHERE user_id = ? ORDER BY id DESC",
        (current_user["id"],),
    ).fetchall()
    return [
        DocumentResponse(
            id=r["id"],
            filename=r["filename"],
            chunk_count=r["chunk_count"],
            status=r["status"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _normalize(q: str) -> str:
    return q.lower().strip()


def _lookup_cache(
    db: sqlite3.Connection,
    user_id: int,
    document_id: Optional[int],
    question: str,
    top_k: int,
    provider: str,
) -> Optional[dict]:
    row = db.execute(
        """
        SELECT answer, citations_json, prompt_tokens, completion_tokens,
               estimated_cost_usd, provider, model
        FROM queries
        WHERE user_id = ?
          AND (document_id IS ? OR (document_id IS NULL AND ? IS NULL))
          AND question_norm = ?
          AND top_k = ?
          AND provider = ?
          AND cache_hit = 0
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id, document_id, document_id, _normalize(question), top_k, provider),
    ).fetchone()
    return dict(row) if row else None


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=200,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def chat(
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    from backend.config import GROQ_API_KEY, OPENAI_API_KEY, LLM_PROVIDER

    provider = LLM_PROVIDER or "groq"

    # Validate document_id
    if payload.document_id is not None:
        doc_row = db.execute(
            "SELECT id FROM documents WHERE id = ? AND user_id = ?",
            (payload.document_id, current_user["id"]),
        ).fetchone()
        if doc_row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {payload.document_id} not found.",
            )

    # Cache lookup
    cached = _lookup_cache(
        db,
        current_user["id"],
        payload.document_id,
        payload.question,
        payload.top_k,
        provider,
    )
    if cached:
        citations_raw = json.loads(cached.get("citations_json") or "[]")
        citations = [Citation(**c) for c in citations_raw]
        db.execute(
            """INSERT INTO queries
               (user_id, document_id, question, question_norm, answer,
                citations_json, provider, model, prompt_tokens, completion_tokens,
                estimated_cost_usd, cache_hit, top_k)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (
                current_user["id"], payload.document_id, payload.question,
                _normalize(payload.question), cached["answer"],
                cached.get("citations_json") or "[]",
                cached["provider"], cached["model"],
                cached["prompt_tokens"], cached["completion_tokens"],
                cached["estimated_cost_usd"], payload.top_k,
            ),
        )
        db.commit()
        return ChatResponse(
            answer=cached["answer"],
            citations=citations,
            cache_hit=True,
            provider=cached["provider"],
            model=cached["model"],
            prompt_tokens=cached["prompt_tokens"],
            completion_tokens=cached["completion_tokens"],
            estimated_cost_usd=cached["estimated_cost_usd"],
        )

    # Call LLM via threadpool (non-blocking)
    try:
        result = await run_in_threadpool(
            run_rag_query,
            document_id=payload.document_id,
            question=payload.question,
            top_k=payload.top_k,
            provider=provider,
            groq_api_key=GROQ_API_KEY,
            openai_api_key=OPENAI_API_KEY,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        )

    citations = [Citation(**c) for c in result["citations"]]
    citations_json = json.dumps([c.model_dump() for c in citations])

    # Persist query (cache_hit = 0 = cache source row)
    db.execute(
        """INSERT INTO queries
           (user_id, document_id, question, question_norm, answer,
            citations_json, provider, model, prompt_tokens, completion_tokens,
            estimated_cost_usd, cache_hit, top_k)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)""",
        (
            current_user["id"], payload.document_id, payload.question,
            _normalize(payload.question), result["answer"],
            citations_json, result["provider"], result["model"],
            result["prompt_tokens"], result["completion_tokens"],
            result["estimated_cost_usd"], payload.top_k,
        ),
    )
    db.commit()

    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        cache_hit=False,
        provider=result["provider"],
        model=result["model"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        estimated_cost_usd=result["estimated_cost_usd"],
    )
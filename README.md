# Chatbotai: RAG Document Q&A Backend

Upload PDF, TXT, or MD documents and ask questions about them. Every answer comes with source citations.

## Problem and 10x claim

Finding an answer in long notes or PDFs means skimming pages for minutes. DocuQuery returns a cited answer in seconds.

## Concepts implemented

| Concept | Where it lives |
|---|---|
| API endpoints (FastAPI, Pydantic validation, correct status codes) | `backend/main.py` |
| Database (SQLite: users, documents, queries) | `backend/database.py` |
| Authentication (JWT, hashed passwords, protected routes) | `backend/auth.py` |
| Caching (answer cache keyed on user, document, question, top_k, provider) | `backend/main.py` (`_lookup_cache`) |
| LLM integration with cost log (token usage saved per query) | `backend/rag.py`, `queries` table |

Swaps used: none. Background ingestion (Step 8, Option A) is also implemented in `backend/main.py` (`_ingest_document_bg`).

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements-backend.txt
copy .env.example .env
```

Edit `.env`:
- `SECRET_KEY`: generate one with `python -c "import secrets; print(secrets.token_hex(32))"` (use at least 32 bytes)
- `GROQ_API_KEY` or `OPENAI_API_KEY`: needed for LLM answers

Never commit `.env`. It is in `.gitignore`.

## Run

```
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

## Tests

```
set PYTHONPATH=.
pytest tests -q
```

PowerShell users: `$env:PYTHONPATH="."` instead of `set`.

## 5-minute demo

1. `POST /api/auth/register` with a username and password. Copy the `access_token` from the response.
2. In `/docs`, click **Authorize** and paste the token.
3. `POST /api/documents` and upload `data/sample_docs/ml_basics_notes.txt`. Wait until `status` shows `ready` in `GET /api/documents`.
4. `POST /api/chat` with `{"question": "What is supervised learning?", "top_k": 4}`.
5. Send the same request again. The second response shows `cache_hit: true`.

Shortcut: `python scripts/seed_demo.py` creates a demo user and uploads the sample document.

## Project structure

```
backend/        FastAPI app (main, auth, database, rag, schemas, config)
src/            RAG core (ingest, llm, chain), reused by backend
tests/          pytest suite (auth, validation, upload limits, cache)
scripts/        seed_demo.py
legacy/         original Vercel and Streamlit code, not part of the capstone backend
data/sample_docs/  sample documents
```

## Future ideas

Streamlit UI, PDF export of chat history, email reports, multi-user document sharing.
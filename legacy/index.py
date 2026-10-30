"""Vercel Serverless Function entry point for DocuQuery AI RAG Chatbot.

Provides lightweight, ultra-fast RAG query endpoints:
- POST /api/chat : Query RAG index with question, provider, and API key
- GET /api/documents : List indexed documents
- GET /api/health : Health check
"""

import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Resolve data path
ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = ROOT_DIR / "data" / "sample_docs"
SAMPLE_FILE = SAMPLE_DOCS_DIR / "ml_basics_notes.txt"

# In-memory document storage for serverless runtime
DOCUMENTS_CACHE = []
CHUNKS_CACHE = []


def chunk_text(text: str, chunk_size: int = 450, chunk_overlap: int = 80, source: str = "document"):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    chunk_id = 1
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "id": f"{source}-chunk-{chunk_id}",
                "content": chunk_content,
                "source": source,
                "page": 1,
            })
            chunk_id += 1
        if end >= text_len:
            break
        start += (chunk_size - chunk_overlap)
    return chunks


def load_cached_chunks():
    """Load and index documents into cache."""
    global CHUNKS_CACHE, DOCUMENTS_CACHE
    if CHUNKS_CACHE:
        return CHUNKS_CACHE

    all_chunks = []
    docs_info = []

    # Check sample docs directory
    if SAMPLE_DOCS_DIR.exists():
        for file in sorted(SAMPLE_DOCS_DIR.iterdir()):
            if file.is_file() and file.suffix.lower() in [".txt", ".md"]:
                try:
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    chunks = chunk_text(text, source=file.name)
                    all_chunks.extend(chunks)
                    docs_info.append({"name": file.name, "chunks": len(chunks), "size": len(text)})
                except Exception as e:
                    print(f"Error loading {file.name}: {e}")

    # Fallback default knowledge if file not found
    if not all_chunks:
        default_notes = (
            "Machine Learning Basics - Sample Notes\n"
            "Supervised Learning is where the model is trained on labeled data. "
            "Examples: Classification (predicting discrete classes like spam vs not spam) "
            "and Regression (predicting continuous values like house prices). Common algorithms include "
            "Linear Regression, Logistic Regression, Random Forest, and Gradient Boosted Trees.\n\n"
            "Unsupervised Learning deals with unlabeled data to discover hidden patterns. "
            "Examples: Clustering (K-Means, DBSCAN), Dimensionality Reduction (PCA, t-SNE).\n\n"
            "Vector Embeddings: Dense numerical vectors that capture semantic meaning. "
            "Similar concepts map to nearby points in high-dimensional vector space. "
            "Enables semantic similarity search in vector databases like ChromaDB and FAISS.\n\n"
            "Retrieval-Augmented Generation (RAG): Combines vector retrieval with generative LLMs "
            "to answer questions accurately grounded in private documents with source citations."
        )
        chunks = chunk_text(default_notes, source="ml_basics_notes.txt")
        all_chunks.extend(chunks)
        docs_info.append({"name": "ml_basics_notes.txt", "chunks": len(chunks), "size": len(default_notes)})

    CHUNKS_CACHE = all_chunks
    DOCUMENTS_CACHE = docs_info
    return CHUNKS_CACHE


def retrieve_relevant_chunks(query: str, top_k: int = 4):
    """Retrieve top-k chunks using term-overlap and cosine similarity."""
    chunks = load_cached_chunks()
    if not chunks:
        return []

    def tokenize(text: str):
        return re.findall(r"\b\w{3,}\b", text.lower())

    q_tokens = tokenize(query)
    if not q_tokens:
        return chunks[:top_k]

    q_set = set(q_tokens)
    scored = []
    for chunk in chunks:
        d_tokens = tokenize(chunk["content"])
        if not d_tokens:
            continue
        overlap = sum(1 for t in d_tokens if t in q_set)
        # Cosine-like normalization
        score = overlap / (math.sqrt(len(d_tokens)) * math.sqrt(len(q_tokens)) + 1e-6)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for score, c in scored[:top_k]]
    return top if top else chunks[:top_k]


def call_groq_llm(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq API directly via urllib without third party dependencies."""
    import urllib.request
    import urllib.error

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "DocuQuery-RAG/1.0",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuQuery AI, a professional RAG assistant. "
                    "Answer questions accurately based on the provided context. "
                    "Always mention which document or section your answer comes from. "
                    "If the answer is not in the context, state that clearly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        return f"Groq API Error ({e.code}): {err_body}"
    except Exception as e:
        return f"Failed to call Groq API: {e}"


def call_openai_llm(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API directly via urllib without third party dependencies."""
    import urllib.request
    import urllib.error

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "DocuQuery-RAG/1.0",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuQuery AI, a helpful RAG assistant. "
                    "Answer questions strictly using the provided context. "
                    "Keep answers structured, clear, and cite references."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        return f"OpenAI API Error ({e.code}): {err_body}"
    except Exception as e:
        return f"Failed to call OpenAI API: {e}"


def extractive_fallback(query: str, chunks: list) -> str:
    """Extractive answer generator when no LLM API key is provided."""
    if not chunks:
        return "No relevant context found in documents."

    full_context = " ".join([c["content"] for c in chunks])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_context) if len(s.strip()) > 15]

    q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    scored_sentences = []
    for s in sentences:
        s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
        score = len(q_words & s_words)
        scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for score, s in scored_sentences if score > 0][:4]
    if not top_sentences:
        top_sentences = sentences[:3]

    ans = " ".join(top_sentences)
    return (
        f"{ans}\n\n"
        f"*(Answer synthesized by DocuQuery Local RAG. "
        f"To enable natural conversational answers, provide a free Groq API key or OpenAI API key in Settings.)*"
    )


class handler(BaseHTTPRequestHandler):
    """Vercel serverless request handler."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/api/health", "/api/health/"]:
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "service": "DocuQuery AI RAG Engine",
                "version": "1.0.0",
                "indexed_documents": len(DOCUMENTS_CACHE or load_cached_chunks()),
            }).encode("utf-8"))
            return

        if path in ["/api/documents", "/api/documents/"]:
            load_cached_chunks()
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "documents": DOCUMENTS_CACHE,
                "total_chunks": len(CHUNKS_CACHE),
            }).encode("utf-8"))
            return

        # Fallback
        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}

        if path in ["/api/chat", "/api/chat/"]:
            question = data.get("question", "").strip()
            if not question:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Field 'question' is required."}).encode("utf-8"))
                return

            provider = data.get("provider", "groq").lower()
            api_key = data.get("apiKey", "") or os.getenv("GROQ_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            top_k = int(data.get("topK", 4))

            # Retrieve chunks
            chunks = retrieve_relevant_chunks(question, top_k=top_k)

            # Build prompt
            context_text = "\n\n---\n\n".join([f"[{c['source']}] {c['content']}" for c in chunks])
            prompt = (
                f"You have retrieved the following excerpts from documents:\n\n"
                f"{context_text}\n\n"
                f"User Question: {question}\n\n"
                f"Answer clearly and concisely based on the context above:"
            )

            # Generate response
            if provider == "groq" and api_key:
                answer = call_groq_llm(prompt, api_key=api_key)
            elif provider == "openai" and api_key:
                answer = call_openai_llm(prompt, api_key=api_key)
            else:
                answer = extractive_fallback(question, chunks)

            sources_response = [
                {
                    "source": c["source"],
                    "page": c.get("page", 1),
                    "snippet": c["content"][:300] + ("..." if len(c["content"]) > 300 else ""),
                }
                for c in chunks
            ]

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "answer": answer,
                "sources": sources_response,
                "provider_used": provider if api_key else "local-extractive",
            }).encode("utf-8"))
            return

        if path in ["/api/upload", "/api/upload/"]:
            filename = data.get("filename", "uploaded_doc.txt")
            text = data.get("content", "").strip()
            if not text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Content cannot be empty."}).encode("utf-8"))
                return

            new_chunks = chunk_text(text, source=filename)
            global CHUNKS_CACHE, DOCUMENTS_CACHE
            load_cached_chunks()
            CHUNKS_CACHE.extend(new_chunks)
            DOCUMENTS_CACHE.append({"name": filename, "chunks": len(new_chunks), "size": len(text)})

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "message": f"Successfully indexed {filename}",
                "chunks_added": len(new_chunks),
                "total_chunks": len(CHUNKS_CACHE),
            }).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

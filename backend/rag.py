"""RAG query runner reusing src/ modules.

Run with run_in_threadpool from FastAPI so the async event loop is not blocked.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Token price table (per 1 000 tokens, USD) ─────────────────────────────────
_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    # Groq models (approximate; Groq pricing is very low)
    "llama-3.1-8b-instant":   {"prompt": 0.00005, "completion": 0.00008},
    "llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079},
    "llama3-8b-8192":         {"prompt": 0.00005, "completion": 0.00008},
    # OpenAI models
    "gpt-4o-mini":            {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o":                 {"prompt": 0.00250, "completion": 0.01000},
}
_DEFAULT_PRICE = {"prompt": 0.00010, "completion": 0.00020}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICE_TABLE.get(model, _DEFAULT_PRICE)
    cost = (prompt_tokens / 1000) * prices["prompt"] + (completion_tokens / 1000) * prices["completion"]
    return round(cost, 8)


def _get_chroma_dir(document_id: Optional[int]) -> Path:
    from backend.config import DB_PATH
    base = Path(DB_PATH).parent
    if document_id is not None:
        return base / "chroma_db" / str(document_id)
    return base / "chroma_db"


def _extract_usage(raw_response: Any) -> Tuple[int, int]:
    """Pull prompt/completion token counts from a LangChain AIMessage or raw response."""
    # LangChain >=0.2 stores usage_metadata on AIMessage
    if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
        meta = raw_response.usage_metadata
        prompt = getattr(meta, "input_tokens", 0) or meta.get("input_tokens", 0) if hasattr(meta, "get") else getattr(meta, "input_tokens", 0)
        completion = getattr(meta, "output_tokens", 0) or meta.get("output_tokens", 0) if hasattr(meta, "get") else getattr(meta, "output_tokens", 0)
        return int(prompt), int(completion)
    # Some wrappers expose .response_metadata
    if hasattr(raw_response, "response_metadata"):
        meta = raw_response.response_metadata or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    return 0, 0


def run_rag_query(
    *,
    document_id: Optional[int],
    question: str,
    top_k: int = 4,
    provider: Optional[str] = None,
    groq_api_key: str = "",
    openai_api_key: str = "",
) -> Dict[str, Any]:
    """Synchronous RAG query. Call via run_in_threadpool in async endpoints.

    Returns a dict with: answer, citations, prompt_tokens, completion_tokens,
    estimated_cost_usd, provider, model, raw_llm_response.
    """
    from backend.config import GROQ_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
    from src.llm import get_llm
    from src.ingest import load_vectorstore, vectorstore_exists, SimpleVectorStore

    # Resolve effective provider and API key
    eff_provider = (provider or LLM_PROVIDER or "groq").lower()
    if eff_provider == "groq":
        api_key = groq_api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured on the server.")
    elif eff_provider == "openai":
        api_key = openai_api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured on the server.")
    else:
        api_key = ""

    # Determine vector store path
    chroma_dir = _get_chroma_dir(document_id)

    if not vectorstore_exists(chroma_dir):
        # Fall back to global store
        from src.ingest import CHROMA_DIR
        chroma_dir = CHROMA_DIR
        if not vectorstore_exists(chroma_dir):
            raise FileNotFoundError(
                "No vector store found. Upload and process documents first."
            )

    vectorstore = load_vectorstore(chroma_dir)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    llm = get_llm(provider=eff_provider, api_key=api_key)

    # Retrieve relevant chunks
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)

    context_texts = [getattr(d, "page_content", str(d)) for d in docs]
    context_str = "\n\n---\n\n".join(context_texts)

    prompt = (
        "Use the following pieces of context from documents to answer the question.\n"
        "If you don't know the answer, say that you don't know.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\nAnswer:"
    )

    if hasattr(llm, "invoke"):
        raw = llm.invoke(prompt)
        answer = getattr(raw, "content", str(raw))
    elif callable(llm):
        raw = llm(prompt)
        answer = str(raw)
    else:
        raw = None
        answer = "Unable to generate an answer."

    prompt_tokens, completion_tokens = _extract_usage(raw)

    # Determine model name
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    cost = _estimate_cost(str(model_name), prompt_tokens, completion_tokens)

    # Build citations from retrieved docs
    citations = []
    for i, doc in enumerate(docs):
        meta = getattr(doc, "metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", None)
        snippet = getattr(doc, "page_content", "")[:200]
        citations.append({
            "source": Path(source).name if source != "unknown" else "unknown",
            "page": page,
            "chunk": i + 1,
            "text_snippet": snippet,
        })

    return {
        "answer": answer,
        "citations": citations,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": cost,
        "provider": eff_provider,
        "model": str(model_name),
    }

import os
from pathlib import Path
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.ingest import rebuild_index, vectorstore_exists, DATA_DIR, load_documents
from src.chain import build_qa_chain

st.set_page_config(page_title="DocuQuery AI — RAG Chatbot", page_icon="📄", layout="wide")

# Custom UI styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        background-color: rgba(66, 153, 225, 0.15);
        color: #63b3ed;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📄 DocuQuery AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-caption">Ask questions about your documents — powered by RAG (LangChain + Vector DB + LLMs)</div>',
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM Provider selection
    provider_options = ["Groq", "OpenAI", "Local", "Fallback"]
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    default_idx = 0 if env_provider == "groq" else (1 if env_provider == "openai" else 0)

    selected_provider_label = st.selectbox(
        "LLM Provider",
        provider_options,
        index=default_idx,
        help="Groq offers ultra-fast free tier models (Llama-3.1). OpenAI requires an OpenAI key. Local uses HuggingFace.",
    )
    selected_provider = selected_provider_label.lower()

    api_key_input = ""
    if selected_provider == "groq":
        default_groq = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=default_groq,
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com",
        )
    elif selected_provider == "openai":
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=default_openai,
            type="password",
            placeholder="sk-...",
            help="Get your key at https://platform.openai.com",
        )

    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=8, value=4)

    st.divider()
    st.subheader("📁 Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Files will be saved to data/sample_docs and indexed.",
    )

    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for uf in uploaded_files:
            target_path = DATA_DIR / uf.name
            if not target_path.exists() or target_path.stat().st_size != uf.size:
                with open(target_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_count += 1
        if saved_count > 0:
            st.toast(f"Saved {saved_count} new file(s) to sample_docs!", icon="💾")

    # List existing documents
    existing_docs = [f.name for f in DATA_DIR.glob("*") if f.is_file()] if DATA_DIR.exists() else []
    with st.expander(f"📚 Documents in library ({len(existing_docs)})", expanded=False):
        if existing_docs:
            for doc_name in existing_docs:
                st.markdown(f"- `{doc_name}`")
        else:
            st.caption("No documents in library yet.")

    # Rebuild Index Button
    if st.button("🔨 Build / Rebuild Vector DB", use_container_width=True, type="primary"):
        with st.spinner("Chunking documents, computing embeddings, and saving to ChromaDB..."):
            try:
                stats = rebuild_index()
                st.session_state.qa_chain = None  # Force chain rebuild
                st.success(f"Indexed {stats['documents']} docs → {stats['chunks']} chunks successfully!")
            except Exception as e:
                st.error(f"Error rebuilding index: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.rerun()

# ---------------- Session State & Chain Management ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

current_config = {
    "top_k": top_k,
    "provider": selected_provider,
    "api_key": api_key_input,
}

if st.session_state.get("chain_config") != current_config:
    # Config changed, reset chain
    st.session_state.qa_chain = None
    st.session_state.chain_config = current_config

# ---------------- Pre-check & Welcome Card ----------------
has_vectorstore = vectorstore_exists()
if not has_vectorstore and not st.session_state.messages:
    st.info(
        "👋 **Welcome to DocuQuery AI!** To get started:\n\n"
        "1. Click **'🔨 Build / Rebuild Vector DB'** in the sidebar to index sample documents.\n"
        "2. Or upload your own PDF/TXT/MD files above.\n"
        "3. Provide an API key (Groq is free and recommended) or run with local fallback.",
        icon="ℹ️",
    )

# Quick sample question pills
if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    col1, col2 = st.columns(2)
    sample_q = None
    with col1:
        if st.button("What is the difference between supervised and unsupervised learning?"):
            sample_q = "What is the difference between supervised and unsupervised learning?"
    with col2:
        if st.button("What is the difference between classification and regression?"):
            sample_q = "What is the difference between classification and regression?"
else:
    sample_q = None

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Citations & Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Unknown Document")
                    page_num = src.get("page", "N/A")
                    st.markdown(f'<div class="source-badge">Source {i} — {source_name} (Page {page_num})</div>', unsafe_allow_html=True)
                    st.caption(src.get("content", ""))

# Handle User Input
chat_input = st.chat_input("Ask a question about your documents...")
question = chat_input or sample_q

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                if st.session_state.qa_chain is None:
                    st.session_state.qa_chain = build_qa_chain(
                        top_k=top_k,
                        provider=selected_provider,
                        api_key=api_key_input if api_key_input else None,
                    )

                result = st.session_state.qa_chain.invoke({"question": question})
                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)

                sources = []
                for doc in result.get("source_documents", []):
                    src_path = doc.metadata.get("source", "Document")
                    filename = Path(src_path).name if src_path else "Document"
                    page = doc.metadata.get("page", "N/A")
                    snippet = doc.page_content.strip()
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "..."
                    sources.append({"source": filename, "page": page, "content": snippet})

                if sources:
                    with st.expander("🔍 Citations & Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-badge">Source {i} — {src["source"]} (Page {src["page"]})</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(src["content"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                err_msg = str(e)
                st.error(f"⚠️ **Error:** {err_msg}")
                st.info(
                    "💡 **Tip:** If the vector database is missing, click **'Build / Rebuild Vector DB'** in the sidebar. "
                    "If using Groq or OpenAI, make sure your API key is correct."
                )


"""Load documents, split into chunks, embed and store in a vector database."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Define project root and default directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
FALLBACK_STORE_FILE = PROJECT_ROOT / "chroma_db" / "inmemory_store.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SimpleDocument:
    """Lightweight document representation compatible with LangChain Document interface."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def to_dict(self):
        return {"page_content": self.page_content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(page_content=data.get("page_content", ""), metadata=data.get("metadata", {}))


class SimpleRetriever:
    """Lightweight in-memory vector retriever using TF-IDF / term overlap cosine similarity."""
    def __init__(self, documents: List[SimpleDocument], k: int = 4):
        self.documents = documents
        self.k = k

    def get_relevant_documents(self, query: str) -> List[SimpleDocument]:
        return self._search(query)

    def invoke(self, query_dict: Any) -> List[SimpleDocument]:
        query = query_dict if isinstance(query_dict, str) else query_dict.get("question", "")
        return self._search(query)

    def _search(self, query: str) -> List[SimpleDocument]:
        if not self.documents:
            return []

        def tokenize(text: str) -> List[str]:
            return re.findall(r"\b\w{3,}\b", text.lower())

        q_tokens = tokenize(query)
        if not q_tokens:
            return self.documents[: self.k]

        q_set = set(q_tokens)
        scores = []
        for doc in self.documents:
            d_tokens = tokenize(doc.page_content)
            if not d_tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for t in d_tokens if t in q_set)
            # Normalized term score
            score = overlap / (math.sqrt(len(d_tokens)) * math.sqrt(len(q_tokens)) + 1e-6)
            scores.append(score)

        # Sort descending by score
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_docs = [self.documents[idx] for idx, _ in indexed[: self.k]]
        return top_docs


class SimpleVectorStore:
    """Zero-dependency vector store when ChromaDB is not installed or SQLite is incompatible."""
    def __init__(self, documents: List[SimpleDocument]):
        self.documents = documents

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None):
        k = (search_kwargs or {}).get("k", 4)
        return SimpleRetriever(self.documents, k=k)

    def save(self, filepath: Path = FALLBACK_STORE_FILE):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [d.to_dict() for d in self.documents]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path = FALLBACK_STORE_FILE):
        if not filepath.exists():
            raise FileNotFoundError(f"Store file not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [SimpleDocument.from_dict(d) for d in data]
        return cls(docs)


def get_embeddings(provider: Optional[str] = None, api_key: Optional[str] = None):
    """Return embedding function: local HuggingFace, OpenAI, or None for fallback."""
    if provider == "openai" or (provider is None and os.getenv("OPENAI_API_KEY")):
        try:
            from langchain_openai import OpenAIEmbeddings
            key = api_key or os.getenv("OPENAI_API_KEY")
            return OpenAIEmbeddings(openai_api_key=key, model="text-embedding-3-small")
        except Exception as e:
            print(f"Failed to load OpenAIEmbeddings: {e}. Falling back to HuggingFace/Simple.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception:
            return None


def load_documents(folder: Path | str = DATA_DIR):
    """Load all PDF, TXT, and MD files from a folder as LangChain Documents."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    docs = []

    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        has_loaders = True
    except Exception:
        has_loaders = False

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        try:
            if has_loaders and ext == ".pdf":
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
            elif ext in [".txt", ".md"]:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs.append(SimpleDocument(page_content=content, metadata={"source": str(file), "page": 1}))
            elif ext == ".pdf":
                # Fallback pypdf reader if langchain_community is unavailable
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file))
                    for i, page in enumerate(reader.pages, 1):
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append(SimpleDocument(page_content=text, metadata={"source": str(file), "page": i}))
                except Exception as pe:
                    print(f"Error reading PDF {file.name}: {pe}")
        except Exception as e:
            print(f"Warning: Failed to load {file.name}: {e}")

    if not docs:
        raise FileNotFoundError(f"No valid PDF, TXT, or MD files found in '{folder}'")
    return docs


def split_chunks(documents, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split documents into overlapping chunks for better retrieval."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)
    except Exception:
        # Fallback character splitter
        chunks = []
        for doc in documents:
            text = doc.page_content
            meta = getattr(doc, "metadata", {})
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append(SimpleDocument(page_content=chunk_text, metadata=meta))
                if end == len(text):
                    break
                start += (chunk_size - chunk_overlap)
        return chunks


def build_vectorstore(chunks, persist_dir: Path | str = CHROMA_DIR, embedding_function=None):
    """Create and persist a vector store (ChromaDB or SimpleVectorStore)."""
    persist_dir_path = Path(persist_dir)
    persist_dir_path.mkdir(parents=True, exist_ok=True)

    emb = embedding_function if embedding_function is not None else get_embeddings()

    # Try Chroma if installed and embeddings are available
    if emb is not None:
        try:
            from langchain_community.vectorstores import Chroma
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=emb,
                persist_directory=str(persist_dir_path),
            )
            if hasattr(vectorstore, "persist"):
                try:
                    vectorstore.persist()
                except Exception:
                    pass
            return vectorstore
        except Exception as e:
            print(f"ChromaDB initialization failed ({e}). Falling back to SimpleVectorStore.")

    # Lightweight fallback store
    simple_docs = [
        SimpleDocument(
            page_content=getattr(c, "page_content", str(c)),
            metadata=getattr(c, "metadata", {}),
        )
        for c in chunks
    ]
    store = SimpleVectorStore(simple_docs)
    store.save(persist_dir_path / "inmemory_store.json")
    return store


def vectorstore_exists(persist_dir: Path | str = CHROMA_DIR) -> bool:
    """Check if a populated ChromaDB or SimpleVectorStore exists."""
    p = Path(persist_dir)
    if not p.exists() or not p.is_dir():
        return False
    if (p / "inmemory_store.json").exists():
        return True
    files = list(p.glob("*"))
    return len(files) > 0


def load_vectorstore(persist_dir: Path | str = CHROMA_DIR, embedding_function=None):
    """Load an existing ChromaDB or SimpleVectorStore."""
    persist_dir_path = Path(persist_dir)
    if not vectorstore_exists(persist_dir_path):
        raise FileNotFoundError(
            f"No vector store found at '{persist_dir_path}'. Please build the vector database first."
        )

    # Check for simple fallback store first
    json_path = persist_dir_path / "inmemory_store.json"
    if json_path.exists():
        return SimpleVectorStore.load(json_path)

    emb = embedding_function if embedding_function is not None else get_embeddings()
    if emb is not None:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma(
                persist_directory=str(persist_dir_path),
                embedding_function=emb,
            )
        except Exception as e:
            print(f"Failed to load Chroma ({e}). Checking for fallback store.")

    if json_path.exists():
        return SimpleVectorStore.load(json_path)

    raise RuntimeError("Unable to load ChromaDB and no in-memory store found. Rebuild index.")


def rebuild_index(folder: Path | str = DATA_DIR, persist_dir: Path | str = CHROMA_DIR, embedding_function=None):
    """Full pipeline: load -> split -> embed -> store. Returns stats."""
    docs = load_documents(folder)
    chunks = split_chunks(docs)
    build_vectorstore(chunks, persist_dir=persist_dir, embedding_function=embedding_function)
    return {"documents": len(docs), "chunks": len(chunks)}


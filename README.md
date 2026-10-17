# DocuQuery AI — RAG based Document Q&A Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB%20%2F%20In--Memory-orange)](https://www.trychroma.com/)
[![Vercel](https://img.shields.io/badge/Deployment-Vercel%20Live-black)](https://vercel.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that lets you upload and query PDF, TXT, and Markdown documents using modern vector search and Large Language Models — with **verifiable source citations**.

Dual-deployment ready:
1. **Live Vercel Web Application** — Glassmorphic, ultra-fast serverless web UI and API (`public/` & `api/`).
2. **Streamlit App** — Full-featured interactive dashboard for local development & Streamlit Cloud (`app.py`).

---

## ✨ Features

- 📄 **Multi-Format Ingestion** — Load and parse PDF, TXT, and Markdown (`.md`) documents.
- ✂️ **Smart Chunking** — Overlapping chunking with metadata preservation for precise document citations.
- 🧠 **Embeddings & Vector Stores** — Supports ChromaDB, OpenAI Embeddings, HuggingFace embeddings, and lightweight zero-dependency vector similarity.
- 💬 **Conversational Memory** — Contextual follow-up memory across conversation turns.
- 🔍 **Source Citations** — Verifiable page numbers and text excerpts cited for every answer.
- 🤖 **Pluggable LLMs** — Groq (ultra-fast Llama 3.1), OpenAI (GPT-4o-mini), and local extractive mode (no API key required).
- 🌐 **Vercel Serverless Ready** — Pre-configured `vercel.json` and `api/index.py` for immediate deployment to Vercel without memory or size limit errors.

---

## 🏗️ Architecture

```
User Document (PDF / TXT / MD)
          │
          ▼
┌────────────────────────┐
│     Text Splitter      │  (Recursive chunking with overlap)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Vector Database / Index│  (ChromaDB / Cosine similarity)
└───────────┬────────────┘
            │
            ▼ Top-k relevant chunks
┌────────────────────────┐
│  LLM Synthesis Engine  │ ◀──── User Question
│  (Groq / OpenAI / Local│
└───────────┬────────────┘
            │
            ▼
Answer + Traceable Source Citations (Document & Page)
```

---

## 🚀 Live Vercel Deployment

This project includes a production-ready Vercel serverless application in `public/` and `api/index.py`.

### Deploy to Vercel in 2 Minutes:

#### Option 1: Via GitHub & Vercel Dashboard (Recommended)
1. Push this repository to your GitHub account:
   ```bash
   git remote add origin https://github.com/<your-username>/docuquery-rag-chatbot.git
   git branch -M main
   git push -u origin main
   ```
2. Open [vercel.com](https://vercel.com/) and click **"Add New Project"**.
3. Import your GitHub repository `docuquery-rag-chatbot`.
4. (Optional) In **Environment Variables**, add:
   - `GROQ_API_KEY`: Your free Groq API key (from [console.groq.com](https://console.groq.com/keys))
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)
5. Click **Deploy**. Your app is immediately live with HTTPS!

#### Option 2: Via Vercel CLI
```bash
npm install -g vercel
vercel
```

---

## 💻 Running the Streamlit App Locally

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/<your-username>/docuquery-rag-chatbot.git
cd docuquery-rag-chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY or OPENAI_API_KEY
```

### 3. Run Streamlit Dashboard
```bash
streamlit run app.py
```

### 4. Build Vector DB from Command Line (Optional)
```bash
python scripts/build_index.py
```

---

## 📁 Project Structure

```
docuquery-rag-chatbot/
├── app.py                  # Streamlit UI dashboard
├── vercel.json             # Vercel deployment configuration
├── api/
│   └── index.py            # Vercel Serverless Python RAG API
├── public/                 # Vercel Web App (HTML5, CSS3, JavaScript)
│   ├── index.html          # Modern glassmorphism web interface
│   ├── style.css           # Premium dark theme & responsive styles
│   └── app.js              # Client chat logic & citation renderer
├── src/
│   ├── ingest.py           # Load, split, embed & store documents
│   ├── llm.py              # Pluggable LLM provider (Groq/OpenAI/Local)
│   └── chain.py            # Conversational RAG chain with memory
├── scripts/
│   └── build_index.py      # CLI script to rebuild vector store
├── data/sample_docs/       # Document library (PDF, TXT, MD)
│   └── ml_basics_notes.txt # Preloaded sample machine learning notes
├── chroma_db/              # Persistent vector store (auto-generated)
├── requirements.txt        # Production dependencies (Vercel & Streamlit)
├── requirements-local.txt  # Optional offline PyTorch & ChromaDB
├── .env.example
├── .gitignore
└── LICENSE
```

---

## 📄 License

MIT License — Feel free to use and customize for your own projects.

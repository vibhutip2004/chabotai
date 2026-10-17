"""Conversational RAG chain with chat memory and source citations."""

from typing import Any, List, Optional
from src.ingest import load_vectorstore, vectorstore_exists, rebuild_index, DATA_DIR
from src.llm import get_llm


class SimpleConversationalChain:
    """Lightweight conversational RAG runner when LangChain chains are unavailable."""
    def __init__(self, retriever: Any, llm: Any):
        self.retriever = retriever
        self.llm = llm
        self.chat_history: List[tuple] = []

    def invoke(self, inputs: dict) -> dict:
        question = inputs.get("question", "")
        # Retrieve relevant chunks
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(question)
        elif hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(question)
        else:
            docs = []

        context_texts = [getattr(d, "page_content", str(d)) for d in docs]
        context_str = "\n\n---\n\n".join(context_texts)

        prompt = (
            f"Use the following pieces of context from documents to answer the question at the end.\n"
            f"If you don't know the answer, say that you don't know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            if hasattr(self.llm, "invoke"):
                raw_ans = self.llm.invoke(prompt)
                answer = getattr(raw_ans, "content", str(raw_ans))
            elif hasattr(self.llm, "_call"):
                answer = self.llm._call(prompt)
            elif callable(self.llm):
                answer = str(self.llm(prompt))
            else:
                answer = str(self.llm)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        self.chat_history.append((question, answer))
        return {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "chat_history": self.chat_history,
        }


def build_qa_chain(
    top_k: int = 4,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
):
    """Build a conversational QA chain over the vector store.

    - top_k: number of relevant chunks retrieved per question
    - provider: LLM provider ("groq", "openai", "local", etc.)
    - api_key: optional explicit API key
    - model: optional model name
    - temperature: sampling temperature
    """
    # Auto-initialize vector store if missing and sample docs exist
    if not vectorstore_exists():
        sample_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
        if any(f.is_file() for f in sample_files):
            print("Vector store not found. Auto-building index from sample documents...")
            rebuild_index()
        else:
            raise FileNotFoundError(
                "Vector database not found and no sample documents available. "
                "Please upload documents and build the vector database first."
            )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    llm = get_llm(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )

    try:
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        return qa_chain
    except Exception as e:
        print(f"Notice: Using SimpleConversationalChain ({e})")
        return SimpleConversationalChain(retriever=retriever, llm=llm)


"""Pluggable LLM provider: OpenAI / Groq / Google Gemini / local HuggingFace / Fallback."""

import os
import re
from typing import Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
):
    """Return a LangChain Chat/LLM instance based on provider and API keys.

    - groq   : ultra-fast, free tier (llama-3.1-8b-instant or llama-3.3-70b-versatile)
    - openai : GPT-4o-mini / GPT-4o
    - local  : offline HuggingFace pipeline (google/flan-t5-small)
    - fallback: zero-dependency extractive answerer when keys & torch are unavailable
    """
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()

    # 1. Groq Provider
    groq_key = api_key if provider == "groq" else (api_key or os.getenv("GROQ_API_KEY"))
    if selected_provider == "groq" or (not selected_provider and groq_key):
        if not groq_key:
            raise ValueError("GROQ_API_KEY is required to use the Groq provider.")
        from langchain_groq import ChatGroq
        model_name = model or "llama-3.1-8b-instant"
        return ChatGroq(model=model_name, groq_api_key=groq_key, temperature=temperature)

    # 2. OpenAI Provider
    openai_key = api_key if provider == "openai" else (api_key or os.getenv("OPENAI_API_KEY"))
    if selected_provider == "openai" or (not selected_provider and openai_key):
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is required to use the OpenAI provider.")
        from langchain_openai import ChatOpenAI
        model_name = model or "gpt-4o-mini"
        return ChatOpenAI(model=model_name, openai_api_key=openai_key, temperature=temperature)

    # 3. Local HuggingFace Pipeline
    if selected_provider == "local":
        try:
            from transformers import pipeline
            try:
                from langchain_huggingface import HuggingFacePipeline
            except Exception:
                from langchain_community.llms import HuggingFacePipeline

            hf_pipe = pipeline(
                task="text2text-generation",
                model=model or "google/flan-t5-small",
                max_new_tokens=256,
            )
            return HuggingFacePipeline(pipeline=hf_pipe)
        except Exception as e:
            print(f"Warning: Failed to load local HuggingFace model ({e}). Using Lightweight Fallback.")

    # 4. Fallback Extractive Model
    return _create_fallback_llm()


def _synthesize_answer(prompt: str) -> str:
    """Extract and synthesize an answer from a RAG prompt context."""
    clean_prompt = prompt.strip()
    context_match = re.search(r"Context:\s*(.*?)(?:Question:|$)", clean_prompt, re.DOTALL | re.IGNORECASE)
    question_match = re.search(r"Question:\s*(.*?)(?:Answer:|$)", clean_prompt, re.DOTALL | re.IGNORECASE)

    question = question_match.group(1).strip() if question_match else clean_prompt.splitlines()[-1]
    context = context_match.group(1).strip() if context_match else clean_prompt

    if not context:
        return "I couldn't find relevant information in the provided documents."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 15]
    
    # Simple semantic keyword matching between question and sentences
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    scored_sentences = []
    for s in sentences:
        s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
        score = len(q_words & s_words)
        scored_sentences.append((score, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for score, s in scored_sentences if score > 0][:3]
    if not top_sentences:
        top_sentences = sentences[:3] if sentences else [context[:400]]

    ans = " ".join(top_sentences)
    return (
        f"{ans}\n\n"
        f"*(Answer synthesized from indexed documents. "
        f"For conversational LLM synthesis, select Groq or OpenAI in the settings.)*"
    )


def _create_fallback_llm():
    """Lightweight LangChain-compatible LLM that provides extractive answers from RAG context."""
    try:
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun

        class LightweightSynthesizerLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "lightweight-extractive"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                return _synthesize_answer(prompt)

        return LightweightSynthesizerLLM()
    except Exception:
        class SimpleCallableLLM:
            def invoke(self, prompt: str) -> str:
                return _synthesize_answer(prompt)

            def _call(self, prompt: str, **kwargs) -> str:
                return _synthesize_answer(prompt)

            def __call__(self, prompt: str, **kwargs) -> str:
                return _synthesize_answer(prompt)

        return SimpleCallableLLM()



from pydantic import BaseModel, Field
from typing import Optional, List


class HealthResponse(BaseModel):
    status: str


# ── Auth schemas ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str


# ── Document schemas ──────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    status: str
    message: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    status: str
    created_at: str


# ── Chat schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The query to ask")
    document_id: Optional[int] = Field(default=None, description="Specific document ID to search")
    top_k: int = Field(default=4, ge=1, le=8, description="Number of document chunks to retrieve")


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk: Optional[int] = None
    text_snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    cache_hit: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


# ── Error schema ──────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str

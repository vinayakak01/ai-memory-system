from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


MemoryKind = Literal["semantic", "episodic", "procedural"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    kind: MemoryKind = "semantic"


class MemoryView(BaseModel):
    id: int
    content: str
    kind: MemoryKind
    source: str
    importance: float
    confidence: float
    access_count: int
    is_user_confirmed: bool
    created_at: datetime
    last_accessed_at: datetime | None
    relevance_score: float | None = None


class ChatResponse(BaseModel):
    reply: str
    memories_used: list[MemoryView]
    new_memories: list[MemoryView]

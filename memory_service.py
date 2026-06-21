from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import SQLITE_LOCK_RETRY_ATTEMPTS, SQLITE_LOCK_RETRY_DELAY_SECONDS
from llm_client import OllamaClient
from models import Memory, Message
from schemas import MemoryView


SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{20,}",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
]


@dataclass
class CandidateMemory:
    content: str
    kind: str
    importance: float
    confidence: float
    source: str = "auto"
    is_user_confirmed: bool = False


class MemoryService:
    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def handle_chat_turn(self, db: Session, user_message: str) -> tuple[str, list[MemoryView], list[MemoryView]]:
        try:
            retrieved = self.retrieve_memories(db, user_message, limit=6)
        except SQLAlchemyError:
            db.rollback()
            retrieved = []

        assistant_reply = self._generate_reply(db, user_message, retrieved)
        candidates = self.extract_candidates(db, user_message, assistant_reply)
        new_memories = self._persist_turn_best_effort(db, user_message, assistant_reply, candidates)
        return assistant_reply, retrieved, new_memories

    def add_manual_memory(self, db: Session, content: str, kind: str = "semantic") -> MemoryView:
        candidate = CandidateMemory(
            content=content.strip(),
            kind=kind,
            importance=9.0,
            confidence=1.0,
            source="manual",
            is_user_confirmed=True,
        )
        stored = self._run_with_retry(db, lambda: self._upsert_memory(db, candidate))
        self._commit_with_retry(db)
        return self._to_view(stored)

    def store_latest_user_message(self, db: Session) -> MemoryView | None:
        latest_user = db.execute(
            select(Message).where(Message.role == "user").order_by(Message.created_at.desc())
        ).scalars().first()
        if latest_user is None:
            return None
        return self.add_manual_memory(db, latest_user.content, "semantic")

    def delete_memory(self, db: Session, memory_id: int) -> bool:
        memory = db.get(Memory, memory_id)
        if memory is None:
            return False
        db.delete(memory)
        self._commit_with_retry(db)
        return True

    def list_memories(self, db: Session) -> list[MemoryView]:
        memories = db.execute(select(Memory).order_by(Memory.updated_at.desc())).scalars().all()
        return [self._to_view(memory) for memory in memories]

    def retrieve_memories(self, db: Session, query: str, limit: int = 5) -> list[MemoryView]:
        memories = db.execute(select(Memory)).scalars().all()
        if not memories:
            return []

        query_embedding = self._embed_text(query)
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, Memory]] = []

        for memory in memories:
            memory_embedding = np.array(json.loads(memory.embedding_json), dtype=np.float32)
            semantic = self._cosine_similarity(query_embedding, memory_embedding)
            age_days = max((now - self._as_utc(memory.updated_at)).total_seconds() / 86400.0, 0.0)
            recency = 1.0 / (1.0 + age_days / 7.0)
            importance = max(min(memory.importance / 10.0, 1.0), 0.0)
            score = (0.65 * semantic) + (0.20 * recency) + (0.15 * importance)
            scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:limit]
        results: list[MemoryView] = []

        for score, memory in selected:
            memory.access_count += 1
            memory.last_accessed_at = datetime.utcnow()
            results.append(self._to_view(memory, relevance_score=round(score, 4)))

        return results

    def extract_candidates(self, db: Session, user_message: str, assistant_reply: str) -> list[CandidateMemory]:
        recent_messages = db.execute(select(Message).order_by(Message.created_at.desc()).limit(6)).scalars().all()
        recent_messages.reverse()
        transcript = "\n".join(f"{m.role}: {m.content}" for m in recent_messages)

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract durable user memories from the conversation.\n"
                    "Return valid JSON only. No markdown. No explanation.\n"
                    "Schema:\n"
                    "{\"memories\":[{\"content\":\"short memory sentence\",\"kind\":\"semantic|episodic|procedural\","
                    "\"importance\":7,\"confidence\":0.82}]}\n"
                    "At most 3 memories.\n"
                    "Keep only memories that are likely to help future conversations.\n"
                    "Good memories: stable preferences, ongoing projects, long-term goals, durable user facts.\n"
                    "Do not store: greetings, one-time small talk, temporary requests, secrets, API keys, passwords.\n"
                    "Write each memory as a short standalone sentence.\n"
                    "If nothing is worth storing, return {\"memories\":[]}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Conversation:\n"
                    f"{transcript}\n\n"
                    "Focus only on facts about the user, not the assistant.\n\n"
                    f"Latest user message:\n{user_message}\n\n"
                    f"Assistant reply:\n{assistant_reply}\n\n"
                    "Examples:\n"
                    "- I prefer concise answers -> procedural\n"
                    "- I am building a memory system MVP -> semantic\n"
                    "- I launched my startup last month -> episodic"
                ),
            },
        ]

        try:
            payload = self.llm_client.extract_json(messages)
        except Exception:
            return self._fallback_candidates(user_message)

        raw_memories = payload.get("memories", []) if isinstance(payload, dict) else []
        candidates: list[CandidateMemory] = []
        for raw in raw_memories:
            content = str(raw.get("content", "")).strip()
            kind = str(raw.get("kind", "semantic")).strip().lower()
            importance = float(raw.get("importance", 5))
            confidence = float(raw.get("confidence", 0.5))
            if not content:
                continue
            candidates.append(
                CandidateMemory(
                    content=content,
                    kind=kind if kind in {"semantic", "episodic", "procedural"} else "semantic",
                    importance=max(1.0, min(10.0, importance)),
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )
        return candidates

    def store_candidates(self, db: Session, candidates: Iterable[CandidateMemory]) -> list[MemoryView]:
        stored: list[MemoryView] = []
        for candidate in candidates:
            if not self._passes_write_gate(candidate):
                continue
            memory = self._run_with_retry(db, lambda candidate=candidate: self._upsert_memory(db, candidate))
            stored.append(self._to_view(memory))
        return stored

    def _generate_reply(self, db: Session, user_message: str, retrieved: list[MemoryView]) -> str:
        recent_messages = db.execute(select(Message).order_by(Message.created_at.desc()).limit(8)).scalars().all()
        recent_messages.reverse()

        memory_lines = [f"- ({m.kind}) {m.content}" for m in retrieved]
        memory_block = "\n".join(memory_lines) if memory_lines else "- No relevant long-term memory found."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, warm assistant with lightweight memory.\n"
                    "Use the provided memory context only when it is relevant.\n"
                    "Do not mention memory unless the user asks.\n"
                    "Be concise, clear, and personalized when useful.\n"
                    "If the memory context is not relevant, ignore it."
                ),
            },
            {
                "role": "system",
                "content": f"Memory context:\n{memory_block}",
            },
        ]
        for message in recent_messages[-6:]:
            messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": user_message})

        return self.llm_client.chat(messages, temperature=0.5)

    def _passes_write_gate(self, candidate: CandidateMemory) -> bool:
        if len(candidate.content) < 8:
            return False
        if candidate.importance < 6 and not candidate.is_user_confirmed:
            return False
        if candidate.confidence < 0.55 and not candidate.is_user_confirmed:
            return False
        lowered = candidate.content.lower()
        if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in SECRET_PATTERNS):
            return False
        if "today" in lowered and candidate.kind == "episodic" and not candidate.is_user_confirmed:
            return False
        return True

    def _upsert_memory(self, db: Session, candidate: CandidateMemory) -> Memory:
        normalized = self._normalize_text(candidate.content)
        existing = db.execute(select(Memory)).scalars().all()

        candidate_embedding = self._embed_text(candidate.content)
        best_match: Memory | None = None
        best_similarity = -1.0

        for memory in existing:
            memory_normalized = self._normalize_text(memory.content)
            if memory_normalized == normalized and memory.kind == candidate.kind:
                best_match = memory
                best_similarity = 1.0
                break

            memory_embedding = np.array(json.loads(memory.embedding_json), dtype=np.float32)
            similarity = self._cosine_similarity(candidate_embedding, memory_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = memory

        if best_match and best_similarity >= 0.92 and best_match.kind == candidate.kind:
            best_match.content = candidate.content
            best_match.source = candidate.source
            best_match.importance = max(best_match.importance, candidate.importance)
            best_match.confidence = max(best_match.confidence, candidate.confidence)
            best_match.is_user_confirmed = best_match.is_user_confirmed or candidate.is_user_confirmed
            best_match.embedding_json = json.dumps(candidate_embedding.tolist())
            return best_match

        memory = Memory(
            content=candidate.content,
            kind=candidate.kind,
            source=candidate.source,
            importance=candidate.importance,
            confidence=candidate.confidence,
            embedding_json=json.dumps(candidate_embedding.tolist()),
            is_user_confirmed=candidate.is_user_confirmed,
        )
        db.add(memory)
        db.flush()
        return memory

    def _store_message(self, db: Session, role: str, content: str) -> None:
        db.add(Message(role=role, content=content))
        db.flush()

    def _persist_turn_best_effort(
        self,
        db: Session,
        user_message: str,
        assistant_reply: str,
        candidates: Iterable[CandidateMemory],
    ) -> list[MemoryView]:
        try:
            self._run_with_retry(db, lambda: self._store_message(db, "user", user_message))
            self._run_with_retry(db, lambda: self._store_message(db, "assistant", assistant_reply))
            new_memories = self.store_candidates(db, candidates)
            self._commit_with_retry(db)
            return new_memories
        except SQLAlchemyError:
            db.rollback()
            return []

    def _fallback_candidates(self, user_message: str) -> list[CandidateMemory]:
        lowered = user_message.lower()
        triggers = ["i like", "i prefer", "i am building", "i work on", "my project", "i want to"]
        if not any(trigger in lowered for trigger in triggers):
            return []

        kind = "procedural" if ("prefer" in lowered or "like" in lowered) else "semantic"
        return [
            CandidateMemory(
                content=user_message.strip(),
                kind=kind,
                importance=7.0,
                confidence=0.72,
            )
        ]

    def _to_view(self, memory: Memory, relevance_score: float | None = None) -> MemoryView:
        return MemoryView(
            id=memory.id,
            content=memory.content,
            kind=memory.kind,  # type: ignore[arg-type]
            source=memory.source,
            importance=memory.importance,
            confidence=memory.confidence,
            access_count=memory.access_count,
            is_user_confirmed=memory.is_user_confirmed,
            created_at=memory.created_at,
            last_accessed_at=memory.last_accessed_at,
            relevance_score=relevance_score,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    def _commit_with_retry(self, db: Session) -> None:
        self._run_with_retry(db, db.commit)

    def _run_with_retry(self, db: Session, operation):
        last_error: SQLAlchemyError | None = None
        for attempt in range(SQLITE_LOCK_RETRY_ATTEMPTS):
            try:
                return operation()
            except OperationalError as exc:
                if not self._is_locked_error(exc):
                    raise
                db.rollback()
                last_error = exc
                if attempt == SQLITE_LOCK_RETRY_ATTEMPTS - 1:
                    break
                time.sleep(SQLITE_LOCK_RETRY_DELAY_SECONDS * (attempt + 1))
        if last_error is not None:
            raise last_error
        return operation()

    @staticmethod
    def _is_locked_error(exc: OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database table is locked" in message

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if math.isclose(denom, 0.0):
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

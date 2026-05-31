"""Redis-backed session manager with in-memory fallback."""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List

from adapt_ai.config import settings

logger = logging.getLogger(__name__)

_memory_store: Dict[str, Dict] = {}


class SessionManager:
    """Manages conversation context and history across requests."""

    _instance: "SessionManager | None" = None

    def __init__(self) -> None:
        self._redis = None
        if not settings.redis_fallback_memory:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
                logger.info("Redis session manager initialised")
            except Exception as e:
                logger.warning("Redis unavailable (%s) — using in-memory fallback", e)
        else:
            logger.info("Session manager using in-memory store (REDIS_FALLBACK_MEMORY=true)")

    @classmethod
    def get(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_raw(self, key: str) -> Dict:
        if self._redis:
            val = await self._redis.get(key)
            return json.loads(val) if val else {}
        return _memory_store.get(key, {})

    async def _set_raw(self, key: str, data: Dict, ttl: int = 3600) -> None:
        if self._redis:
            await self._redis.set(key, json.dumps(data), ex=ttl)
        else:
            _memory_store[key] = data

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_context(self, session_id: str) -> Dict[str, Any]:
        return await self._get_raw(f"ctx:{session_id}")

    async def save_context(self, session_id: str, context: Dict[str, Any]) -> None:
        existing = await self.get_context(session_id)
        existing.update(context)
        await self._set_raw(f"ctx:{session_id}", existing)

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        history = await self.get_conversation_history(session_id)
        history.append({"role": role, "content": content})
        await self._set_raw(f"hist:{session_id}", {"messages": history})

    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        data = await self._get_raw(f"hist:{session_id}")
        return data.get("messages", [])

    async def clear(self, session_id: str) -> None:
        for prefix in ("ctx:", "hist:"):
            key = f"{prefix}{session_id}"
            if self._redis:
                await self._redis.delete(key)
            else:
                _memory_store.pop(key, None)

"""WebSocket connection manager — tracks active connections for broadcast."""

import asyncio
from typing import Any


class ConnectionManager:
    """Manages per-client asyncio queues for WebSocket message delivery.

    Each connected WebSocket gets a private queue.  The *replay* loop
    pushes into that queue directly; the *demo trigger* uses
    :meth:`broadcast` to reach every connected client at once.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []

    # ── lifecycle ──────────────────────────────────────────────────
    def register(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues.append(q)
        return q

    def unregister(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        if q in self._queues:
            self._queues.remove(q)

    # ── messaging ──────────────────────────────────────────────────
    async def broadcast(self, message: dict[str, Any]) -> None:
        """Push *message* to every currently-registered client queue."""
        for q in self._queues:
            await q.put(message)

    @property
    def active_count(self) -> int:
        return len(self._queues)

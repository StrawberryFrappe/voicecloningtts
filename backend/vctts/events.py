"""In-process pub/sub used to push live events to UIs (WebSocket clients today,
other frontends such as a Discord bot later)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, max_queue: int = 2000):
        self._subs: set[asyncio.Queue] = set()
        self._max = max_queue

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def publish(self, event: dict[str, Any]) -> None:
        self.publish_nowait(event)

    def publish_nowait(self, event: dict[str, Any]) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("dropping slow event subscriber")
                self._subs.discard(q)

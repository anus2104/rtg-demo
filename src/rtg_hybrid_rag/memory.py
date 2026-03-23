from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from supermemory import Supermemory

from rtg_hybrid_rag.settings import Settings


@dataclass
class MemoryContext:
    static_facts: list[str]
    dynamic_facts: list[str]
    relevant_memories: list[str]

    def render(self) -> str:
        parts: list[str] = []
        if self.static_facts:
            parts.append("Static profile:\n" + "\n".join(f"- {item}" for item in self.static_facts))
        if self.dynamic_facts:
            parts.append("Dynamic profile:\n" + "\n".join(f"- {item}" for item in self.dynamic_facts))
        if self.relevant_memories:
            parts.append(
                "Relevant memories:\n" + "\n".join(f"- {item}" for item in self.relevant_memories)
            )
        return "\n\n".join(parts) if parts else "No stored memory found."


class MemoryManager:
    def __init__(self, settings: Settings, container_tag: str) -> None:
        self.settings = settings
        self.container_tag = container_tag
        self.enabled = bool(settings.supermemory_api_key and container_tag)
        self.client = (
            Supermemory(api_key=settings.supermemory_api_key)
            if self.enabled and settings.supermemory_api_key
            else None
        )

    def fetch_context(self, query: str) -> MemoryContext:
        if not self.client or not self.enabled:
            return MemoryContext([], [], [])

        response = self.client.profile(
            container_tag=self.container_tag,
            q=query,
            threshold=self.settings.supermemory_threshold,
        )

        static_facts = list(getattr(response.profile, "static", []) or [])
        dynamic_facts = list(getattr(response.profile, "dynamic", []) or [])
        search_results = getattr(response, "search_results", None)
        raw_results = list(getattr(search_results, "results", []) or [])

        relevant_memories: list[str] = []
        for item in raw_results:
            memory = None
            if isinstance(item, dict):
                memory = item.get("memory") or item.get("chunk")
            else:
                memory = getattr(item, "memory", None) or getattr(item, "chunk", None)
            if memory:
                relevant_memories.append(str(memory))

        return MemoryContext(static_facts, dynamic_facts, relevant_memories)

    def add_memory(self, content: str, metadata: dict[str, str | float | bool] | None = None) -> None:
        if not self.client or not self.enabled:
            return

        payload = {
            "source": "rtg-hybrid-rag-agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            payload.update(metadata)

        self.client.add(
            content=content,
            container_tag=self.container_tag,
            metadata=payload,
        )

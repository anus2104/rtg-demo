"""
RTG ShopPilot — FastAPI server
Exposes POST /chat as a Server-Sent Events stream.

The frontend (aiService.ts) sends:
  { message, history, profile, pageContext, sessionId? }

Each SSE event is one of:
  data: {"type": "delta",   "text": "..."}
  data: {"type": "profile", "profile": {...}}
  data: [DONE]

The backend:
  1. Converts history to pydantic-ai ModelMessage list
  2. Enriches the user message with page context
  3. Runs the pydantic-ai agent with streaming
  4. Emits delta events for every token
  5. Emits a profile update event at the end
  6. Emits [DONE]
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from rtg_hybrid_rag.agent_layer import AgentDeps, build_agent
from rtg_hybrid_rag.assistant import RecommendationAssistant
from rtg_hybrid_rag.memory import MemoryManager
from rtg_hybrid_rag.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons - built once at startup, shared across all requests
# ---------------------------------------------------------------------------
_recommender: RecommendationAssistant | None = None
_agent = None  # pydantic-ai Agent[AgentDeps, str]
_index_ready: bool = False


def _build_index_sync() -> None:
    global _recommender, _agent, _index_ready
    try:
        logger.info("Building retrieval index - this may take a few minutes on first run ...")
        _recommender = RecommendationAssistant(settings)
        _agent = build_agent(settings)
        _index_ready = True
        logger.info("Index ready. Server accepting chat requests.")
    except Exception as exc:
        logger.exception("Index build failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    thread = threading.Thread(target=_build_index_sync, daemon=True)
    thread.start()
    logger.info("Server started - index building in background ...")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="RTG ShopPilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten to your Shopify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class CurrentProduct(BaseModel):
    title: str
    price: float
    handle: str
    description: str | None = None


class PageContext(BaseModel):
    pageType: str  # "product" | "collection" | "cart" | "home" | "other"
    currentProduct: CurrentProduct | None = None
    collectionHandle: str | None = None


class CustomerProfile(BaseModel):
    name: str | None = None
    sleepIssues: list[str] = Field(default_factory=list)
    firmnessPref: str | None = None
    sleepPosition: str | None = None
    budget: dict[str, float | None] | None = None  # {min, max}
    partnerSharing: bool | None = None
    previousMattress: str | None = None
    previousMattressIssue: str | None = None
    dealbreakers: list[str] = Field(default_factory=list)
    interestedProducts: list[str] = Field(default_factory=list)
    turnCount: int = 0


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    profile: CustomerProfile | None = None
    pageContext: PageContext | None = None
    sessionId: str | None = None  # used as Supermemory container tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_pydantic_ai_history(history: list[ChatMessage]) -> list[ModelMessage]:
    """Convert frontend ChatMessage list to pydantic-ai ModelMessage list."""
    messages: list[ModelMessage] = []
    for msg in history:
        if msg.role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=msg.content)]))
    return messages


def _enrich_message(
    message: str,
    page_context: PageContext | None,
    profile: CustomerProfile | None,
) -> str:
    """
    Prepend page-level and profile context so the agent starts with full
    awareness of where the customer is and what we already know about them.
    """
    parts: list[str] = []

    if page_context:
        if page_context.pageType == "product" and page_context.currentProduct:
            p = page_context.currentProduct
            parts.append(
                f"[Page context: customer is viewing '{p.title}' (handle: {p.handle}) "
                f"priced at ${p.price:.2f}"
                + (f" — {p.description[:200]}" if p.description else "")
                + "]"
            )
        elif page_context.pageType == "collection" and page_context.collectionHandle:
            parts.append(
                f"[Page context: customer is browsing collection '{page_context.collectionHandle}']"
            )
        elif page_context.pageType == "cart":
            parts.append("[Page context: customer is on the cart page]")

    if profile:
        profile_parts: list[str] = []
        if profile.sleepIssues:
            profile_parts.append(f"sleep issues: {', '.join(profile.sleepIssues)}")
        if profile.firmnessPref:
            profile_parts.append(f"prefers {profile.firmnessPref} firmness")
        if profile.sleepPosition:
            profile_parts.append(f"sleeps {profile.sleepPosition}")
        if profile.budget:
            mn = profile.budget.get("min")
            mx = profile.budget.get("max")
            if mn and mx:
                profile_parts.append(f"budget ${mn:.0f}–${mx:.0f}")
            elif mx:
                profile_parts.append(f"budget under ${mx:.0f}")
            elif mn:
                profile_parts.append(f"budget over ${mn:.0f}")
        if profile.previousMattressIssue:
            profile_parts.append(f"previous issue: {profile.previousMattressIssue}")
        if profile_parts:
            parts.append(f"[Known profile: {'; '.join(profile_parts)}]")

    if parts:
        context_block = "\n".join(parts)
        return f"{context_block}\n\n{message}"

    return message


def _build_updated_profile(
    profile: CustomerProfile | None,
    message: str,
) -> dict[str, Any]:
    """
    Return an updated profile dict. In the minimal version we just increment
    the turn counter; a richer implementation would run NLP extraction here.
    """
    base = profile.model_dump() if profile else CustomerProfile().model_dump()
    base["turnCount"] = base.get("turnCount", 0) + 1

    # Simple heuristic: detect budget mention and update if not already set
    lower = message.lower()
    if base.get("budget") is None:
        import re
        under_m = re.search(r"(?:under|below|less than|max)\s*\$?(\d+)", lower)
        if under_m:
            base["budget"] = {"max": float(under_m.group(1))}

    return base


def _sse(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    if _agent is None or _recommender is None:
        raise HTTPException(503, "Server is still initialising — retry in a moment.")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # ── Build pydantic-ai message history ──────────────────────────
            pydantic_history = _build_pydantic_ai_history(request.history[:-20])
            # Keep last 20 turns in history; the current message is sent separately

            # ── Enrich the current user message ────────────────────────────
            enriched_message = _enrich_message(
                request.message,
                request.pageContext,
                request.profile,
            )

            # ── Memory manager — one per request, keyed by session ─────────
            memory_tag = (
                request.sessionId
                or settings.supermemory_container_tag
            )
            memory = MemoryManager(settings, container_tag=memory_tag)

            deps = AgentDeps(recommender=_recommender, memory=memory)

            # ── Stream agent response ───────────────────────────────────────
            async with _agent.run_stream(
                enriched_message,
                deps=deps,
                message_history=pydantic_history,
            ) as result:
                async for delta in result.stream_text(delta=True):
                    if delta:
                        yield _sse({"type": "delta", "text": delta})

            # ── Emit updated profile ────────────────────────────────────────
            updated_profile = _build_updated_profile(request.profile, request.message)
            yield _sse({"type": "profile", "profile": updated_profile})

            # ── Done ────────────────────────────────────────────────────────
            yield _sse("[DONE]")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during /chat stream: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse("[DONE]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    # Always return 200 so Railway healthcheck passes immediately.
    # index_ready tells the frontend whether chat is available yet.
    return {
        "status": "ok",
        "index_ready": _index_ready,
        "agent_ready": _agent is not None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

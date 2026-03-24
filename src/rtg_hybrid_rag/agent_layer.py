from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rtg_hybrid_rag.assistant import RecommendationAssistant
from rtg_hybrid_rag.memory import MemoryManager
from rtg_hybrid_rag.models import UserTurn
from rtg_hybrid_rag.settings import Settings
from rtg_hybrid_rag.skills import list_skill_names, render_skill_guides


@dataclass
class AgentDeps:
    recommender: RecommendationAssistant
    memory: MemoryManager


def _build_model(settings: Settings) -> OpenAIChatModel:
    if settings.agent_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        provider = OpenAIProvider(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        return OpenAIChatModel(settings.agent_model, provider=provider)

    if settings.agent_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        provider = OpenAIProvider(api_key=settings.openai_api_key)
        return OpenAIChatModel(settings.agent_model, provider=provider)

    raise ValueError(f"Unsupported agent provider: {settings.agent_provider}")


def build_agent(settings: Settings) -> Agent[AgentDeps, str]:
    model = _build_model(settings)
    agent = Agent(
        model,
        deps_type=AgentDeps,
        name="rtg-mattress-agent",
        instructions=dedent(
            """
            You are a conversational mattress and sleep-product assistant.
            You do not invent products. Use the `recommend_products` tool whenever the user is asking
            what product to buy, compare, shortlist, refine, or follow up on prior preferences.
            Use `get_sales_skills` when you need structured selling guidance for scenarios like cooling,
            pressure relief, firmness, budget, brand-locked requests, adjustable bases, or kids mattresses.
            Use `save_memory` when the conversation reveals durable preferences like size, firmness,
            brand preference, budget range, medical/sleep issues, or product decisions.

            Important:
            - On every tool call, pass a self-contained request that includes all active constraints
              from the conversation so far, not just the latest delta.
            - Preserve prior constraints unless the user clearly changes them.
            - Respect the tool output as the source of truth for product names, prices, and ranking.
            - Keep answers concise and helpful. If the tool output already contains a good shortlist,
              summarize it rather than rewriting everything from scratch.
            """
        ).strip(),
    )

    @agent.instructions
    def memory_instructions(ctx: RunContext[AgentDeps]) -> str:
        if not ctx.deps.memory.enabled:
            return "Persistent memory is disabled for this session."
        user_query = ""
        try:
            if ctx.messages:
                last_message = ctx.messages[-1]
                parts = getattr(last_message, "parts", []) or []
                text_parts: list[str] = []
                for part in parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        text_parts.append(content)
                user_query = "\n".join(text_parts).strip()
        except Exception:
            user_query = ""

        memory_context = ctx.deps.memory.fetch_context(user_query or "user preferences")
        return (
            "Persistent memory is enabled through Supermemory.\n"
            f"Memory container tag: {ctx.deps.memory.container_tag}\n\n"
            "Injected memory context for this turn:\n"
            f"{memory_context.render()}\n\n"
            "Use this memory context directly when helpful. "
            "Use `save_memory` when the conversation reveals durable preferences worth keeping."
        )

    @agent.tool_plain
    def get_sales_skills(skill_names: list[str] | None = None) -> str:
        """Return structured domain guidance for common mattress-sales conversation types.

        Args:
            skill_names: Optional skill names such as cooling, back_pain, pressure_relief,
                firmness, budget, brand, adjustable_base, or kids.
        """

        return render_skill_guides(skill_names or list_skill_names())

    @agent.tool
    def save_memory(ctx: RunContext[AgentDeps], memory: str) -> str:
        """Save a durable user preference or conversation summary to Supermemory.

        Args:
            memory: A concise memory worth keeping for future conversations.
        """

        ctx.deps.memory.add_memory(memory, metadata={"type": "preference"})
        return "Memory saved."

    @agent.tool
    def recommend_products(ctx: RunContext[AgentDeps], request: str) -> str:
        """Run hybrid retrieval plus reranking for a self-contained product request.

        Args:
            request: A complete user-need summary including all active constraints such as size,
                budget, sleep position, temperature needs, comfort, brand, or product type.
        """

        result = ctx.deps.recommender.recommend([UserTurn(role="user", content=request)])
        lines = [
            "Grounded recommendation result:",
            f"Normalized query: {result.state.normalized_query}",
            "",
            "Top ranked products:",
        ]
        for rank, candidate in enumerate(result.candidates, start=1):
            product = candidate.product
            lines.append(
                (
                    f"{rank}. {product.theme} | brand={product.specialty_brand or '-'} | "
                    f"category={product.product_category} | price=${product.sale_price:.0f} | "
                    f"size={product.mattress_size or '-'} | type={product.mattress_type or '-'} | "
                    f"comfort={product.comfort or '-'} | temp={product.temperature_management or '-'} | "
                    f"support={product.support_level or '-'} | pressure={product.pressure_relief or '-'} | "
                    f"rerank_score={candidate.rerank_score:.3f}"
                )
            )
            if candidate.match_reasons:
                lines.append(f"   reason: {candidate.match_reasons[0]}")

        lines.extend(
            [
                "",
                "Grounded response draft:",
                result.answer,
            ]
        )
        return "\n".join(lines)

    return agent

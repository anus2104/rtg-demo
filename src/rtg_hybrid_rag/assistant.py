from __future__ import annotations

from textwrap import dedent

from rtg_hybrid_rag.conversation import ConversationAnalyzer
from rtg_hybrid_rag.indexer import IndexBuilder
from rtg_hybrid_rag.llm import OpenAICompatibleClient
from rtg_hybrid_rag.models import RecommendationResult, RetrievedCandidate, UserTurn
from rtg_hybrid_rag.reranker import CohereReranker
from rtg_hybrid_rag.retrieval import HybridRetriever
from rtg_hybrid_rag.settings import Settings


class RecommendationAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index = IndexBuilder(settings).build_or_load()
        self.analyzer = ConversationAnalyzer()
        self.retriever = HybridRetriever(self.index, settings)
        self.reranker = CohereReranker(settings)
        self.client = OpenAICompatibleClient(settings)

    def recommend(self, turns: list[UserTurn]) -> RecommendationResult:
        state = self.analyzer.build_state(turns)
        retrieved = self.retriever.retrieve(state)
        constrained = self._apply_hard_filters(state, retrieved)
        reranked = self.reranker.rerank(state, constrained)
        top_candidates = reranked[: self.settings.final_top_n]
        answer = self._draft_answer(state, top_candidates)
        return RecommendationResult(
            state=state,
            candidates=top_candidates,
            answer=answer,
        )

    def _apply_hard_filters(
        self,
        state,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        working = list(candidates)

        if state.product_category:
            matching = [
                candidate
                for candidate in working
                if candidate.product.product_category == state.product_category
            ]
            if matching:
                working = matching

        if state.preferred_sizes:
            matching = [
                candidate
                for candidate in working
                if candidate.product.mattress_size in state.preferred_sizes
            ]
            if matching:
                working = matching

        if state.budget_max is not None:
            strict_matching = [
                candidate
                for candidate in working
                if candidate.product.sale_price <= state.budget_max
            ]
            if len(strict_matching) >= min(3, self.settings.final_top_n):
                working = strict_matching
            else:
                budget_ceiling = state.budget_max * 1.1
                near_matching = [
                    candidate
                    for candidate in working
                    if candidate.product.sale_price <= budget_ceiling
                ]
                if near_matching:
                    working = near_matching

        return working or candidates

    def _draft_answer(self, state, candidates: list[RetrievedCandidate]) -> str:
        shortlist = []
        for rank, candidate in enumerate(candidates, start=1):
            product = candidate.product
            shortlist.append(
                {
                    "rank": rank,
                    "theme": product.theme,
                    "sku_number": product.sku_number,
                    "brand": product.specialty_brand,
                    "price": product.sale_price,
                    "size": product.mattress_size,
                    "type": product.mattress_type,
                    "comfort": product.comfort,
                    "pressure_relief": product.pressure_relief,
                    "sleep_position": product.sleep_position,
                    "support_level": product.support_level,
                    "temperature_management": product.temperature_management,
                    "reason": ", ".join(candidate.match_reasons) or "Strong overall fit.",
                }
            )

        system_prompt = dedent(
            """
            You are a grounded mattress recommendation assistant.
            Use only the supplied shortlisted products.
            Answer in a conversational retail-assistant tone.
            Mention the top 3-5 products, explain why they fit, and call out tradeoffs.
            Respect hard constraints like size and budget. If a product is above budget, say so clearly and do not lead with it if in-budget options exist.
            If the conversation is multi-turn, explicitly treat earlier user constraints as active unless contradicted.
            If the user profile implies uncertainty, ask one sharp follow-up question at the end.
            """
        ).strip()
        user_prompt = (
            f"Conversation state:\n{state.model_dump_json(indent=2)}\n\n"
            f"Shortlisted products:\n{shortlist}\n\n"
            "Write the answer."
        )
        return self.client.text_completion(
            provider=self.settings.generation_provider,
            model=self.settings.generation_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )

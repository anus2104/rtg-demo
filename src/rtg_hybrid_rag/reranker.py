from __future__ import annotations

import json

from rtg_hybrid_rag.llm import OpenAICompatibleClient
from rtg_hybrid_rag.models import ConversationState, RetrievedCandidate
from rtg_hybrid_rag.settings import Settings


class CohereReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAICompatibleClient(settings)

    def rerank(
        self,
        state: ConversationState,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        shortlist = candidates[: self.settings.rerank_top_n]
        payload = []
        for candidate in shortlist:
            product = candidate.product
            payload.append(
                {
                    "product_id": product.product_id,
                    "theme": product.theme,
                    "brand": product.specialty_brand,
                    "category": product.product_category,
                    "size": product.mattress_size,
                    "type": product.mattress_type,
                    "comfort": product.comfort,
                    "pressure_relief": product.pressure_relief,
                    "sleep_position": product.sleep_position,
                    "support_level": product.support_level,
                    "temperature_management": product.temperature_management,
                    "sale_price": product.sale_price,
                    "summary": product.collection_copy or product.customer_description,
                    "hybrid_score": round(candidate.hybrid_score, 4),
                }
            )

        system_prompt = (
            "You are a product reranker for mattress and sleep-product retrieval. "
            "Given a user profile and candidate products, return strict JSON with a key "
            "'ranked' whose value is an array of objects. Each object must include "
            "'product_id', 'score' (0-1), and 'reason'. Favor products that satisfy the "
            "latest user intent, hard constraints like size and budget, then softer fit such "
            "as cooling, support, and comfort. Keep reasons short."
        )
        user_prompt = (
            f"User profile:\n{state.model_dump_json(indent=2)}\n\n"
            f"Candidates:\n{json.dumps(payload, indent=2)}\n\n"
            "Return JSON only."
        )

        response = self.client.json_completion(
            provider=self.settings.rerank_provider,
            model=self.settings.rerank_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        ranking = response.get("ranked", [])
        rank_map = {
            item["product_id"]: (float(item.get("score", 0.0)), str(item.get("reason", "")))
            for item in ranking
            if item.get("product_id")
        }

        rescored: list[RetrievedCandidate] = []
        for candidate in shortlist:
            llm_score, reason = rank_map.get(candidate.product.product_id, (0.0, ""))
            candidate.rerank_score = candidate.hybrid_score * 0.35 + llm_score * 0.65
            candidate.match_reasons = [reason] if reason else []
            rescored.append(candidate)

        return sorted(rescored, key=lambda item: item.rerank_score, reverse=True)

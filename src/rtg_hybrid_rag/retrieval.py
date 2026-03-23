from __future__ import annotations

from rtg_hybrid_rag.catalog import numeric_rating, tokenize
from rtg_hybrid_rag.indexer import CatalogIndex
from rtg_hybrid_rag.llm import OpenAICompatibleClient
from rtg_hybrid_rag.models import ConversationState, RetrievedCandidate
from rtg_hybrid_rag.settings import Settings


class HybridRetriever:
    def __init__(self, index: CatalogIndex, settings: Settings) -> None:
        self.index = index
        self.settings = settings
        self.client = OpenAICompatibleClient(settings)

    def retrieve(self, state: ConversationState) -> list[RetrievedCandidate]:
        query_embedding = self.client.embed_texts([state.normalized_query])[0]
        where_filter = self._build_where_filter(state)
        dense_results = self.index.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.settings.initial_recall_k,
            where=where_filter if where_filter else None,
        )

        dense_scores: dict[str, float] = {}
        dense_ids = dense_results["ids"][0]
        dense_distances = dense_results["distances"][0]
        for product_id, distance in zip(dense_ids, dense_distances, strict=False):
            dense_scores[product_id] = 1.0 - float(distance)

        sparse_tokens = tokenize(state.normalized_query)
        sparse_raw_scores = self.index.bm25.get_scores(sparse_tokens)
        sparse_order = sorted(
            zip(self.index.products, sparse_raw_scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )[: self.settings.initial_recall_k]
        sparse_scores = {product.product_id: float(score) for product, score in sparse_order}

        max_sparse = max(sparse_scores.values(), default=1.0) or 1.0
        combined_ids = set(dense_scores) | set(sparse_scores)
        candidates: list[RetrievedCandidate] = []
        for product_id in combined_ids:
            product = self.index.by_id[product_id]
            dense_score = dense_scores.get(product_id, 0.0)
            sparse_score = sparse_scores.get(product_id, 0.0) / max_sparse
            hybrid_score = (
                dense_score * self.settings.hybrid_dense_weight
                + sparse_score * self.settings.hybrid_sparse_weight
            )
            candidates.append(
                RetrievedCandidate(
                    product=product,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    hybrid_score=hybrid_score,
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item: self._apply_business_bias(item, state),
            reverse=True,
        )
        return ranked[: self.settings.initial_recall_k]

    def _build_where_filter(self, state: ConversationState) -> dict | None:
        clauses: list[dict] = []

        if state.product_category:
            clauses.append({"category": {"$eq": state.product_category}})
        if state.preferred_sizes:
            if len(state.preferred_sizes) == 1:
                clauses.append({"size": {"$eq": state.preferred_sizes[0]}})
            else:
                clauses.append({"size": {"$in": state.preferred_sizes}})
        if state.preferred_brands:
            if len(state.preferred_brands) == 1:
                clauses.append({"brand": {"$eq": state.preferred_brands[0]}})
            else:
                clauses.append({"brand": {"$in": state.preferred_brands}})
        if state.mattress_types:
            if len(state.mattress_types) == 1:
                clauses.append({"type": {"$eq": state.mattress_types[0]}})
            else:
                clauses.append({"type": {"$in": state.mattress_types}})
        if state.budget_min is not None:
            clauses.append({"sale_price": {"$gte": state.budget_min}})
        if state.budget_max is not None:
            clauses.append({"sale_price": {"$lte": state.budget_max}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _apply_business_bias(self, candidate: RetrievedCandidate, state: ConversationState) -> float:
        score = candidate.hybrid_score
        product = candidate.product

        if state.product_category and product.product_category == state.product_category:
            score += 0.18
        elif state.product_category:
            score -= 0.1

        if state.preferred_sizes and product.mattress_size in state.preferred_sizes:
            score += 0.12
        elif state.preferred_sizes and product.mattress_size and product.mattress_size not in state.preferred_sizes:
            score -= 0.06

        if state.preferred_brands and (product.specialty_brand or "") in state.preferred_brands:
            score += 0.08

        if state.mattress_types and (product.mattress_type or "") in state.mattress_types:
            score += 0.08

        if state.comfort_preferences and (product.comfort or "") in state.comfort_preferences:
            score += 0.07

        if state.budget_max is not None:
            if product.sale_price <= state.budget_max:
                score += 0.08
            else:
                score -= min(0.12, (product.sale_price - state.budget_max) / max(state.budget_max, 1))

        if state.budget_min is not None and product.sale_price >= state.budget_min:
            score += 0.04

        if "cooling" in state.needs:
            temp_score = numeric_rating(product.temperature_management) or 0
            score += temp_score * 0.02
        if "pressure_relief" in state.needs:
            pressure_score = numeric_rating(product.pressure_relief) or 0
            score += pressure_score * 0.02
        if "back_pain" in state.needs:
            support_score = numeric_rating(product.support_level) or 0
            score += support_score * 0.02

        return score

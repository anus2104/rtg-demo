from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductDocument(BaseModel):
    product_id: str
    sku_number: str
    theme: str
    customer_description: str
    sale_price: float
    regular_price: float
    specialty_brand: str | None = None
    proprietary: str | None = None
    collection_copy: str | None = None
    mattress_designation: str | None = None
    mattress_size: str | None = None
    mattress_type: str | None = None
    pressure_relief: str | None = None
    sleep_position: str | None = None
    support_level: str | None = None
    temperature_management: str | None = None
    comfort: str | None = None
    product_category: str
    search_text: str


class UserTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationState(BaseModel):
    raw_query: str
    normalized_query: str
    product_category: str | None = None
    preferred_sizes: list[str] = Field(default_factory=list)
    preferred_brands: list[str] = Field(default_factory=list)
    mattress_types: list[str] = Field(default_factory=list)
    sleep_positions: list[str] = Field(default_factory=list)
    comfort_preferences: list[str] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list)
    budget_min: float | None = None
    budget_max: float | None = None
    user_profile: str = ""
    is_multi_turn: bool = False
    turn_count: int = 1


class RetrievedCandidate(BaseModel):
    product: ProductDocument
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rerank_score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    state: ConversationState
    candidates: list[RetrievedCandidate]
    answer: str

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str
    description: str
    turns: list[str]
    expected_category: str | None = None
    expected_sizes: list[str] = Field(default_factory=list)
    expected_brands: list[str] = Field(default_factory=list)
    expected_types: list[str] = Field(default_factory=list)
    expected_comforts: list[str] = Field(default_factory=list)
    expected_sleep_positions: list[str] = Field(default_factory=list)
    must_include_themes_any: list[str] = Field(default_factory=list)
    max_price: float | None = None
    min_price: float | None = None
    top_k: int = 5


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="single_turn_hot_side_budget_queen",
        description="Cooling-focused queen mattress under budget for a side sleeper.",
        turns=[
            "I sleep hot, mostly on my side, and want a queen mattress under $2000.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        max_price=2000,
        must_include_themes_any=[
            "SMART MATCH 2.0",
            "TUFT & NEEDLE STITCH TN3.1",
            "TUFT & NEEDLE STITCH TN9.1",
            "LEVEL THREE MEDIUM PILLOWTOP",
            "LEVEL THREE PLUSH",
        ],
    ),
    EvalCase(
        case_id="single_turn_adjustable_base_queen",
        description="Explicit adjustable base request should not return mattresses.",
        turns=[
            "I need a queen adjustable base under $2000 with head and foot adjustability.",
        ],
        expected_category="adjustable_base",
        expected_sizes=["Queen"],
        max_price=2000,
        must_include_themes_any=[
            "BASELOGIC SILVER",
            "BASELOGIC PLATINUM",
            "TEMPUR-ERGO 3.0",
            "EASE 4.0",
        ],
    ),
    EvalCase(
        case_id="single_turn_tempur_cooling_queen",
        description="Brand-constrained Tempur-Pedic cooling request.",
        turns=[
            "Show me a queen Tempur-Pedic mattress with strong cooling and a medium feel.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        expected_brands=["TEMPURPEDIC"],
        must_include_themes_any=[
            "ADAPT 2.0 MEDIUM",
            "ADAPT 2.0 MEDIUM HYBRID",
            "PROBREEZE 2.0 MEDIUM HYBRID",
            "PROADAPT 2.0 MEDIUM",
        ],
    ),
    EvalCase(
        case_id="single_turn_firm_king_stomach_under_1500",
        description="Firm king mattress for stomach/back sleeper on a budget.",
        turns=[
            "I need a king mattress under $1500 with a firmer feel. I mostly sleep on my stomach and back.",
        ],
        expected_category="mattress",
        expected_sizes=["King"],
        max_price=1500,
        expected_comforts=["Firm", "Extra Firm"],
        must_include_themes_any=[
            "TUFT & NEEDLE STITCH TN5.1",
            "TUFT & NEEDLE TN1",
            "FORESTER",
        ],
    ),
    EvalCase(
        case_id="multi_turn_back_pain_hot_budget",
        description="Multi-turn refinement should preserve earlier constraints.",
        turns=[
            "I need a queen mattress for back pain.",
            "I sleep hot and want medium comfort.",
            "Keep it under $2200 and I mostly sleep on my side.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        max_price=2200,
        expected_comforts=["Medium"],
        must_include_themes_any=[
            "CASPER SNOW 2.0",
            "ADAPT 2.0 MEDIUM",
            "SMART MATCH 2.0",
            "LEVEL THREE MEDIUM PILLOWTOP",
        ],
    ),
    EvalCase(
        case_id="multi_turn_pivot_to_adjustable_base",
        description="Later turn pivots the category from mattress to adjustable base.",
        turns=[
            "I was looking for a queen mattress around $2000.",
            "Actually I already picked the mattress. Now I just need a queen adjustable base under $1700.",
        ],
        expected_category="adjustable_base",
        expected_sizes=["Queen"],
        max_price=1700,
        must_include_themes_any=[
            "BASELOGIC SILVER",
            "BASELOGIC PLATINUM",
            "TEMPUR-ERGO 3.0",
            "EASE 4.0",
        ],
    ),
    EvalCase(
        case_id="single_turn_soft_queen_side_under_1800",
        description="User wants a softer queen mattress for side sleeping without going too expensive.",
        turns=[
            "I’m a side sleeper and want a softer queen mattress under $1800.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        max_price=1800,
        expected_comforts=["Soft", "Extra Soft"],
        must_include_themes_any=[
            "LEVEL THREE PLUSH",
            "ARADA CANYON",
            "KERRY PARK",
            "HASTINGS",
        ],
    ),
    EvalCase(
        case_id="single_turn_queen_hybrid_under_1200",
        description="Budget hybrid request should stay inside the hybrid lane.",
        turns=[
            "Show me a queen hybrid mattress under $1200.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        expected_types=["Hybrid"],
        max_price=1200,
        must_include_themes_any=[
            "TUFT & NEEDLE STITCH TN3.1",
            "TUFT & NEEDLE STITCH TN5.1",
            "BROAD PEAK MEDIUM HYBRID",
            "HASTINGS",
            "OCEANA MIST",
        ],
    ),
    EvalCase(
        case_id="single_turn_queen_foam_under_1000",
        description="Explicit foam request with a tight budget.",
        turns=[
            "I want a queen foam mattress under $1000.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        expected_types=["Foam"],
        max_price=1000,
        must_include_themes_any=[
            "CASPER CLOUD ONE",
            "CASPER ONYX 2.0",
            "MOLECULE CORE",
            "BRADFORD HILL",
            "NORTHGATE",
        ],
    ),
    EvalCase(
        case_id="single_turn_split_cal_king_adjustable_base",
        description="Specific base-size request should return only matching adjustable bases.",
        turns=[
            "I need a split California king adjustable base under $1700.",
        ],
        expected_category="adjustable_base",
        expected_sizes=["Split California King"],
        max_price=1700,
        must_include_themes_any=[
            "BASELOGIC SILVER",
            "BASELOGIC PLATINUM",
            "TEMPUR-ERGO 3.0",
            "EASE 4.0",
        ],
    ),
    EvalCase(
        case_id="single_turn_pressure_relief_queen_under_2500",
        description="Pressure-relief focused shopper with shoulder and hip pain.",
        turns=[
            "My shoulders and hips get sore. I want a queen mattress under $2500 with really good pressure relief.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        max_price=2500,
        must_include_themes_any=[
            "ADAPT 2.0 MEDIUM",
            "TUFT & NEEDLE STITCH TN9.1",
            "LEVEL FOUR PLUSH PILLOWTOP",
            "LEVEL THREE MEDIUM PILLOWTOP",
            "NORTHGATE",
        ],
    ),
    EvalCase(
        case_id="multi_turn_hybrid_budget_refinement",
        description="Multi-turn refinement should narrow a broad request into an affordable hybrid shortlist.",
        turns=[
            "I need a queen mattress.",
            "Actually I want a hybrid if possible.",
            "Try to keep it under $1200.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        expected_types=["Hybrid"],
        max_price=1200,
        must_include_themes_any=[
            "TUFT & NEEDLE STITCH TN3.1",
            "TUFT & NEEDLE STITCH TN5.1",
            "BROAD PEAK MEDIUM HYBRID",
            "HASTINGS",
            "OCEANA MIST",
        ],
    ),
    EvalCase(
        case_id="multi_turn_tempur_medium_cooling",
        description="Multi-turn premium shopper narrows into a Tempur-Pedic medium cooling request.",
        turns=[
            "I’m looking for a queen mattress and I usually sleep hot.",
            "I want to stay with Tempur-Pedic.",
            "Prefer a medium feel.",
        ],
        expected_category="mattress",
        expected_sizes=["Queen"],
        expected_brands=["TEMPURPEDIC"],
        expected_comforts=["Medium"],
        must_include_themes_any=[
            "ADAPT 2.0 MEDIUM",
            "ADAPT 2.0 MEDIUM HYBRID",
            "PROBREEZE 2.0 MEDIUM HYBRID",
            "PROADAPT 2.0 MEDIUM",
            "PROADAPT 2.0 MEDIUM HYBRID",
        ],
    ),
]

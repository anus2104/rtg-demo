from __future__ import annotations

import re
from pathlib import Path

from rtg_hybrid_rag.models import ConversationState, UserTurn


SIZE_PATTERNS = {
    "Twin XL": ("twin xl", "twxl", "twx"),
    "Twin": (" twin ",),
    "Full": (" full ",),
    "Queen": (" queen ",),
    "King": (" king ",),
    "Split King": ("split king",),
    "California King": ("california king", "cal king"),
    "Split California King": ("split california king",),
}

BRANDS = [
    "CASPER",
    "TEMPURPEDIC",
    "BEDGEAR",
    "BEAUTYREST",
    "HELIX",
    "TUFT & NEEDLE",
    "SEALY",
    "SIMMONS",
    "STEARNS & FOSTER",
    "KINGSDOWN",
]


def _extract_sizes(query: str) -> list[str]:
    lowered = f" {query.lower()} "
    matches: list[str] = []
    for canonical, patterns in SIZE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            matches.append(canonical)
    return matches


def _extract_budget(query: str) -> tuple[float | None, float | None]:
    lowered = query.lower().replace(",", "")
    under_match = re.search(r"(under|below|less than|max(?:imum)?)\s+\$?(\d+(?:\.\d+)?)", lowered)
    if under_match:
        return None, float(under_match.group(2))
    over_match = re.search(r"(above|over|at least|min(?:imum)?)\s+\$?(\d+(?:\.\d+)?)", lowered)
    if over_match:
        return float(over_match.group(2)), None
    range_match = re.search(
        r"\$?(\d+(?:\.\d+)?)\s*(?:-|to)\s*\$?(\d+(?:\.\d+)?)",
        lowered,
    )
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    return None, None


def _extract_brands(query: str) -> list[str]:
    lowered = query.lower()
    return [brand for brand in BRANDS if brand.lower() in lowered]


def _extract_comfort(query: str) -> list[str]:
    lowered = query.lower()
    matches: list[str] = []
    if any(token in lowered for token in ("plush", "soft", "medium-soft", "pressure relief")):
        matches.append("Soft")
    if "medium" in lowered:
        matches.append("Medium")
    if any(token in lowered for token in ("firm", "extra firm", "sturdy")):
        matches.append("Firm")
    return sorted(set(matches))


def _extract_sleep_positions(query: str) -> list[str]:
    lowered = query.lower()
    matches: list[str] = []
    for pos in ("side", "back", "stomach"):
        if pos in lowered:
            matches.append(pos.title())
    return matches


def _extract_mattress_types(query: str) -> list[str]:
    lowered = query.lower()
    matches: list[str] = []
    for mattress_type in ("foam", "hybrid", "innerspring"):
        if mattress_type in lowered:
            matches.append(mattress_type.title())
    return matches


def _extract_needs(query: str) -> list[str]:
    lowered = query.lower()
    rules = {
        "cooling": ("sleep hot", "hot sleeper", "cooling", "runs hot", "warm at night"),
        "back_pain": ("back pain", "lower back", "lumbar"),
        "pressure_relief": ("pressure", "shoulder pain", "hip pain", "pressure relief"),
        "motion_isolation": ("motion transfer", "partner moves", "motion isolation"),
        "snoring": ("snore", "snoring"),
        "adjustable_base": ("adjustable base", "power base", "base"),
    }
    matches: list[str] = []
    for need, patterns in rules.items():
        if any(pattern in lowered for pattern in patterns):
            matches.append(need)
    return matches


def _extract_category(query: str) -> str | None:
    lowered = query.lower()
    if "base" in lowered or "adjustable" in lowered:
        return "adjustable_base"
    if "mattress" in lowered or any(token in lowered for token in ("foam", "hybrid", "innerspring")):
        return "mattress"
    return None


def _merge_values(existing: list[str], fresh: list[str]) -> list[str]:
    merged = list(existing)
    for value in fresh:
        if value not in merged:
            merged.append(value)
    return merged


class ConversationAnalyzer:
    def build_state(self, turns: list[UserTurn]) -> ConversationState:
        user_turns = [turn.content for turn in turns if turn.role == "user"]
        if not user_turns:
            raise ValueError("At least one user turn is required")

        latest_query = user_turns[-1]
        state = ConversationState(
            raw_query=latest_query,
            normalized_query=latest_query.strip(),
            is_multi_turn=len(user_turns) > 1,
            turn_count=len(user_turns),
        )

        for query in user_turns:
            state.preferred_sizes = _merge_values(state.preferred_sizes, _extract_sizes(query))
            state.preferred_brands = _merge_values(state.preferred_brands, _extract_brands(query))
            state.mattress_types = _merge_values(state.mattress_types, _extract_mattress_types(query))
            state.sleep_positions = _merge_values(state.sleep_positions, _extract_sleep_positions(query))
            state.comfort_preferences = _merge_values(
                state.comfort_preferences, _extract_comfort(query)
            )
            state.needs = _merge_values(state.needs, _extract_needs(query))
            budget_min, budget_max = _extract_budget(query)
            if budget_min is not None:
                state.budget_min = budget_min
            if budget_max is not None:
                state.budget_max = budget_max
            category = _extract_category(query)
            if category:
                state.product_category = category

        summary_parts = []
        if state.product_category:
            summary_parts.append(f"Category: {state.product_category}")
        if state.preferred_sizes:
            summary_parts.append(f"Sizes: {', '.join(state.preferred_sizes)}")
        if state.preferred_brands:
            summary_parts.append(f"Brands: {', '.join(state.preferred_brands)}")
        if state.mattress_types:
            summary_parts.append(f"Types: {', '.join(state.mattress_types)}")
        if state.sleep_positions:
            summary_parts.append(f"Sleep positions: {', '.join(state.sleep_positions)}")
        if state.comfort_preferences:
            summary_parts.append(f"Comfort: {', '.join(state.comfort_preferences)}")
        if state.needs:
            summary_parts.append(f"Needs: {', '.join(state.needs)}")
        if state.budget_min is not None or state.budget_max is not None:
            summary_parts.append(f"Budget: {state.budget_min or 0:.0f}-{state.budget_max or 999999:.0f}")

        state.user_profile = " | ".join(summary_parts)
        if state.user_profile:
            state.normalized_query = f"{state.user_profile}. Latest ask: {latest_query}"
        return state

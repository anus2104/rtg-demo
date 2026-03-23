from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillGuide:
    name: str
    when_to_use: str
    guidance: tuple[str, ...]


SKILL_GUIDES: dict[str, SkillGuide] = {
    "cooling": SkillGuide(
        name="cooling",
        when_to_use="Use when the user says they sleep hot, run warm, or wants a cooler bed.",
        guidance=(
            "Prioritize temperature management, then mattress construction and price.",
            "Cooling-focused shoppers usually care about hybrid airflow, breathable materials, and premium cooling lines.",
            "Do not overstate cooling if the catalog only shows 2-Good temperature management.",
        ),
    ),
    "back_pain": SkillGuide(
        name="back_pain",
        when_to_use="Use when the user mentions back pain, lumbar support, or wants more support/alignment.",
        guidance=(
            "Prioritize support level first, then pressure relief and mattress type.",
            "Explain support tradeoffs clearly: too soft may feel plush but can reduce alignment for some users.",
            "Avoid saying a mattress is ideal for back pain unless support and pressure relief are both strong.",
        ),
    ),
    "pressure_relief": SkillGuide(
        name="pressure_relief",
        when_to_use="Use when the user mentions shoulder pain, hip pain, side-sleeper discomfort, or pressure relief.",
        guidance=(
            "Prioritize pressure relief first, then comfort and sleep position.",
            "For side sleepers, softer or medium models with stronger pressure relief usually fit better.",
            "Do not recommend firm stomach-sleeper products near the top unless the user explicitly likes firm feel.",
        ),
    ),
    "firmness": SkillGuide(
        name="firmness",
        when_to_use="Use when the user is choosing between soft, medium, firm, plush, or extra firm feels.",
        guidance=(
            "Treat firmness as a high-priority filter, not a cosmetic preference.",
            "Soft/plush usually pairs better with side-sleeper pressure relief; firm usually pairs better with stomach/back support.",
            "If you include off-firmness alternatives, label them explicitly as tradeoffs.",
        ),
    ),
    "budget": SkillGuide(
        name="budget",
        when_to_use="Use when the user has a price ceiling or asks for best value.",
        guidance=(
            "Keep in-budget options first whenever enough relevant options exist.",
            "If showing an above-budget option, explain why it is worth stretching for and say it is above budget.",
            "Separate best overall from best value if those are different products.",
        ),
    ),
    "brand": SkillGuide(
        name="brand",
        when_to_use="Use when the user names a brand such as Tempur-Pedic, Casper, or Tuft & Needle.",
        guidance=(
            "Treat explicit brand requests as hard constraints unless the user invites alternatives.",
            "Do not mix off-brand options into the main shortlist for brand-locked requests.",
            "If you mention off-brand comparisons, keep them separate and clearly optional.",
        ),
    ),
    "adjustable_base": SkillGuide(
        name="adjustable_base",
        when_to_use="Use when the user asks about bases, adjustable bases, or power bases.",
        guidance=(
            "Stay in the adjustable base category only.",
            "Treat exact base size as a hard constraint, especially split sizes.",
            "Explain value tiers using concrete differences when available, not just price.",
        ),
    ),
    "kids": SkillGuide(
        name="kids",
        when_to_use="Use when the user is shopping for a child, teen, guest kid room, or bunk room.",
        guidance=(
            "Prefer kid-designated products when the catalog has them.",
            "Explain the tradeoff between price, durability, and comfort simply.",
            "Do not overcomplicate the answer with premium features unless the user asks for them.",
        ),
    ),
}


def list_skill_names() -> list[str]:
    return sorted(SKILL_GUIDES)


def render_skill_guides(skill_names: list[str]) -> str:
    if not skill_names:
        skill_names = list_skill_names()

    lines: list[str] = []
    for skill_name in skill_names:
        guide = SKILL_GUIDES.get(skill_name)
        if not guide:
            continue
        lines.append(f"{guide.name}")
        lines.append(f"When to use: {guide.when_to_use}")
        for item in guide.guidance:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() or "No matching skill guidance found."

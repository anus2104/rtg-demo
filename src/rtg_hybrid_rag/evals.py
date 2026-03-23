from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from rtg_hybrid_rag.assistant import RecommendationAssistant
from rtg_hybrid_rag.eval_cases import EVAL_CASES, EvalCase
from rtg_hybrid_rag.models import ProductDocument, UserTurn
from rtg_hybrid_rag.settings import Settings


class DeterministicCheck(BaseModel):
    name: str
    passed: bool
    details: str


class JudgeScore(BaseModel):
    grounding: int
    constraint_adherence: int
    recommendation_quality: int
    explanation_quality: int
    user_satisfaction: int
    overall_pass: bool
    summary: str


class EvalCaseResult(BaseModel):
    case_id: str
    description: str
    turns: list[str]
    deterministic_pass: bool
    deterministic_checks: list[DeterministicCheck]
    judge: JudgeScore | None = None
    candidate_themes: list[str]
    oracle_themes: list[str]
    answer: str


class EvalSuiteResult(BaseModel):
    cases: list[EvalCaseResult]


@dataclass
class EvalRunner:
    settings: Settings
    assistant: RecommendationAssistant

    def run_case(self, case: EvalCase, use_judge: bool = True) -> EvalCaseResult:
        turns = [UserTurn(role="user", content=turn) for turn in case.turns]
        result = self.assistant.recommend(turns)
        top_candidates = result.candidates[: case.top_k]
        oracle_products = self._oracle_products(case)
        checks = self._deterministic_checks(case, top_candidates, oracle_products)
        judge = self._judge_case(case, result, oracle_products) if use_judge else None
        return EvalCaseResult(
            case_id=case.case_id,
            description=case.description,
            turns=case.turns,
            deterministic_pass=all(check.passed for check in checks),
            deterministic_checks=checks,
            judge=judge,
            candidate_themes=[candidate.product.theme for candidate in top_candidates],
            oracle_themes=[product.theme for product in oracle_products[:10]],
            answer=result.answer,
        )

    def run_suite(self, use_judge: bool = True) -> EvalSuiteResult:
        return EvalSuiteResult(cases=[self.run_case(case, use_judge=use_judge) for case in EVAL_CASES])

    def _oracle_products(self, case: EvalCase) -> list[ProductDocument]:
        products = list(self.assistant.index.products)
        if case.expected_category:
            products = [product for product in products if product.product_category == case.expected_category]
        if case.expected_sizes:
            products = [product for product in products if product.mattress_size in case.expected_sizes]
        if case.expected_brands:
            products = [product for product in products if (product.specialty_brand or "") in case.expected_brands]
        if case.expected_types:
            products = [product for product in products if (product.mattress_type or "") in case.expected_types]
        if case.expected_comforts:
            products = [product for product in products if (product.comfort or "") in case.expected_comforts]
        if case.min_price is not None:
            products = [product for product in products if product.sale_price >= case.min_price]
        if case.max_price is not None:
            products = [product for product in products if product.sale_price <= case.max_price]
        return sorted(products, key=self._oracle_score, reverse=True)

    def _oracle_score(self, product: ProductDocument) -> tuple[float, float]:
        def rating(value: str | None) -> int:
            if not value:
                return 0
            return int(str(value).split("-", 1)[0])

        score = (
            rating(product.pressure_relief)
            + rating(product.support_level)
            + rating(product.temperature_management)
        )
        return (float(score), -product.sale_price)

    def _deterministic_checks(
        self,
        case: EvalCase,
        top_candidates,
        oracle_products: list[ProductDocument],
    ) -> list[DeterministicCheck]:
        checks: list[DeterministicCheck] = []
        products = [candidate.product for candidate in top_candidates]

        if case.expected_category:
            passed = all(product.product_category == case.expected_category for product in products)
            checks.append(
                DeterministicCheck(
                    name="category_match",
                    passed=passed,
                    details=f"Expected {case.expected_category}; got {[p.product_category for p in products]}",
                )
            )
        if case.expected_sizes:
            passed = all(product.mattress_size in case.expected_sizes for product in products)
            checks.append(
                DeterministicCheck(
                    name="size_match",
                    passed=passed,
                    details=f"Expected one of {case.expected_sizes}; got {[p.mattress_size for p in products]}",
                )
            )
        if case.max_price is not None:
            passed = all(product.sale_price <= case.max_price * 1.1 for product in products[:3])
            checks.append(
                DeterministicCheck(
                    name="budget_respect_top3",
                    passed=passed,
                    details=f"Top 3 prices: {[p.sale_price for p in products[:3]]}; budget={case.max_price}",
                )
            )
        if case.expected_brands:
            passed = all((product.specialty_brand or "") in case.expected_brands for product in products[:3])
            checks.append(
                DeterministicCheck(
                    name="brand_match_top3",
                    passed=passed,
                    details=f"Expected brands {case.expected_brands}; got {[(p.specialty_brand or '') for p in products[:3]]}",
                )
            )
        if case.expected_comforts:
            passed = any((product.comfort or "") in case.expected_comforts for product in products[:3])
            checks.append(
                DeterministicCheck(
                    name="comfort_present_top3",
                    passed=passed,
                    details=f"Expected one of {case.expected_comforts}; got {[p.comfort for p in products[:3]]}",
                )
            )
        if case.must_include_themes_any:
            top_themes = {product.theme for product in products}
            passed = any(theme in top_themes for theme in case.must_include_themes_any)
            checks.append(
                DeterministicCheck(
                    name="oracle_theme_overlap",
                    passed=passed,
                    details=f"Top themes={sorted(top_themes)}; expected any of {case.must_include_themes_any}",
                )
            )

        oracle_top = {product.theme for product in oracle_products[:10]}
        overlap = oracle_top.intersection({product.theme for product in products})
        checks.append(
            DeterministicCheck(
                name="oracle_top10_overlap",
                passed=bool(overlap),
                details=f"Overlap with oracle top 10: {sorted(overlap)}",
            )
        )
        return checks

    def _judge_case(self, case: EvalCase, result, oracle_products: list[ProductDocument]) -> JudgeScore | None:
        oracle_payload = [
            {
                "theme": product.theme,
                "brand": product.specialty_brand,
                "price": product.sale_price,
                "size": product.mattress_size,
                "type": product.mattress_type,
                "comfort": product.comfort,
                "sleep_position": product.sleep_position,
                "pressure_relief": product.pressure_relief,
                "support_level": product.support_level,
                "temperature_management": product.temperature_management,
            }
            for product in oracle_products[:10]
        ]
        candidate_payload = [
            {
                "theme": candidate.product.theme,
                "brand": candidate.product.specialty_brand,
                "price": candidate.product.sale_price,
                "size": candidate.product.mattress_size,
                "type": candidate.product.mattress_type,
                "comfort": candidate.product.comfort,
                "sleep_position": candidate.product.sleep_position,
                "pressure_relief": candidate.product.pressure_relief,
                "support_level": candidate.product.support_level,
                "temperature_management": candidate.product.temperature_management,
                "rerank_score": candidate.rerank_score,
            }
            for candidate in result.candidates
        ]

        rubric = (
            "You are grading a mattress recommendation system. Score from 1 to 5. "
            "Grounding means the answer stays faithful to the returned candidates and product facts. "
            "Constraint adherence means budget/size/category/brand constraints are respected. "
            "Recommendation quality means the shortlist is sensible compared to the oracle products from the catalog. "
            "Explanation quality means the user-facing answer clearly explains tradeoffs. "
            "User satisfaction means: if you were the user asking this exact question, how satisfied would you be with this answer and shortlist."
        )
        prompt = (
            f"Eval case:\n{case.model_dump_json(indent=2)}\n\n"
            f"Returned candidates:\n{json.dumps(candidate_payload, indent=2)}\n\n"
            f"Oracle products from the catalog:\n{json.dumps(oracle_payload, indent=2)}\n\n"
            f"User-facing answer:\n{result.answer}\n\n"
            "Return strict JSON with keys grounding, constraint_adherence, recommendation_quality, "
            "explanation_quality, user_satisfaction, overall_pass, summary."
        )

        from rtg_hybrid_rag.llm import OpenAICompatibleClient

        client = OpenAICompatibleClient(self.settings)
        try:
            payload = client.json_completion(
                provider=self.settings.eval_judge_provider,
                model=self.settings.eval_judge_model,
                system_prompt=rubric,
                user_prompt=prompt,
                temperature=0.0,
            )
            return JudgeScore.model_validate(payload)
        except Exception as exc:
            return JudgeScore(
                grounding=0,
                constraint_adherence=0,
                recommendation_quality=0,
                explanation_quality=0,
                user_satisfaction=0,
                overall_pass=False,
                summary=f"Judge failed: {exc}",
            )


def render_eval_report(console: Console, suite: EvalSuiteResult) -> None:
    summary = Table(title="Eval Summary")
    summary.add_column("Case")
    summary.add_column("Deterministic")
    summary.add_column("Judge")
    summary.add_column("Satisfaction")
    summary.add_column("Summary")
    for case in suite.cases:
        judge_status = "n/a"
        satisfaction = "-"
        judge_summary = "-"
        if case.judge:
            judge_status = "pass" if case.judge.overall_pass else "fail"
            satisfaction = f"{case.judge.user_satisfaction}/5"
            judge_summary = case.judge.summary
        summary.add_row(
            case.case_id,
            "pass" if case.deterministic_pass else "fail",
            judge_status,
            satisfaction,
            judge_summary,
        )
    console.print(summary)


def render_eval_details(console: Console, suite: EvalSuiteResult) -> None:
    for case in suite.cases:
        console.print(f"\n[bold]{case.case_id}[/bold]")
        if case.judge:
            judge_table = Table(title=f"Judge Output: {case.case_id}")
            judge_table.add_column("Metric")
            judge_table.add_column("Value")
            judge_table.add_row("Grounding", f"{case.judge.grounding}/5")
            judge_table.add_row("Constraint Adherence", f"{case.judge.constraint_adherence}/5")
            judge_table.add_row("Recommendation Quality", f"{case.judge.recommendation_quality}/5")
            judge_table.add_row("Explanation Quality", f"{case.judge.explanation_quality}/5")
            judge_table.add_row("User Satisfaction", f"{case.judge.user_satisfaction}/5")
            judge_table.add_row("Overall Pass", "pass" if case.judge.overall_pass else "fail")
            console.print(judge_table)
            console.print(case.judge.summary)
        else:
            console.print("Judge output not available.")


def save_eval_report(path: Path, suite: EvalSuiteResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")


def save_eval_markdown(path: Path, suite: EvalSuiteResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_cases = len(suite.cases)
    deterministic_passes = sum(1 for case in suite.cases if case.deterministic_pass)
    judge_cases = [case for case in suite.cases if case.judge is not None]
    judge_passes = sum(1 for case in judge_cases if case.judge and case.judge.overall_pass)
    satisfaction_values = [
        case.judge.user_satisfaction for case in judge_cases if case.judge is not None
    ]
    avg_satisfaction = (
        sum(satisfaction_values) / len(satisfaction_values) if satisfaction_values else 0.0
    )

    lines: list[str] = [
        "# RTG Hybrid RAG Eval Report",
        "",
        "## Summary",
        "",
        f"- Cases run: {total_cases}",
        f"- Deterministic pass rate: {deterministic_passes}/{total_cases}",
        f"- Judge pass rate: {judge_passes}/{len(judge_cases) if judge_cases else 0}",
        f"- Average user satisfaction: {avg_satisfaction:.2f}/5" if judge_cases else "- Average user satisfaction: n/a",
        "",
    ]

    for case in suite.cases:
        lines.extend(
            [
                f"## {case.case_id}",
                "",
                f"{case.description}",
                "",
                f"- Deterministic: {'pass' if case.deterministic_pass else 'fail'}",
            ]
        )
        if case.judge:
            lines.extend(
                [
                    f"- Judge: {'pass' if case.judge.overall_pass else 'fail'}",
                    f"- Grounding: {case.judge.grounding}/5",
                    f"- Constraint adherence: {case.judge.constraint_adherence}/5",
                    f"- Recommendation quality: {case.judge.recommendation_quality}/5",
                    f"- Explanation quality: {case.judge.explanation_quality}/5",
                    f"- User satisfaction: {case.judge.user_satisfaction}/5",
                    "",
                    "### Judge Summary",
                    "",
                    case.judge.summary,
                ]
            )
        lines.extend(
            [
                "",
                "### Question",
                "",
            ]
        )
        if len(case.turns) == 1:
            lines.append(case.turns[0])
        else:
            for index, turn in enumerate(case.turns, start=1):
                lines.append(f"{index}. {turn}")
        lines.extend(
            [
                "",
                "### Deterministic Checks",
                "",
            ]
        )
        for check in case.deterministic_checks:
            lines.append(
                f"- {'PASS' if check.passed else 'FAIL'} `{check.name}`: {check.details}"
            )
        lines.extend(
            [
                "",
                "### Returned Candidate Themes",
                "",
            ]
        )
        for theme in case.candidate_themes:
            lines.append(f"- {theme}")
        lines.extend(
            [
                "",
                "### Oracle Themes",
                "",
            ]
        )
        for theme in case.oracle_themes:
            lines.append(f"- {theme}")
        lines.extend(
            [
                "",
                "### Agent Answer",
                "",
                case.answer,
                "",
            ]
        )

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

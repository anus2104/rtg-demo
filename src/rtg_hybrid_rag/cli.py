from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rtg_hybrid_rag.agent_layer import AgentDeps, build_agent
from rtg_hybrid_rag.assistant import RecommendationAssistant
from rtg_hybrid_rag.evals import (
    EvalRunner,
    render_eval_details,
    render_eval_report,
    save_eval_markdown,
    save_eval_report,
)
from rtg_hybrid_rag.indexer import IndexBuilder
from rtg_hybrid_rag.memory import MemoryManager
from rtg_hybrid_rag.models import UserTurn
from rtg_hybrid_rag.settings import settings


app = typer.Typer(no_args_is_help=True)
console = Console()


def _render_candidates(result) -> None:
    table = Table(title="Top Candidates")
    table.add_column("Rank")
    table.add_column("Theme")
    table.add_column("Brand")
    table.add_column("Category")
    table.add_column("Price")
    table.add_column("Size")
    table.add_column("Type")
    table.add_column("Score")
    for rank, candidate in enumerate(result.candidates, start=1):
        product = candidate.product
        table.add_row(
            str(rank),
            product.theme,
            product.specialty_brand or "-",
            product.product_category,
            f"${product.sale_price:.0f}",
            product.mattress_size or "-",
            product.mattress_type or "-",
            f"{candidate.rerank_score:.3f}",
        )
    console.print(table)


@app.command()
def build_index(force: bool = typer.Option(False, "--force", help="Rebuild the full local index.")) -> None:
    index = IndexBuilder(settings).build_or_load(force_rebuild=force)
    console.print(
        f"Indexed {len(index.products)} products into Chroma collection `{index.collection.name}` "
        f"with fingerprint `{index.fingerprint}`."
    )


@app.command()
def ask(
    query: str = typer.Argument(..., help="The user's latest message."),
    history_file: Path | None = typer.Option(
        None,
        "--history-file",
        help="Optional JSON file containing prior conversation turns.",
    ),
) -> None:
    turns: list[UserTurn] = []
    if history_file:
        payload = json.loads(history_file.read_text(encoding="utf-8"))
        turns.extend(UserTurn.model_validate(item) for item in payload)
    turns.append(UserTurn(role="user", content=query))

    assistant = RecommendationAssistant(settings)
    result = assistant.recommend(turns)
    console.print("\n[bold]Assistant Answer[/bold]\n")
    console.print(result.answer)
    console.print()
    _render_candidates(result)


@app.command("agent-chat")
def agent_chat(
    memory_tag: str | None = typer.Option(
        None,
        "--memory-tag",
        help="Supermemory container tag to use for persistent user memory.",
    ),
) -> None:
    recommender = RecommendationAssistant(settings)
    agent = build_agent(settings)
    memory = MemoryManager(settings, container_tag=memory_tag or settings.supermemory_container_tag)
    console.print("PydanticAI interactive chat. Type `exit` to stop.\n")
    if memory.enabled:
        console.print(f"Supermemory enabled with container tag: {memory.container_tag}\n")
    else:
        console.print("Supermemory disabled. Set `SUPERMEMORY_API_KEY` to enable persistent memory.\n")
    agent.to_cli_sync(
        deps=AgentDeps(recommender=recommender, memory=memory),
        prog_name="rtg-hybrid-rag",
    )


@app.command("run-evals")
def run_evals(
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip GPT-5.4 grading."),
    show_judge_output: bool = typer.Option(
        False,
        "--show-judge-output",
        help="Print per-case judge scores and summaries in the terminal.",
    ),
) -> None:
    runner = EvalRunner(settings=settings, assistant=RecommendationAssistant(settings))
    suite = runner.run_suite(use_judge=not no_judge)
    render_eval_report(console, suite)
    if show_judge_output:
        render_eval_details(console, suite)
    output_dir = settings.cache_dir / "evals"
    json_path = output_dir / "latest.json"
    markdown_path = output_dir / "latest.md"
    save_eval_report(json_path, suite)
    save_eval_markdown(markdown_path, suite)
    console.print(f"\nSaved eval JSON report to {json_path}")
    console.print(f"Saved eval Markdown report to {markdown_path}")

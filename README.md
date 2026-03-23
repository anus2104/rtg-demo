# RTG Hybrid RAG

Hybrid RAG prototype for mattress and sleep-product recommendations over
`Mattress SKUS RTG.xlsx`.

The system is built around the actual RTG catalog and supports:

- dense retrieval with `ChromaDB`
- sparse retrieval with `BM25`
- hybrid score fusion
- metadata filtering
- Cohere reranking through OpenRouter
- grounded answer generation with OpenAI-compatible models
- conversational CLI interaction through `pydantic-ai`
- Supermemory-based persistent memory injected into the agent each turn
- a sales-skills tool for domain-specific conversation guidance
- scenario-based evals with deterministic checks and GPT-5.4 judging

## Architecture

The recommendation pipeline is:

1. Parse the user request or multi-turn conversation into structured constraints.
2. Run dense retrieval in Chroma with metadata filters when available.
3. Run BM25 retrieval over the same catalog.
4. Merge dense and sparse results into a hybrid shortlist.
5. Apply hard filtering for category, size, and budget before reranking.
6. Rerank the shortlist with a small Cohere model.
7. Generate a grounded final answer using only the reranked candidates.

The conversational agent layer is separate from retrieval:

- `agent-chat` uses `pydantic-ai` as the terminal conversation shell.
- Supermemory context is injected into the agent instructions on every turn when enabled.
- the agent can also save durable preferences back into Supermemory.
- the agent has a `get_sales_skills` tool for structured guidance across conversation types like cooling, back pain, firmness, budget, brand-locked requests, adjustable bases, and kids mattresses.
- the agent calls a tool that invokes the existing hybrid retrieval pipeline
- retrieval and ranking logic remain custom and debuggable

## Retrieval Behavior

Single-turn behavior:

- uses the current user message as the main request
- extracts constraints such as size, budget, product type, comfort, and brand
- runs dense + sparse retrieval
- reranks the merged shortlist
- answers from the final candidate set only

Multi-turn behavior:

- merges all prior user turns into one active conversation state
- preserves earlier constraints unless the user clearly changes them
- supports pivots, for example from mattress to adjustable base
- uses the merged state for retrieval, reranking, and final response generation

## Metadata Filtering

Metadata filtering is applied in multiple places:

- retrieval-time Chroma filtering for fields like:
  - `category`
  - `size`
  - `brand`
  - `type`
  - `sale_price`
- hybrid ranking bias based on parsed user constraints
- hard filtering before reranking, especially for category, size, and budget

So the system is not just semantic search. It is hybrid search with structured constraints.

## Project Layout

Core files:

- [catalog.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/catalog.py): workbook ingestion and search text construction
- [conversation.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/conversation.py): turn parsing and conversation-state extraction
- [indexer.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/indexer.py): Chroma index + BM25 index build/load
- [retrieval.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/retrieval.py): dense retrieval, BM25 retrieval, hybrid fusion, metadata-aware ranking
- [reranker.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/reranker.py): Cohere rerank stage
- [assistant.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/assistant.py): end-to-end recommendation orchestration
- [agent_layer.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/agent_layer.py): `pydantic-ai` conversational agent wrapper
- [memory.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/memory.py): Supermemory wrapper and per-turn memory retrieval
- [skills.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/skills.py): structured sales-guidance tool definitions
- [eval_cases.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/eval_cases.py): scenario definitions
- [evals.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/evals.py): deterministic checks, GPT-5.4 judging, JSON/Markdown report generation
- [cli.py](/home/aditya/Desktop/RTG-demo/src/rtg_hybrid_rag/cli.py): CLI commands

## Environment

The project loads `.env` and `.env.local` with `load_dotenv(...)`.

Useful variables:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `SUPERMEMORY_API_KEY`
- `SUPERMEMORY_CONTAINER_TAG` default: `rtg-default-user`
- `SUPERMEMORY_THRESHOLD` default: `0.6`
- `GENERATION_PROVIDER` with `openai` or `openrouter`
- `GENERATION_MODEL` default: `gpt-5.4-mini`
- `AGENT_PROVIDER` with `openai` or `openrouter`
- `AGENT_MODEL` default: `gpt-5.4-mini`
- `EMBEDDING_PROVIDER` with `openai` or `openrouter`
- `EMBEDDING_MODEL` default: `text-embedding-3-small`
- `RERANK_PROVIDER` default: `openrouter`
- `RERANK_MODEL` default: `cohere/command-r7b-12-2024`
- `EVAL_JUDGE_PROVIDER` default: `openai`
- `EVAL_JUDGE_MODEL` default: `gpt-5.4`

## Setup

This project is `uv`-first. Use `uv` for environment creation, dependency sync,
and command execution.

### Quick Start

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and sync dependencies:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync
```

Create a local env file with the API keys you need:

```bash
cat > .env.local <<'EOF'
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
SUPERMEMORY_API_KEY=...
EOF
```

Build the local retrieval index:

```bash
uv run rtg-hybrid-rag build-index
```

This creates local cache artifacts under `.cache/`.

### Setup Summary

From a fresh clone, the normal flow is:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync
uv run rtg-hybrid-rag build-index
uv run rtg-hybrid-rag --help
```

### Dependency Files

- `pyproject.toml` is the source of truth for project metadata, dependencies, and the CLI entrypoint.
- `uv.lock` pins the resolved dependency set for reproducible installs with `uv sync`.
- `requirements.txt` is included only as a compatibility export for tools that expect it, but the preferred workflow is still `uv`.

## Commands

Show CLI help:

```bash
uv run rtg-hybrid-rag --help
```

### Single Request

Run a single request directly through the recommender:

```bash
uv run rtg-hybrid-rag ask 'I sleep hot, mostly on my side, and want a queen mattress under $2000'
```

### Request With Prior History

Create a `history.json` file:

```json
[
  {"role": "user", "content": "I need a queen mattress for back pain."},
  {"role": "assistant", "content": "What comfort level do you prefer?"},
  {"role": "user", "content": "Medium or slightly firm, and I sleep hot."}
]
```

Then run:

```bash
uv run rtg-hybrid-rag ask 'Keep it under $2200.' --history-file history.json
```

### Interactive Agent Mode

Use the `pydantic-ai` terminal chat wrapper:

```bash
uv run rtg-hybrid-rag agent-chat
```

This is the only interactive mode now.

When Supermemory is enabled:

- relevant memory is fetched and injected into the agent instructions every turn
- the agent can save durable user preferences for future conversations

The agent also has these internal tools:

- `recommend_products`
- `save_memory`
- `get_sales_skills`

Use a specific Supermemory container tag:

```bash
uv run rtg-hybrid-rag agent-chat --memory-tag customer_123
```

## Evaluation

Run the scenario suite:

```bash
uv run rtg-hybrid-rag run-evals
```

Skip GPT-5.4 judging and run only deterministic checks:

```bash
uv run rtg-hybrid-rag run-evals --no-judge
```

### How the Eval Works

Each eval case is a realistic user scenario, either single-turn or multi-turn.

For every case, the system:

1. runs the real recommender pipeline
2. derives an oracle product pool from the actual workbook using the case constraints
3. runs deterministic checks such as category, size, budget, brand, comfort, and oracle overlap
4. asks GPT-5.4 to judge:
   - grounding
   - constraint adherence
   - recommendation quality
   - explanation quality
   - user satisfaction

### Eval Outputs

Each eval run writes:

- JSON report: `.cache/evals/latest.json`
- Markdown report: `.cache/evals/latest.md`

The Markdown report includes:

- summary metrics
- pass/fail status
- judge scores
- satisfaction score
- the scenario question or turn sequence
- deterministic check details
- returned candidate themes
- oracle themes
- the final agent answer

## Current Eval Status

The eval suite includes both passing and intentionally challenging scenarios.

The newer scenarios are useful for finding ranking gaps, for example:

- softness not being weighted strongly enough
- size leakage for closely related adjustable-base sizes
- pressure-relief intent not being modeled strongly enough

That is expected and useful: the eval suite is meant to drive improvements, not just report green checks.

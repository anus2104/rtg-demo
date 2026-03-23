from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]


for env_name in (".env", ".env.local"):
    env_path = ROOT / env_name
    if env_path.exists():
        load_dotenv(env_path, override=False)


class Settings(BaseModel):
    data_path: Path = Field(default=ROOT / "Mattress SKUS RTG.xlsx")
    sheet_name: str = "Products (2)"
    cache_dir: Path = Field(default=ROOT / ".cache")
    chroma_dir: Path = Field(default=ROOT / ".cache" / "chroma")
    bm25_dir: Path = Field(default=ROOT / ".cache" / "bm25")
    conversation_dir: Path = Field(default=ROOT / ".cache" / "conversations")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    generation_provider: str = os.getenv("GENERATION_PROVIDER", "openai")
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-5.4-mini")
    agent_provider: str = os.getenv("AGENT_PROVIDER", os.getenv("GENERATION_PROVIDER", "openai"))
    agent_model: str = os.getenv("AGENT_MODEL", os.getenv("GENERATION_MODEL", "gpt-5.4-mini"))
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "openrouter")
    rerank_model: str = os.getenv("RERANK_MODEL", "cohere/command-r7b-12-2024")
    eval_judge_provider: str = os.getenv("EVAL_JUDGE_PROVIDER", "openai")
    eval_judge_model: str = os.getenv("EVAL_JUDGE_MODEL", "gpt-5.4")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    supermemory_api_key: str | None = os.getenv("SUPERMEMORY_API_KEY")
    supermemory_container_tag: str = os.getenv("SUPERMEMORY_CONTAINER_TAG", "rtg-default-user")
    supermemory_threshold: float = float(os.getenv("SUPERMEMORY_THRESHOLD", "0.6"))
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )

    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.55"))
    hybrid_sparse_weight: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.45"))
    initial_recall_k: int = int(os.getenv("INITIAL_RECALL_K", "20"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "10"))
    final_top_n: int = int(os.getenv("FINAL_TOP_N", "5"))

    def ensure_directories(self) -> None:
        for path in (
            self.cache_dir,
            self.chroma_dir,
            self.bm25_dir,
            self.conversation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()

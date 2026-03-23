from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from rtg_hybrid_rag.catalog import catalog_fingerprint, load_catalog, save_catalog_snapshot, tokenize
from rtg_hybrid_rag.llm import OpenAICompatibleClient, batched
from rtg_hybrid_rag.models import ProductDocument
from rtg_hybrid_rag.settings import Settings


class CatalogIndex:
    def __init__(
        self,
        *,
        products: list[ProductDocument],
        collection: Collection,
        bm25: BM25Okapi,
        fingerprint: str,
    ) -> None:
        self.products = products
        self.collection = collection
        self.bm25 = bm25
        self.fingerprint = fingerprint
        self.by_id = {product.product_id: product for product in products}


class IndexBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAICompatibleClient(settings)
        self.settings.ensure_directories()

    def _bm25_paths(self, fingerprint: str) -> tuple[Path, Path]:
        return (
            self.settings.bm25_dir / f"{fingerprint}.pkl",
            self.settings.bm25_dir / f"{fingerprint}.jsonl",
        )

    def _collection_name(self, fingerprint: str) -> str:
        return f"rtg_products_{fingerprint}"

    def build_or_load(self, force_rebuild: bool = False) -> CatalogIndex:
        products = load_catalog(self.settings.data_path, self.settings.sheet_name)
        fingerprint = catalog_fingerprint(self.settings.data_path, self.settings.embedding_model)

        snapshot_path = self._bm25_paths(fingerprint)[1]
        bm25_path = self._bm25_paths(fingerprint)[0]

        chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        collection_name = self._collection_name(fingerprint)

        if force_rebuild:
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass

        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if force_rebuild or collection.count() != len(products):
            existing_ids = set(collection.get(include=[])["ids"])
            if existing_ids:
                collection.delete(ids=list(existing_ids))

            documents = [product.search_text for product in products]
            metadatas = [
                {
                    "product_id": product.product_id,
                    "sku_number": product.sku_number,
                    "theme": product.theme,
                    "brand": product.specialty_brand or "",
                    "category": product.product_category,
                    "sale_price": product.sale_price,
                    "size": product.mattress_size or "",
                    "comfort": product.comfort or "",
                    "type": product.mattress_type or "",
                }
                for product in products
            ]
            embeddings: list[list[float]] = []
            for batch in batched(documents, self.settings.embedding_batch_size):
                embeddings.extend(self.client.embed_texts(batch))
            collection.add(
                ids=[product.product_id for product in products],
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        if force_rebuild or not bm25_path.exists():
            bm25 = BM25Okapi([tokenize(product.search_text) for product in products])
            with bm25_path.open("wb") as handle:
                pickle.dump(bm25, handle)
            save_catalog_snapshot(products, snapshot_path)
        else:
            with bm25_path.open("rb") as handle:
                bm25 = pickle.load(handle)

        return CatalogIndex(
            products=products,
            collection=collection,
            bm25=bm25,
            fingerprint=fingerprint,
        )

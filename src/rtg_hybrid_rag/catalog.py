from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from rtg_hybrid_rag.models import ProductDocument


RATING_MAP = {
    "1-Basic": 1,
    "2-Good": 2,
    "3-Superior": 3,
    "4-Premier": 4,
    "5-Ultimate": 5,
}


SIZE_ALIASES = {
    "twx": "Twin XL",
    "twxl": "Twin XL",
    "twin xl": "Twin XL",
    "twin": "Twin",
    "full": "Full",
    "queen": "Queen",
    "king": "King",
    "split king": "Split King",
    "cal king": "California King",
    "california king": "California King",
    "split california king": "Split California King",
}


def _clean_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def _clean_price(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return round(float(value), 2)


def _category_for_row(row: pd.Series) -> str:
    mattress_type = _clean_text(row.get("Mattress Type"))
    theme = (_clean_text(row.get("Theme")) or "").lower()
    desc = (_clean_text(row.get("Customer Description")) or "").lower()
    copy = (_clean_text(row.get("Collection Copy")) or "").lower()
    haystack = " ".join((theme, desc, copy))
    if mattress_type:
        return "mattress"
    if any(token in haystack for token in ("adjustable base", "power base", "ergo", "baselogic", "ease 4.0", "base")):
        return "adjustable_base"
    if "pillow" in haystack:
        return "pillow"
    return "sleep_product"


def _normalize_size(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower().strip()
    return SIZE_ALIASES.get(lowered, value)


def _rating_value(value: str | None) -> int | None:
    if not value:
        return None
    return RATING_MAP.get(value)


def build_search_text(product: ProductDocument) -> str:
    parts = [
        f"category: {product.product_category}",
        f"theme: {product.theme}",
        f"description: {product.customer_description}",
        f"brand: {product.specialty_brand or 'unknown'}",
        f"size: {product.mattress_size or 'unknown'}",
        f"type: {product.mattress_type or 'unknown'}",
        f"comfort: {product.comfort or 'unknown'}",
        f"pressure relief: {product.pressure_relief or 'unknown'}",
        f"sleep position: {product.sleep_position or 'unknown'}",
        f"support level: {product.support_level or 'unknown'}",
        f"temperature management: {product.temperature_management or 'unknown'}",
        f"price: sale {product.sale_price}, regular {product.regular_price}",
    ]
    if product.collection_copy:
        parts.append(product.collection_copy)
    return "\n".join(parts)


def load_catalog(path: Path, sheet_name: str) -> list[ProductDocument]:
    dataframe = pd.read_excel(path, sheet_name=sheet_name)
    products: list[ProductDocument] = []
    for record in dataframe.to_dict(orient="records"):
        sku_number = str(record["Sku Number"])
        base_product = ProductDocument(
            product_id=sku_number,
            sku_number=sku_number,
            theme=_clean_text(record.get("Theme")) or sku_number,
            customer_description=_clean_text(record.get("Customer Description")) or sku_number,
            sale_price=_clean_price(record.get("Sale Price")),
            regular_price=_clean_price(record.get("Regular Price")),
            specialty_brand=_clean_text(record.get("Specialty Brand")),
            proprietary=_clean_text(record.get("Proprietary (P)")),
            collection_copy=_clean_text(record.get("Collection Copy")),
            mattress_designation=_clean_text(record.get("Mattress Designation")),
            mattress_size=_normalize_size(_clean_text(record.get("Mattress Size"))),
            mattress_type=_clean_text(record.get("Mattress Type")),
            pressure_relief=_clean_text(record.get("Pressure Relief")),
            sleep_position=_clean_text(record.get("Sleep Position")),
            support_level=_clean_text(record.get("Support Level")),
            temperature_management=_clean_text(record.get("Temperature Management")),
            comfort=_clean_text(record.get("Comfort")),
            product_category=_category_for_row(pd.Series(record)),
            search_text="",
        )
        base_product.search_text = build_search_text(base_product)
        products.append(base_product)
    return products


def catalog_fingerprint(data_path: Path, embedding_model: str) -> str:
    stat = data_path.stat()
    payload = f"{data_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}::{embedding_model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_catalog_snapshot(products: list[ProductDocument], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for product in products:
            handle.write(json.dumps(product.model_dump(), ensure_ascii=True) + "\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def numeric_rating(value: str | None) -> int | None:
    return _rating_value(value)

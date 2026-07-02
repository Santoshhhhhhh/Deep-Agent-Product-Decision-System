"""Product catalog loader (ground truth for the critic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .state import Product

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "data" / "products.json"


class Catalog:
    def __init__(self, path: str | Path = DEFAULT_CATALOG) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._products = [Product.model_validate(p) for p in raw]
        self._by_id = {p.id: p for p in self._products}

    def get(self, product_id: str) -> Optional[Product]:
        return self._by_id.get(product_id)

    def candidates(self, category: Optional[str] = None, limit: int = 40) -> list[Product]:
        if category:
            cat = category.lower().rstrip("s")
            pool = [p for p in self._products if cat in p.category.lower()]
            if pool:
                return pool[:limit]
        return self._products[:limit]

    def __len__(self) -> int:
        return len(self._products)

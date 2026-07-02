"""Generate a deterministic synthetic product catalog (data/products.json).

Run: python data/generate_catalog.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

BRANDS = ["Nexon", "Aurel", "Vanta", "Kirei", "Bolt", "Norde", "Peak", "Lumo",
          "Astra", "Crest"]

CATEGORIES = {
    "laptop": {
        "price": (450, 2800),
        "attrs": lambda: {
            "ram_gb": random.choice([8, 16, 16, 32]),
            "storage_gb": random.choice([256, 512, 512, 1024]),
            "weight_kg": round(random.uniform(0.9, 2.6), 2),
            "battery_hours": random.randint(5, 20),
            "gpu": random.choice(["integrated", "integrated", "RTX 4050", "RTX 4060", "RTX 4070"]),
            "screen_inches": random.choice([13.3, 14, 15.6, 16]),
        },
        "desc": lambda a: (
            f"{a['screen_inches']}\" laptop with {a['ram_gb']}GB RAM, "
            f"{a['storage_gb']}GB SSD, {a['gpu']} GPU, "
            f"{a['battery_hours']}h battery"
            + (", lightweight portable build" if a["weight_kg"] < 1.4 else "")
        ),
    },
    "headphones": {
        "price": (30, 550),
        "attrs": lambda: {
            "wireless": random.choice([True, True, False]),
            "noise_cancelling": random.choice([True, False]),
            "battery_hours": random.randint(8, 60),
            "weight_g": random.randint(180, 400),
        },
        "desc": lambda a: (
            ("wireless " if a["wireless"] else "wired ")
            + ("noise cancelling ANC headphones" if a["noise_cancelling"] else "headphones")
            + f" with {a['battery_hours']}h battery"
        ),
    },
    "smartphone": {
        "price": (200, 1400),
        "attrs": lambda: {
            "ram_gb": random.choice([6, 8, 12]),
            "storage_gb": random.choice([128, 256, 512]),
            "battery_mah": random.randint(3800, 5500),
            "camera_mp": random.choice([48, 50, 108, 200]),
            "fast_charging": random.choice([True, True, False]),
        },
        "desc": lambda a: (
            f"smartphone with {a['camera_mp']}MP camera, {a['ram_gb']}GB RAM, "
            f"{a['battery_mah']}mAh battery"
            + (", fast charging" if a["fast_charging"] else "")
        ),
    },
    "monitor": {
        "price": (120, 1100),
        "attrs": lambda: {
            "size_inches": random.choice([24, 27, 27, 32, 34]),
            "resolution": random.choice(["1080p", "1440p", "4k", "4k"]),
            "refresh_hz": random.choice([60, 75, 144, 165, 240]),
            "panel": random.choice(["IPS", "VA", "OLED"]),
            "hdr": random.choice([True, False]),
        },
        "desc": lambda a: (
            f"{a['size_inches']}\" {a['resolution']} {a['panel']} monitor, "
            f"{a['refresh_hz']}Hz refresh" + (", HDR" if a["hdr"] else "")
        ),
    },
    "office chair": {
        "price": (90, 1300),
        "attrs": lambda: {
            "lumbar_support": random.choice([True, True, False]),
            "adjustable_arms": random.choice([True, False]),
            "material": random.choice(["mesh", "fabric", "leather"]),
            "max_weight_kg": random.choice([110, 120, 136, 150]),
        },
        "desc": lambda a: (
            f"ergonomic {a['material']} office chair"
            + (" with lumbar support" if a["lumbar_support"] else "")
            + (", adjustable arms" if a["adjustable_arms"] else "")
        ),
    },
    "coffee maker": {
        "price": (40, 800),
        "attrs": lambda: {
            "type": random.choice(["drip", "espresso", "espresso", "french press"]),
            "built_in_grinder": random.choice([True, False, False]),
            "capacity_cups": random.choice([4, 8, 10, 12]),
        },
        "desc": lambda a: (
            f"{a['type']} coffee maker, {a['capacity_cups']} cups"
            + (", built-in grinder to grind fresh beans" if a["built_in_grinder"] else "")
        ),
    },
    "backpack": {
        "price": (25, 350),
        "attrs": lambda: {
            "capacity_l": random.choice([18, 22, 26, 30, 40]),
            "waterproof": random.choice([True, False]),
            "laptop_sleeve": random.choice([True, True, False]),
            "weight_kg": round(random.uniform(0.5, 1.8), 2),
        },
        "desc": lambda a: (
            f"{a['capacity_l']}L backpack"
            + (", waterproof" if a["waterproof"] else "")
            + (", padded laptop sleeve" if a["laptop_sleeve"] else "")
        ),
    },
    "keyboard": {
        "price": (25, 300),
        "attrs": lambda: {
            "mechanical": random.choice([True, True, False]),
            "wireless": random.choice([True, False]),
            "rgb": random.choice([True, False]),
            "layout": random.choice(["full", "tkl", "60%"]),
        },
        "desc": lambda a: (
            ("mechanical " if a["mechanical"] else "membrane ")
            + ("wireless " if a["wireless"] else "")
            + f"{a['layout']} keyboard" + (", RGB backlight" if a["rgb"] else "")
        ),
    },
}

ADJ = ["Pro", "Air", "Max", "Lite", "Ultra", "Edge", "Prime", "Core", "Flex", "One"]


def main() -> None:
    products = []
    i = 0
    per_cat = 60
    for cat, spec in CATEGORIES.items():
        lo, hi = spec["price"]
        for _ in range(per_cat):
            i += 1
            attrs = spec["attrs"]()
            price = round(random.uniform(lo, hi), 2)
            rating = round(random.uniform(3.0, 4.9), 1)
            products.append({
                "id": f"P{i:04d}",
                "name": f"{random.choice(BRANDS)} {random.choice(ADJ)} "
                        f"{cat.title()} {random.randint(100, 999)}",
                "category": cat,
                "price": price,
                "rating": rating,
                "review_count": random.randint(12, 4800),
                "attributes": attrs,
                "description": spec["desc"](attrs),
            })
    out = Path(__file__).parent / "products.json"
    out.write_text(json.dumps(products, indent=2), encoding="utf-8")
    print(f"Wrote {len(products)} products to {out}")


if __name__ == "__main__":
    main()

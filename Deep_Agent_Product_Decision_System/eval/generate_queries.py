"""Generate the evaluation query set (deterministic, templated).

Queries combine a category, 0-2 hard constraints, and 1-2 soft
preferences, producing realistic shopping intents like:
  "lightweight laptop under $1400 with good battery life"
Run: python eval/generate_queries.py --n 1200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

random.seed(7)

TEMPLATES = {
    "laptop": {
        "budgets": [800, 1000, 1200, 1500, 2000],
        "prefs": ["lightweight", "good battery life", "gpu for gaming",
                  "portable for travel", "16gb ram"],
    },
    "headphones": {
        "budgets": [80, 150, 250, 400],
        "prefs": ["wireless", "noise cancelling", "long battery"],
    },
    "smartphone": {
        "budgets": [400, 700, 1000, 1300],
        "prefs": ["great camera", "fast charging", "long battery"],
    },
    "monitor": {
        "budgets": [200, 350, 600, 900],
        "prefs": ["4k", "144hz refresh", "hdr", "oled"],
    },
    "office chair": {
        "budgets": [200, 400, 700],
        "prefs": ["lumbar support", "ergonomic", "mesh back"],
    },
    "coffee maker": {
        "budgets": [100, 250, 500],
        "prefs": ["built-in grinder", "espresso", "large capacity"],
    },
    "backpack": {
        "budgets": [60, 120, 250],
        "prefs": ["waterproof", "laptop sleeve", "lightweight"],
    },
    "keyboard": {
        "budgets": [60, 120, 250],
        "prefs": ["mechanical", "wireless", "rgb"],
    },
}

PATTERNS = [
    "{prefs} {cat} under ${budget}",
    "best {cat} below ${budget} with {prefs}",
    "{cat} with {prefs}, max ${budget}",
    "looking for a {cat} under ${budget}, needs {prefs}",
    "{prefs} {cat} under ${budget} with rating at least {minr}",
]


def make_queries(n: int) -> list[str]:
    queries = []
    cats = list(TEMPLATES)
    while len(queries) < n:
        cat = random.choice(cats)
        spec = TEMPLATES[cat]
        budget = random.choice(spec["budgets"])
        prefs = ", ".join(random.sample(spec["prefs"],
                                        k=random.choice([1, 1, 2])))
        pattern = random.choice(PATTERNS)
        queries.append(pattern.format(
            cat=cat, budget=budget, prefs=prefs,
            minr=random.choice([3.5, 4.0, 4.2]),
        ))
    return queries[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()
    out = Path(__file__).parent / "queries.json"
    out.write_text(json.dumps(make_queries(args.n), indent=2), encoding="utf-8")
    print(f"Wrote {args.n} queries to {out}")

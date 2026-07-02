"""Deterministic mock backend for offline runs and tests.

Each agent's system prompt carries a ``ROLE: <name>`` marker and the user
prompt embeds its context as JSON between ``<context>`` tags, so the mock
can parse the real inputs and produce schema-valid, plausible outputs
without any network calls. Behavior is fully deterministic, which makes
the evaluation harness reproducible.
"""
from __future__ import annotations

import json
import re


def mock_route(system: str, user: str) -> str:
    role_m = re.search(r"ROLE:\s*(\w+)", system)
    role = role_m.group(1).lower() if role_m else "generic"
    ctx = _context(user)
    if role == "planner":
        return _mock_planner(ctx)
    if role == "specialist":
        return _mock_specialist(ctx)
    if role == "critic_notes":
        return json.dumps({"notes": "Verified hard constraints against catalog attributes."})
    if role == "judge":
        return _mock_judge(ctx)
    return json.dumps({"text": "ok"})


def _context(user: str) -> dict:
    m = re.search(r"<context>(.*?)</context>", user, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------------ planner
_CATEGORIES = ["laptop", "headphones", "smartphone", "monitor", "keyboard",
               "backpack", "coffee maker", "office chair", "tablet", "camera"]

_FEATURE_LEXICON = [
    "battery", "lightweight", "noise cancel", "wireless", "mechanical",
    "ergonomic", "4k", "oled", "waterproof", "gpu", "ram", "storage",
    "portable", "fast charging", "camera", "zoom", "grind", "lumbar",
    "usb-c", "silent", "rgb", "hdr", "refresh", "anc", "stylus",
]


def _mock_planner(ctx: dict) -> str:
    query = (ctx.get("query") or "").lower()

    category = next((c for c in _CATEGORIES if c.rstrip("s") in query), None)

    constraints = []
    m = re.search(r"(?:under|below|less than|max(?:imum)?(?: of)?)\s*\$?\s*([\d,]+)", query)
    if m:
        constraints.append({"field": "price", "op": "lte",
                            "value": float(m.group(1).replace(",", ""))})
    m = re.search(r"(?:over|above|at least|more than)\s*\$?\s*([\d,]+)", query)
    if m:
        constraints.append({"field": "price", "op": "gte",
                            "value": float(m.group(1).replace(",", ""))})
    m = re.search(r"rating (?:of )?(?:at least |above |over )?([\d.]+)", query)
    if m:
        constraints.append({"field": "rating", "op": "gte", "value": float(m.group(1))})
    for feat, field in [("16gb", "ram_gb"), ("32gb", "ram_gb")]:
        if feat in query:
            constraints.append({"field": field, "op": "gte",
                                "value": int(feat.replace("gb", ""))})

    prefs = [{"description": ph, "weight": 2.0}
             for ph in _extract_pref_phrases(query, category)]
    if not prefs:
        prefs = [{"description": f, "weight": 2.0}
                 for f in _FEATURE_LEXICON if f in query]
    if not prefs:
        prefs = [{"description": "overall quality", "weight": 1.0}]

    subtasks = [
        {"agent": "budget",
         "instruction": "Rank candidates by price-to-value fit given budget constraints.",
         "constraints": [c for c in constraints if c["field"] == "price"],
         "preferences": []},
        {"agent": "feature",
         "instruction": "Rank candidates by how well attributes match requested features.",
         "constraints": [c for c in constraints if c["field"] != "price"],
         "preferences": prefs},
        {"agent": "review",
         "instruction": "Rank candidates by review quality and volume.",
         "constraints": [], "preferences": []},
    ]
    return json.dumps({
        "query": ctx.get("query", ""),
        "category": category,
        "hard_constraints": constraints,
        "preferences": prefs,
        "subtasks": subtasks,
    })


_STOP_PHRASES = ("rating", "$", "under", "below", "max", "best", "looking")


def _extract_pref_phrases(query: str, category: str | None) -> list[str]:
    """Pull soft-preference noun phrases out of the query text."""
    phrases: list[str] = []
    # after "with"/"needs", up to a budget clause or end
    for m in re.finditer(r"(?:with|needs)\s+(.+?)(?=,?\s*(?:max|under|below)\s*\$|$)",
                         query):
        phrases.append(m.group(1))
    # before the category word (pattern: "{prefs} {cat} under $...")
    if category:
        m = re.match(rf"(?:best\s+|looking for a\s+)?(.+?)\s+{re.escape(category)}",
                     query)
        if m:
            phrases.append(m.group(1))
    out = []
    for chunk in phrases:
        for part in re.split(r",| and ", chunk):
            part = part.strip()
            if part and not any(s in part for s in _STOP_PHRASES):
                out.append(part)
    return list(dict.fromkeys(out))


# Semantic expansion: how an LLM grounds a preference phrase into catalog
# attribute vocabulary ("gpu for gaming" -> discrete RTX GPUs, etc.)
_SEMANTIC = {
    "gpu": ["rtx"], "gaming": ["rtx"], "portable": ["lightweight"],
    "144hz": ["144"], "refresh": ["144", "165", "240"],
    "camera": ["camera"], "grinder": ["grinder"], "grind": ["grinder"],
    "capacity": ["12 cups", "10 cups"], "battery": ["battery"],
    "16gb": ["16gb", "32gb"], "mesh": ["mesh"], "espresso": ["espresso"],
    "waterproof": ["waterproof"], "sleeve": ["laptop sleeve"],
    "lumbar": ["lumbar"], "ergonomic": ["ergonomic"],
    "mechanical": ["mechanical"], "wireless": ["wireless"], "rgb": ["rgb"],
    "4k": ["4k"], "hdr": ["hdr"], "oled": ["oled"],
    "noise": ["noise cancelling"], "cancelling": ["noise cancelling"],
    "anc": ["noise cancelling"], "charging": ["fast charging"],
    "fast": ["fast charging"], "lightweight": ["lightweight"],
}


def _pref_matches(phrase: str, blob: str) -> bool:
    if phrase in blob:
        return True
    keys = []
    for word in re.findall(r"[\w%]+", phrase.lower()):
        keys.extend(_SEMANTIC.get(word, []))
    return any(k in blob for k in keys)


# --------------------------------------------------------------- specialist
def _mock_specialist(ctx: dict) -> str:
    agent = ctx.get("agent", "feature")
    products = ctx.get("products", [])
    prefs = [p["description"].lower() for p in ctx.get("preferences", [])]
    budget = next((c["value"] for c in ctx.get("constraints", [])
                   if c.get("field") == "price" and c.get("op") == "lte"), None)

    out = []
    for p in products:
        attrs = p.get("attributes", {})
        blob = " ".join([p.get("name", ""), p.get("description", ""),
                         json.dumps(attrs)]).lower()
        if agent == "budget":
            price = p.get("price", 0.0)
            if budget:
                score = 0.0 if price > budget else 0.55 + 0.45 * (1 - price / budget)
                rationale = (f"${price:.0f} vs ${budget:.0f} budget"
                             if price <= budget else "over budget")
            else:
                score = max(0.05, min(1.0, 1.0 - price / 3000.0))
                rationale = f"absolute price ${price:.0f}"
        elif agent == "review":
            rating = p.get("rating", 0.0)
            n = p.get("review_count", 0)
            score = min(1.0, (rating / 5.0) * 0.8 + min(n, 2000) / 2000.0 * 0.2)
            rationale = f"{rating}★ across {n} reviews"
        else:  # feature
            violated = [c for c in ctx.get("constraints", [])
                        if not _check(c, {**p, "attributes": attrs})]
            if violated:
                score = 0.0
                rationale = f"violates {violated[0].get('field')} constraint"
            else:
                hits = [f for f in prefs if _pref_matches(f, blob)]
                score = min(1.0, 0.3 + 0.7 * (len(hits) / max(len(prefs), 1)))
                rationale = ("matches: " + ", ".join(hits)) if hits else "generic fit"
        out.append({"product_id": p["id"], "score": round(score, 4),
                    "rationale": rationale})
    out.sort(key=lambda c: c["score"], reverse=True)
    return json.dumps({"agent": agent, "candidates": out})


# -------------------------------------------------------------------- judge
def _mock_judge(ctx: dict) -> str:
    """Deterministic LLM-as-judge fallback: grades a top-5 list 0-1 per item."""
    query_constraints = ctx.get("hard_constraints", [])
    prefs = [p["description"].lower() for p in ctx.get("preferences", [])]
    items = ctx.get("recommendations", [])
    scores = []
    for it in items:
        attrs = it.get("attributes", {})
        blob = " ".join([it.get("name", ""), it.get("description", ""),
                         json.dumps(attrs)]).lower()
        ok = all(_check(c, it) for c in query_constraints)
        pref_hits = sum(1 for f in prefs if _pref_matches(f, blob)) / max(len(prefs), 1)
        quality = max(0.0, (it.get("rating", 0) - 3.0) / 1.9)
        # Strict rubric: constraint pass is necessary but NOT sufficient;
        # relevance additionally requires genuine preference fit.
        s = 0.0 if not ok else min(1.0, 0.3 + 0.45 * min(pref_hits, 1.0)
                                   + 0.25 * quality)
        scores.append({"product_id": it.get("product_id"),
                       "relevant": s >= 0.6, "score": round(s, 3)})
    return json.dumps({"item_scores": scores})


def _check(c: dict, item: dict) -> bool:
    field, op, value = c.get("field"), c.get("op"), c.get("value")
    actual = item.get(field, item.get("attributes", {}).get(field))
    if actual is None:
        return False
    try:
        if op == "lte":
            return float(actual) <= float(value)
        if op == "gte":
            return float(actual) >= float(value)
        if op == "eq":
            return str(actual).lower() == str(value).lower()
        if op == "contains":
            return str(value).lower() in str(actual).lower()
    except (TypeError, ValueError):
        return False
    return False

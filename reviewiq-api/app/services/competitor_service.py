"""ReviewIQ — Competitor Intelligence Service"""

import pandas as pd
import numpy as np
import json
import anthropic
from typing import List
import logging

logger = logging.getLogger(__name__)

ASPECT_DIMENSIONS = {
    "quality":          {"pos": ["durable","sturdy","well-made","excellent quality","high quality","great quality","solid"], "neg": ["broke","flimsy","cheap","poor quality","falls apart","defective"]},
    "value":            {"pos": ["great value","worth it","affordable","reasonable price","good deal"], "neg": ["overpriced","too expensive","not worth","rip off","waste of money"]},
    "effectiveness":    {"pos": ["works great","works well","effective","really works","amazing results"], "neg": ["doesn't work","ineffective","no results","waste","useless"]},
    "packaging":        {"pos": ["well packaged","nice packaging","arrived safely","securely packed"], "neg": ["damaged packaging","poorly packaged","arrived damaged","broken","leaking"]},
    "shipping":         {"pos": ["fast shipping","quick delivery","arrived fast","on time"], "neg": ["slow shipping","late delivery","took forever","delayed"]},
    "usability":        {"pos": ["easy to use","user friendly","simple","intuitive","easy setup"], "neg": ["confusing","hard to use","complicated","difficult"]},
    "customer_service": {"pos": ["great service","helpful","responsive","excellent support"], "neg": ["terrible service","no response","ignored","unhelpful","rude"]},
    "scent_taste":      {"pos": ["smells great","delicious","love the scent","tastes good","great flavor"], "neg": ["bad smell","terrible taste","awful scent","nasty","chemical smell"]},
}


def score_aspect(text: str, aspect_dict: dict) -> float | None:
    pos_hits = sum(1 for kw in aspect_dict["pos"] if kw in text)
    neg_hits = sum(1 for kw in aspect_dict["neg"] if kw in text)
    total = pos_hits + neg_hits
    if total == 0:
        return None
    return ((pos_hits - neg_hits) / total) * 100


def get_asin_aspect_scores(asin: str, reviews_df: pd.DataFrame) -> dict:
    texts = reviews_df[reviews_df["asin"] == asin]["review_text"].fillna("").str.lower()
    combined = " ".join(texts.tolist())
    return {
        aspect: round(score, 1)
        for aspect, kws in ASPECT_DIMENSIONS.items()
        if (score := score_aspect(combined, kws)) is not None
    }


def get_competitors(
    target_asin: str,
    reviews_df: pd.DataFrame,
    ris_df: pd.DataFrame,
    top_n: int = 3,
) -> List[str]:
    if target_asin not in ris_df.index:
        return []

    category = ris_df.loc[target_asin, "category"]
    cat_df = ris_df[ris_df["category"] == category].copy()
    review_counts = reviews_df.groupby("asin").size().rename("review_count")
    cat_df = cat_df.join(review_counts, how="left")
    cat_df = cat_df[cat_df.index != target_asin]
    cat_df["competitor_score"] = (
        cat_df["ris_score"] * 0.5 +
        cat_df["review_count"].fillna(0).rank(pct=True) * 50
    )
    return cat_df.sort_values("competitor_score", ascending=False).head(top_n).index.tolist()


def build_aspect_gap_matrix(
    target_asin: str,
    competitor_asins: List[str],
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    all_asins = [target_asin] + competitor_asins
    matrix = {}
    for asin in all_asins:
        label = "TARGET" if asin == target_asin else f"COMP_{competitor_asins.index(asin)+1}"
        matrix[label] = get_asin_aspect_scores(asin, reviews_df)

    gap_df = pd.DataFrame(matrix).T
    comp_rows = gap_df.drop("TARGET", errors="ignore")
    if not comp_rows.empty:
        gap_df.loc["COMP_AVG"] = comp_rows.mean()
        gap_df.loc["GAP"] = gap_df.loc["COMP_AVG"] - gap_df.loc["TARGET"]
    return gap_df


async def generate_action_brief_claude(
    asin: str,
    gap_matrix: pd.DataFrame,
    ris_df: pd.DataFrame,
    api_key: str,
) -> dict:
    ris_row = ris_df.loc[asin] if asin in ris_df.index else {}
    gap_row = gap_matrix.loc["GAP"].dropna().sort_values(ascending=False) if "GAP" in gap_matrix.index else pd.Series()

    prompt = f"""You are a senior Amazon marketplace strategist for ReviewIQ.

## Target Product
- ASIN: {asin}
- RIS Score: {ris_row.get('ris_score', 'N/A')}/100
- Category: {ris_row.get('category', 'N/A')}
- Grade: {ris_row.get('ris_grade', 'N/A')}

## Aspect Gap vs Top Competitors (positive = competitor leads you, negative = you lead)
{gap_row.to_string() if not gap_row.empty else "No gap data available"}

## Full Aspect Matrix
{gap_matrix.round(1).to_string()}

Generate a prescriptive action brief. Return ONLY valid JSON with this exact structure:
{{
  "actions": [
    {{
      "rank": 1,
      "aspect": "quality",
      "what_to_fix": "specific actionable fix",
      "why_it_matters": "data-driven reason",
      "estimated_lift_pct_range": [5, 12],
      "timeframe": "60 days"
    }}
  ],
  "strength": "one thing this product does better than competitors",
  "strategic_recommendation": "2 sentence overall strategy"
}}

Include top 3 actions. Be specific and data-driven."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Claude returned non-JSON: {raw[:200]}")
        return {"actions": [], "strength": "", "strategic_recommendation": raw}

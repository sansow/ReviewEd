"""ReviewIQ — RIS Service"""

import pandas as pd
from typing import Optional


WEIGHTS = {
    "sentiment_score": 0.25,
    "authenticity_score": 0.25,
    "aspect_coverage_score": 0.20,
    "velocity_trend_score": 0.15,
    "competitive_position_score": 0.15,
}


def get_ris_for_asin(asin: str, ris_df: pd.DataFrame) -> dict:
    if asin not in ris_df.index:
        return {"error": f"ASIN {asin} not found in RIS data"}

    row = ris_df.loc[asin]
    return {
        "asin": asin,
        "ris_score": float(row["ris_score"]),
        "ris_grade": str(row["ris_grade"]),
        "category": str(row["category"]),
        "category_percentile": float(row.get("category_percentile", 50.0)),
        "components": {
            "sentiment_score": round(float(row.get("sentiment_score", 50.0)), 1),
            "authenticity_score": round(float(row.get("authenticity_score", 50.0)), 1),
            "aspect_coverage_score": round(float(row.get("aspect_coverage_score", 50.0)), 1),
            "velocity_trend_score": round(float(row.get("velocity_trend_score", 50.0)), 1),
            "competitive_position_score": round(float(row.get("competitive_position_score", 50.0)), 1),
        },
        "weights": WEIGHTS,
    }


def get_top_asins_by_category(
    category: str,
    ris_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    top_n: int = 10,
    min_reviews: int = 50,
) -> pd.DataFrame:
    cat_df = ris_df[ris_df["category"] == category].copy()
    if cat_df.empty:
        return pd.DataFrame()

    if not reviews_df.empty:
        review_counts = reviews_df[reviews_df["asin"].isin(cat_df.index)].groupby("asin").size()
        cat_df = cat_df.join(review_counts.rename("review_count"), how="left")
        cat_df = cat_df[cat_df["review_count"].fillna(0) >= min_reviews]

    return cat_df.sort_values("ris_score", ascending=False).head(top_n)

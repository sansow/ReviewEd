"""ReviewIQ — Revenue Impact Service"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

CATEGORY_MULTIPLIERS = {
    "Health_and_Household": 1.15,
    "Beauty_and_Personal_Care": 1.20,
    "Baby_Products": 1.25,
    "Grocery_and_Gourmet_Food": 1.10,
    "Sports_and_Outdoors": 1.05,
    "Home_and_Kitchen": 1.08,
    "Pet_Supplies": 1.12,
    "Electronics": 0.95,
}

ASPECT_RATING_IMPACT = {
    "quality":          {"conversion_base": 0.08, "timeframe": "90 days"},
    "effectiveness":    {"conversion_base": 0.10, "timeframe": "90 days"},
    "value":            {"conversion_base": 0.06, "timeframe": "30 days"},
    "packaging":        {"conversion_base": 0.04, "timeframe": "30 days"},
    "shipping":         {"conversion_base": 0.05, "timeframe": "30 days"},
    "usability":        {"conversion_base": 0.07, "timeframe": "60 days"},
    "customer_service": {"conversion_base": 0.05, "timeframe": "30 days"},
    "scent_taste":      {"conversion_base": 0.06, "timeframe": "60 days"},
}


def estimate_sales_lift(aspect: str, gap_score: float, category: str, current_ris: float) -> Optional[dict]:
    if aspect not in ASPECT_RATING_IMPACT or gap_score <= 0:
        return None

    impact = ASPECT_RATING_IMPACT[aspect]
    cat_mult = CATEGORY_MULTIPLIERS.get(category, 1.0)
    gap_fraction = min(abs(gap_score) / 100, 1.0)
    ris_upside_mult = 1.0 + max(0, (70 - current_ris) / 100)

    point_est = impact["conversion_base"] * gap_fraction * cat_mult * ris_upside_mult
    lower = point_est * 0.6
    upper = point_est * 1.4

    confidence = "high" if abs(gap_score) > 50 else "medium" if abs(gap_score) > 25 else "low"

    return {
        "aspect": aspect,
        "gap_score": round(gap_score, 1),
        "lift_pct": round(point_est * 100, 1),
        "lift_range": [round(lower * 100, 1), round(upper * 100, 1)],
        "timeframe": impact["timeframe"],
        "confidence": confidence,
    }


def build_prescriptive_card(
    asin: str,
    gap_scores: dict,
    category: str,
    current_ris: float,
) -> dict:
    actions = []
    for aspect, gap in sorted(gap_scores.items(), key=lambda x: x[1], reverse=True):
        if gap > 5:
            lift = estimate_sales_lift(aspect, gap, category, current_ris)
            if lift:
                actions.append(lift)

    actions.sort(key=lambda x: x["lift_pct"], reverse=True)
    top3 = actions[:3]

    for i, a in enumerate(top3):
        a["rank"] = i + 1
        a["cta"] = f"Improve {a['aspect'].replace('_', ' ')} to match top competitors"

    total_lift = sum(a["lift_pct"] * (0.85 ** i) for i, a in enumerate(top3))
    total_lower = sum(a["lift_range"][0] * (0.85 ** i) for i, a in enumerate(top3))
    total_upper = sum(a["lift_range"][1] * (0.85 ** i) for i, a in enumerate(top3))

    timeframes = {"30 days": 30, "60 days": 60, "90 days": 90}
    primary_tf = max(
        (a["timeframe"] for a in top3), key=lambda t: timeframes.get(t, 60)
    ) if top3 else "60 days"

    return {
        "asin": asin,
        "current_ris": current_ris,
        "top_actions": top3,
        "projected_total_lift": {
            "point_estimate": round(total_lift, 1),
            "range": [round(total_lower, 1), round(total_upper, 1)],
            "timeframe": primary_tf,
        },
    }


def forecast_sentiment_trajectory(asin: str, reviews_df: pd.DataFrame, periods: int = 90):
    """Wrapper around Prophet forecast — returns weekly forecast DataFrame or None."""
    if "review_date" not in reviews_df.columns:
        return None

    asin_df = reviews_df[reviews_df["asin"] == asin].copy()
    if len(asin_df) < 30:
        return None

    try:
        from prophet import Prophet

        ts = (
            asin_df.set_index("review_date")
            .resample("W")["star_rating"]
            .mean()
            .reset_index()
            .rename(columns={"review_date": "ds", "star_rating": "y"})
            .dropna()
        )

        if len(ts) < 8:
            return None

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.1,
            interval_width=0.8,
        )
        model.fit(ts)
        future = model.make_future_dataframe(periods=int(periods / 7), freq="W")
        forecast = model.predict(future)

        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(1, 5)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    except Exception as e:
        logger.warning(f"Prophet forecast failed: {e}")
        return None

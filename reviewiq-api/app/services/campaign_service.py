"""ReviewIQ — Campaign Service (in-memory store, swap for DB in production)"""

import uuid
import pandas as pd
from datetime import datetime, date
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# In-memory campaign store
# In production: replace with PostgreSQL/Redis
_campaigns: dict = {}


def create_campaign(
    name: str,
    asin: str,
    start_date: str,
    end_date: Optional[str],
    goal: Optional[str],
    baseline_ris: Optional[float],
) -> dict:
    campaign_id = str(uuid.uuid4())[:8]
    campaign = {
        "id": campaign_id,
        "name": name,
        "asin": asin,
        "start_date": start_date,
        "end_date": end_date,
        "goal": goal,
        "baseline_ris": baseline_ris,
        "created_at": datetime.utcnow().isoformat(),
    }
    _campaigns[campaign_id] = campaign
    return campaign


def get_campaign(campaign_id: str) -> Optional[dict]:
    return _campaigns.get(campaign_id)


def list_campaigns() -> List[dict]:
    return list(_campaigns.values())


def delete_campaign(campaign_id: str) -> bool:
    if campaign_id in _campaigns:
        del _campaigns[campaign_id]
        return True
    return False


def get_campaign_performance(
    campaign: dict,
    reviews_df: pd.DataFrame,
    ris_df: pd.DataFrame,
) -> dict:
    asin = campaign["asin"]
    start = pd.to_datetime(campaign["start_date"])
    end = pd.to_datetime(campaign.get("end_date") or datetime.utcnow().isoformat())

    if "review_date" not in reviews_df.columns or reviews_df.empty:
        return {"sentiment_trend": [], "current_ris": None, "ris_delta": None}

    asin_reviews = reviews_df[
        (reviews_df["asin"] == asin) &
        (reviews_df["review_date"] >= start) &
        (reviews_df["review_date"] <= end)
    ].copy()

    # Weekly sentiment trend
    trend = []
    if not asin_reviews.empty:
        weekly = (
            asin_reviews.set_index("review_date")
            .resample("W")
            .agg(
                avg_rating=("star_rating", "mean"),
                review_count=("star_rating", "count"),
            )
            .reset_index()
        )

        for _, row in weekly.iterrows():
            trend.append({
                "date": str(row["review_date"].date()),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "review_count": int(row["review_count"]),
                "positive_pct": round(
                    float((asin_reviews[
                        (asin_reviews["review_date"] >= row["review_date"] - pd.Timedelta("7D")) &
                        (asin_reviews["review_date"] <= row["review_date"])
                    ]["star_rating"] >= 4).mean() * 100), 1
                ),
            })

    # Current RIS
    current_ris = None
    ris_delta = None
    if asin in ris_df.index:
        current_ris = float(ris_df.loc[asin, "ris_score"])
        if campaign.get("baseline_ris") is not None:
            ris_delta = round(current_ris - campaign["baseline_ris"], 1)

    # Alert: if last week avg_rating < baseline avg rating
    alert = None
    if len(trend) >= 2:
        recent_rating = trend[-1]["avg_rating"]
        early_rating = trend[0]["avg_rating"]
        if recent_rating < early_rating - 0.3:
            alert = f"⚠️ Sentiment declining: {recent_rating:.2f}★ vs {early_rating:.2f}★ at campaign start"
        elif ris_delta is not None and ris_delta < -5:
            alert = f"⚠️ RIS dropped {abs(ris_delta):.1f} points since campaign start"

    return {
        "sentiment_trend": trend,
        "current_ris": current_ris,
        "ris_delta": ris_delta,
        "alert": alert,
    }

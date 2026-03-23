"""
ReviewIQ — Sentiment Router
POST /api/sentiment/score     → Score a list of review texts
GET  /api/sentiment/asin/{asin} → Aggregate sentiment for an ASIN
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import SentimentRequest, SentimentResponse, SentimentResult
from app.core.state import app_state
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/score", response_model=SentimentResponse)
async def score_sentiment(request: SentimentRequest):
    """
    Score sentiment for up to 100 review texts using the DistilBERT/RoBERTa ensemble.

    Returns per-review label, probabilities, confidence, and aggregate stats.
    """
    if app_state.sentiment_service is None:
        raise HTTPException(503, "Sentiment model not loaded. Run notebook 03 first.")

    results_raw = app_state.sentiment_service.predict(request.texts)

    results = []
    for i, text in enumerate(request.texts):
        pos_prob = float(results_raw["positive_prob"][i])
        neg_prob = float(results_raw["negative_prob"][i])
        confidence = float(results_raw["confidence"][i])
        label = results_raw["labels"][i]

        # Estimate star rating from positive probability
        # positive_prob 0 → 1 star, 1.0 → 5 stars
        star_est = round(1 + pos_prob * 4, 1)

        results.append(SentimentResult(
            text=text[:200],  # truncate for response
            label=label,
            positive_prob=round(pos_prob, 4),
            negative_prob=round(neg_prob, 4),
            confidence=round(confidence, 4),
            star_rating_estimate=star_est,
        ))

    pos_probs = [r.positive_prob for r in results]
    aggregate = {
        "avg_positive_prob": round(float(np.mean(pos_probs)), 4),
        "pct_positive": round(sum(1 for r in results if r.label == "positive") / len(results) * 100, 1),
        "avg_confidence": round(float(np.mean([r.confidence for r in results])), 4),
        "avg_star_estimate": round(float(np.mean([r.star_rating_estimate for r in results])), 2),
    }

    return SentimentResponse(results=results, aggregate=aggregate)


@router.get("/asin/{asin}", response_model=SentimentResponse)
async def get_asin_sentiment(asin: str, max_reviews: int = 500):
    """
    Get aggregate sentiment for all reviews of an ASIN.
    Samples up to max_reviews for efficiency.
    """
    if app_state.reviews_df.empty:
        raise HTTPException(503, "Reviews data not loaded.")
    if app_state.sentiment_service is None:
        raise HTTPException(503, "Sentiment model not loaded.")

    asin_reviews = app_state.reviews_df[
        app_state.reviews_df["asin"] == asin.upper()
    ]["full_text"].dropna()

    if asin_reviews.empty:
        raise HTTPException(404, f"No reviews found for ASIN {asin}")

    if len(asin_reviews) > max_reviews:
        asin_reviews = asin_reviews.sample(max_reviews, random_state=42)

    return await score_sentiment(SentimentRequest(
        texts=asin_reviews.tolist(),
        asin=asin,
    ))

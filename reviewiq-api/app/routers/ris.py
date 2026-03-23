"""
ReviewIQ — RIS Router
GET  /api/ris/{asin}         → Single ASIN RIS breakdown
POST /api/ris/batch          → Batch RIS for multiple ASINs
GET  /api/ris/category/{cat} → Top ASINs in a category by RIS
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.models.schemas import RISResponse, RISComponents, RISBatchRequest, RISBatchResponse
from app.core.state import app_state
from app.services.ris_service import get_ris_for_asin, get_top_asins_by_category
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

WEIGHTS = {
    "sentiment_score": 0.25,
    "authenticity_score": 0.25,
    "aspect_coverage_score": 0.20,
    "velocity_trend_score": 0.15,
    "competitive_position_score": 0.15,
}


@router.get("/{asin}", response_model=RISResponse)
async def get_ris(asin: str):
    """
    Get the full RIS (Review Intelligence Score) breakdown for a single ASIN.

    Returns a 0-100 composite score with component breakdown, grade, and
    category percentile rank.
    """
    if app_state.ris_df.empty:
        raise HTTPException(503, "RIS data not loaded. Run notebook 04 first.")

    result = get_ris_for_asin(asin.upper(), app_state.ris_df)
    if "error" in result:
        raise HTTPException(404, result["error"])

    return RISResponse(
        asin=result["asin"],
        ris_score=result["ris_score"],
        ris_grade=result["ris_grade"],
        category=result["category"],
        category_percentile=result["category_percentile"],
        components=RISComponents(**result["components"]),
        weights=WEIGHTS,
    )


@router.post("/batch", response_model=RISBatchResponse)
async def get_ris_batch(request: RISBatchRequest):
    """
    Get RIS scores for up to 50 ASINs in a single request.
    Useful for competitor comparison tables.
    """
    if app_state.ris_df.empty:
        raise HTTPException(503, "RIS data not loaded.")

    results = []
    not_found = []

    for asin in request.asins:
        result = get_ris_for_asin(asin.upper(), app_state.ris_df)
        if "error" in result:
            not_found.append(asin)
        else:
            results.append(RISResponse(
                asin=result["asin"],
                ris_score=result["ris_score"],
                ris_grade=result["ris_grade"],
                category=result["category"],
                category_percentile=result["category_percentile"],
                components=RISComponents(**result["components"]),
                weights=WEIGHTS,
            ))

    return RISBatchResponse(results=results, not_found=not_found)


@router.get("/category/{category}", response_model=List[RISResponse])
async def get_category_leaderboard(
    category: str,
    top_n: int = Query(10, ge=1, le=50),
    min_reviews: int = Query(50, ge=0),
):
    """
    Get the top N ASINs in a category ranked by RIS score.
    Useful for competitive benchmarking.
    """
    if app_state.ris_df.empty:
        raise HTTPException(503, "RIS data not loaded.")

    top = get_top_asins_by_category(
        category, app_state.ris_df, app_state.reviews_df, top_n, min_reviews
    )

    if top.empty:
        raise HTTPException(404, f"Category '{category}' not found or no ASINs with enough reviews.")

    return [
        RISResponse(
            asin=row.name,
            ris_score=row["ris_score"],
            ris_grade=row["ris_grade"],
            category=row["category"],
            category_percentile=row["category_percentile"],
            components=RISComponents(
                sentiment_score=row["sentiment_score"],
                authenticity_score=row["authenticity_score"],
                aspect_coverage_score=row["aspect_coverage_score"],
                velocity_trend_score=row["velocity_trend_score"],
                competitive_position_score=row["competitive_position_score"],
            ),
            weights=WEIGHTS,
        )
        for _, row in top.iterrows()
    ]

"""
ReviewIQ — Revenue Impact Predictor Router
GET  /api/revenue/{asin}             → Full revenue impact report + Prophet forecast
GET  /api/revenue/{asin}/quick       → Quick lift estimates (no forecast)
"""

from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import RevenueImpactResponse, ActionLift
from app.core.state import app_state
from app.services.revenue_service import (
    build_prescriptive_card,
    forecast_sentiment_trajectory,
)
from app.services.competitor_service import get_competitors, build_aspect_gap_matrix
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{asin}", response_model=RevenueImpactResponse)
async def get_revenue_impact(
    asin: str,
    top_n_competitors: int = Query(3, ge=1, le=5),
):
    """
    Full revenue impact report for an ASIN.

    Returns:
    - Top 3 ranked actions with estimated sales lift % and confidence intervals
    - Combined projected lift (e.g. "+13-20% in 60 days")
    - Prophet 30/60/90 day rating trajectory forecast
    """
    if app_state.ris_df.empty or app_state.reviews_df.empty:
        raise HTTPException(503, "Data not loaded. Run notebooks 01 and 04 first.")

    asin = asin.upper()
    if asin not in app_state.ris_df.index:
        raise HTTPException(404, f"ASIN {asin} not found.")

    try:
        ris_row = app_state.ris_df.loc[asin]
        category = str(ris_row["category"])
        current_ris = float(ris_row["ris_score"])

        # Get gap scores
        competitors = get_competitors(asin, app_state.reviews_df, app_state.ris_df, top_n_competitors)
        gap_matrix_df = build_aspect_gap_matrix(asin, competitors, app_state.reviews_df)
        gap_scores = {
            k: float(v) for k, v in gap_matrix_df.loc["GAP"].dropna().items()
            if float(v) > 0  # only gaps where competitors lead
        }

        # Build prescriptive card
        card = build_prescriptive_card(asin, gap_scores, category, current_ris)

        # Prophet forecast
        forecast_30d = forecast_60d = forecast_90d = None
        try:
            forecast = forecast_sentiment_trajectory(asin, app_state.reviews_df)
            if forecast is not None and len(forecast) > 0:
                # Get predicted ratings at day 30, 60, 90
                import pandas as pd
                last_hist_date = app_state.reviews_df[
                    app_state.reviews_df["asin"] == asin
                ]["review_date"].max() if "review_date" in app_state.reviews_df.columns else None

                if last_hist_date is not None:
                    future_rows = forecast[forecast["ds"] > last_hist_date]
                    if len(future_rows) >= 4:
                        forecast_30d = round(float(future_rows.iloc[4]["yhat"]), 2)
                    if len(future_rows) >= 8:
                        forecast_60d = round(float(future_rows.iloc[8]["yhat"]), 2)
                    if len(future_rows) >= 12:
                        forecast_90d = round(float(future_rows.iloc[12]["yhat"]), 2)
        except Exception as fe:
            logger.warning(f"Forecast failed for {asin}: {fe}")

        # Build response
        top_actions = [
            ActionLift(
                rank=a["rank"],
                aspect=a["aspect"],
                gap_score=a["gap_score"],
                lift_pct=a["lift_pct"],
                lift_range=a["lift_range"],
                timeframe=a["timeframe"],
                confidence=a["confidence"],
                cta=a["cta"],
            )
            for a in card["top_actions"]
        ]

        return RevenueImpactResponse(
            asin=asin,
            category=category,
            current_ris=current_ris,
            top_actions=top_actions,
            projected_total_lift=card["projected_total_lift"],
            forecast_30d=forecast_30d,
            forecast_60d=forecast_60d,
            forecast_90d=forecast_90d,
        )

    except Exception as e:
        logger.error(f"Revenue impact failed for {asin}: {e}")
        raise HTTPException(500, f"Revenue impact calculation failed: {str(e)}")


@router.get("/{asin}/quick")
async def get_quick_lift_estimate(asin: str):
    """
    Fast lift estimate using pre-computed RIS components.
    No competitor analysis or Prophet forecast — returns in <100ms.
    """
    if app_state.ris_df.empty:
        raise HTTPException(503, "RIS data not loaded.")

    asin = asin.upper()
    if asin not in app_state.ris_df.index:
        raise HTTPException(404, f"ASIN {asin} not found.")

    row = app_state.ris_df.loc[asin]
    current_ris = float(row["ris_score"])

    # Quick estimate: each 10 RIS points below 80 = ~5% upside
    upside_ris_pts = max(0, 80 - current_ris)
    quick_lift_est = round(upside_ris_pts * 0.5, 1)

    return {
        "asin": asin,
        "current_ris": current_ris,
        "ris_grade": row["ris_grade"],
        "quick_lift_estimate_pct": quick_lift_est,
        "quick_lift_range": [round(quick_lift_est * 0.6, 1), round(quick_lift_est * 1.4, 1)],
        "note": "Quick estimate only. Call GET /api/revenue/{asin} for full analysis.",
    }

"""
ReviewIQ — Competitor Intelligence Router
GET  /api/competitor/{asin}          → Full competitor analysis with gap matrix
GET  /api/competitor/{asin}/gaps     → Just the aspect gap scores (fast)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from app.models.schemas import CompetitorAnalysisResponse, AspectGapMatrix, CompetitorAction
from app.core.state import app_state
from app.services.competitor_service import (
    get_competitors,
    build_aspect_gap_matrix,
    generate_action_brief_claude,
)
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{asin}", response_model=CompetitorAnalysisResponse)
async def get_competitor_analysis(
    asin: str,
    top_n: int = Query(3, ge=1, le=5),
):
    """
    Full competitor analysis for an ASIN:
    - Identifies top N competitors in same category
    - Builds aspect gap matrix (8 dimensions)
    - Calls Claude API to generate ranked prescriptive action brief

    Note: First call may take 5-10 seconds due to Claude API call.
    """
    if app_state.ris_df.empty or app_state.reviews_df.empty:
        raise HTTPException(503, "Data not loaded. Run notebooks 01 and 04 first.")

    asin = asin.upper()

    if asin not in app_state.ris_df.index:
        raise HTTPException(404, f"ASIN {asin} not found in RIS data.")

    try:
        # Find competitors
        competitors = get_competitors(asin, app_state.reviews_df, app_state.ris_df, top_n)

        # Build gap matrix
        gap_matrix_df = build_aspect_gap_matrix(asin, competitors, app_state.reviews_df)

        # Claude API brief
        brief = await generate_action_brief_claude(
            asin, gap_matrix_df, app_state.ris_df, settings.ANTHROPIC_API_KEY
        )

        # Build response
        gap_row = gap_matrix_df.loc["GAP"].dropna()
        target_row = gap_matrix_df.loc["TARGET"].dropna()
        aspects = gap_matrix_df.columns.tolist()

        comp_scores = {}
        for comp_label in [f"COMP_{i+1}" for i in range(len(competitors))]:
            if comp_label in gap_matrix_df.index:
                comp_scores[comp_label] = {
                    k: (None if v != v else round(float(v), 1))
                    for k, v in gap_matrix_df.loc[comp_label].items()
                }

        gap_matrix = AspectGapMatrix(
            aspects=aspects,
            target_scores={k: (None if v != v else round(float(v), 1)) for k, v in target_row.items()},
            competitor_scores=comp_scores,
            gaps={k: (None if v != v else round(float(v), 1)) for k, v in gap_row.items()},
        )

        actions = [
            CompetitorAction(
                rank=i + 1,
                aspect=a.get("aspect", a.get("dimension", "general")),
                what_to_fix=a.get("what_to_fix", a.get("action", "")),
                why_it_matters=a.get("why_it_matters", a.get("reason", "")),
                estimated_lift_pct_range=a.get("estimated_lift_pct_range", [0, 0]),
                timeframe=a.get("timeframe", "60 days"),
            )
            for i, a in enumerate(brief.get("actions", []))
        ]

        return CompetitorAnalysisResponse(
            asin=asin,
            competitors=competitors,
            target_ris=float(app_state.ris_df.loc[asin, "ris_score"]),
            gap_matrix=gap_matrix,
            top_actions=actions,
            strength=brief.get("strength", ""),
            strategic_recommendation=brief.get("strategic_recommendation", ""),
        )

    except Exception as e:
        logger.error(f"Competitor analysis failed for {asin}: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.get("/{asin}/gaps")
async def get_aspect_gaps_only(asin: str, top_n: int = Query(3, ge=1, le=5)):
    """
    Lightweight endpoint: returns just the aspect gap scores without Claude synthesis.
    Use for real-time UI updates.
    """
    if app_state.ris_df.empty or app_state.reviews_df.empty:
        raise HTTPException(503, "Data not loaded.")

    asin = asin.upper()
    if asin not in app_state.ris_df.index:
        raise HTTPException(404, f"ASIN {asin} not found.")

    competitors = get_competitors(asin, app_state.reviews_df, app_state.ris_df, top_n)
    gap_matrix_df = build_aspect_gap_matrix(asin, competitors, app_state.reviews_df)
    gap_row = gap_matrix_df.loc["GAP"].dropna()

    return {
        "asin": asin,
        "competitors": competitors,
        "gaps": {k: round(float(v), 1) for k, v in gap_row.items()},
        "target_scores": {
            k: round(float(v), 1)
            for k, v in gap_matrix_df.loc["TARGET"].dropna().items()
        },
    }

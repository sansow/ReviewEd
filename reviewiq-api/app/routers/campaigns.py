"""
ReviewIQ — Campaigns Router
POST /api/campaigns/             → Create a campaign
GET  /api/campaigns/             → List all campaigns
GET  /api/campaigns/{id}         → Get campaign details
GET  /api/campaigns/{id}/performance → Sentiment trend + RIS delta
DELETE /api/campaigns/{id}       → Delete campaign
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    CampaignCreateRequest, CampaignPerformanceResponse, CampaignSentimentPoint
)
from app.core.state import app_state
from app.services.campaign_service import (
    create_campaign, get_campaign, list_campaigns,
    delete_campaign, get_campaign_performance,
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", status_code=201)
async def create_new_campaign(request: CampaignCreateRequest):
    """
    Create a new campaign to track review sentiment over a time period.

    Captures baseline RIS at campaign start for delta tracking.
    """
    if app_state.ris_df.empty:
        raise HTTPException(503, "RIS data not loaded.")

    asin = request.asin.upper()
    baseline_ris = request.baseline_ris

    if baseline_ris is None and asin in app_state.ris_df.index:
        baseline_ris = float(app_state.ris_df.loc[asin, "ris_score"])

    campaign = create_campaign(
        name=request.name,
        asin=asin,
        start_date=request.start_date,
        end_date=request.end_date,
        goal=request.goal,
        baseline_ris=baseline_ris,
    )

    return {"campaign_id": campaign["id"], "message": "Campaign created", "campaign": campaign}


@router.get("/")
async def list_all_campaigns():
    """List all campaigns."""
    return {"campaigns": list_campaigns()}


@router.get("/{campaign_id}")
async def get_campaign_detail(campaign_id: str):
    """Get campaign metadata."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(404, f"Campaign {campaign_id} not found.")
    return campaign


@router.get("/{campaign_id}/performance", response_model=CampaignPerformanceResponse)
async def get_performance(campaign_id: str):
    """
    Get campaign performance:
    - Weekly sentiment trend over campaign window
    - RIS delta (current vs baseline)
    - Alert if sentiment dropped below baseline
    """
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(404, f"Campaign {campaign_id} not found.")

    if app_state.reviews_df.empty:
        raise HTTPException(503, "Reviews data not loaded.")

    asin = campaign["asin"]
    perf = get_campaign_performance(campaign, app_state.reviews_df, app_state.ris_df)

    trend = [
        CampaignSentimentPoint(
            date=pt["date"],
            avg_rating=pt["avg_rating"],
            review_count=pt["review_count"],
            ris_score=pt.get("ris_score"),
            positive_pct=pt["positive_pct"],
        )
        for pt in perf["sentiment_trend"]
    ]

    return CampaignPerformanceResponse(
        campaign_id=campaign_id,
        asin=asin,
        name=campaign["name"],
        baseline_ris=campaign.get("baseline_ris"),
        current_ris=perf.get("current_ris"),
        ris_delta=perf.get("ris_delta"),
        sentiment_trend=trend,
        alert=perf.get("alert"),
    )


@router.delete("/{campaign_id}", status_code=204)
async def delete_campaign_endpoint(campaign_id: str):
    """Delete a campaign."""
    success = delete_campaign(campaign_id)
    if not success:
        raise HTTPException(404, f"Campaign {campaign_id} not found.")

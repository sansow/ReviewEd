"""
ReviewIQ — Pydantic Models
Request and response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ── Shared ────────────────────────────────────────────────────────────────────

class RISGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ── RIS ───────────────────────────────────────────────────────────────────────

class RISComponents(BaseModel):
    sentiment_score: float = Field(..., ge=0, le=100, description="Weighted positive sentiment")
    authenticity_score: float = Field(..., ge=0, le=100, description="Inverse fake review probability")
    aspect_coverage_score: float = Field(..., ge=0, le=100, description="Breadth of aspects discussed")
    velocity_trend_score: float = Field(..., ge=0, le=100, description="Recent review momentum")
    competitive_position_score: float = Field(..., ge=0, le=100, description="vs category average")


class RISResponse(BaseModel):
    asin: str
    ris_score: float = Field(..., ge=0, le=100)
    ris_grade: RISGrade
    category: str
    category_percentile: float = Field(..., ge=0, le=100)
    components: RISComponents
    weights: Dict[str, float]


class RISBatchRequest(BaseModel):
    asins: List[str] = Field(..., max_items=50)


class RISBatchResponse(BaseModel):
    results: List[RISResponse]
    not_found: List[str]


# ── Sentiment ─────────────────────────────────────────────────────────────────

class SentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100, description="Review texts to score")
    asin: Optional[str] = None


class SentimentResult(BaseModel):
    text: str
    label: str  # "positive" | "negative"
    positive_prob: float
    negative_prob: float
    confidence: float
    star_rating_estimate: Optional[float] = None


class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    aggregate: Dict[str, float]  # avg_positive_prob, pct_positive, avg_confidence


# ── Competitor ────────────────────────────────────────────────────────────────

class AspectGapMatrix(BaseModel):
    aspects: List[str]
    target_scores: Dict[str, Optional[float]]
    competitor_scores: Dict[str, Dict[str, Optional[float]]]  # comp_id -> aspect -> score
    gaps: Dict[str, Optional[float]]  # aspect -> gap (positive = competitor leads)


class CompetitorAction(BaseModel):
    rank: int
    aspect: str
    what_to_fix: str
    why_it_matters: str
    estimated_lift_pct_range: List[float]
    timeframe: str


class CompetitorAnalysisResponse(BaseModel):
    asin: str
    competitors: List[str]
    target_ris: float
    gap_matrix: AspectGapMatrix
    top_actions: List[CompetitorAction]
    strength: str
    strategic_recommendation: str


# ── Revenue Impact ────────────────────────────────────────────────────────────

class ActionLift(BaseModel):
    rank: int
    aspect: str
    gap_score: float
    lift_pct: float
    lift_range: List[float]  # [lower, upper]
    timeframe: str
    confidence: str  # "high" | "medium" | "low"
    cta: str


class RevenueImpactResponse(BaseModel):
    asin: str
    category: str
    current_ris: float
    top_actions: List[ActionLift]
    projected_total_lift: Dict[str, Any]  # point_estimate, range, timeframe
    forecast_30d: Optional[float] = None  # predicted avg rating in 30 days
    forecast_60d: Optional[float] = None
    forecast_90d: Optional[float] = None


# ── Ad Copy ───────────────────────────────────────────────────────────────────

class AdCopyRequest(BaseModel):
    asin: str
    min_rating: float = Field(4.0, ge=1, le=5)
    max_reviews: int = Field(200, ge=10, le=1000)
    formats: List[str] = Field(
        default=["amazon_bullets", "facebook_ad", "instagram_caption"],
        description="Output formats to generate"
    )
    brand_tone: Optional[str] = Field(None, description="e.g. 'professional', 'fun', 'scientific'")


class AdCopyVariant(BaseModel):
    format: str
    copy: str
    source_phrases: List[str]  # actual customer phrases used


class AdCopyResponse(BaseModel):
    asin: str
    reviews_analyzed: int
    top_phrases: List[str]
    variants: List[AdCopyVariant]


# ── Campaigns ─────────────────────────────────────────────────────────────────

class CampaignCreateRequest(BaseModel):
    name: str
    asin: str
    start_date: str  # ISO date
    end_date: Optional[str] = None
    goal: Optional[str] = None  # e.g. "improve packaging score"
    baseline_ris: Optional[float] = None


class CampaignSentimentPoint(BaseModel):
    date: str
    avg_rating: float
    review_count: int
    ris_score: Optional[float] = None
    positive_pct: float


class CampaignPerformanceResponse(BaseModel):
    campaign_id: str
    asin: str
    name: str
    baseline_ris: Optional[float]
    current_ris: Optional[float]
    ris_delta: Optional[float]
    sentiment_trend: List[CampaignSentimentPoint]
    alert: Optional[str] = None  # e.g. "Sentiment dropped below baseline"

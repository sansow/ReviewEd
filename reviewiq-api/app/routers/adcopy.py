"""
ReviewIQ — Ad Copy Generator Router
POST /api/adcopy/generate   → Generate ad copy from top positive reviews
GET  /api/adcopy/phrases/{asin} → Extract top power phrases only
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import AdCopyRequest, AdCopyResponse, AdCopyVariant
from app.core.state import app_state
from app.core.config import settings
from app.services.adcopy_service import extract_power_phrases, generate_ad_copy_claude
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=AdCopyResponse)
async def generate_ad_copy(request: AdCopyRequest):
    """
    Generate ad copy from authentic customer review language.

    Uses top-rated reviews to extract genuine customer phrases, then
    calls Claude to generate platform-specific ad copy variants.

    Supported formats:
    - amazon_bullets: 5 A+ content bullet points
    - facebook_ad: Short-form FB/Instagram ad copy (3 variants)
    - instagram_caption: IG caption with hashtag suggestions
    - google_rsa: Google Responsive Search Ad headlines + descriptions
    - email_subject: Email subject line variants
    """
    if app_state.reviews_df.empty:
        raise HTTPException(503, "Reviews data not loaded.")
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured.")

    asin = request.asin.upper()
    asin_reviews = app_state.reviews_df[
        (app_state.reviews_df["asin"] == asin) &
        (app_state.reviews_df["star_rating"] >= request.min_rating)
    ]

    if asin_reviews.empty:
        raise HTTPException(404, f"No reviews found for ASIN {asin} with rating >= {request.min_rating}")

    if len(asin_reviews) > request.max_reviews:
        asin_reviews = asin_reviews.sample(request.max_reviews, random_state=42)

    # Extract power phrases from review text
    texts = asin_reviews["review_text"].dropna().tolist()
    power_phrases = extract_power_phrases(texts, top_n=20)

    if not power_phrases:
        raise HTTPException(422, "Could not extract meaningful phrases from reviews.")

    # Generate ad copy via Claude
    variants_raw = await generate_ad_copy_claude(
        asin=asin,
        power_phrases=power_phrases,
        formats=request.formats,
        brand_tone=request.brand_tone,
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.CLAUDE_MODEL,
    )

    variants = [
        AdCopyVariant(
            format=v["format"],
            copy=v["copy"],
            source_phrases=v.get("source_phrases", []),
        )
        for v in variants_raw
    ]

    return AdCopyResponse(
        asin=asin,
        reviews_analyzed=len(texts),
        top_phrases=power_phrases[:10],
        variants=variants,
    )


@router.get("/phrases/{asin}")
async def get_power_phrases(
    asin: str,
    min_rating: float = 4.0,
    max_reviews: int = 300,
    top_n: int = 20,
):
    """
    Extract top customer power phrases for an ASIN without generating ad copy.
    Fast endpoint for populating phrase libraries.
    """
    if app_state.reviews_df.empty:
        raise HTTPException(503, "Reviews data not loaded.")

    asin = asin.upper()
    asin_reviews = app_state.reviews_df[
        (app_state.reviews_df["asin"] == asin) &
        (app_state.reviews_df["star_rating"] >= min_rating)
    ]

    if asin_reviews.empty:
        raise HTTPException(404, f"No reviews found for ASIN {asin}")

    if len(asin_reviews) > max_reviews:
        asin_reviews = asin_reviews.sample(max_reviews, random_state=42)

    from app.services.adcopy_service import extract_power_phrases
    phrases = extract_power_phrases(asin_reviews["review_text"].dropna().tolist(), top_n=top_n)

    return {
        "asin": asin,
        "reviews_analyzed": len(asin_reviews),
        "power_phrases": phrases,
    }

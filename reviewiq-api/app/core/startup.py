"""
ReviewIQ — Startup Model Loader
Loads trained models and RIS data into app state on startup.
"""

import pandas as pd
from pathlib import Path
from app.core.config import settings
from app.core.state import app_state
import logging

logger = logging.getLogger(__name__)


async def load_models():
    """Load all models and data into memory at startup."""

    logger.info("🚀 ReviewIQ API starting up — loading models...")

    # ── RIS scores ────────────────────────────────────────────────────────────
    ris_path = settings.DATA_DIR / "ris_outputs" / "ris_scores.parquet"
    if ris_path.exists():
        app_state.ris_df = pd.read_parquet(ris_path).set_index("asin")
        logger.info(f"✅ RIS scores loaded: {len(app_state.ris_df):,} ASINs")
    else:
        logger.warning(f"⚠️  RIS data not found at {ris_path}. Run notebook 04 first.")
        app_state.ris_df = pd.DataFrame()

    # ── Reviews data ──────────────────────────────────────────────────────────
    reviews_path = settings.DATA_DIR / "processed" / "reviews_clean_all.parquet"
    if reviews_path.exists():
        app_state.reviews_df = pd.read_parquet(reviews_path)
        logger.info(f"✅ Reviews loaded: {len(app_state.reviews_df):,} rows")
    else:
        logger.warning(f"⚠️  Reviews data not found. Run notebook 01 first.")
        app_state.reviews_df = pd.DataFrame()

    # ── Sentiment ensemble ────────────────────────────────────────────────────
    try:
        from app.services.sentiment_service import SentimentService
        app_state.sentiment_service = SentimentService()
        logger.info("✅ Sentiment ensemble loaded")
    except Exception as e:
        logger.warning(f"⚠️  Sentiment model not loaded (run notebook 03): {e}")
        app_state.sentiment_service = None

    logger.info("✅ ReviewIQ API ready")

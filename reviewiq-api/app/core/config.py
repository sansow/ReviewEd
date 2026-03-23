"""
ReviewIQ — App Configuration
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # API
    APP_NAME: str = "ReviewIQ API"
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # Paths
    DATA_DIR: Path = Path("../../data")
    MODELS_DIR: Path = Path("../../models")

    # CORS — add your frontend URL here
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://reviewiq.yourdomain.com",
    ]

    # Model settings
    SENTIMENT_BATCH_SIZE: int = 32
    SENTIMENT_MAX_LENGTH: int = 256
    ENSEMBLE_WEIGHTS: List[float] = [0.4, 0.6]  # distilbert, roberta

    # Revenue model
    DEFAULT_CATEGORY_MULTIPLIER: float = 1.0

    # Cache
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

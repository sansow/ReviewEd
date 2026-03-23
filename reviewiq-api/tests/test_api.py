"""
ReviewIQ API — Test Suite
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.core import state as app_state_module

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_app_state():
    """Inject mock data into app_state for all tests."""
    from app.core.state import app_state

    # Minimal RIS dataframe
    ris_data = {
        "ris_score": [72.5, 65.0, 80.1],
        "ris_grade": ["A", "B", "A+"],
        "category": ["Health_and_Household"] * 3,
        "category_percentile": [75.0, 55.0, 90.0],
        "sentiment_score": [78.0, 65.0, 82.0],
        "authenticity_score": [85.0, 70.0, 88.0],
        "aspect_coverage_score": [60.0, 55.0, 70.0],
        "velocity_trend_score": [65.0, 60.0, 75.0],
        "competitive_position_score": [70.0, 62.0, 80.0],
    }
    app_state.ris_df = pd.DataFrame(ris_data, index=["B001TEST01", "B001TEST02", "B001TEST03"])

    # Minimal reviews dataframe
    reviews_data = {
        "asin": ["B001TEST01"] * 10 + ["B001TEST02"] * 10 + ["B001TEST03"] * 10,
        "review_text": ["Great product works well"] * 30,
        "full_text": ["Great product works well"] * 30,
        "star_rating": [5.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0, 4.0, 5.0] * 3,
        "category": ["Health_and_Household"] * 30,
        "word_count": [5] * 30,
        "review_length": [25] * 30,
        "fake_risk": [0.1] * 30,
    }
    app_state.reviews_df = pd.DataFrame(reviews_data)
    app_state.sentiment_service = None  # no model in tests

    yield


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── RIS ───────────────────────────────────────────────────────────────────────

def test_get_ris_found():
    r = client.get("/api/ris/B001TEST01")
    assert r.status_code == 200
    data = r.json()
    assert data["asin"] == "B001TEST01"
    assert 0 <= data["ris_score"] <= 100
    assert data["ris_grade"] in ["A+", "A", "B", "C", "D", "F"]
    assert "components" in data
    assert "weights" in data


def test_get_ris_not_found():
    r = client.get("/api/ris/NOTEXIST01")
    assert r.status_code == 404


def test_ris_batch():
    r = client.post("/api/ris/batch", json={"asins": ["B001TEST01", "B001TEST02", "NOTEXIST"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    assert "NOTEXIST" in data["not_found"]


def test_ris_category_leaderboard():
    r = client.get("/api/ris/category/Health_and_Household?top_n=5&min_reviews=0")
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)
    assert len(results) <= 5
    # Should be sorted by RIS descending
    if len(results) > 1:
        assert results[0]["ris_score"] >= results[1]["ris_score"]


# ── Sentiment ─────────────────────────────────────────────────────────────────

def test_sentiment_no_model():
    r = client.post("/api/sentiment/score", json={"texts": ["Great product!"]})
    assert r.status_code == 503  # model not loaded in tests


def test_sentiment_asin_no_model():
    r = client.get("/api/sentiment/asin/B001TEST01")
    assert r.status_code == 503


# ── Competitor ────────────────────────────────────────────────────────────────

def test_competitor_gaps_fast():
    r = client.get("/api/competitor/B001TEST01/gaps?top_n=2")
    assert r.status_code == 200
    data = r.json()
    assert data["asin"] == "B001TEST01"
    assert "gaps" in data
    assert "competitors" in data


def test_competitor_not_found():
    r = client.get("/api/competitor/NOTEXIST01/gaps")
    assert r.status_code == 404


# ── Revenue ───────────────────────────────────────────────────────────────────

def test_revenue_quick():
    r = client.get("/api/revenue/B001TEST01/quick")
    assert r.status_code == 200
    data = r.json()
    assert "current_ris" in data
    assert "quick_lift_estimate_pct" in data
    assert data["quick_lift_estimate_pct"] >= 0


def test_revenue_quick_not_found():
    r = client.get("/api/revenue/NOTEXIST01/quick")
    assert r.status_code == 404


# ── Ad Copy ───────────────────────────────────────────────────────────────────

def test_adcopy_phrases():
    r = client.get("/api/adcopy/phrases/B001TEST01?min_rating=4&max_reviews=50")
    assert r.status_code == 200
    data = r.json()
    assert data["asin"] == "B001TEST01"
    assert "power_phrases" in data


# ── Campaigns ────────────────────────────────────────────────────────────────

def test_campaign_lifecycle():
    # Create
    r = client.post("/api/campaigns/", json={
        "name": "Q2 Packaging Fix",
        "asin": "B001TEST01",
        "start_date": "2025-04-01",
        "goal": "Improve packaging score",
    })
    assert r.status_code == 201
    campaign_id = r.json()["campaign_id"]

    # Get
    r = client.get(f"/api/campaigns/{campaign_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "Q2 Packaging Fix"

    # List
    r = client.get("/api/campaigns/")
    assert r.status_code == 200
    assert any(c["id"] == campaign_id for c in r.json()["campaigns"])

    # Delete
    r = client.delete(f"/api/campaigns/{campaign_id}")
    assert r.status_code == 204

    # Confirm gone
    r = client.get(f"/api/campaigns/{campaign_id}")
    assert r.status_code == 404


def test_campaign_not_found():
    r = client.get("/api/campaigns/doesnotexist")
    assert r.status_code == 404

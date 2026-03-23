"""
ReviewIQ — FastAPI Backend
Serves RIS scores, competitor intelligence, revenue predictions,
sentiment scoring, and ad copy generation to the React frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.routers import ris, sentiment, competitor, revenue, adcopy, campaigns
from app.core.config import settings
from app.core.startup import load_models

app = FastAPI(
    title="ReviewIQ API",
    description="AI-powered Amazon review intelligence platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load models into memory ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    await load_models()

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ris.router,        prefix="/api/ris",        tags=["RIS"])
app.include_router(sentiment.router,  prefix="/api/sentiment",  tags=["Sentiment"])
app.include_router(competitor.router, prefix="/api/competitor", tags=["Competitor"])
app.include_router(revenue.router,    prefix="/api/revenue",    tags=["Revenue"])
app.include_router(adcopy.router,     prefix="/api/adcopy",     tags=["Ad Copy"])
app.include_router(campaigns.router,  prefix="/api/campaigns",  tags=["Campaigns"])

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

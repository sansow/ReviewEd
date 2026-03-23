# ReviewIQ — ML Notebook Suite
## Amazon Reviews 2023 → Revenue Intelligence Platform

---

## Architecture Overview

```
McAuley-Lab/Amazon-Reviews-2023 (HuggingFace)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  01_dataset_ingestion     → data/processed/*.parquet    │
│  02_exploratory_analysis  → reports/eda_plots/          │
│  03_sentiment_training    → models/distilbert_sentiment/ │
│                           → models/roberta_sentiment/    │
│  04_ris_score_engine      → data/ris_outputs/           │
│  05_competitor_intel      → data/competitor_outputs/    │
│  06_revenue_predictor     → data/revenue_outputs/       │
└─────────────────────────────────────────────────────────┘
          │
          ▼
    FastAPI Backend  →  React/Next.js Frontend
```

---

## Notebook Guide

| # | Notebook | Purpose | Est. Runtime |
|---|----------|---------|--------------|
| 01 | `01_dataset_ingestion` | Load HuggingFace dataset, clean, split | 20-45 min |
| 02 | `02_exploratory_analysis` | EDA, distributions, wordclouds | 10-15 min |
| 03 | `03_sentiment_model_training` | Train DistilBERT + RoBERTa ensemble | 2-3 hrs (GPU) |
| 04 | `04_ris_score_engine` | Compute RIS for all ASINs | 30-60 min |
| 05 | `05_competitor_intelligence` | Aspect gap matrix + Claude brief | 15-20 min |
| 06 | `06_revenue_impact_predictor` | Prophet forecast + sales lift model | 20-30 min |

---

## Setup (RHOAI / OpenShift AI Workbench)

```bash
# 1. Create workbench with:
#    - Image: PyTorch (CUDA 11.8+)
#    - Container size: Large (8 CPU, 32GB RAM)
#    - GPU: 1x NVIDIA A10G (for notebook 03)

# 2. Clone repo
git clone https://github.com/Sansow/reviewiq.git
cd reviewiq

# 3. Install requirements
pip install -r requirements-notebooks.txt

# 4. Set env var for Claude API (notebook 05, 06)
export ANTHROPIC_API_KEY=your_key_here

# 5. Run notebooks IN ORDER
jupyter nbconvert --to notebook --execute notebooks/01_data/01_dataset_ingestion.ipynb
# ... etc
```

---

## Data Flow

```
Notebook 01 → data/processed/
  ├── reviews_clean_all.parquet    (all reviews)
  ├── reviews_train.parquet        (80% split)
  ├── reviews_val.parquet          (10% split)
  ├── reviews_test.parquet         (10% split)
  └── reviews_{CATEGORY}.parquet  (per category)

Notebook 03 → models/
  ├── distilbert_sentiment/        (HuggingFace model dir)
  ├── roberta_sentiment/           (HuggingFace model dir)
  └── ensemble_config.json

Notebook 04 → data/ris_outputs/
  └── ris_scores.parquet           (RIS for all ASINs)

Notebook 05 → data/competitor_outputs/
  ├── radar_{ASIN}.html
  └── brief_{ASIN}.json

Notebook 06 → data/revenue_outputs/
  └── revenue_waterfall_{ASIN}.html
```

---

## RIS Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Sentiment Score | 25% | Bayesian avg star rating → 0-100 |
| Authenticity Score | 25% | Fake review risk inverse |
| Aspect Coverage | 20% | Breadth of product qualities mentioned |
| Velocity Trend | 15% | Recent review momentum |
| Competitive Position | 15% | vs category average rating |

**Grades:** A+ (85-100) | A (75-84) | B (65-74) | C (55-64) | D (40-54) | F (<40)

---

## Revenue Impact Model

Sales lift estimation per aspect improvement:

```
lift = base_conversion_delta × gap_fraction × category_multiplier × ris_upside_factor
```

**Category multipliers:** Baby Products (1.25) → Beauty (1.20) → Health (1.15) → Electronics (0.95)

**Timeframes:** Shipping/Packaging fixes → 30 days | Usability → 60 days | Quality/Effectiveness → 90 days

---

## API Endpoints (FastAPI)

These notebooks feed the following backend endpoints:

```
GET  /api/ris/{asin}                    → RIS score + components
GET  /api/competitor-analysis/{asin}   → Gap matrix + action brief
GET  /api/revenue-impact/{asin}        → Sales lift projections
POST /api/sentiment                     → Real-time review scoring
POST /api/ad-copy                       → Generate ad copy from reviews
```

---

## Model Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| DistilBERT (sentiment) | F1 | >0.92 |
| RoBERTa (sentiment) | F1 | >0.94 |
| Ensemble | F1 | >0.95 |
| Fake review detector | Precision | >0.85 |
| Revenue predictor | MAE | <3% lift |

---

*ReviewIQ — AI Product Intelligence Platform*
*Built on OpenShift AI (RHOAI) · Powered by HuggingFace + Anthropic Claude*

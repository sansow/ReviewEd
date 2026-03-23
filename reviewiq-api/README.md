# ReviewIQ API

FastAPI backend serving the ReviewIQ intelligence platform.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/ris/{asin}` | RIS score + component breakdown |
| POST | `/api/ris/batch` | Batch RIS for up to 50 ASINs |
| GET | `/api/ris/category/{cat}` | Category leaderboard |
| POST | `/api/sentiment/score` | Score review texts |
| GET | `/api/sentiment/asin/{asin}` | Aggregate ASIN sentiment |
| GET | `/api/competitor/{asin}` | Full competitor analysis + Claude brief |
| GET | `/api/competitor/{asin}/gaps` | Aspect gaps only (fast) |
| GET | `/api/revenue/{asin}` | Revenue impact + Prophet forecast |
| GET | `/api/revenue/{asin}/quick` | Quick lift estimate (<100ms) |
| POST | `/api/adcopy/generate` | Generate ad copy from reviews |
| GET | `/api/adcopy/phrases/{asin}` | Extract top customer phrases |
| POST | `/api/campaigns/` | Create campaign |
| GET | `/api/campaigns/` | List campaigns |
| GET | `/api/campaigns/{id}/performance` | Sentiment trend + RIS delta |

Interactive docs at `http://localhost:8000/docs`

## Quick Start

```bash
# 1. Copy and fill env
cp .env.example .env

# 2. Install deps
pip install -r requirements.txt

# 3. Run notebooks 01–04 first to generate data + models

# 4. Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker-compose up --build
```

## OpenShift

```bash
# Create namespace
oc new-project reviewiq

# Create secret
oc create secret generic reviewiq-secrets \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY

# Deploy
oc apply -f openshift-deploy.yaml
```

## Prerequisites

Run notebooks in order before starting the API:
1. `01_dataset_ingestion` → populates `data/processed/`
2. `03_sentiment_model_training` → populates `models/`
3. `04_ris_score_engine` → populates `data/ris_outputs/`

The API gracefully degrades if models are missing (returns 503 for those endpoints).

## Tests

```bash
pytest tests/ -v
```

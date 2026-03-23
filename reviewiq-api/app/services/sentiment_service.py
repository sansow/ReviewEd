"""ReviewIQ — Sentiment Service (DistilBERT + RoBERTa Ensemble)"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID2LABEL = {0: "negative", 1: "positive"}


class SentimentService:
    def __init__(self):
        db_dir = str(settings.MODELS_DIR / "distilbert_sentiment")
        rb_dir = str(settings.MODELS_DIR / "roberta_sentiment")
        weights = settings.ENSEMBLE_WEIGHTS

        logger.info(f"Loading sentiment ensemble on {DEVICE}...")

        self.weights = weights
        self.db_tok = AutoTokenizer.from_pretrained(db_dir)
        self.db_model = AutoModelForSequenceClassification.from_pretrained(db_dir).to(DEVICE).eval()

        self.rb_tok = AutoTokenizer.from_pretrained(rb_dir)
        self.rb_model = AutoModelForSequenceClassification.from_pretrained(rb_dir).to(DEVICE).eval()

        logger.info("Sentiment ensemble loaded ✅")

    def predict(self, texts: list, batch_size: int = None) -> dict:
        batch_size = batch_size or settings.SENTIMENT_BATCH_SIZE
        max_length = settings.SENTIMENT_MAX_LENGTH
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                db_enc = self.db_tok(batch, truncation=True, padding=True,
                                     max_length=max_length, return_tensors="pt").to(DEVICE)
                db_probs = torch.softmax(self.db_model(**db_enc).logits, dim=-1).cpu().numpy()

                rb_enc = self.rb_tok(batch, truncation=True, padding=True,
                                     max_length=max_length, return_tensors="pt").to(DEVICE)
                rb_probs = torch.softmax(self.rb_model(**rb_enc).logits, dim=-1).cpu().numpy()

            ensemble_probs = self.weights[0] * db_probs + self.weights[1] * rb_probs
            all_probs.append(ensemble_probs)

        all_probs = np.vstack(all_probs)
        predictions = np.argmax(all_probs, axis=-1)
        confidence = np.max(all_probs, axis=-1)

        return {
            "predictions": predictions,
            "confidence": confidence,
            "positive_prob": all_probs[:, 1],
            "negative_prob": all_probs[:, 0],
            "labels": [ID2LABEL[p] for p in predictions],
        }

"""Model loading and probability prediction."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from app.config import settings
from app.ml.feature_engineering import features_to_row
from app.utils.logger import logger

try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGB_AVAILABLE = False
    logger.warning("xgboost not available: {}", exc)


@lru_cache(maxsize=1)
def _load_model(path: str, mtime: float) -> Any | None:
    """Load and cache the XGBoost model.

    ``mtime`` participates in the cache key so the model auto-reloads after a
    fresh training run overwrites the file.
    """
    if not XGB_AVAILABLE:
        return None
    if not os.path.exists(path):
        return None
    try:
        model = XGBClassifier()
        model.load_model(path)
        logger.success("Loaded XGBoost model from {}", path)
        return model
    except Exception as exc:
        logger.error("Failed to load model {}: {}", path, exc)
        return None


def get_model() -> Any | None:
    """Return the active model instance (or None if not trained yet)."""
    path = settings.model_path
    if not os.path.exists(path):
        return None
    return _load_model(path, os.path.getmtime(path))


def predict_signal(features: dict[str, float]) -> dict[str, float]:
    """Return BUY/SELL probabilities from the model.

    The model predicts P(next candle up). buy = that probability, sell = 1-buy.
    If no model exists, returns a neutral 50/50 so the bot degrades gracefully.
    """
    model = get_model()
    if model is None:
        return {"buy": 0.5, "sell": 0.5, "model": False}

    try:
        X = features_to_row(features)
        proba = model.predict_proba(X)[0]
        # Class 1 == "up" probability.
        prob_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return {"buy": round(prob_up, 4), "sell": round(1.0 - prob_up, 4), "model": True}
    except Exception as exc:
        logger.error("predict_signal failed: {} - returning neutral.", exc)
        return {"buy": 0.5, "sell": 0.5, "model": False}

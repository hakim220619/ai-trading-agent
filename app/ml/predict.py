"""Ensemble model loading and probability prediction."""
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

try:
    import joblib
    from lightgbm import LGBMClassifier  # type: ignore  # noqa: F401

    LGB_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    joblib = None  # type: ignore
    LGBMClassifier = None  # type: ignore
    LGB_AVAILABLE = False
    logger.warning("lightgbm not available: {}", exc)


@lru_cache(maxsize=4)
def _load_xgboost(path: str, mtime: float) -> Any | None:
    if not XGB_AVAILABLE or not os.path.exists(path):
        return None
    try:
        model = XGBClassifier()
        model.load_model(path)
        logger.success("Loaded XGBoost model from {}", path)
        return model
    except Exception as exc:
        logger.error("Failed to load XGBoost model {}: {}", path, exc)
        return None


@lru_cache(maxsize=4)
def _load_lightgbm(path: str, mtime: float) -> Any | None:
    if not LGB_AVAILABLE or not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        logger.success("Loaded LightGBM model from {}", path)
        return model
    except Exception as exc:
        logger.error("Failed to load LightGBM model {}: {}", path, exc)
        return None


def get_models() -> dict[str, Any]:
    """Return every trained model currently available."""
    models: dict[str, Any] = {}
    if os.path.exists(settings.model_path):
        model = _load_xgboost(settings.model_path, os.path.getmtime(settings.model_path))
        if model is not None:
            models["xgboost"] = model
    if os.path.exists(settings.lightgbm_model_path):
        model = _load_lightgbm(
            settings.lightgbm_model_path,
            os.path.getmtime(settings.lightgbm_model_path),
        )
        if model is not None:
            models["lightgbm"] = model
    return models


def get_model() -> Any | None:
    """Backward-compatible active-model lookup used by status endpoints."""
    models = get_models()
    return models.get("xgboost") or models.get("lightgbm")


def _up_probability(model: Any, features: Any) -> float:
    proba = model.predict_proba(features)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def predict_signal(features: dict[str, float]) -> dict[str, Any]:
    """Return weighted BUY/SELL probabilities from XGBoost and LightGBM."""
    models = get_models()
    if not models:
        return {"buy": 0.5, "sell": 0.5, "model": False, "models": []}

    try:
        row = features_to_row(features)
        probabilities = {name: _up_probability(model, row) for name, model in models.items()}
        xgb_weight = float(settings.xgboost_ensemble_weight)
        configured = {"xgboost": xgb_weight, "lightgbm": 1.0 - xgb_weight}
        weight_total = sum(configured[name] for name in probabilities)
        if weight_total <= 0:
            weight_total = float(len(probabilities))
            configured = {name: 1.0 for name in probabilities}
        prob_up = sum(probabilities[name] * configured[name] for name in probabilities) / weight_total
        return {
            "buy": round(prob_up, 4),
            "sell": round(1.0 - prob_up, 4),
            "model": True,
            "models": list(probabilities),
            "components": {name: round(value, 4) for name, value in probabilities.items()},
        }
    except Exception as exc:
        logger.error("ensemble predict_signal failed: {} - returning neutral.", exc)
        return {"buy": 0.5, "sell": 0.5, "model": False, "models": []}

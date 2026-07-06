"""Train the XGBoost + LightGBM ensemble from CSV candle data.

Usage:
    python -m app.ml.train_xgboost --csv data/XAUUSD_M5.csv
    python -m app.ml.train_xgboost --csv data/XAUUSD_M5.csv --horizon 1 --atr-mult 0.5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.config import settings
from app.ml.feature_engineering import FEATURE_COLUMNS, make_dataset
from app.mt5.market_data import load_candles_csv
from app.utils.logger import logger

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGB_IMPORT_ERROR = exc

try:
    import joblib
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover
    joblib = None  # type: ignore
    LGBMClassifier = None  # type: ignore
    LGB_IMPORT_ERROR = exc


def train(
    csv_path: str,
    model_path: str | None = None,
    lightgbm_model_path: str | None = None,
    horizon: int = 1,
    atr_mult: float = 0.5,
    test_size: float = 0.2,
) -> dict[str, float]:
    """Train and persist both ensemble models; return evaluation metrics."""
    if XGBClassifier is None:
        raise RuntimeError(f"xgboost is required for ensemble training: {XGB_IMPORT_ERROR}")
    if LGBMClassifier is None or joblib is None:
        raise RuntimeError(f"lightgbm and joblib are required for ensemble training: {LGB_IMPORT_ERROR}")
    model_path = model_path or settings.model_path
    lightgbm_model_path = lightgbm_model_path or settings.lightgbm_model_path

    df = load_candles_csv(csv_path)
    if df.empty:
        raise ValueError(f"No data loaded from {csv_path}")

    X, y = make_dataset(df, horizon=horizon, atr_mult=atr_mult)
    if len(X) < 100:
        raise ValueError(f"Not enough samples after cleaning: {len(X)}")

    logger.info("Dataset: {} rows, positive rate={:.3f}", len(X), y.mean())

    # Time series: do NOT shuffle so test set is the most recent data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    lightgbm_model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    xgboost_model.fit(X_train, y_train)
    lightgbm_model.fit(X_train, y_train)

    xgb_probability = xgboost_model.predict_proba(X_test)[:, 1]
    lgb_probability = lightgbm_model.predict_proba(X_test)[:, 1]
    xgb_weight = float(settings.xgboost_ensemble_weight)
    ensemble_probability = xgb_probability * xgb_weight + lgb_probability * (1.0 - xgb_weight)
    y_pred = (ensemble_probability >= 0.5).astype(int)
    xgb_pred = (xgb_probability >= 0.5).astype(int)
    lgb_pred = (lgb_probability >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "xgboost_accuracy": float(accuracy_score(y_test, xgb_pred)),
        "lightgbm_accuracy": float(accuracy_score(y_test, lgb_pred)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    logger.success("Accuracy : {:.4f}", metrics["accuracy"])
    logger.success("Precision: {:.4f}", metrics["precision"])
    logger.success("Recall   : {:.4f}", metrics["recall"])
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    xgboost_model.save_model(model_path)
    os.makedirs(os.path.dirname(lightgbm_model_path), exist_ok=True)
    joblib.dump(lightgbm_model, lightgbm_model_path)
    logger.success("XGBoost model saved to {}", model_path)
    logger.success("LightGBM model saved to {}", lightgbm_model_path)

    # Log feature importances for insight.
    importances = sorted(
        zip(FEATURE_COLUMNS, np.mean([xgboost_model.feature_importances_, lightgbm_model.feature_importances_], axis=0)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    logger.info("Top features: {}", importances[:8])

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost + LightGBM ensemble")
    p.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    p.add_argument("--model", default=None, help="Output model path")
    p.add_argument("--lightgbm-model", default=None, help="LightGBM output model path")
    p.add_argument("--horizon", type=int, default=1, help="Future candle horizon")
    p.add_argument("--atr-mult", type=float, default=0.5, help="ATR multiple for target")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        csv_path=args.csv,
        model_path=args.model,
        lightgbm_model_path=args.lightgbm_model,
        horizon=args.horizon,
        atr_mult=args.atr_mult,
        test_size=args.test_size,
    )

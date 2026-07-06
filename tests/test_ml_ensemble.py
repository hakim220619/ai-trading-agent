from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.ml.predict import predict_signal


class _ProbabilityModel:
    def __init__(self, upward: float) -> None:
        self.upward = upward

    def predict_proba(self, _features):
        return [[1.0 - self.upward, self.upward]]


class EnsemblePredictionTests(unittest.TestCase):
    def test_combines_xgboost_and_lightgbm_probabilities(self) -> None:
        models = {"xgboost": _ProbabilityModel(0.8), "lightgbm": _ProbabilityModel(0.6)}
        with (
            patch("app.ml.predict.get_models", return_value=models),
            patch("app.ml.predict.settings", SimpleNamespace(xgboost_ensemble_weight=0.5)),
        ):
            result = predict_signal({})

        self.assertEqual(result["buy"], 0.7)
        self.assertEqual(result["sell"], 0.3)
        self.assertEqual(result["models"], ["xgboost", "lightgbm"])

    def test_uses_available_model_when_other_model_is_missing(self) -> None:
        with (
            patch("app.ml.predict.get_models", return_value={"lightgbm": _ProbabilityModel(0.65)}),
            patch("app.ml.predict.settings", SimpleNamespace(xgboost_ensemble_weight=0.5)),
        ):
            result = predict_signal({})

        self.assertEqual(result["buy"], 0.65)
        self.assertTrue(result["model"])

    def test_returns_neutral_without_trained_models(self) -> None:
        with patch("app.ml.predict.get_models", return_value={}):
            result = predict_signal({})

        self.assertEqual(result["buy"], 0.5)
        self.assertEqual(result["sell"], 0.5)
        self.assertFalse(result["model"])


if __name__ == "__main__":
    unittest.main()

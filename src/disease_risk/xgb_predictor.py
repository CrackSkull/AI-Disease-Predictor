"""
src/disease_risk/xgb_predictor.py
---------------------------------
Combined XGBoost + KDE disease risk predictor.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

try:
    import shap
except Exception:  # pragma: no cover - graceful fallback when SHAP is unavailable
    shap = None

from .engine import DISEASES, MODEL_FEATURE_NAMES, _haversine_km, load_cases_from_directory


def _uniform_risk() -> Dict[str, float]:
    """Return an equal distribution across all supported diseases."""
    share = round(1.0 / len(DISEASES), 6)
    return {disease: share for disease in DISEASES}


class XGBRiskPredictor:
    """
    Combined XGBoost + KDE disease risk predictor.

    Usage:
        predictor = XGBRiskPredictor.load(
            model_dir=Path("ml_output"),
            dataset_dir=Path("dataset"),
        )
        result = predictor.predict(latitude=13.08, longitude=80.27, age=30, gender="male")
    """

    KDE_BANDWIDTH_KM = 15.0
    KDE_MAX_RADIUS_KM = 50.0

    TRAINING_REGION = {
        "name": "Tamil Nadu, India",
        "lat_min": 8.0,
        "lat_max": 13.6,
        "lon_min": 76.2,
        "lon_max": 80.4,
    }

    def __init__(self):
        self._model = None
        self._classes: List[str] = []
        self._meta: Dict[str, object] = {}
        self._cases: List[Dict[str, object]] = []
        self._shap_explainer = None

    @classmethod
    def load(cls, model_dir: Path, dataset_dir: Path) -> "XGBRiskPredictor":
        """Load the trained XGBoost model, metadata, and raw case data."""
        predictor = cls()

        model_path = model_dir / "disease_xgb_model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at: {model_path}\n"
                "Please run train_ml_model.py first to generate the model."
            )

        predictor._model = xgb.XGBClassifier()
        predictor._model.load_model(str(model_path))

        meta_path = model_dir / "model_meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                predictor._meta = json.load(handle)
            predictor._classes = list(predictor._meta.get("classes", []))

        predictor._init_shap_explainer()

        if dataset_dir.exists():
            try:
                cases, _ = load_cases_from_directory(dataset_dir)
                predictor._cases = cases
                print(f"  Loaded {len(cases)} raw cases for KDE + spatial context.")
            except Exception as exc:
                print(f"  Warning: Could not load raw cases ({exc}). KDE will be skipped.")
                predictor._cases = []

        return predictor

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP explainability if the dependency is available."""
        self._shap_explainer = None
        if self._model is None or shap is None:
            return
        try:
            self._shap_explainer = shap.TreeExplainer(self._model)
        except Exception as exc:
            print(f"  Warning: Could not initialize SHAP explainer ({exc}).")
            self._shap_explainer = None

    def _build_features(
        self,
        latitude: float,
        longitude: float,
        age: Optional[float],
        gender: Optional[str],
    ) -> np.ndarray:
        """Build the 6-feature vector used during model training."""
        now = datetime.now()
        month = now.month
        season = 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8, 9] else 3

        age_val = float(age) if age is not None else 30.0
        gender_text = (gender or "").strip().lower()
        gender_val = 1.0 if gender_text in ("male", "m") else 0.0 if gender_text in ("female", "f") else 0.5

        return np.array([[latitude, longitude, age_val, gender_val, month, season]])

    def _normalize_shap_values(
        self,
        shap_values: object,
        class_index: int,
        feature_count: int,
    ) -> np.ndarray:
        """Normalize SHAP output shapes into a single vector for one class."""
        if isinstance(shap_values, list):
            if not shap_values:
                return np.zeros(feature_count)
            class_values = np.asarray(shap_values[min(class_index, len(shap_values) - 1)])
            return class_values[0] if class_values.ndim > 1 else class_values

        values = np.asarray(shap_values)
        if values.ndim == 1:
            return values
        if values.ndim == 2:
            return values[0]
        if values.ndim == 3:
            if values.shape[1] == feature_count:
                return values[0, :, min(class_index, values.shape[2] - 1)]
            return values[min(class_index, values.shape[0] - 1), 0, :]
        return np.zeros(feature_count)

    def _top_factors(
        self,
        features: np.ndarray,
        target_disease: Optional[str],
    ) -> List[Dict[str, float]]:
        """Compute the top SHAP contributors for the dominant predicted disease."""
        if self._shap_explainer is None or target_disease is None:
            return []

        class_lookup = {name.lower(): idx for idx, name in enumerate(self._classes)}
        class_index = class_lookup.get(target_disease.lower())
        if class_index is None:
            return []

        try:
            shap_values = self._shap_explainer.shap_values(features)
            contributions = self._normalize_shap_values(
                shap_values,
                class_index=class_index,
                feature_count=len(MODEL_FEATURE_NAMES),
            )
        except Exception as exc:
            print(f"  Warning: Could not compute SHAP values ({exc}).")
            return []

        factors = [
            {
                "feature": feature_name,
                "impact": round(float(abs(impact)), 6),
            }
            for feature_name, impact in zip(MODEL_FEATURE_NAMES, contributions)
        ]
        factors.sort(key=lambda item: item["impact"], reverse=True)
        return factors[:3]

    def _base_metadata(self) -> Dict[str, object]:
        """Return shared metadata used in all prediction responses."""
        return {
            "model_type": "XGBoost + KDE",
            "model_accuracy": self._meta.get("accuracy", "N/A"),
            "f1_score": self._meta.get("f1_score", "N/A"),
            "best_iteration": self._meta.get("best_iteration", "N/A"),
            "trained_at": self._meta.get("trained_at", "N/A"),
            "kde_bandwidth_km": self.KDE_BANDWIDTH_KM,
            "shap_enabled": self._shap_explainer is not None,
        }

    def _response_payload(
        self,
        *,
        balanced_risk: Dict[str, float],
        raw_case_distribution: Dict[str, float],
        nearby_cases_25km: Dict[str, int],
        location_density_level: str,
        disclaimer: str,
        top_factors: Optional[List[Dict[str, float]]] = None,
        metadata: Optional[Dict[str, object]] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Build a consistent response shape for all prediction outcomes."""
        payload: Dict[str, object] = {
            "balanced_risk": balanced_risk,
            "raw_case_distribution": raw_case_distribution,
            "nearby_cases_25km": nearby_cases_25km,
            "location_density_level": location_density_level,
            "top_factors": top_factors or [],
            "metadata": metadata or self._base_metadata(),
            "disclaimer": disclaimer,
        }
        if extra:
            payload.update(extra)
        return payload

    def _kde_scores(self, latitude: float, longitude: float) -> Dict[str, float]:
        """Calculate Gaussian KDE scores for each disease at the given location."""
        scores = {disease: 0.0 for disease in DISEASES}
        if not self._cases:
            return scores

        deg_cutoff = self.KDE_MAX_RADIUS_KM / 111.0
        for case in self._cases:
            if abs(case["latitude"] - latitude) > deg_cutoff:
                continue
            if abs(case["longitude"] - longitude) > deg_cutoff:
                continue

            disease = case.get("disease", "")
            if disease not in DISEASES:
                continue

            dist_km = _haversine_km(latitude, longitude, case["latitude"], case["longitude"])
            if dist_km > self.KDE_MAX_RADIUS_KM:
                continue

            weight = math.exp(-0.5 * (dist_km / self.KDE_BANDWIDTH_KM) ** 2)
            scores[disease] += weight

        return scores

    def _count_nearby_cases(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 25.0,
    ) -> Dict[str, int]:
        """Count historical cases inside the given radius for frontend display."""
        counts = {disease: 0 for disease in DISEASES}
        deg_cutoff = radius_km / 111.0

        for case in self._cases:
            if abs(case["latitude"] - latitude) > deg_cutoff:
                continue
            if abs(case["longitude"] - longitude) > deg_cutoff:
                continue
            dist_km = _haversine_km(latitude, longitude, case["latitude"], case["longitude"])
            if dist_km <= radius_km:
                disease = case.get("disease", "")
                if disease in counts:
                    counts[disease] += 1
        return counts

    def _density_level(self, nearby_total: int) -> str:
        """Map the nearby case count to a human-readable density label."""
        if nearby_total >= 500:
            return "very_high"
        if nearby_total >= 100:
            return "high"
        if nearby_total >= 20:
            return "medium"
        return "low"

    def _is_in_training_region(self, latitude: float, longitude: float) -> bool:
        """Check whether a location falls inside the model's training region."""
        region = self.TRAINING_REGION
        return (
            region["lat_min"] <= latitude <= region["lat_max"]
            and region["lon_min"] <= longitude <= region["lon_max"]
        )

    def predict(
        self,
        latitude: float,
        longitude: float,
        age: Optional[float] = None,
        gender: Optional[str] = None,
    ) -> Dict[str, object]:
        """Predict disease risk using XGBoost probabilities blended with KDE."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call XGBRiskPredictor.load() first.")

        if not self._is_in_training_region(latitude, longitude):
            uniform_risk = _uniform_risk()
            return self._response_payload(
                balanced_risk=uniform_risk,
                raw_case_distribution=uniform_risk,
                nearby_cases_25km={disease: 0 for disease in DISEASES},
                location_density_level="unknown",
                metadata={
                    **self._base_metadata(),
                    "kde_used": False,
                    "in_training_region": False,
                },
                disclaimer=(
                    "This model is trained on Tamil Nadu, India data only. "
                    "Your location is outside this region — predictions are not valid here. "
                    "Not a medical diagnosis."
                ),
                extra={
                    "out_of_region": True,
                    "out_of_region_message": (
                        f"Your location ({latitude:.4f}, {longitude:.4f}) is outside the "
                        f"training data region ({self.TRAINING_REGION['name']}). "
                        "No historical case data is available for this area. "
                        "Risk percentages shown are not reliable for your location."
                    ),
                },
            )

        features = self._build_features(latitude, longitude, age, gender)
        probabilities = self._model.predict_proba(features)[0]

        xgb_scores = {}
        for idx, class_name in enumerate(self._classes):
            normalized_name = class_name.lower()
            if normalized_name in DISEASES:
                xgb_scores[normalized_name] = float(probabilities[idx])

        kde_scores = self._kde_scores(latitude, longitude)
        kde_total = sum(kde_scores.values())

        if kde_total > 0:
            combined_scores = {
                disease: xgb_scores.get(disease, 0.0) * kde_scores.get(disease, 0.0)
                for disease in DISEASES
            }
        else:
            combined_scores = {disease: xgb_scores.get(disease, 0.0) for disease in DISEASES}

        total_score = sum(combined_scores.values())
        if total_score > 0:
            balanced_risk = {
                disease: round(combined_scores[disease] / total_score, 6)
                for disease in DISEASES
            }
        else:
            balanced_risk = _uniform_risk()

        top_disease = max(balanced_risk, key=balanced_risk.get) if balanced_risk else None
        top_factors = self._top_factors(features, top_disease)

        nearby_counts = self._count_nearby_cases(latitude, longitude, radius_km=25.0)
        nearby_total = sum(nearby_counts.values())

        return self._response_payload(
            balanced_risk=balanced_risk,
            raw_case_distribution=balanced_risk,
            nearby_cases_25km=nearby_counts,
            location_density_level=self._density_level(nearby_total),
            top_factors=top_factors,
            metadata={
                **self._base_metadata(),
                "kde_used": kde_total > 0,
                "explained_disease": top_disease,
            },
            disclaimer=(
                "Risk is estimated using XGBoost (92.81% accuracy) combined with "
                "KDE on historical case data from Tamil Nadu, India (2021). "
                "This reflects relative disease activity in your area — "
                "not absolute probability of infection. Not a medical diagnosis."
            ),
        )

    @property
    def training_summary(self) -> Dict[str, object]:
        """Used by api_server.py GET /health endpoint."""
        return {
            "total_cases": len(self._cases),
            "model_type": "XGBoost + KDE",
            "model_accuracy": self._meta.get("accuracy", "N/A"),
        }

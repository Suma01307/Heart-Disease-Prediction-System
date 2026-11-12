"""Loads trained models and runs predictions. Raises ModelNotTrained if a model file is missing."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from .imagefeat import extract_features

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

MODEL_FILES = {
    "clinical": "heart_clinical_model.joblib",
    "xray": "xray_model.joblib",
    "ecg": "ecg_model.joblib",
}


class ModelNotTrained(Exception):
    pass


@lru_cache(maxsize=None)
def _load(kind: str):
    path = settings.ML_MODELS_DIR / MODEL_FILES[kind]
    if not path.exists():
        raise ModelNotTrained(kind)
    return joblib.load(path)


def reload_models():
    _load.cache_clear()


def predict_clinical(data: dict):
    """Returns (is_positive, confidence_percent, probability_of_disease_percent)."""
    model = _load("clinical")
    row = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
    p_disease = float(model.predict_proba(row)[0][1])
    positive = p_disease >= 0.5
    confidence = (p_disease if positive else 1 - p_disease) * 100
    return positive, confidence, p_disease * 100


def predict_image(kind: str, file_obj):
    """kind = 'xray' or 'ecg'. Returns (is_abnormal, confidence_percent)."""
    model = _load(kind)
    feats = extract_features(file_obj).reshape(1, -1)
    p_abnormal = float(model.predict_proba(feats)[0][list(model.classes_).index(1)])
    positive = p_abnormal >= 0.5
    confidence = (p_abnormal if positive else 1 - p_abnormal) * 100
    return positive, confidence

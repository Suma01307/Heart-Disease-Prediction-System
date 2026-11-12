"""
Train the clinical heart-disease model.

Usage (from the project root, venv active):
    python -m ml_training.train_tabular                 # uses dataset/heart.csv
    python -m ml_training.train_tabular --csv path.csv  # custom CSV
    python -m ml_training.train_tabular --synthetic     # DEMO ONLY: generates fake data

Expected CSV columns (UCI Cleveland / Kaggle "heart.csv"):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET_NAMES = ["target", "num", "HeartDisease", "output", "condition"]


def synthetic_data(n=1500, seed=42) -> pd.DataFrame:
    """Fake data with plausible risk relationships. For demonstrating the pipeline ONLY."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.normal(131, 17, n).clip(90, 200).astype(int),
        "chol": rng.normal(246, 51, n).clip(120, 560).astype(int),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(149, 23, n).clip(70, 205).astype(int),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.gamma(1.2, 0.9, n).clip(0, 6).round(1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.choice([0, 1, 2, 3], n, p=[.58, .22, .13, .07]),
        "thal": rng.choice([1, 2, 3], n, p=[.06, .55, .39]),
    })
    z = (0.04 * (df.age - 54) + 0.6 * df.sex + 0.5 * (df.cp == 3) - 0.03 * (df.thalach - 149)
         + 0.9 * df.exang + 0.6 * df.oldpeak + 0.8 * df.ca + 0.7 * (df.thal == 3)
         + 0.005 * (df.chol - 246) + 0.01 * (df.trestbps - 131) - 2.6)
    p = 1 / (1 + np.exp(-z))
    df["target"] = (rng.random(n) < p).astype(int)
    return df


def load_csv(path: Path, invert: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    target = next((c for c in TARGET_NAMES if c in df.columns), None)
    if target is None:
        raise SystemExit(f"No target column found. Expected one of {TARGET_NAMES}; got {list(df.columns)}")
    df = df.rename(columns={target: "target"})
    df["target"] = (df["target"] > 0).astype(int)   # UCI 'num' 1-4 -> 1
    if invert:
        df["target"] = 1 - df["target"]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    df = df[FEATURES + ["target"]].replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    return df.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "dataset" / "heart.csv"))
    ap.add_argument("--synthetic", action="store_true", help="DEMO ONLY: train on generated data")
    ap.add_argument("--invert-target", action="store_true",
                    help="Flip labels (some Kaggle copies of heart.csv have 1 = healthy)")
    args = ap.parse_args()

    if args.synthetic:
        print("!! Using SYNTHETIC data. The model is for demo purposes only and has no clinical meaning.")
        df = synthetic_data()
    else:
        path = Path(args.csv)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path}\nDownload heart.csv (see README) or run with --synthetic for a demo.")
        df = load_csv(path, args.invert_target)

    print(f"Rows: {len(df)} | Positive rate: {df.target.mean():.2%}")
    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    best_name, best_model, best_cv = None, None, -1
    for name, model in candidates.items():
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        print(f"{name:20s} 5-fold CV ROC-AUC: {cv:.3f}")
        if cv > best_cv:
            best_name, best_model, best_cv = name, model, cv

    best_model.fit(X_tr, y_tr)
    pred = best_model.predict(X_te)
    proba = best_model.predict_proba(X_te)[:, 1]
    print(f"\nBest model: {best_name}")
    print(f"Hold-out accuracy: {accuracy_score(y_te, pred):.3f} | ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, pred, target_names=["No disease", "Disease"]))

    best_model.fit(X, y)  # refit on all data for deployment
    out = ROOT / "ml_models" / "heart_clinical_model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

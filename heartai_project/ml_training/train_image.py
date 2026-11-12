"""
Train the chest X-ray or ECG image classifier.

Put images in:
    dataset/xray/normal/    dataset/xray/abnormal/
    dataset/ecg/normal/     dataset/ecg/abnormal/

Usage (from the project root):
    python -m ml_training.train_image --kind xray
    python -m ml_training.train_image --kind ecg
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from predictor.imagefeat import extract_features

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_folder(folder: Path, label: int):
    X, y = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in EXTS:
            try:
                X.append(extract_features(p))
                y.append(label)
            except Exception as e:  # unreadable image
                print(f"skip {p.name}: {e}")
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["xray", "ecg"], required=True)
    args = ap.parse_args()

    base = ROOT / "dataset" / args.kind
    Xn, yn = load_folder(base / "normal", 0)
    Xa, ya = load_folder(base / "abnormal", 1)
    print(f"{args.kind}: normal={len(yn)} abnormal={len(ya)}")
    if min(len(yn), len(ya)) < 20:
        raise SystemExit("Need at least ~20 images in EACH folder (hundreds+ recommended). See README for datasets.")

    X = np.array(Xn + Xa)
    y = np.array(yn + ya)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te), target_names=["normal", "abnormal"]))

    model.fit(X, y)
    out = ROOT / "ml_models" / f"{args.kind}_model.joblib"
    joblib.dump(model, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

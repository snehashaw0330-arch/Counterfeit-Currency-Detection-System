"""Compute REAL ROC curves for every ML technique on the held-out test split.

Reproduces the exact evaluation path of scripts/benchmark_models.py, but instead
of hard 0/1 predictions it collects the probability of *genuine* per test image
so a true ROC curve (and its AUC) can be drawn. Writes the curve data to
docs/_report_figs/roc_data.json, which scripts/build_report_doc.py reads to embed
Figure 10.2 in the project report.

Positive class = genuine (label 1), matching backend/classical.py and main.py.

Run:
    venv\\Scripts\\python.exe scripts\\compute_roc.py
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402

from backend.features import extract_feature_vector  # noqa: E402

INDEX = os.path.join(ROOT, "dataset", "index.json")
CLF_DIR = os.path.join(ROOT, "models", "classical")
CNN_PATH = os.path.join(ROOT, "models", "mobilenet_counterfeit_detector.keras")
OUT = os.path.join(ROOT, "docs", "_report_figs", "roc_data.json")

# Drawn order / labels (best-first is decided later by AUC).
CLASSICAL = ["svm_rbf", "random_forest", "logistic_regression", "knn"]
PRETTY = {
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "mobilenet_cnn": "MobileNetV2 (CNN)",
}


def _load_test():
    with open(INDEX, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    test = [s for s in index["samples"] if s["split"] == "test"]
    imgs, y = [], []
    for s in test:
        img = cv2.imread(os.path.join(ROOT, s["path"].replace("/", os.sep)))
        if img is None:
            continue
        imgs.append(img)
        y.append(s["y"])
    return imgs, np.array(y)


def _prob_genuine(model, X):
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        gi = classes.index(1) if 1 in classes else len(classes) - 1
        return model.predict_proba(X)[:, gi]
    # decision_function fallback (SVM without probability)
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return (d - d.min()) / (d.max() - d.min() + 1e-9)
    return model.predict(X).astype(float)


def _classical_scores(imgs):
    scaler = joblib.load(os.path.join(CLF_DIR, "scaler.joblib"))
    X = scaler.transform(
        np.array([extract_feature_vector(im) for im in imgs], dtype=np.float32)
    )
    scores = {}
    for name in CLASSICAL:
        model = joblib.load(os.path.join(CLF_DIR, f"{name}.joblib"))
        scores[name] = _prob_genuine(model, X)
    return scores


def _cnn_scores(imgs):
    import importlib
    try:
        tf = importlib.import_module("tensorflow")
        load_model = tf.keras.models.load_model
    except ImportError:
        from keras.models import load_model
    model = load_model(CNN_PATH)
    out = []
    for im in imgs:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(cv2.resize(rgb, (224, 224)) / 255.0, axis=0)
        out.append(float(model.predict(x, verbose=0)[0][0]))  # P(genuine)
    return np.array(out)


def main():
    imgs, y = _load_test()
    print(f"Computing ROC on {len(imgs)} held-out test images "
          f"({int((y == 1).sum())} genuine / {int((y == 0).sum())} fake)...")

    scores = _classical_scores(imgs)
    if os.path.exists(CNN_PATH):
        try:
            scores["mobilenet_cnn"] = _cnn_scores(imgs)
        except Exception as exc:
            print(f"  CNN ROC skipped ({exc})")

    out = {"meta": {"n_test": len(imgs),
                    "n_genuine": int((y == 1).sum()),
                    "n_fake": int((y == 0).sum()),
                    "pos_label": "genuine (1)"},
           "curves": {}}
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s, pos_label=1)
        a = float(auc(fpr, tpr))
        out["curves"][name] = {
            "label": PRETTY[name],
            "fpr": [round(float(v), 5) for v in fpr],
            "tpr": [round(float(v), 5) for v in tpr],
            "auc": round(a, 4),
        }
        print(f"  {PRETTY[name]:<22} AUC = {a:.3f}  ({len(fpr)} points)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

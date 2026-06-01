"""
Phase D.3 — train the classical ML techniques.

Reads dataset/index.json, builds a feature matrix with
backend.features (augmenting TRAIN images via backend.augment to
expand the small base corpus and balance the classes), then trains
and cross-validates four classical techniques:

    Logistic Regression · SVM (RBF) · Random Forest · KNN

For each it records 5-fold CV macro-F1 on the training split and
full metrics (accuracy / per-class precision-recall / confusion
matrix) on the held-out TEST split (test images are NOT augmented).

Artifacts (all git-ignored, regenerable):
    models/classical/<name>.joblib
    models/classical/scaler.joblib
    models/classical/metrics.json

Run:
    venv\\Scripts\\python.exe scripts\\train_classical.py

Deterministic: fixed seeds throughout, so re-running reproduces the
same models and numbers.
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score,
)

from backend.features import extract_feature_vector, FEATURE_DIM, FEATURE_NAMES  # noqa: E402
from backend.augment import augment_variants  # noqa: E402

SEED = 42
TARGET_PER_CLASS = 250          # approx augmented rows per class in TRAIN
INDEX = os.path.join(ROOT, "dataset", "index.json")
OUT_DIR = os.path.join(ROOT, "models", "classical")


def _load_index():
    with open(INDEX, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _imread(rel_path):
    img = cv2.imread(os.path.join(ROOT, rel_path.replace("/", os.sep)))
    return img


def _aug_per_image(n_in_class):
    """Augmented variants per base image so each class reaches
    ~TARGET_PER_CLASS rows (incl. the original)."""
    if n_in_class <= 0:
        return 0
    return max(0, round(TARGET_PER_CLASS / n_in_class) - 1)


def _build_matrix(index):
    samples = index["samples"]
    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]

    n_gen = sum(1 for s in train if s["y"] == 1)
    n_fake = sum(1 for s in train if s["y"] == 0)
    aug_gen = _aug_per_image(n_gen)
    aug_fake = _aug_per_image(n_fake)

    print(f"TRAIN base: {n_gen} genuine / {n_fake} fake")
    print(f"Augmenting +{aug_gen}/genuine, +{aug_fake}/fake "
          f"(target ~{TARGET_PER_CLASS} rows per class)")

    # groups_tr ties every augmented row back to its base image so
    # group-aware CV never splits a note's variants across folds
    # (otherwise CV macro-F1 is inflated by near-duplicate leakage).
    X_tr, y_tr, groups_tr = [], [], []
    for i, s in enumerate(train):
        img = _imread(s["path"])
        if img is None:
            print(f"  WARN unreadable: {s['path']}")
            continue
        X_tr.append(extract_feature_vector(img))
        y_tr.append(s["y"])
        groups_tr.append(i)
        n_aug = aug_gen if s["y"] == 1 else aug_fake
        for v in augment_variants(img, n_aug, seed=SEED + i):
            X_tr.append(extract_feature_vector(v))
            y_tr.append(s["y"])
            groups_tr.append(i)

    X_te, y_te = [], []
    for s in test:
        img = _imread(s["path"])
        if img is None:
            print(f"  WARN unreadable: {s['path']}")
            continue
        X_te.append(extract_feature_vector(img))
        y_te.append(s["y"])

    return (
        np.array(X_tr, dtype=np.float32), np.array(y_tr),
        np.array(groups_tr),
        np.array(X_te, dtype=np.float32), np.array(y_te),
        {"aug_genuine": aug_gen, "aug_fake": aug_fake},
    )


def _models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=SEED,
        ),
        "svm_rbf": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def _evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    # labels: 0=fake, 1=genuine
    p, r, f, _ = precision_recall_fscore_support(
        y_te, pred, labels=[0, 1], zero_division=0,
    )
    cm = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    macro_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    return {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "precision_fake": round(float(p[0]), 4),
        "recall_fake": round(float(r[0]), 4),
        "precision_genuine": round(float(p[1]), 4),
        "recall_genuine": round(float(r[1]), 4),
        "confusion_matrix": cm,  # [[TN(fake->fake), FP(fake->gen)], [FN, TP]]
    }


def main():
    if not os.path.exists(INDEX):
        print("dataset/index.json missing — run scripts/build_dataset.py first.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    index = _load_index()

    X_tr, y_tr, groups_tr, X_te, y_te, aug_info = _build_matrix(index)
    print(f"Feature matrix: train {X_tr.shape}, test {X_te.shape} "
          f"(dim {FEATURE_DIM})")
    n_groups = len(set(groups_tr.tolist()))

    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    # Group-aware, stratified CV: keeps a base note's augmented
    # variants in a single fold so the CV score is honest. Clamp
    # the fold count to the number of groups in the smaller class.
    n_splits = max(2, min(5, n_groups // 2))
    cv = StratifiedGroupKFold(n_splits=n_splits)
    print(f"Group-aware CV: {n_splits} folds over {n_groups} base groups")

    results = {}
    best_name, best_score = None, -1.0

    for name, model in _models().items():
        cv_scores = cross_val_score(
            model, Xs_tr, y_tr, groups=groups_tr,
            cv=cv, scoring="f1_macro",
        )
        model.fit(Xs_tr, y_tr)
        metrics = _evaluate(model, Xs_te, y_te)
        metrics["cv_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_macro_f1_std"] = round(float(cv_scores.std()), 4)
        results[name] = metrics
        joblib.dump(model, os.path.join(OUT_DIR, f"{name}.joblib"))

        print(f"\n[{name}]")
        print(f"  CV macro-F1 : {metrics['cv_macro_f1_mean']:.3f} "
              f"± {metrics['cv_macro_f1_std']:.3f}")
        print(f"  test acc    : {metrics['test_accuracy']:.3f}   "
              f"test macro-F1: {metrics['test_macro_f1']:.3f}")
        print(f"  confusion   : {metrics['confusion_matrix']} "
              f"(rows=actual [fake,genuine], cols=pred)")

        # Select on CV (test set is tiny — CV is the more stable signal).
        if metrics["cv_macro_f1_mean"] > best_score:
            best_score = metrics["cv_macro_f1_mean"]
            best_name = name

    payload = {
        "meta": {
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "feature_names": FEATURE_NAMES,
            "train_rows": int(X_tr.shape[0]),
            "test_rows": int(X_te.shape[0]),
            "augmentation": aug_info,
            "label_scheme": "1=genuine, 0=fake",
            "selection_metric": "cv_macro_f1_mean",
            "best_model": best_name,
            "data_caveat": (
                f"Held-out TEST split is {int(X_te.shape[0])} images, so test "
                f"metrics are higher-variance than the group-aware CV estimate "
                f"on the augmented train split. Model selection uses CV (no "
                f"test-set peeking). Add more images into dataset/real|fake and "
                f"re-run build_dataset.py + this script to improve generalisation."
            ),
        },
        "models": results,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nBest by CV macro-F1: {best_name} ({best_score:.3f})")
    print(f"Saved models + metrics to {os.path.relpath(OUT_DIR, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

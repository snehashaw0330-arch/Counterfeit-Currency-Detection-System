"""
Phase T2 — train a Bangladesh (BDT) counterfeit classifier on JaalTaka.

Mirrors the INR classical pipeline (scripts/train_classical.py) for methodology
consistency: the same `backend.features` extractor and the same four techniques
(LogReg / SVM-RBF / RandomForest / KNN), selected by group-aware CV macro-F1.

BDT-specific correctness:
- JaalTaka gives SIX security-region crops per physical note. Splitting must be
  GROUP-AWARE BY NOTE so a note's crops never straddle train/test (else leakage
  inflates the score). The group key is `<label>_<note_id>` because real and fake
  reuse the same `note_id` strings for different physical notes.
- No augmentation: with 336 images/class the base data is already plentiful, so
  we train on real images only (honest, no synthetic variants).
- We report BOTH per-image metrics (the deployable unit — a single uploaded
  photo) and per-note majority-vote metrics (JaalTaka's intended multi-view use).

Artifacts (git-ignored, regenerable):
    models/bdt/<name>.joblib · scaler.joblib · metrics.json

Run:
    venv\\Scripts\\python.exe scripts\\train_bdt_counterfeit.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

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

SEED = 42
TEST_FRACTION = 0.2
BDT_DIR = os.path.join(ROOT, "dataset", "foreign", "bdt", "security_crops")
OUT_DIR = os.path.join(ROOT, "models", "bdt")


def _load_samples():
    """[{path, y(1=real/0=fake), note_id, group, split}] from the locked layout."""
    samples = []
    for label, y in (("real", 1), ("fake", 0)):
        for img in sorted(glob.glob(os.path.join(BDT_DIR, label, "*.jpg"))):
            meta_path = os.path.splitext(img)[0] + ".json"
            note_id = ""
            if os.path.exists(meta_path):
                try:
                    note_id = json.load(open(meta_path, encoding="utf-8")).get("note_id", "")
                except Exception:
                    pass
            if not note_id:  # fall back to filename: note_017_3.jpg -> note_017
                base = os.path.basename(img)
                note_id = "_".join(base.split("_")[:2])
            samples.append({
                "path": img, "y": y, "note_id": note_id,
                "group": f"{label}_{note_id}",
            })
    return samples


def _assign_splits(samples):
    """Deterministic group-aware, class-balanced train/test split by note."""
    notes_by_class = defaultdict(list)
    for s in samples:
        notes_by_class[s["y"]].append(s["group"])
    test_groups = set()
    rng = np.random.default_rng(SEED)
    for y, groups in notes_by_class.items():
        uniq = sorted(set(groups))
        rng.shuffle(uniq)
        k = max(1, int(round(len(uniq) * TEST_FRACTION)))
        test_groups.update(uniq[:k])
    for s in samples:
        s["split"] = "test" if s["group"] in test_groups else "train"
    return samples


def _matrix(samples, split):
    rows, ys, groups, notes = [], [], [], []
    for s in samples:
        if s["split"] != split:
            continue
        img = cv2.imread(s["path"])
        if img is None:
            print(f"  WARN unreadable: {s['path']}")
            continue
        rows.append(extract_feature_vector(img))
        ys.append(s["y"])
        groups.append(s["group"])
        notes.append(s["group"])
    return (np.array(rows, dtype=np.float32), np.array(ys),
            np.array(groups), notes)


def _models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=SEED),
        "svm_rbf": SVC(kernel="rbf", probability=True,
                       class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def _image_metrics(pred, y_te):
    acc = float(accuracy_score(y_te, pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    return {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "precision_fake": round(float(p[0]), 4),
        "recall_fake": round(float(r[0]), 4),
        "precision_genuine": round(float(p[1]), 4),
        "recall_genuine": round(float(r[1]), 4),
        "confusion_matrix": cm,  # rows=actual[fake,genuine], cols=pred
    }


def _note_metrics(pred, y_te, note_groups):
    """Majority-vote each note's crops into one note-level prediction."""
    votes = defaultdict(list)
    truth = {}
    for p, y, g in zip(pred, y_te, note_groups):
        votes[g].append(int(p))
        truth[g] = int(y)
    gp, gy = [], []
    for g, vs in votes.items():
        gp.append(1 if sum(vs) * 2 >= len(vs) else 0)  # tie -> genuine
        gy.append(truth[g])
    return {
        "note_count": len(gy),
        "note_accuracy": round(float(accuracy_score(gy, gp)), 4),
        "note_macro_f1": round(float(f1_score(gy, gp, average="macro", zero_division=0)), 4),
        "note_confusion_matrix": confusion_matrix(gy, gp, labels=[0, 1]).tolist(),
    }


def main():
    if not os.path.isdir(BDT_DIR):
        print(f"{BDT_DIR} missing — run scripts/fetch_jaaltaka.py first.")
        return 1

    samples = _assign_splits(_load_samples())
    if not samples:
        print("No BDT images found.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    X_tr, y_tr, groups_tr, _ = _matrix(samples, "train")
    X_te, y_te, _, notes_te = _matrix(samples, "test")
    n_groups = len(set(groups_tr.tolist()))
    print(f"BDT samples: {len(samples)}  ({sum(s['y'] for s in samples)} real / "
          f"{sum(1 for s in samples if s['y'] == 0)} fake)")
    print(f"Feature matrix: train {X_tr.shape}, test {X_te.shape} (dim {FEATURE_DIM})")
    print(f"Train note-groups: {n_groups}; test images: {len(y_te)} "
          f"over {len(set(notes_te))} notes")

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    n_splits = max(2, min(5, n_groups // 2))
    cv = StratifiedGroupKFold(n_splits=n_splits)
    print(f"Group-aware CV: {n_splits} folds over {n_groups} note-groups")

    results, best_name, best_score = {}, None, -1.0
    for name, model in _models().items():
        cv_scores = cross_val_score(model, Xs_tr, y_tr, groups=groups_tr,
                                    cv=cv, scoring="f1_macro")
        model.fit(Xs_tr, y_tr)
        pred = model.predict(Xs_te)
        m = _image_metrics(pred, y_te)
        m["cv_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
        m["cv_macro_f1_std"] = round(float(cv_scores.std()), 4)
        m.update(_note_metrics(pred, y_te, notes_te))
        results[name] = m
        joblib.dump(model, os.path.join(OUT_DIR, f"{name}.joblib"))
        print(f"\n[{name}]")
        print(f"  CV macro-F1 : {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f}")
        print(f"  image  acc  : {m['test_accuracy']:.3f}   macro-F1: {m['test_macro_f1']:.3f}")
        print(f"  note   acc  : {m['note_accuracy']:.3f}   macro-F1: {m['note_macro_f1']:.3f}")
        print(f"  image confusion {m['confusion_matrix']} (rows=actual[fake,gen])")
        if m["cv_macro_f1_mean"] > best_score:
            best_score, best_name = m["cv_macro_f1_mean"], name

    payload = {
        "meta": {
            "dataset": "JaalTaka (BDT 500/1000, genuine + physical counterfeit)",
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "feature_names": FEATURE_NAMES,
            "n_images": len(samples),
            "train_images": int(X_tr.shape[0]),
            "test_images": int(X_te.shape[0]),
            "split": "group-aware by note (label_note_id), class-balanced, no augmentation",
            "label_scheme": "1=genuine, 0=fake",
            "selection_metric": "cv_macro_f1_mean",
            "best_model": best_name,
        },
        "models": results,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nBest by CV macro-F1: {best_name} ({best_score:.3f})")
    print(f"Saved to {os.path.relpath(OUT_DIR, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

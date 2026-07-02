"""
Generic per-currency counterfeit classifier for foreign notes.

Mirrors scripts/train_bdt_counterfeit.py (same `backend.features` extractor, same
four techniques, group-aware CV) but is parameterised by currency code so any
foreign currency with a genuine+fake dataset in the locked layout can be trained
with ONE script — no per-currency copy-paste. BDT keeps its dedicated script;
this covers AUD and every future currency your partner supplies.

    dataset/foreign/<ccy>/{full_note,security_crops}/{real,fake}/<file>[.json]

Splitting is GROUP-AWARE by the sidecar `group` (falls back to the filename), so
a note's crops — or a real image and a *derived* fake twin — never straddle
train/test (that would inflate the score).

HONESTY: the script inspects each fake's sidecar `source` and records whether the
fakes look SYNTHETIC (digitally-altered copies of reals). A high score on
synthetic fakes measures "can it spot that manipulation", NOT real-counterfeit
detection — the metrics.json says so explicitly so downstream reporting can't
overstate the capability.

Artifacts (git-ignored, regenerable):
    models/<ccy>/<name>.joblib · scaler.joblib · metrics.json

Run:
    venv\\Scripts\\python.exe scripts\\train_foreign_counterfeit.py aud
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
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_samples(base):
    """[{path, y, group}] from full_note/ + security_crops/ real|fake trees."""
    samples = []
    synthetic_hits = 0
    fake_total = 0
    for label, y in (("real", 1), ("fake", 0)):
        for sub in ("full_note", "security_crops"):
            for ext in IMG_EXT:
                for img in sorted(glob.glob(os.path.join(base, sub, label, "*" + ext))):
                    meta = {}
                    mp = os.path.splitext(img)[0] + ".json"
                    if os.path.exists(mp):
                        try:
                            meta = json.load(open(mp, encoding="utf-8"))
                        except Exception:
                            meta = {}
                    group = meta.get("group") or os.path.splitext(os.path.basename(img))[0]
                    if y == 0:
                        fake_total += 1
                        if "synthetic" in str(meta.get("source", "")).lower():
                            synthetic_hits += 1
                    samples.append({"path": img, "y": y, "group": str(group)})
    return samples, synthetic_hits, fake_total


def _assign_splits(samples):
    """Deterministic group-aware, class-balanced train/test split."""
    groups_by_class = defaultdict(list)
    for s in samples:
        groups_by_class[s["y"]].append(s["group"])
    test_groups, rng = set(), np.random.default_rng(SEED)
    for _y, groups in groups_by_class.items():
        uniq = sorted(set(groups))
        rng.shuffle(uniq)
        k = max(1, int(round(len(uniq) * TEST_FRACTION)))
        test_groups.update(uniq[:k])
    for s in samples:
        s["split"] = "test" if s["group"] in test_groups else "train"
    return samples


def _matrix(samples, split):
    rows, ys, groups = [], [], []
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
    return np.array(rows, dtype=np.float32), np.array(ys), np.array(groups)


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
    return {
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "precision_fake": round(float(p[0]), 4),
        "recall_fake": round(float(r[0]), 4),
        "precision_genuine": round(float(p[1]), 4),
        "recall_genuine": round(float(r[1]), 4),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: train_foreign_counterfeit.py <ccy>  (e.g. aud)")
        return 2
    ccy = sys.argv[1].lower()
    base = os.path.join(ROOT, "dataset", "foreign", ccy)
    out_dir = os.path.join(ROOT, "models", ccy)
    if not os.path.isdir(base):
        print(f"{base} missing.")
        return 1

    raw, synth_hits, fake_total = _load_samples(base)
    samples = _assign_splits(raw)
    if not samples:
        print(f"No {ccy.upper()} images found under {base}.")
        return 1

    n_real = sum(s["y"] for s in samples)
    n_fake = sum(1 for s in samples if s["y"] == 0)
    if n_real == 0 or n_fake == 0:
        print(f"Need BOTH real and fake {ccy.upper()} images (have {n_real} real / {n_fake} fake).")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    X_tr, y_tr, g_tr = _matrix(samples, "train")
    X_te, y_te, _ = _matrix(samples, "test")
    n_groups = len(set(g_tr.tolist()))
    synthetic = fake_total > 0 and synth_hits >= fake_total * 0.5

    print(f"{ccy.upper()} samples: {len(samples)} ({n_real} real / {n_fake} fake)")
    print(f"Feature matrix: train {X_tr.shape}, test {X_te.shape} (dim {FEATURE_DIM})")
    print(f"Train groups: {n_groups}; test images: {len(y_te)}")
    if synthetic:
        print("  !! fakes look SYNTHETIC (digitally-altered reals) — metrics measure\n"
              "     manipulation-detection, NOT real physical counterfeit detection.")

    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    n_splits = max(2, min(5, n_groups // 2))
    cv = StratifiedGroupKFold(n_splits=n_splits)
    print(f"Group-aware CV: {n_splits} folds over {n_groups} groups")

    results, best_name, best_score = {}, None, -1.0
    for name, model in _models().items():
        cv_scores = cross_val_score(model, Xs_tr, y_tr, groups=g_tr, cv=cv, scoring="f1_macro")
        model.fit(Xs_tr, y_tr)
        m = _image_metrics(model.predict(Xs_te), y_te)
        m["cv_macro_f1_mean"] = round(float(cv_scores.mean()), 4)
        m["cv_macro_f1_std"] = round(float(cv_scores.std()), 4)
        results[name] = m
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))
        print(f"\n[{name}]")
        print(f"  CV macro-F1 : {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f}")
        print(f"  image  acc  : {m['test_accuracy']:.3f}   macro-F1: {m['test_macro_f1']:.3f}")
        print(f"  confusion {m['confusion_matrix']} (rows=actual[fake,gen])")
        if m["cv_macro_f1_mean"] > best_score:
            best_score, best_name = m["cv_macro_f1_mean"], name

    payload = {
        "meta": {
            "currency": ccy.upper(),
            "seed": SEED,
            "feature_dim": FEATURE_DIM,
            "feature_names": FEATURE_NAMES,
            "n_images": len(samples),
            "n_real": n_real,
            "n_fake": n_fake,
            "train_images": int(X_tr.shape[0]),
            "test_images": int(X_te.shape[0]),
            "split": "group-aware (sidecar group / filename), class-balanced, no augmentation",
            "label_scheme": "1=genuine, 0=fake",
            "selection_metric": "cv_macro_f1_mean",
            "best_model": best_name,
            "fakes_are_synthetic": bool(synthetic),
            "honesty_note": (
                "Fakes are SYNTHETIC (digitally-altered copies of the real images); "
                "these metrics reflect detection of that manipulation, not of real "
                "physical counterfeits. Re-train on real physical counterfeit photos "
                "for a credible benchmark."
            ) if synthetic else "Fakes sourced as physical counterfeits.",
        },
        "models": results,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nBest by CV macro-F1: {best_name} ({best_score:.3f})")
    print(f"Saved to {os.path.relpath(out_dir, ROOT)}"
          + ("  [SYNTHETIC-FAKE caveat recorded in metrics.json]" if synthetic else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

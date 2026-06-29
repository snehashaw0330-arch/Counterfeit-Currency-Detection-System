"""
Phase T3 — genuine-only (one-class) counterfeit screening, benchmarked on BDT.

Tests the "train on genuine, flag deviations as fake" idea HONESTLY, on the one
foreign currency where we can measure it (JaalTaka BDT has genuine + real fakes):

  - Train Isolation Forest + One-Class SVM on GENUINE-ONLY training crops.
  - Score held-out genuine + fake crops; higher score = more genuine.
  - Report threshold-free ROC-AUC (genuine vs fake) and, at an operating point
    fixed to ~95% genuine pass-rate on TRAIN, the fake-catch rate + genuine FPR.
  - Compare to the SUPERVISED model (models/bdt/metrics.json), which sees fakes.

Same group-aware split (by <label>_<note_id>, seed 42) as the supervised trainer,
so the comparison is fair. No augmentation.

Output: models/bdt/oneclass_metrics.json  (tracked record)

Run:
    venv\\Scripts\\python.exe scripts\\train_oneclass_anomaly.py
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

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from backend.features import extract_feature_vector  # noqa: E402

SEED = 42
TEST_FRACTION = 0.2
GENUINE_PASS_TARGET = 0.95  # operating point: keep 95% of genuine on TRAIN
BDT_DIR = os.path.join(ROOT, "dataset", "foreign", "bdt", "security_crops")
OUT = os.path.join(ROOT, "models", "bdt", "oneclass_metrics.json")


def _load_samples():
    samples = []
    for label, y in (("real", 1), ("fake", 0)):
        for img in sorted(glob.glob(os.path.join(BDT_DIR, label, "*.jpg"))):
            base = os.path.basename(img)
            note_id = "_".join(base.split("_")[:2])
            samples.append({"path": img, "y": y, "group": f"{label}_{note_id}",
                            "note": f"{label}_{note_id}"})
    return samples


def _assign_splits(samples):
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["y"]].append(s["group"])
    test_groups = set()
    rng = np.random.default_rng(SEED)
    for groups in by_class.values():
        uniq = sorted(set(groups))
        rng.shuffle(uniq)
        k = max(1, int(round(len(uniq) * TEST_FRACTION)))
        test_groups.update(uniq[:k])
    for s in samples:
        s["split"] = "test" if s["group"] in test_groups else "train"
    return samples


_CACHE = os.path.join(ROOT, "models", "bdt", "_features_cache.npz")


def _features(samples):
    """Extract (and cache) features. Re-runs reuse the cache when the exact file
    list matches — feature extraction over 672 large crops is the slow step."""
    paths = [s["path"] for s in samples]
    if os.path.exists(_CACHE):
        try:
            d = np.load(_CACHE, allow_pickle=True)
            if list(d["paths"]) == paths:
                print("  (using cached features)")
                return d["X"], samples
        except Exception:
            pass
    X = []
    for i, s in enumerate(samples):
        img = cv2.imread(s["path"])
        X.append(extract_feature_vector(img) if img is not None
                 else np.zeros_like(X[0]) if X else None)
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(samples)}")
    X = np.array([x for x in X if x is not None], dtype=np.float32)
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        np.savez(_CACHE, paths=np.array(paths), X=X)
    except Exception:
        pass
    return X, samples


def _op_point(scores_gen_train, scores_test, y_test):
    """Threshold at the genuine-pass target; report fake-catch + genuine pass."""
    thr = float(np.quantile(scores_gen_train, 1.0 - GENUINE_PASS_TARGET))
    pred_gen = scores_test >= thr               # >= thr -> called genuine
    y = np.array(y_test)
    gen_pass = float(np.mean(pred_gen[y == 1])) if np.any(y == 1) else 0.0
    fake_catch = float(np.mean(~pred_gen[y == 0])) if np.any(y == 0) else 0.0
    return round(thr, 5), round(gen_pass, 4), round(fake_catch, 4)


def _note_auc(scores, metas):
    """Note-level: mean score per note, then AUC over notes."""
    agg, lab = defaultdict(list), {}
    for sc, m in zip(scores, metas):
        agg[m["note"]].append(sc); lab[m["note"]] = m["y"]
    notes = list(agg)
    s = [float(np.mean(agg[n])) for n in notes]
    y = [lab[n] for n in notes]
    return round(float(roc_auc_score(y, s)), 4) if len(set(y)) > 1 else None


def main():
    if not os.path.isdir(BDT_DIR):
        print(f"{BDT_DIR} missing — run scripts/fetch_jaaltaka.py first.")
        return 1
    samples = _assign_splits(_load_samples())
    print("Extracting features (672 crops) ...")
    X, meta = _features(samples)

    tr = np.array([m["split"] == "train" for m in meta])
    y = np.array([m["y"] for m in meta])
    gen_tr = tr & (y == 1)                      # GENUINE-ONLY training
    te = ~tr

    scaler = StandardScaler().fit(X[gen_tr])
    Xs = scaler.transform(X)

    models = {
        "isolation_forest": IsolationForest(n_estimators=300, random_state=SEED,
                                            contamination="auto"),
        "oneclass_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
    }

    results = {}
    y_te = y[te].tolist()
    metas_te = [m for m, t in zip(meta, te) if t]
    for name, model in models.items():
        model.fit(Xs[gen_tr])                   # fit on genuine only
        s_all = model.decision_function(Xs)     # higher = more inlier (genuine)
        s_te = s_all[te]
        auc = round(float(roc_auc_score(y_te, s_te)), 4)
        thr, gen_pass, fake_catch = _op_point(s_all[gen_tr], s_te, y_te)
        results[name] = {
            "roc_auc": auc,
            "note_roc_auc": _note_auc(s_te, metas_te),
            "genuine_pass_at_op": gen_pass,
            "fake_catch_at_op": fake_catch,
            "operating_point": f"threshold for ~{int(GENUINE_PASS_TARGET*100)}% genuine pass on train",
        }
        print(f"\n[{name}]  (trained on GENUINE only)")
        print(f"  image ROC-AUC : {auc:.3f}   note ROC-AUC: {results[name]['note_roc_auc']}")
        print(f"  @op: genuine pass {gen_pass:.3f}  fake catch {fake_catch:.3f}")

    # supervised reference
    sup = None
    sup_path = os.path.join(ROOT, "models", "bdt", "metrics.json")
    if os.path.exists(sup_path):
        m = json.load(open(sup_path, encoding="utf-8"))
        best = m["meta"]["best_model"]
        sup = {"best_model": best,
               "image_accuracy": m["models"][best]["test_accuracy"],
               "note_accuracy": m["models"][best]["note_accuracy"]}

    payload = {
        "meta": {
            "dataset": "JaalTaka BDT (genuine + physical counterfeit)",
            "approach": "one-class (trained on GENUINE crops only); fakes used for evaluation only",
            "split": "group-aware by note, seed 42 (same as supervised)",
            "train_genuine": int(gen_tr.sum()),
            "test_images": int(te.sum()),
            "supervised_reference": sup,
            "honest_note": (
                "One-class screening needs only genuine data to TRAIN, but needs "
                "counterfeits to VALIDATE. It is measured here on BDT; for "
                "AUD/CAD/GBP/PHP it cannot be validated (no public fakes)."
            ),
        },
        "models": results,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    if sup:
        print(f"\nSupervised reference ({sup['best_model']}): "
              f"image acc {sup['image_accuracy']}, note acc {sup['note_accuracy']}")
    print(f"Saved {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

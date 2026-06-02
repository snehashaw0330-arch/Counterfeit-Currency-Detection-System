"""
Phase F.1 — retrain the MobileNetV2 counterfeit classifier.

Transfer learning, overfitting-aware for our small corpus:
  - ImageNet-pretrained MobileNetV2 base, FROZEN, used as a feature
    extractor (1280-d global-avg-pool vector). Base features are
    precomputed once (fast on CPU) and a small trainable head is fit
    on them with heavy augmentation + class balancing.
  - Honest eval on the held-out TEST split (no augmentation).

Saved as a NEW model file (…_v2.keras) so the production model in
backend/main.py is untouched until we confirm the retrain is better.

Input/label convention matches backend/main.py exactly:
  - input is RGB in [0,1] (i.e. pixels/255). A Rescaling layer inside
    the saved model maps [0,1] -> [-1,1] for MobileNetV2, so main.py
    needs NO change.
  - output is sigmoid P(genuine); >= 0.5 -> REAL.

Run:
    venv\\Scripts\\python.exe scripts\\train_cnn.py
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, confusion_matrix,
)

from backend.augment import augment_variants  # noqa: E402

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG = 224
TARGET_PER_CLASS = 300
INDEX = os.path.join(ROOT, "dataset", "index.json")
OUT = os.path.join(ROOT, "models", "mobilenet_counterfeit_detector_v2.keras")
CURRENT_CNN_TEST_MACRO_F1 = 0.564   # from docs/BENCHMARK.md (for comparison)


def _prep(bgr):
    """BGR uint8 -> RGB float32 [0,1] at 224x224 (matches main.py)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG, IMG)), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _imread(rel):
    return cv2.imread(os.path.join(ROOT, rel.replace("/", os.sep)))


def _aug_per(n):
    return max(0, round(TARGET_PER_CLASS / n) - 1) if n else 0


def _build_data(index):
    samples = index["samples"]
    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]

    n_gen = sum(1 for s in train if s["y"] == 1)
    n_fake = sum(1 for s in train if s["y"] == 0)
    aug_gen, aug_fake = _aug_per(n_gen), _aug_per(n_fake)
    print(f"TRAIN base: {n_gen} genuine / {n_fake} fake  "
          f"(+{aug_gen}/gen, +{aug_fake}/fake)")

    X_tr, y_tr = [], []
    for i, s in enumerate(train):
        img = _imread(s["path"])
        if img is None:
            continue
        X_tr.append(_prep(img))
        y_tr.append(s["y"])
        for v in augment_variants(img, aug_gen if s["y"] == 1 else aug_fake,
                                  seed=SEED + i):
            X_tr.append(_prep(v))
            y_tr.append(s["y"])

    X_te, y_te = [], []
    for s in test:
        img = _imread(s["path"])
        if img is None:
            continue
        X_te.append(_prep(img))
        y_te.append(s["y"])

    return (
        np.array(X_tr, dtype=np.float32), np.array(y_tr),
        np.array(X_te, dtype=np.float32), np.array(y_te),
    )


def main():
    if not os.path.exists(INDEX):
        print("dataset/index.json missing — run scripts/build_dataset.py first.")
        return 1

    with open(INDEX, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    X_tr, y_tr, X_te, y_te = _build_data(index)
    print(f"train {X_tr.shape}  test {X_te.shape}")

    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG, IMG, 3), include_top=False,
            weights="imagenet", pooling="avg",
        )
    except Exception as exc:
        print(f"Could not load ImageNet weights ({exc}). "
              f"Need network access on first run.")
        return 1
    base.trainable = False

    # Precompute frozen-base features ([-1,1] input for MobileNetV2).
    feats_tr = base.predict(X_tr * 2.0 - 1.0, batch_size=32, verbose=0)
    feats_te = base.predict(X_te * 2.0 - 1.0, batch_size=32, verbose=0)

    n_gen = int((y_tr == 1).sum())
    n_fake = int((y_tr == 0).sum())
    total = n_gen + n_fake
    class_weight = {
        0: total / (2.0 * n_fake),
        1: total / (2.0 * n_gen),
    }

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feats_tr.shape[1],)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="binary_crossentropy", metrics=["accuracy"])

    head.fit(
        feats_tr, y_tr,
        validation_split=0.15,
        epochs=80, batch_size=32,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        )],
        verbose=0,
    )

    # ---- honest test eval ----
    p = head.predict(feats_te, verbose=0).ravel()
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])

    print("\n=== retrained CNN (v2) on held-out test ===")
    print(f"  test acc      : {acc:.3f}")
    print(f"  test macro-F1 : {macro_f1:.3f}   "
          f"(current production CNN: {CURRENT_CNN_TEST_MACRO_F1:.3f})")
    print(f"  confusion [[TN,FP],[FN,TP]] (0=fake,1=genuine): {cm.tolist()}")

    # ---- assemble full inference model (matches main.py input) ----
    final = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG, IMG, 3)),       # RGB [0,1]
        tf.keras.layers.Rescaling(scale=2.0, offset=-1.0),  # -> [-1,1]
        base,
        head,
    ])
    final.save(OUT)
    print(f"\nSaved retrained model -> {os.path.relpath(OUT, ROOT)}")

    if macro_f1 > CURRENT_CNN_TEST_MACRO_F1:
        print(f"BETTER than current CNN (+{macro_f1 - CURRENT_CNN_TEST_MACRO_F1:.3f} "
              f"macro-F1). Candidate for adoption (swap path in backend/main.py).")
    else:
        print("NOT better than current CNN on test — keep production model; "
              "document and revisit with more data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

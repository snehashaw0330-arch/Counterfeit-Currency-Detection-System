"""
Train shallow multi-currency classifiers on the downloaded BankNote-Net CSV.

This is a separate foreign-currency branch from the repo's current INR
genuine-vs-fake image pipeline. It trains directly on BankNote-Net embeddings.

Input:
    dataset/foreign/banknote_net/banknote_net.csv

Outputs:
    models/banknote_net/
        scaler.joblib
        logistic_regression.joblib
        random_forest.joblib
        metrics.json

Run:
    venv\\Scripts\\python.exe scripts\\train_banknote_net_currency.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402


CSV_PATH = os.path.join(ROOT, "dataset", "foreign", "banknote_net", "banknote_net.csv")
OUT_DIR = os.path.join(ROOT, "models", "banknote_net")
SEED = 42
TEST_FRACTION = 0.2


def _find_col(columns, wanted):
    wanted = wanted.lower()
    for col in columns:
        if str(col).strip().lower() == wanted:
            return col
    for col in columns:
        if wanted in str(col).strip().lower():
            return col
    return None


def _load():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "BankNote-Net CSV missing. Run scripts/fetch_banknote_net.py first."
        )

    with open(CSV_PATH, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise RuntimeError("BankNote-Net CSV is empty.")

        currency_col = _find_col(header, "currency")
        if currency_col is None:
            raise RuntimeError("Could not find a currency column in BankNote-Net CSV.")

        currency_idx = header.index(currency_col)
        exclude_idx = {currency_idx}
        for wanted in ("denomination", "face"):
            col = _find_col(header, wanted)
            if col is not None:
                exclude_idx.add(header.index(col))

        numeric_idx = []
        for i, col in enumerate(header):
            low = str(col).strip().lower()
            if i in exclude_idx:
                continue
            if not low or low in {"index", "id"}:
                continue
            if low.startswith("v_"):
                numeric_idx.append(i)

        if not numeric_idx:
            raise RuntimeError("No numeric embedding columns found in BankNote-Net CSV.")

        X_rows = []
        y_raw = []
        for row in reader:
            if not row or currency_idx >= len(row):
                continue
            cur = str(row[currency_idx]).strip()
            if not cur:
                continue
            try:
                feats = [float(row[i]) for i in numeric_idx]
            except (ValueError, IndexError):
                continue
            X_rows.append(feats)
            y_raw.append(cur)

    if not X_rows:
        raise RuntimeError("No valid BankNote-Net rows could be parsed from the CSV.")
    X = np.array(X_rows, dtype=np.float32)

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    return X, y, enc, numeric_idx, currency_col


def _metrics(model, X_te, y_te):
    pred = model.predict(X_te)
    return {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
    }


def main() -> int:
    X, y, enc, numeric_cols, currency_col = _load()
    print(f"Loaded BankNote-Net: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Target currencies ({len(enc.classes_)}): {', '.join(enc.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr)
    Xs_te = scaler.transform(X_te)

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xs_tr, y_tr)
        joblib.dump(model, os.path.join(OUT_DIR, f"{name}.joblib"))
        results[name] = _metrics(model, Xs_te, y_te)
        print(
            f"{name:<20} "
            f"acc={results[name]['accuracy']:.4f} "
            f"macro_f1={results[name]['macro_f1']:.4f}"
        )

    payload = {
        "meta": {
            "dataset": "BankNote-Net embeddings CSV",
            "csv_path": os.path.relpath(CSV_PATH, ROOT).replace(os.sep, "/"),
            "currency_column": currency_col,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_currencies": int(len(enc.classes_)),
            "currencies": enc.classes_.tolist(),
            "test_fraction": TEST_FRACTION,
            "seed": SEED,
            "note": (
                "These metrics describe multi-currency classification on "
                "BankNote-Net embeddings, not INR genuine-vs-fake detection."
            ),
        },
        "models": results,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved models + metrics to {os.path.relpath(OUT_DIR, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

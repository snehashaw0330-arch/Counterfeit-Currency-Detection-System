"""
Phase D.1 — dataset indexer.

Builds a single labelled index (dataset/index.json) of every image
we can train/evaluate on, drawn from two sources:

  1. The fixture corpus described by tests/sample_notes/manifest.json
     (the ground-truth source of truth). grade == "fake" -> label
     "fake"; everything else (clean / phone / edge_case) -> "genuine".

  2. ANY images placed anywhere under dataset/ (recursively). The
     label is inferred from the folder path — a path component (or
     substring) like real/genuine/authentic -> "genuine";
     fake/counterfeit/forged -> "fake". Images whose label cannot be
     inferred are SKIPPED (and reported) — we never guess a label.

This means you can drop a downloaded dataset folder anywhere under
dataset/ (e.g. dataset/kaggle_indian_currency/real/...,
dataset/.../fake/...) and re-run this script; it just works. See
docs/DATASET.md for the drop-in rules.

The index is split deterministically (stratified by label, fixed
seed) into train / test. Augmentation is applied PER-SPLIT at
training time (train_classical.py), so variants never leak across
the split boundary. With a large real dataset, train_classical
automatically stops augmenting (target rows already met).

Run:
    venv\\Scripts\\python.exe scripts\\build_dataset.py [--validate]

    --validate  also opens every newly-ingested image with OpenCV and
                reports unreadable/corrupt files (slower; one-time).

Output:
    dataset/index.json   (git-ignored, regenerable)

Deterministic: same inputs + same seed => same split.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.model_selection import train_test_split  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "tests", "sample_notes")
MANIFEST = os.path.join(SAMPLE_DIR, "manifest.json")
DATASET_DIR = os.path.join(ROOT, "dataset")
INDEX_OUT = os.path.join(DATASET_DIR, "index.json")

SEED = 42
TEST_FRACTION = 0.25
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_GENUINE_TOKENS = ("real", "genuine", "authentic", "original", "legit", "true")
_FAKE_TOKENS = ("fake", "counterfeit", "forged", "false", "fraud")

# Folders that hold cropped security-feature templates or notebook
# checkpoints — NOT full banknote images. These must never enter the
# whole-note genuine/fake classifier (they'd poison it). Reserved for
# Phase E (template/motif matching). Matched as a path-component substring.
_EXCLUDE_TOKENS = ("feature", "template")
_CHECKPOINT = ".ipynb_checkpoints"


def _is_excluded(parts):
    """True if any folder in the path marks a feature-crop /
    template / notebook-checkpoint tree (not full notes)."""
    for p in parts:
        pl = p.lower()
        if pl == _CHECKPOINT or any(tok in pl for tok in _EXCLUDE_TOKENS):
            return True
    return False


def _rel(path):
    """Repo-relative POSIX path — portable across machines/OSes."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _label_from_grade(grade):
    return "fake" if grade == "fake" else "genuine"


def _infer_label_from_parts(parts):
    """parts: folder names below dataset/, closest-folder first.

    Exact-token match wins first (most reliable), then substring
    match (handles 'fake_notes', 'real_500', 'counterfeit-set')."""

    lowered = [p.lower() for p in parts]
    for p in lowered:
        if p in _FAKE_TOKENS:
            return "fake"
        if p in _GENUINE_TOKENS:
            return "genuine"
    for p in lowered:
        if any(tok in p for tok in _FAKE_TOKENS):
            return "fake"
        if any(tok in p for tok in _GENUINE_TOKENS):
            return "genuine"
    return None


def _collect_from_manifest():
    """Yield (abs_path, label, source) for each manifest fixture."""

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    for key, meta in manifest.items():
        if key == "_schema":
            continue
        abs_path = os.path.join(SAMPLE_DIR, key.replace("/", os.sep))
        if not os.path.exists(abs_path):
            print(f"  WARN manifest entry missing on disk: {key}")
            continue
        yield abs_path, _label_from_grade(meta.get("grade", "")), "fixture"


def _collect_from_dataset_dir(skipped, excluded):
    """Recursively yield (abs_path, label, source) for images under
    dataset/, inferring the label from the folder path.

    Feature-crop / template / checkpoint subtrees are skipped into
    `excluded` (reserved for Phase E). Images with no inferable
    label go to `skipped` (reported, never guessed)."""

    if not os.path.isdir(DATASET_DIR):
        return

    for root, _dirs, files in os.walk(DATASET_DIR):
        rel_root = os.path.relpath(root, DATASET_DIR)
        root_parts = [] if rel_root == "." else rel_root.split(os.sep)

        # Foreign-currency data (dataset/foreign/<ccy>/...) is handled by the
        # separate country-aware Phase-S pipeline — never pool it into the INR
        # whole-note genuine/fake classifier (would contaminate the benchmark).
        if root_parts and root_parts[0] == "foreign":
            continue

        if _is_excluded(root_parts):
            for name in files:
                if name.lower().endswith(_IMG_EXT):
                    excluded.append(_rel(os.path.join(root, name)))
            continue

        for name in files:
            if not name.lower().endswith(_IMG_EXT):
                continue
            abs_path = os.path.join(root, name)
            # folder components below dataset/, closest folder first
            parts = list(reversed(root_parts))
            label = _infer_label_from_parts(parts)
            if label is None:
                skipped.append(_rel(abs_path))
                continue
            yield abs_path, label, "dataset_dir"


def _validate_readable(paths):
    """Return the subset of paths OpenCV cannot decode."""
    import cv2
    bad = []
    for p in paths:
        if cv2.imread(p) is None:
            bad.append(_rel(p))
    return bad


def build(validate=False):
    os.makedirs(os.path.join(DATASET_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "fake"), exist_ok=True)

    samples = {}  # abs_path -> (label, source); dedup by path
    skipped = []
    excluded = []

    for abs_path, label, source in _collect_from_manifest():
        samples[abs_path] = (label, source)
    for abs_path, label, source in _collect_from_dataset_dir(skipped, excluded):
        samples.setdefault(abs_path, (label, source))

    if excluded:
        print(f"Excluded {len(excluded)} feature-crop/template/checkpoint "
              f"image(s) from the whole-note classifier (reserved for Phase E).")

    if skipped:
        print(f"Skipped {len(skipped)} image(s) under dataset/ with no "
              f"inferable real/fake label (place them inside a real/ or "
              f"fake/ folder). First few:")
        for s in skipped[:5]:
            print(f"  - {s}")

    if not samples:
        print("No labelled images found. Add images under dataset/real "
              "and dataset/fake, or check tests/sample_notes/manifest.json.")
        return 1

    if validate:
        print("Validating image readability (this can take a while)...")
        bad = _validate_readable(list(samples.keys()))
        for b in bad:
            print(f"  DROP unreadable: {b}")
            # remove from samples by matching repo-relative path
        if bad:
            bad_set = set(bad)
            samples = {
                p: v for p, v in samples.items() if _rel(p) not in bad_set
            }

    paths = sorted(samples.keys())
    labels = [samples[p][0] for p in paths]
    sources = [samples[p][1] for p in paths]

    n_fake = labels.count("fake")
    n_genuine = labels.count("genuine")

    stratify = labels if (n_fake >= 2 and n_genuine >= 2) else None

    idx = list(range(len(paths)))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=stratify,
    )
    split_of = {i: "train" for i in train_idx}
    split_of.update({i: "test" for i in test_idx})

    records = []
    for i, p in enumerate(paths):
        records.append({
            "path": _rel(p),
            "label": labels[i],
            "y": 1 if labels[i] == "genuine" else 0,
            "split": split_of[i],
            "source": sources[i],
        })

    out = {
        "meta": {
            "seed": SEED,
            "test_fraction": TEST_FRACTION,
            "total": len(records),
            "genuine": n_genuine,
            "fake": n_fake,
            "train": sum(1 for r in records if r["split"] == "train"),
            "test": sum(1 for r in records if r["split"] == "test"),
            "skipped_unlabelled": len(skipped),
            "excluded_feature_crops": len(excluded),
            "label_scheme": "genuine=1 (real/phone/edge_case), fake=0 (grade=fake)",
            "note": (
                "Drop more images anywhere under dataset/ inside a real/ or "
                "fake/ folder and re-run. train_classical.py augments TRAIN "
                "samples toward a balanced target and stops augmenting once "
                "the base data is plentiful."
            ),
        },
        "samples": records,
    }

    with open(INDEX_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"Indexed {len(records)} images "
          f"({n_genuine} genuine / {n_fake} fake)")
    print(f"  train: {out['meta']['train']}   test: {out['meta']['test']}")
    print(f"  sources: fixture={sources.count('fixture')}, "
          f"dataset_dir={sources.count('dataset_dir')}")
    print(f"Wrote {_rel(INDEX_OUT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the labelled dataset index.")
    parser.add_argument(
        "--validate", action="store_true",
        help="open every image with OpenCV and drop unreadable ones",
    )
    args = parser.parse_args()
    raise SystemExit(build(validate=args.validate))

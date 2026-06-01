# Dataset — how to add more data

The ML techniques (Phase D) and the CNN (Phase F) get better the more
labelled images they train on. The `dataset/` folder is **git-ignored**
(images are large and not ours to redistribute), so each machine builds it
locally. This guide is how.

## 1. Where to put images

Drop images **anywhere under `dataset/`**, as long as the folder path says
whether they are genuine or fake. `scripts/build_dataset.py` walks `dataset/`
recursively and infers the label from the path:

| Path contains a folder like… | Label |
|---|---|
| `real`, `genuine`, `authentic`, `original`, `legit`, `true` | **genuine** |
| `fake`, `counterfeit`, `forged`, `false`, `fraud` | **fake** |

So all of these work:

```
dataset/real/500/img001.jpg                         -> genuine
dataset/fake/img_fake_12.png                         -> fake
dataset/kaggle_fake_currency/Real/2000/note.jpg      -> genuine
dataset/kaggle_fake_currency/Fake/note.jpg           -> fake
```

Images whose label can't be inferred are **skipped and reported** — we never
guess. If a download uses different folder names, either rename the top folder
to `real`/`fake` or nest it under `dataset/real/` or `dataset/fake/`.

## 2. Build → train → benchmark

After adding images, re-run the three Phase-D scripts in order:

```
venv\Scripts\python.exe scripts\build_dataset.py --validate
venv\Scripts\python.exe scripts\train_classical.py
venv\Scripts\python.exe scripts\benchmark_models.py
```

- `--validate` opens every image with OpenCV and drops unreadable/corrupt ones
  (worth it the first time after a big download).
- `train_classical.py` augments the TRAIN split toward a balanced target and
  **automatically stops augmenting** once the real data is plentiful, so a large
  dataset trains on real images, not synthetic variants.
- `benchmark_models.py` refreshes `docs/BENCHMARK.md` with the new numbers.

No code changes are needed — the whole pipeline is data-driven.

## 3. Suggested public datasets

Fetch these yourself (they need a Kaggle/Mendeley login + ToS acceptance, which
an automated agent can't do), unzip under `dataset/`, and make sure the real/fake
folder convention above holds.

- **Kaggle — "Fake Currency Detection Dataset"** — has both real and fake Indian
  notes; the best fit for our binary task.
  <https://www.kaggle.com/datasets/sreeharisureshkaggle/fake-currency-detection-dataset>
- **Kaggle — "Indian Currency Note images dataset 2020"** (~3,700 imgs, 7
  denominations) — **all genuine** (denomination classification). Useful to
  enrich the *genuine* class; provides no fakes. Place under `dataset/real/`.
  <https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020>
- **Mendeley / article — "Dataset of Indian and Thai banknotes with
  annotations"** — annotated banknote images.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8907680/>
- **GitHub — aprameya2001/Fake-Currency-Detection-System** — an image-processing
  reference repo that includes sample real/fake note images.
  <https://github.com/aprameya2001/Fake-Currency-Detection-System>

> **Caveat to keep us honest:** many "fake" sets are *digitally degraded* real
> notes (photocopy/scan artefacts), not physical counterfeits. That's fine for a
> first detector, but note it in the final report — true physical-counterfeit
> images are scarce and the model's real-world ceiling depends on getting them.

## 4. What gets committed

- **Committed:** `docs/DATASET.md` (this file), `docs/BENCHMARK.md`,
  `models/classical/metrics.json`.
- **Git-ignored (regenerable / large):** everything under `dataset/`
  (incl. `dataset/index.json`), `models/classical/*.joblib`.

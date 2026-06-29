"""
Phase R.3 — digital PUF (Physical-Unclonable-Function) note identity.

A PUF authenticates a *specific physical object* by a fingerprint of its
intrinsic, hard-to-clone micro-structure. Here we extract a **texture
fingerprint** from a fixed region of the note — a high-pass (fine-grain)
perceptual hash — and run an honest **enroll → verify** loop against a local
registry:

  enroll(image, note_id)  -> store the fingerprint under note_id
  verify(image, note_id)  -> AUTHENTIC / NO_MATCH / UNKNOWN by Hamming distance

Honest scope (stated in the report + UI):
  - This is a software proxy. A true PUF needs controlled capture; matching
    across genuinely different phone photos of the same physical note (lighting,
    angle, focus) is the hard real-world part. The fingerprint is robust to
    re-encoding / mild degradation and clearly separates different captures,
    which is what the closed-loop proof-of-concept demonstrates.
  - It is an IDENTITY check (is this the enrolled note?), not a counterfeit
    detector, and it requires prior enrollment — with no registry there is
    nothing to verify against.

Never raises. Pure numpy + OpenCV + a JSON registry.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local registry (git-ignored). Overridable for tests.
_REGISTRY_PATH = os.path.join(ROOT, "puf_registry.json")
_LOCK = threading.Lock()

# Fingerprint geometry: a GRID x GRID bit hash of the fine-texture residual of a
# fixed central region. 16 -> 256 bits.
_GRID = 16
_BITS = _GRID * _GRID
# Hamming-distance fraction at/below which two fingerprints are the same note.
# Calibrated in R.3: re-encoded/degraded same-capture stays well under this,
# different captures sit near 0.5.
_MATCH_MAX_DISTANCE = 0.25


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def compute_fingerprint(image) -> str:
    """Return a hex texture-fingerprint of the note's fixed central region.

    High-pass residual (fine grain) -> GRIDxGRID -> bits vs median -> hex.
    Stable under JPEG / resize; distinct between different captures. Never
    raises (returns an all-zero hash on bad input)."""
    try:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = arr.astype(np.uint8)

        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return "0" * (_BITS // 4)

        # Fixed central region (60%) so the same physical area is compared.
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.3), int(w * 0.3)
        region = gray[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx]
        region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_AREA)

        # Fine-texture residual (suppress design/lighting; keep micro-grain).
        blur = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 3.0)
        residual = region.astype(np.float32) - blur
        grid = cv2.resize(residual, (_GRID, _GRID), interpolation=cv2.INTER_AREA)

        bits = (grid > np.median(grid)).flatten()
        return _bits_to_hex(bits)
    except Exception:
        return "0" * (_BITS // 4)


def _bits_to_hex(bits) -> str:
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return format(val, "0{}x".format(_BITS // 4))


def _hex_to_bits(hexstr: str) -> np.ndarray:
    val = int(hexstr, 16)
    out = np.zeros(_BITS, dtype=np.uint8)
    for i in range(_BITS):
        out[_BITS - 1 - i] = (val >> i) & 1
    return out


def hamming_distance(fp_a: str, fp_b: str) -> float:
    """Normalised Hamming distance in [0,1] between two hex fingerprints."""
    try:
        a, b = _hex_to_bits(fp_a), _hex_to_bits(fp_b)
        return float(np.count_nonzero(a != b)) / float(_BITS)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Registry (local JSON, never raises)
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_registry(reg: dict) -> None:
    try:
        tmp = _REGISTRY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(reg, fh, indent=2)
        os.replace(tmp, _REGISTRY_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Enroll / verify
# ---------------------------------------------------------------------------

def enroll(image, note_id, meta=None) -> dict:
    """Register `image`'s fingerprint under `note_id`. Returns a result dict;
    never raises. Re-enrolling an existing id overwrites it."""
    note_id = str(note_id).strip()
    if not note_id:
        return {"status": "ERROR", "message": "note_id is required", "note_id": None}

    fp = compute_fingerprint(image)
    with _LOCK:
        reg = _load_registry()
        existed = note_id in reg
        reg[note_id] = {
            "fingerprint": fp,
            "meta": meta or {},
            "enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save_registry(reg)

    return {
        "status": "REENROLLED" if existed else "ENROLLED",
        "note_id": note_id,
        "fingerprint": fp,
        "registry_size": len(reg),
    }


def verify(image, note_id=None) -> dict:
    """Verify `image` against the registry.

    With `note_id`: compare to that entry -> AUTHENTIC / NO_MATCH / UNKNOWN.
    Without: return the best match in the registry (AUTHENTIC if close enough,
    else NO_MATCH). Never raises."""
    fp = compute_fingerprint(image)
    reg = _load_registry()

    result = {
        "verdict": "UNKNOWN",
        "note_id": str(note_id).strip() if note_id else None,
        "fingerprint": fp,
        "distance": None,
        "similarity": None,
        "threshold": _MATCH_MAX_DISTANCE,
        "note": "",
    }

    if not reg:
        result["note"] = "Registry is empty — enroll a note before verifying."
        return result

    try:
        if note_id and str(note_id).strip():
            nid = str(note_id).strip()
            entry = reg.get(nid)
            if entry is None:
                result["note"] = f"No note enrolled under id '{nid}'."
                return result
            dist = hamming_distance(fp, entry["fingerprint"])
            result["distance"] = round(dist, 4)
            result["similarity"] = round(1.0 - dist, 4)
            if dist <= _MATCH_MAX_DISTANCE:
                result["verdict"] = "AUTHENTIC"
                result["note"] = "Fingerprint matches the enrolled note."
            else:
                result["verdict"] = "NO_MATCH"
                result["note"] = ("Fingerprint does not match the enrolled note "
                                  "for this id.")
            return result

        # No id: search for the closest enrolled note.
        best_id, best_dist = None, 1.0
        for nid, entry in reg.items():
            d = hamming_distance(fp, entry["fingerprint"])
            if d < best_dist:
                best_id, best_dist = nid, d
        result["note_id"] = best_id
        result["distance"] = round(best_dist, 4)
        result["similarity"] = round(1.0 - best_dist, 4)
        if best_dist <= _MATCH_MAX_DISTANCE:
            result["verdict"] = "AUTHENTIC"
            result["note"] = f"Closest enrolled note: '{best_id}'."
        else:
            result["verdict"] = "NO_MATCH"
            result["note"] = "No enrolled note matches this image."
        return result
    except Exception as e:
        result["note"] = f"verification error: {e}"
        return result

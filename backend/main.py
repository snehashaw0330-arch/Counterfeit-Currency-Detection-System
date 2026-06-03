from fastapi import FastAPI, UploadFile, File, Body, Query, Response
from fastapi.middleware.cors import CORSMiddleware

import importlib

try:
    tf = importlib.import_module("tensorflow")
    load_model = tf.keras.models.load_model
except ImportError:
    from keras.models import load_model

import numpy as np
import cv2
from PIL import Image

import io
import os
import threading

from backend.forensic import (
    run_forensic_pipeline,
    warmup_ocr,
    diagnostics,
    assess_readability,
)
from backend.classical import predict_classical, warmup_classical
from backend.genai import explain as genai_explain, llm_available
from backend.security_pattern import pattern_png

# Reject absurd uploads early (DoS / accidental huge files). 25 MB
# comfortably covers a high-res phone photo.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Counterfeit Currency Detection API"
)

# =====================================================
# ENABLE CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "mobilenet_counterfeit_detector.keras"
)

print("\nMODEL PATH:", MODEL_PATH)

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(MODEL_PATH)

print("Model Loaded Successfully\n")


# =====================================================
# OCR WARMUP
# =====================================================
# EasyOCR's first call costs ~3 s to load the detector +
# recogniser weights. Running that in a background thread
# at startup means the first /predict request after launch
# doesn't pay the latency. Safe to fail silently — the
# lazy-init path inside forensic.py still works.

def _background_warmup():
    try:
        ok = warmup_ocr()
        print(f"OCR Warmup: {'OK' if ok else 'unavailable, lazy fallback active'}")
    except Exception as exc:
        print(f"OCR Warmup raised (non-fatal): {exc}")
    try:
        clf = warmup_classical()
        print(f"Classical model: {'loaded' if clf else 'not trained yet (run scripts/train_classical.py)'}")
    except Exception as exc:
        print(f"Classical warmup raised (non-fatal): {exc}")


@app.on_event("startup")
def _kickoff_warmup():
    threading.Thread(
        target=_background_warmup,
        name="ocr-warmup",
        daemon=True,
    ).start()


# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")
def home():

    return {

        "status": "success",

        "message":
        "Counterfeit Currency Detection Backend Running"
    }


# =====================================================
# IMAGE LOADING + VALIDATION
# =====================================================

def _decode_upload(image_bytes):
    """Validate and decode an upload into (rgb_array, bgr_image).

    Raises ValueError with a user-facing message on bad input so the
    endpoints can return a clean error (never a 500 / stack trace)."""

    if not image_bytes:
        raise ValueError("Empty upload — no file received")

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Image too large ({len(image_bytes) // (1024 * 1024)} MB); "
            f"maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("File is not a readable image")

    rgb_array = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return rgb_array, bgr_image


# =====================================================
# CORE ANALYSIS (shared by /predict and /diagnose)
# =====================================================

def _analyze(rgb_array, bgr_image):
    """Run the CNN + forensic pipeline + classical second opinion and
    compute the combined REAL / SUSPICIOUS / FAKE verdict. Returns the
    success payload dict. Pure logic — no request handling — so both
    /predict and /diagnose share exactly one implementation."""

    model_input = np.expand_dims(
        cv2.resize(rgb_array, (224, 224)) / 255.0, axis=0
    )
    prediction = float(model.predict(model_input)[0][0])

    # ---- model (CNN) verdict ----
    if prediction >= 0.5:
        model_verdict = "REAL"
        model_confidence = round(prediction * 100, 2)
    else:
        model_verdict = "FAKE"
        model_confidence = round((1 - prediction) * 100, 2)

    # ---- forensic pipeline ----
    forensic_analysis = run_forensic_pipeline(bgr_image)

    scored_checks = [
        c for k, c in forensic_analysis.items()
        if k != "modular_ai_pipeline" and c["status"] in ("PASS", "FAIL")
    ]
    pass_count = sum(1 for c in scored_checks if c["status"] == "PASS")
    total = max(len(scored_checks), 1)
    forensic_score = pass_count / total

    # Forensic outweighs the CNN (the classifier is brittle on
    # out-of-distribution inputs), so 0.6 / 0.4.
    combined_score = 0.4 * prediction + 0.6 * forensic_score

    # Hard gates: structural sanity (not a banknote at all), colour
    # palette (desaturated/inverted), and proportions (stretched /
    # wrong-size) each veto a REAL verdict.
    structural_failed = (
        forensic_analysis.get("structural_sanity", {}).get("status") == "FAIL"
    )
    colour_failed = (
        forensic_analysis.get("hologram_detection", {}).get("status") == "FAIL"
    )
    proportion_failed = (
        forensic_analysis.get("proportion_analysis", {}).get("status") == "FAIL"
    )

    if structural_failed:
        security_verdict = "FAKE"
    elif (
        combined_score >= 0.65
        and pass_count >= 5
        and not colour_failed
        and not proportion_failed
    ):
        security_verdict = "REAL"
    elif combined_score < 0.35 or (prediction < 0.35 and forensic_score < 0.35):
        security_verdict = "FAKE"
    else:
        security_verdict = "SUSPICIOUS"

    # ---- honesty / readability overlay ----
    # A "REAL"/"SUSPICIOUS" verdict only means something if we could
    # actually read the note. When we couldn't read ANY identity
    # signal (serial, proportions, denomination) the honest answer is
    # "can't verify — retake", NOT a confident REAL. A clear FAKE
    # (gross structural/colour/score failure) is kept, because those
    # gates don't need OCR and are honest signals on their own.
    readability = assess_readability(bgr_image, forensic_analysis)

    if security_verdict in ("REAL", "SUSPICIOUS") and readability["level"] == "none":
        final_verdict = "UNVERIFIED"
    else:
        final_verdict = security_verdict

    final_confidence = round(max(combined_score, 1 - combined_score) * 100, 2)

    # ---- classical ML second opinion (display-only) ----
    classical = predict_classical(bgr_image)
    ml_models = {
        "cnn": {
            "verdict": model_verdict,
            "confidence": f"{model_confidence:.2f}%",
            "prob_genuine": prediction,
        },
        "classical": classical,
        "agreement": (
            classical["verdict"] == model_verdict
            if classical.get("available") else None
        ),
    }

    return {
        "status": "success",
        "prediction": final_verdict,
        "security_verdict": security_verdict,
        "verification_level": readability["level"],
        "verification": readability,
        "guidance": readability["guidance"],
        "confidence": f"{final_confidence:.2f}%",
        "raw_prediction": prediction,
        "model_verdict": model_verdict,
        "model_confidence": f"{model_confidence:.2f}%",
        "forensic_score": round(forensic_score * 100, 2),
        "forensic_pass_count": pass_count,
        "forensic_total_checks": total,
        "ml_models": ml_models,
        "forensic_analysis": forensic_analysis,
    }


# =====================================================
# PREDICT CURRENCY
# =====================================================

@app.post("/predict")
async def predict_currency(file: UploadFile = File(...)):

    try:
        rgb_array, bgr_image = _decode_upload(await file.read())
        return _analyze(rgb_array, bgr_image)

    except ValueError as ve:
        # Expected, user-facing validation problems.
        return {"status": "error", "message": str(ve)}

    except Exception as e:
        print("\nERROR:", str(e))
        return {"status": "error", "message": str(e)}


# =====================================================
# DIAGNOSE (raw OCR + per-check intermediates) — Phase G
# =====================================================

@app.post("/diagnose")
async def diagnose_currency(file: UploadFile = File(...)):
    """Superset of /predict for debugging and demos: the same verdict
    plus a `diagnostics` block with the raw EasyOCR tokens and the
    located-note size, so you can see exactly what the OCR read and
    why a serial / denomination came out the way it did."""

    try:
        rgb_array, bgr_image = _decode_upload(await file.read())
        result = _analyze(rgb_array, bgr_image)
        result["diagnostics"] = diagnostics(bgr_image)
        return result

    except ValueError as ve:
        return {"status": "error", "message": str(ve)}

    except Exception as e:
        print("\nERROR:", str(e))
        return {"status": "error", "message": str(e)}


# =====================================================
# EXPLAIN (GenAI explanation & accessibility) — Phase I
# =====================================================

@app.post("/explain")
async def explain_result(payload: dict = Body(...)):
    """Take a /predict result and return a plain-language explanation plus
    manual-verification guidance. Uses Claude when ANTHROPIC_API_KEY is set,
    otherwise a deterministic template. Never fails the request."""
    try:
        return {
            "status": "success",
            "llm_available": llm_available(),
            "explanation": genai_explain(payload),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =====================================================
# SECURITY-PATTERN ART (generative) — Phase J
# =====================================================

@app.get("/security-pattern")
def security_pattern(
    seed: str = Query("0", description="Any text/number; same seed → same art"),
    size: int = Query(600, ge=128, le=1200),
):
    """Procedurally generate an ABSTRACT guilloché / rosette security-pattern
    artwork and return it as a PNG. Deterministic per seed (a seed can be
    derived from a serial number, so the pattern can be regenerated and compared
    for verification). This is generative *security-pattern art* — it does NOT
    produce currency imagery."""
    png = pattern_png(seed, size)
    return Response(content=png, media_type="image/png")

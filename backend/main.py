from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model

import numpy as np
import cv2
from PIL import Image

import io
import os
import threading

from backend.forensic import run_forensic_pipeline, warmup_ocr

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
# PREDICT CURRENCY
# =====================================================

@app.post("/predict")
async def predict_currency(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        # Decode once into a BGR cv2 image (used for forensic
        # pipeline at original resolution) and a 224x224 tensor
        # (used for the MobileNetV2 classifier).

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        rgb_array = np.array(pil_image)

        bgr_image = cv2.cvtColor(
            rgb_array,
            cv2.COLOR_RGB2BGR
        )

        # Model input

        model_input = cv2.resize(
            rgb_array,
            (224, 224)
        ) / 255.0

        model_input = np.expand_dims(model_input, axis=0)

        print(
            "Processed Shape:",
            model_input.shape
        )

        prediction = model.predict(model_input)[0][0]

        print(
            "Raw Prediction:",
            prediction
        )

        # =============================================
        # MODEL VERDICT (raw ML output)
        # =============================================

        if prediction >= 0.5:
            model_verdict = "REAL"
            model_confidence = round(prediction * 100, 2)
        else:
            model_verdict = "FAKE"
            model_confidence = round((1 - prediction) * 100, 2)

        # =============================================
        # FORENSIC PIPELINE
        # =============================================

        forensic_analysis = run_forensic_pipeline(bgr_image)

        # =============================================
        # COMBINED VERDICT
        # =============================================
        # Aggregate forensic checks. The model alone has
        # ~97% val accuracy on its training distribution
        # but is brittle on out-of-distribution photos,
        # so we cross-check against the forensic features.

        scored_checks = [
            c for k, c in forensic_analysis.items()
            if k != "modular_ai_pipeline"
            and c["status"] in ("PASS", "FAIL")
        ]

        pass_count = sum(
            1 for c in scored_checks if c["status"] == "PASS"
        )

        total = max(len(scored_checks), 1)

        forensic_score = pass_count / total

        # Forensic now weighs more than the ML output. The
        # classifier is unreliable on out-of-distribution
        # inputs (it approves pure noise and colour-inverted
        # notes at 95%+), so we trust the independent
        # forensic checks more heavily.
        combined_score = (
            0.4 * float(prediction) + 0.6 * forensic_score
        )

        # Hard gate: structural_sanity FAIL means the image
        # is not plausibly a banknote at all (blank, noise,
        # half-cropped). Override the verdict outright.
        sanity = forensic_analysis.get("structural_sanity", {})
        structural_failed = sanity.get("status") == "FAIL"

        # The colour-richness check (kept under the legacy
        # hologram_detection key for API stability) is a
        # palette-integrity signal. If it fails we have a
        # desaturated / inverted / hue-shifted print — the
        # ML model is colour-blind so we must veto REAL here.
        colour = forensic_analysis.get("hologram_detection", {})
        colour_failed = colour.get("status") == "FAIL"

        # A REAL verdict requires:
        #   - structural sanity OK
        #   - combined score >= 0.65
        #   - at least 5 of 8 forensic checks PASS
        #     (reverse-side notes naturally lack OCR/face/
        #     denomination, so requiring 6 punishes them)
        #   - colour palette intact

        if structural_failed:
            final_verdict = "FAKE"
        elif (
            combined_score >= 0.65
            and pass_count >= 5
            and not colour_failed
        ):
            final_verdict = "REAL"
        elif combined_score < 0.35 or (
            float(prediction) < 0.35 and forensic_score < 0.35
        ):
            final_verdict = "FAKE"
        else:
            final_verdict = "SUSPICIOUS"

        final_confidence = round(
            max(combined_score, 1 - combined_score) * 100,
            2
        )

        # =============================================
        # FINAL RESPONSE
        # =============================================

        return {

            "status": "success",

            "prediction": final_verdict,

            "confidence": f"{final_confidence:.2f}%",

            "raw_prediction": float(prediction),

            "model_verdict": model_verdict,

            "model_confidence": f"{model_confidence:.2f}%",

            "forensic_score": round(forensic_score * 100, 2),

            "forensic_pass_count": pass_count,

            "forensic_total_checks": total,

            "forensic_analysis": forensic_analysis
        }

    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )

        return {

            "status": "error",

            "message":
            str(e)
        }


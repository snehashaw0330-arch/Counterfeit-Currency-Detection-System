from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model

import numpy as np
import cv2
from PIL import Image

import io
import os
import webbrowser

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
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image_bytes):

    # Convert image bytes to PIL image

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Convert image to numpy array

    image = np.array(image)

    # Resize image

    image = cv2.resize(
        image,
        (224, 224)
    )

    # Normalize image

    image = image / 255.0

    # Add batch dimension

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# =====================================================
# PREDICT CURRENCY
# =====================================================

@app.post("/predict")
async def predict_currency(
    file: UploadFile = File(...)
):

    try:

        # Read uploaded image

        image_bytes = await file.read()

        # Preprocess image

        processed_image = preprocess_image(
            image_bytes
        )

        print(
            "Processed Shape:",
            processed_image.shape
        )

        # Predict

        prediction = model.predict(
            processed_image
        )[0][0]

        print(
            "Raw Prediction:",
            prediction
        )

        # =============================================
        # PREDICTION LOGIC
        # =============================================

        if prediction >= 0.5:

            result = "REAL"

            confidence = round(
                prediction * 100,
                2
            )

        else:

            result = "FAKE"

            confidence = round(
                (1 - prediction) * 100,
                2
            )

        # =============================================
        # FEATURE PLACEHOLDERS
        # =============================================

        forensic_analysis = {

            "uv_light_detection":
            "Pending",

            "watermark_detection":
            "Pending",

            "ocr_serial_number":
            "Pending",

            "gandhi_face_analysis":
            "Pending",

            "security_thread_detection":
            "Pending",

            "hologram_detection":
            "Pending",

            "denomination_classification":
            "Pending",

            "modular_ai_pipeline":
            "Active"
        }

        # =============================================
        # FINAL RESPONSE
        # =============================================

        return {

            "status": "success",

            "prediction":
            result,

            "confidence":
            f"{confidence:.2f}%",

            "raw_prediction":
            float(prediction),

            "forensic_analysis":
            forensic_analysis
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

# =====================================================
# AUTO OPEN SWAGGER DOCS
# =====================================================

webbrowser.open(
    "http://127.0.0.1:8000/docs"
)
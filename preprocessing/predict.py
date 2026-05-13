import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ============================================
# GET PROJECT ROOT PATH
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# MODEL PATH
# ============================================

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "mobilenet_counterfeit_detector.keras"
)

print("Model Path:", model_path)

# ============================================
# LOAD MODEL
# ============================================

model = load_model(model_path)

print("Model Loaded Successfully")

# ============================================
# IMAGE SETTINGS
# ============================================

IMG_SIZE = 224

# ============================================
# PREPROCESS IMAGE
# ============================================

def preprocess_image(image_path):

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize image
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ============================================
# UV LIGHT DETECTION PLACEHOLDER
# ============================================

def uv_light_detection():

    # Future UV detection module
    # Requires UV dataset / UV camera images

    return {
        "uv_feature_detected": "Not Implemented Yet"
    }

# ============================================
# WATERMARK DETECTION PLACEHOLDER
# ============================================

def watermark_detection():

    # Future watermark detection module

    return {
        "watermark_detected": "Not Implemented Yet"
    }

# ============================================
# SECURITY THREAD DETECTION PLACEHOLDER
# ============================================

def security_thread_detection():

    # Future security thread detection module

    return {
        "security_thread_detected": "Not Implemented Yet"
    }

# ============================================
# GANDHI FACE ANALYSIS PLACEHOLDER
# ============================================

def gandhi_face_analysis():

    # Future Gandhi face analysis module

    return {
        "gandhi_face_match": "Not Implemented Yet"
    }

# ============================================
# OCR SERIAL NUMBER PLACEHOLDER
# ============================================

def serial_number_ocr():

    # Future OCR module

    return {
        "serial_number": "Not Implemented Yet"
    }

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_currency(image_path):

    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Model prediction
    prediction = model.predict(processed_image)[0][0]

    print("\nRaw Prediction Score:", prediction)

    # Prediction Logic
    if prediction >= 0.5:

        result = "REAL"
        confidence = round(prediction * 100, 2)

    else:

        result = "FAKE"
        confidence = round((1 - prediction) * 100, 2)

    # ============================================
    # MODULAR FEATURE PIPELINE
    # ============================================

    uv_result = uv_light_detection()

    watermark_result = watermark_detection()

    security_thread_result = security_thread_detection()

    gandhi_result = gandhi_face_analysis()

    serial_result = serial_number_ocr()

    # ============================================
    # FINAL OUTPUT
    # ============================================

    print("\n========== FINAL RESULT ==========")

    print("Prediction:", result)

    print("Confidence:", confidence, "%")

    print("\n----- FORENSIC ANALYSIS -----")

    print("UV Analysis:", uv_result)

    print("Watermark Analysis:", watermark_result)

    print("Security Thread Analysis:", security_thread_result)

    print("Gandhi Face Analysis:", gandhi_result)

    print("OCR Serial Analysis:", serial_result)

    print("=================================")

# ============================================
# CONTINUOUS PREDICTION LOOP
# ============================================

while True:

    image_path = input("\nEnter image path (or type exit): ").strip('"')

    if image_path.lower() == "exit":

        print("Program Closed")

        break

    try:

        predict_currency(image_path)

    except Exception as e:

        print("Error:", str(e))
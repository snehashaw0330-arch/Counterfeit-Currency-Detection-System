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
    "counterfeit_currency_detector.keras"
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

IMG_SIZE = 128

# ============================================
# PREPROCESS IMAGE
# ============================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0

    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    return img

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_currency(image_path):

    processed_image = preprocess_image(image_path)

    prediction = model.predict(processed_image)[0][0]

    print("\nPrediction Score:", prediction)

    if prediction >= 0.5:
        print("Currency is REAL")
        print("Confidence:", round(prediction * 100, 2), "%")

    else:
        print("Currency is FAKE")
        print("Confidence:", round((1 - prediction) * 100, 2), "%")

# ============================================
# CONTINUOUS PREDICTION LOOP
# ============================================

while True:

    image_path = input("\nEnter image path (or type exit): ").strip('"')

    if image_path.lower() == "exit":
        print("Program Closed")
        break

    predict_currency(image_path)
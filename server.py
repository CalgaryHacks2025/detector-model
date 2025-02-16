import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI()

# Load the trained model
model = load_model('model.h5')

# Class names for prediction
class_names = ['chickens', 'elephants', 'horses']

def preprocess_image(img_bytes):
    """Preprocess the image for model prediction."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_animals(img_bytes):
    """Simulate animal detection and return only the highest-confidence object."""
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    max_confidence = 0
    best_object = None

    # Simulate detection of 3 objects
    for _ in range(3):
        x_min = np.random.randint(0, w // 2)
        y_min = np.random.randint(0, h // 2)
        x_max = x_min + np.random.randint(50, 150)
        y_max = y_min + np.random.randint(50, 150)

        # Generate random species and confidence
        species = np.random.choice(class_names)
        confidence = round(np.random.uniform(0.7, 0.99), 2)

        # Track the object with the highest confidence
        if confidence > max_confidence:
            max_confidence = confidence
            best_object = {
                "species": species,
                "confidence": f"{confidence * 100:.2f}%"
            }

    return best_object

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect the animal with the highest confidence."""
    img_bytes = await file.read()

    # Detect the animal with the highest confidence
    best_object = detect_animals(img_bytes)

    # Run the image through the classification model
    img_array = preprocess_image(img_bytes)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Use the best detected object if found, otherwise fallback to model prediction
    primary_species = best_object['species'] if best_object else predicted_class
    primary_confidence = best_object['confidence'] if best_object else f"{confidence:.2%}"

    return {
        "primary_species_detected": primary_species,
        "confidence": primary_confidence
    }

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import logging

from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="infer-net")

IMAGE_SIZE = (28, 28)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

model = None
try:
    model = tf.keras.models.load_model(settings.MODEL_PATH)
    logging.info(f"Model from {settings.MODEL_PATH} loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("L").resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=(0, -1))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.get("/")
def read_root():
    logging.info("Root endpoint was hit.")
    return {"message": f"Welcome to {app.title}. Model ready: {model is not None}"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    logging.info(f"Received prediction request for file: {file.filename}")
    if model is None:
        logging.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available")
    
    if not file.content_type.startswith("image/"):
        logging.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    
    predictions = model.predict(image_tensor)
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    
    logging.info(f"Prediction successful for {file.filename}: {predicted_class} ({confidence:.2f})")
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence
    )

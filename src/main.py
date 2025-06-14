from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = FastAPI(title="infer-net")
MODEL_PATH = "artifacts/infer-net.keras"
IMAGE_SIZE = (28, 28)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model from {MODEL_PATH} loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    image_url: HttpUrl

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

def preprocess_image(image_url: str) -> np.ndarray:
    """Downloads, resizes, and preprocesses an image for the model."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        
        image = image.convert("L").resize(IMAGE_SIZE)
        image_array = np.array(image)
        
        image_array = image_array.astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=(0, -1))
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.get("/")
def read_root():
    """Confirms the API is running."""
    return {"message": f"Welcome to {app.title}. Model ready: {model is not None}"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Accepts an image URL and returns a real prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    
    image_tensor = preprocess_image(str(request.image_url))
    
    predictions = model.predict(image_tensor)
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence
    )

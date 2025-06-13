from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="infer-net")

class PredictionRequest(BaseModel):
    image_url: HttpUrl

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts an image URL and returns a mocked prediction.
    In a real application, this is where the model would be called.
    """
    return PredictionResponse(
        predicted_class="cat",
        confidence=0.95
    )

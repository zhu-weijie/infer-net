from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from io import BytesIO


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app, preprocess_image

client = TestClient(app)


def test_preprocess_image():
    """Tests the image preprocessing helper function."""
    dummy_image = Image.new("L", (10, 10), color="black")

    byte_io = BytesIO()
    dummy_image.save(byte_io, "PNG")
    image_bytes = byte_io.getvalue()

    processed_tensor = preprocess_image(image_bytes)

    assert isinstance(processed_tensor, np.ndarray)
    assert processed_tensor.shape == (1, 28, 28, 1)


def test_predict_endpoint_success():
    """Tests the /predict endpoint with a valid image file."""
    dummy_image = Image.new("L", (28, 28), color="white")
    byte_io = BytesIO()
    dummy_image.save(byte_io, "PNG")
    byte_io.seek(0)

    response = client.post(
        "/predict", files={"file": ("test_image.png", byte_io, "image/png")}
    )

    assert response.status_code == 200
    response_json = response.json()
    assert "predicted_class" in response_json
    assert "confidence" in response_json


def test_predict_endpoint_invalid_file_type():
    """Tests the /predict endpoint with a non-image file."""
    text_file = BytesIO(b"this is not an image")

    response = client.post(
        "/predict", files={"file": ("test.txt", text_file, "text/plain")}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File provided is not an image."}

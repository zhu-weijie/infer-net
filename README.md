# infer-net: Dockerized Image Classification API

[![CI Status](https://github.com/zhu-weijie/infer-net/actions/workflows/ci.yml/badge.svg)](https://github.com/zhu-weijie/infer-net/actions/workflows/ci.yml)

A production-style, end-to-end machine learning application that serves an image classification model through a containerized FastAPI web service.

This project was built to demonstrate best practices in software engineering applied to the ML lifecycle, including a Docker-first approach, automated CI/CD with GitHub Actions, configuration management, and automated testing.

## Key Features

*   **FastAPI Backend**: A high-performance API built with FastAPI.
*   **Dockerized Service**: The entire application is containerized with Docker and orchestrated with Docker Compose for easy setup and deployment.
*   **ML Model Integration**: Serves a TensorFlow/Keras CNN model trained on the Fashion MNIST dataset.
*   **Direct Image Upload**: A robust `/predict` endpoint that accepts direct image file uploads.
*   **CI/CD Pipeline**: Automated workflow using GitHub Actions for:
    *   **Linting** with `ruff`.
    *   **Testing** with `pytest`.
    *   **Build Validation** of the Docker image.
*   **Configuration Management**: Externalized configuration for maintainability (model path is set via environment variables).
*   **Automated Testing**: Unit and integration tests to ensure API reliability.

## Technology Stack

*   **Backend**: Python, FastAPI
*   **ML Framework**: TensorFlow / Keras
*   **Containerization**: Docker, Docker Compose
*   **CI/CD**: GitHub Actions
*   **Testing**: Pytest
*   **Linting**: Ruff

---

## Getting Started

### Prerequisites

*   [Docker](https://www.docker.com/get-started) and Docker Compose installed on your local machine.
*   A trained model file (`infer-net.keras`) available. For CI, this is handled via GitHub Releases.

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zhu-weijie/infer-net.git
    cd infer-net
    ```

2.  **Place the model file:**
    *   Create a directory `artifacts`.
    *   Place your trained `infer-net.keras` model file inside it.

3.  **Build and run the container:**
    Use Docker Compose to build the image and start the service. This command will start the API on `http://localhost:8000`.
    ```bash
    docker-compose up --build
    ```

---

## How to Use the API

Once the application is running, you can access the interactive API documentation at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### Prediction Endpoint

You can send a `POST` request to the `/predict` endpoint with an image file to get a classification.

**`POST /predict`**

#### Example with `curl`

Use `curl` to send an image file and receive a prediction.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -F 'file=@/path/to/your/image.png;type=image/png'
```

Example Response:

```json
{
  "predicted_class": "Ankle boot",
  "confidence": 0.9985
}
```

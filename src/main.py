from fastapi import FastAPI

app = FastAPI(title="ML Classifier API")

@app.get("/")
def read_root():
    return {"message": "API is running"}

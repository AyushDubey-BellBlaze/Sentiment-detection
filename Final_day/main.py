from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict_emotions

app = FastAPI(
    title="Emotion Analysis API",
    description="Detects 30+ emotions from text using a trained ML model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Emotion Analysis API is running!", "version": "1.0.0"}

@app.post("/analyze")
def analyze(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = predict_emotions(input.text)
    return result

@app.get("/emotions")
def list_emotions():
    from model import EMOTIONS
    return {"total": len(EMOTIONS), "emotions": EMOTIONS}

print("Go and search on browser: http://127.0.0.1:8000/ \n and http://127.0.0.1:8000/docs")
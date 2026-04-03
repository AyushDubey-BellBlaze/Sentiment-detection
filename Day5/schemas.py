from pydantic import BaseModel
from typing import List

class TextRequest(BaseModel):
    text: str

class EmotionScore(BaseModel):
    label: str
    score: float

class EmotionResponse(BaseModel):
    top_emotion: str
    confidence: float
    detected_emotions: List[str]
    all_emotions: List[EmotionScore]
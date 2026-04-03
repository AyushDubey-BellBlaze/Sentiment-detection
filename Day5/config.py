import os

class Settings:
    PROJECT_NAME: str = "Advanced Emotion AI API"
    VERSION: str = "3.0"

    MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"

    # Threshold for multi-label detection
    EMOTION_THRESHOLD: float = 0.15

    # Max text length
    MAX_TEXT_LENGTH: int = 1000

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()
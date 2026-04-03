#!/bin/bash
# setup_and_run.sh — Run this once to set up and launch the API

echo "========================================="
echo "   Emotion Analysis API — Setup Script   "
echo "========================================="

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Training the emotion model..."
python model.py

echo ""
echo "[3/3] Starting the API server..."
echo "  API running at: http://127.0.0.1:8000"
echo "  Swagger docs:   http://127.0.0.1:8000/docs"
echo "  Press CTRL+C to stop."
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000
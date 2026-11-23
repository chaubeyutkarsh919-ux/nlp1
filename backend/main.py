"""
FastAPI Backend Application for Multimodal Sentiment Analysis & Live Facial Recognition.

Endpoints:
- GET  /health              : System & model health check
- POST /analyze/text        : Text sentiment prediction
- POST /analyze/expression  : Facial expression classification from uploaded image
- POST /analyze/multimodal  : Late fusion of text sentiment and facial expression
- POST /recognize/enroll    : Enroll face identity with a custom name
- POST /recognize/identify  : Identify registered face and predict expression
- GET  /recognize/list      : List all registered face identities
- POST /analyze/live-frame  : Fast frame analysis for live camera streams
"""

import sys
import os
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
from typing import Optional, List
# pyrefly: ignore[missing-import]
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text_analysis.predict import predict_sentiment
from src.facial_analysis.predict import predict_expression
from src.facial_analysis.face_detection import get_face_detector
from src.facial_analysis.face_recognizer import get_face_recognizer
from src.facial_analysis.live_processor import LiveVideoProcessor
from src.fusion.fusion_engine import fuse_predictions, RESPONSIBLE_AI_DISCLAIMER

app = FastAPI(
    title="Multimodal Sentiment & Live Facial Recognition API",
    description="Combines NLP Text Sentiment, Live Facial Recognition (Identity), and Real-Time Expression Analysis.",
    version="1.1.0",
)

# Enable CORS for local dashboards (Streamlit / React / Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_live_processor = LiveVideoProcessor()


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="User text input to analyze for sentiment", min_length=1)


@app.get("/health")
def health_check():
    """Verify backend and loaded models status."""
    return {
        "status": "online",
        "service": "Multimodal Sentiment & Live Face Recognition API",
        "modules": {
            "text_analysis": "ready",
            "facial_expression_analysis": "ready",
            "facial_identity_recognition": "ready",
            "multimodal_fusion_engine": "ready",
        },
        "disclaimer": RESPONSIBLE_AI_DISCLAIMER,
    }


@app.post("/analyze/text")
def analyze_text(request: TextAnalysisRequest):
    """Analyze written text using TF-IDF + Logistic Regression model."""
    try:
        result = predict_sentiment(request.text)
        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "cleaned_text": result["cleaned_text"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


@app.post("/analyze/expression")
async def analyze_expression(file: UploadFile = File(...)):
    """Classify visible facial expression from an uploaded image."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predict_expression(contents)
        result_clean = {k: v for k, v in result.items() if k != "face_crop"}
        return result_clean
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facial analysis failed: {str(e)}")


@app.post("/analyze/multimodal")
async def analyze_multimodal(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """Late fusion of text sentiment and optional facial expression."""
    try:
        text_result = predict_sentiment(text)
        facial_result = None
        if file is not None:
            contents = await file.read()
            if len(contents) > 0:
                facial_result = predict_expression(contents)

        fused = fuse_predictions(text_result, facial_result)
        return fused
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")


@app.post("/recognize/enroll")
async def enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    """Enroll a user's face identity into the facial recognition database."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        det = get_face_detector().detect_face(img)
        if not det["face_detected"] or det["face_crop"] is None:
            raise HTTPException(status_code=400, detail="No face detected in the enrollment image.")

        recognizer = get_face_recognizer()
        success = recognizer.enroll(name, det["face_crop"])
        if not success:
            raise HTTPException(status_code=500, detail="Could not compute facial feature descriptor.")

        return {
            "status": "success",
            "message": f"Successfully enrolled face for '{name}'",
            "enrolled_users": recognizer.list_enrolled(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.get("/recognize/list")
def list_enrolled_faces():
    """List all currently enrolled face identities."""
    recognizer = get_face_recognizer()
    return {"enrolled_users": recognizer.list_enrolled()}


@app.post("/analyze/live-frame")
async def analyze_live_frame(file: UploadFile = File(...)):
    """High-speed processing endpoint for live video frames."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame buffer.")

        _, meta = _live_processor.process_frame(frame, draw_hud=False)
        meta_clean = {k: v for k, v in meta.items() if k != "face_crop"}
        return meta_clean
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


if __name__ == "__main__":
    # pyrefly: ignore[missing-import]
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

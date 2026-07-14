import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from predict_v2 import predict

# Initialize FastAPI application
app = FastAPI(
    title="Regional SMS Phishing Detector API",
    description="Backend API for detecting regional language SMS scams in Hindi, Hinglish, and English.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Define request schema
class PredictionRequest(BaseModel):
    message: str = Field(..., description="The raw SMS message text to analyze.")
    model: str = Field("transformer", description="Model to use: 'baseline' (TF-IDF + LR) or 'transformer' (MuRIL).")
    sender: Optional[str] = Field(None, description="Optional sender ID (e.g., AD-SBI or phone number).")

# Define response schema
class PredictionResponse(BaseModel):
    label: str
    confidence: float
    decision_threshold: float
    top_triggering_terms: list[str]
    model_used: str

@app.post("/api/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest):
    """
    Main prediction endpoint that routes query to selected classifier path.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message body cannot be empty.")
        
    if request.model not in ["baseline", "transformer"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Select 'baseline' or 'transformer'.")
        
    try:
        result = predict(
            message=request.message, 
            model_type=request.model, 
            sender=request.sender
        )
        return PredictionResponse(
            label=result["label"],
            confidence=float(result["confidence"]),
            decision_threshold=float(result["decision_threshold"]),
            top_triggering_terms=result["top_triggering_terms"],
            model_used=result["model_used"]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Mount static files at root AFTER api endpoints to avoid route shadowing
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Run server locally on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

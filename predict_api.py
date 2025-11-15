"""
FastAPI server for dual model predictions
- Model 1: Question Type Classifier (Terbuka, Sugestif, Tertutup, Reflektif)
- Model 2: Empathy Classifier (Empatik, Netral, Judgemental)

Run: uvicorn predict_api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

app = FastAPI(
    title="Counseling Dual Classifier API",
    description="API for classifying questions and empathy levels",
    version="2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
QUESTION_MODEL_PATH = os.path.join(BASE_DIR, "indobert_counseling_v2")
EMPATHY_MODEL_PATH = os.path.join(BASE_DIR, "empathy_model2_finetuned")

MAX_LENGTH = 128

# Label mappings
QUESTION_LABELS = {0: 'Terbuka', 1: 'Sugestif', 2: 'Tertutup', 3: 'Reflektif'}
EMPATHY_LABELS = {0: 'Empatik', 1: 'Netral', 2: 'Judgemental'}

# Global variables
question_tokenizer = None
question_model = None
empathy_tokenizer = None
empathy_model = None
device = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1)
    
class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify")

class DualPredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify with both models")

class DualPredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts for dual classification")

class QuestionResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

class EmpathyResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

class DualPredictionResult(BaseModel):
    text: str
    question_type: QuestionResult
    empathy_level: EmpathyResult
    processing_time_ms: float

class DualPredictResponse(BaseModel):
    success: bool
    data: DualPredictionResult

class DualPredictBatchResponse(BaseModel):
    success: bool
    data: List[DualPredictionResult]
    total_items: int
    total_processing_time_ms: float

# ============================================================================
# MODEL LOADING
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load both models on startup"""
    global question_tokenizer, question_model, empathy_tokenizer, empathy_model, device
    
    print("="*70)
    print("Loading models...")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load Question Type Model
    try:
        print("\n1. Loading Question Type Model...")
        question_tokenizer = AutoTokenizer.from_pretrained(QUESTION_MODEL_PATH)
        question_model = AutoModelForSequenceClassification.from_pretrained(QUESTION_MODEL_PATH)
        question_model.to(device)
        question_model.eval()
        print("   ✓ Question Type Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load Question Type Model: {e}")
        question_model = None
    
    # Load Empathy Model
    try:
        print("\n2. Loading Empathy Model...")
        empathy_tokenizer = AutoTokenizer.from_pretrained(EMPATHY_MODEL_PATH)
        empathy_model = AutoModelForSequenceClassification.from_pretrained(EMPATHY_MODEL_PATH)
        empathy_model.to(device)
        empathy_model.eval()
        print("   ✓ Empathy Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load Empathy Model: {e}")
        empathy_model = None
    
    print("\n" + "="*70)
    print("Models loaded!")
    print("="*70 + "\n")

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_question_type(text: str) -> QuestionResult:
    """Predict question type"""
    if question_model is None:
        raise HTTPException(status_code=503, detail="Question model not loaded")
    
    inputs = question_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = question_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    all_probs = {
        QUESTION_LABELS[i]: float(probs[0][i])
        for i in range(len(QUESTION_LABELS))
    }
    
    return QuestionResult(
        label=QUESTION_LABELS[predicted_class],
        label_id=predicted_class,
        confidence=confidence,
        probabilities=all_probs
    )

def predict_empathy(text: str) -> EmpathyResult:
    """Predict empathy level"""
    if empathy_model is None:
        raise HTTPException(status_code=503, detail="Empathy model not loaded")
    
    inputs = empathy_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = empathy_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    all_probs = {
        EMPATHY_LABELS[i]: float(probs[0][i])
        for i in range(len(EMPATHY_LABELS))
    }
    
    return EmpathyResult(
        label=EMPATHY_LABELS[predicted_class],
        label_id=predicted_class,
        confidence=confidence,
        probabilities=all_probs
    )

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Counseling Dual Classifier",
        "version": "2.0",
        "status": "running",
        "models": {
            "question_type": {
                "loaded": question_model is not None,
                "labels": list(QUESTION_LABELS.values())
            },
            "empathy": {
                "loaded": empathy_model is not None,
                "labels": list(EMPATHY_LABELS.values())
            }
        },
        "endpoints": {
            "dual_predict": "/predict-dual",
            "dual_predict_batch": "/predict-dual-batch",
            "question_only": "/predict-question",
            "empathy_only": "/predict-empathy",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "question_type": question_model is not None,
            "empathy": empathy_model is not None
        },
        "device": str(device) if device else "not initialized"
    }

@app.post("/predict-dual", response_model=DualPredictResponse)
async def predict_dual(request: DualPredictRequest):
    """Predict both question type and empathy level"""
    start_time = time.time()
    
    try:
        # Predict question type
        question_result = predict_question_type(request.text)
        
        # Predict empathy
        empathy_result = predict_empathy(request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = DualPredictionResult(
            text=request.text,
            question_type=question_result,
            empathy_level=empathy_result,
            processing_time_ms=processing_time
        )
        
        return DualPredictResponse(success=True, data=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-dual-batch", response_model=DualPredictBatchResponse)
async def predict_dual_batch(request: DualPredictBatchRequest):
    """Predict both models for multiple texts"""
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")
    
    start_time = time.time()
    results = []
    
    try:
        for text in request.texts:
            text_start = time.time()
            
            # Predict question type
            question_result = predict_question_type(text)
            
            # Predict empathy
            empathy_result = predict_empathy(text)
            
            text_time = (time.time() - text_start) * 1000
            
            results.append(DualPredictionResult(
                text=text,
                question_type=question_result,
                empathy_level=empathy_result,
                processing_time_ms=text_time
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return DualPredictBatchResponse(
            success=True,
            data=results,
            total_items=len(results),
            total_processing_time_ms=total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-question")
async def predict_question_only(request: PredictRequest):
    """Predict question type only"""
    start_time = time.time()
    
    try:
        result = predict_question_type(request.text)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "data": {
                "text": request.text,
                **result.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-empathy")
async def predict_empathy_only(request: PredictRequest):
    """Predict empathy level only"""
    start_time = time.time()
    
    try:
        result = predict_empathy(request.text)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "data": {
                "text": request.text,
                **result.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
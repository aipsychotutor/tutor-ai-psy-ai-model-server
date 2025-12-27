"""
FastAPI server gabungan:
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import torch
from transformers import AutoTokenizer, BertForSequenceClassification 
from ultralytics import YOLO
from PIL import Image
import time
import os
import io, json, base64
from collections import defaultdict
import uuid

# ======================================================================
# CONFIG & GLOBALS
# ======================================================================

# Perbaikan: Pastikan nama repo sesuai dengan Hugging Face (biasanya pakai strip '-')
QUESTION_MODEL_PATH = "bun1110/question_type" 
EMPATHY_MODEL_PATH = "bun1110/empathy"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "face_emotion_detection", "best.pt")

MAX_LENGTH = 128

QUESTION_LABELS = {0: 'Terbuka', 1: 'Sugestif', 2: 'Tertutup', 3: 'Reflektif'}
EMPATHY_LABELS = {0: 'Empatik', 1: 'Netral', 2: 'Judgemental'}

# Global vars
question_tokenizer = None
question_model = None
empathy_tokenizer = None
empathy_model = None
device = None
yolo_model = None

# ======================================================================
# LIFESPAN
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global question_tokenizer, question_model
    global empathy_tokenizer, empathy_model
    global device, yolo_model

    print("="*70)
    print("Loading Models (Force BERT Architecture)...")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device NLP: {device}")

    # 1. Question Type Model
    try:
        print(f"\n1. Downloading/Loading Question Model from: {QUESTION_MODEL_PATH}")
        question_tokenizer = AutoTokenizer.from_pretrained(QUESTION_MODEL_PATH)
        # Force BERT & ignore size mismatch if config label count is wrong
        question_model = BertForSequenceClassification.from_pretrained(
            QUESTION_MODEL_PATH,
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        question_model.to(device)
        question_model.eval()
        print("   ✓ Question Type Model loaded (BERT)")
    except Exception as e:
        print("   ✗ Failed to load Question Type Model:", e)
        question_model = None

    # 2. Empathy Model
    try:
        print(f"\n2. Downloading/Loading Empathy Model from: {EMPATHY_MODEL_PATH}")
        empathy_tokenizer = AutoTokenizer.from_pretrained(EMPATHY_MODEL_PATH)
        # Force BERT & ignore size mismatch
        empathy_model = BertForSequenceClassification.from_pretrained(
            EMPATHY_MODEL_PATH,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        empathy_model.to(device)
        empathy_model.eval()
        print("   ✓ Empathy Model loaded (BERT)")
    except Exception as e:
        print("   ✗ Failed to load Empathy Model:", e)
        empathy_model = None

    # 3. YOLO Model
    try:
        print(f"\n3. Loading YOLO Model from local: {YOLO_MODEL_PATH}")
        if os.path.exists(YOLO_MODEL_PATH):
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("   ✓ YOLO Model loaded")
        else:
            print(f"   ✗ File not found: {YOLO_MODEL_PATH}")
            yolo_model = None
    except Exception as e:
        print("   ✗ Failed to load YOLO model:", e)
        yolo_model = None

    print("\n" + "="*70)
    yield
    print("Shutting down...")

# ======================================================================
# APP SETUP
# ======================================================================

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# MODELS
# ======================================================================

class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1)

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

# ======================================================================
# LOGIC
# ======================================================================

def predict_question_type(text: str) -> QuestionResult:
    if question_model is None:
        raise HTTPException(status_code=503, detail="Question model not loaded")

    inputs = question_tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = question_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    all_probs = {QUESTION_LABELS[i]: float(probs[0][i]) for i in range(len(QUESTION_LABELS))}
    return QuestionResult(label=QUESTION_LABELS[predicted_class], label_id=predicted_class, confidence=confidence, probabilities=all_probs)

def predict_empathy(text: str) -> EmpathyResult:
    if empathy_model is None:
        raise HTTPException(status_code=503, detail="Empathy model not loaded")

    inputs = empathy_tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = empathy_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    all_probs = {EMPATHY_LABELS[i]: float(probs[0][i]) for i in range(len(EMPATHY_LABELS))}
    return EmpathyResult(label=EMPATHY_LABELS[predicted_class], label_id=predicted_class, confidence=confidence, probabilities=all_probs)

# ======================================================================
# ENDPOINTS
# ======================================================================

@app.post("/predict-dual", response_model=DualPredictResponse)
async def predict_dual(request: DualPredictRequest):
    start_time = time.time()
    try:
        q_res = predict_question_type(request.text)
        e_res = predict_empathy(request.text)
        return DualPredictResponse(success=True, data=DualPredictionResult(text=request.text, question_type=q_res, empathy_level=e_res, processing_time_ms=(time.time() - start_time) * 1000))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- BAGIAN INI YANG SEBELUMNYA HILANG ---
@app.post("/predict-dual-batch", response_model=DualPredictBatchResponse)
async def predict_dual_batch(request: DualPredictBatchRequest):
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")

    start_time = time.time()
    results = []
    try:
        for text in request.texts:
            t_start = time.time()
            q_res = predict_question_type(text)
            e_res = predict_empathy(text)
            t_time = (time.time() - t_start) * 1000
            results.append(
                DualPredictionResult(
                    text=text,
                    question_type=q_res,
                    empathy_level=e_res,
                    processing_time_ms=t_time
                )
            )
        total_time = (time.time() - start_time) * 1000
        return DualPredictBatchResponse(
            success=True,
            data=results,
            total_items=len(results),
            total_processing_time_ms=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ----------------------------------------

@app.post("/predict-question")
async def predict_question_only(request: PredictRequest):
    start_time = time.time()
    try:
        res = predict_question_type(request.text)
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": True,
            "data": {
                "text": request.text,
                **res.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-empathy")
async def predict_empathy_only(request: PredictRequest):
    start_time = time.time()
    try:
        res = predict_empathy(request.text)
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": True,
            "data": {
                "text": request.text,
                **res.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
# WEBSOCKET
# ======================================================================

@app.websocket("/ws")
async def ws_yolo(websocket: WebSocket):
    await websocket.accept()
    if yolo_model is None:
        await websocket.close()
        return
    session_id = str(uuid.uuid4())
    label_counts = defaultdict(int)
    total_frames = 0
    try:
        while True:
            text_data = await websocket.receive_text()
            msg = json.loads(text_data)
            if msg["type"] == "frame":
                header, encoded = msg["image"].split(",", 1)
                img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                r = yolo_model.predict(img, conf=0.5, imgsz=640)[0]
                for box in r.boxes:
                    label_counts[yolo_model.names[int(box.cls[0])]] += 1
                total_frames += 1
            elif msg["type"] == "finish":
                await websocket.send_text(json.dumps({"type": "summary", "data": {"session_id": session_id, "total_frames": total_frames, "labels": [{"label": k, "count": v} for k, v in label_counts.items()]}}))
                await websocket.close()
                break
    except:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

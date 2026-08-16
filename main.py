import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# Constants & Paths
# ======================
MAX_SEQ_LEN = 300
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts"

# Global dictionary lưu trữ các tài nguyên nặng
artifacts = {}

# ======================
# Lifespan Management
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model và artifacts khi ứng dụng khởi chạy."""
    try:
        # Sử dụng legacy_h5_format để bỏ qua xung đột quantization_config của Keras 3
        model_path = MODEL_DIR / "cnn_model.h5"
        artifacts["model"] = legacy_h5_format.load_model_from_hdf5(
            str(model_path), custom_objects=None, compile=False
        )
        
        with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
            artifacts["tokenizer"] = pickle.load(f)

        with open(MODEL_DIR / "label_map.pkl", "rb") as f:
            artifacts["index_to_label"] = pickle.load(f)

        artifacts["rules"] = pd.read_pickle(MODEL_DIR / "apriori_rules.pkl")
        print("Successfully loaded all model artifacts.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        raise e

    yield

    # Clean up khi ứng dụng dừng
    artifacts.clear()

# ======================
# Inference Logic
# ======================
def predict_text(text: str) -> dict:
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]
    index_to_label = artifacts["index_to_label"]
    rules = artifacts["rules"]

    # Preprocess
    tokenized_text = ViTokenizer.tokenize(text.lower())
    seq = tokenizer.texts_to_sequences([tokenized_text])
    pad = pad_sequences(
        seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )

    # Fast TensorFlow inference
    predictions = model(pad, training=False)
    probs = predictions.numpy()[0]

    cnn_idx = int(np.argmax(probs))
    cnn_label = index_to_label[cnn_idx]
    cnn_conf = float(probs[cnn_idx])

    # Rule-based Fallback (Apriori)
    if cnn_conf < 0.85:
        words = set(tokenized_text.split())
        for _, row in rules.iterrows():
            if set(row["antecedents"]).issubset(words):
                rule_label = list(row["consequents"])[0].replace("L__", "")
                if row["confidence"] > 0.9:
                    return {
                        "label": rule_label,
                        "method": "Apriori",
                        "confidence": float(row["confidence"]),
                    }

    return {"label": cnn_label, "method": "CNN", "confidence": cnn_conf}

# ======================
# FastAPI App & Schemas
# ======================
app = FastAPI(title="Text Classification Service", lifespan=lifespan)

class NewsRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Thủ tướng phát biểu tại hội nghị kinh tế.")

class NewsResponse(BaseModel):
    label: str
    method: str
    confidence: float

@app.get("/health")
def health_check():
    """Endpoint kiểm tra trạng thái dịch vụ."""
    return {"status": "ok", "model_loaded": "model" in artifacts}

@app.post("/predict", response_model=NewsResponse)
def predict(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text standard cannot be empty.")
    
    return predict_text(request.text)

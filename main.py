from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from pyvi import ViTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 300
MODEL_DIR = "model_artifacts"

# Khai báo các biến toàn cục lưu trữ mô hình
model = None
tokenizer = None
index_to_label = None
rules = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo và load mô hình khi API bắt đầu chạy
    global model, tokenizer, index_to_label, rules
    print("Đang load mô hình và các artifacts...")
    
    try:
        model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn_model.keras", compile=False)
        
        with open(f"{MODEL_DIR}/tokenizer.pickle", "rb") as file:
            tokenizer = pickle.load(file)
            
        with open(f"{MODEL_DIR}/label_map.pickle", "rb") as file:
            index_to_label = pickle.load(file)
            
        rules = pd.read_pickle(f"{MODEL_DIR}/apriori_rules.pkl")
        print("Load thành công! Sẵn sàng nhận requests.")
    except Exception as e:
        print(f"Lỗi khi load mô hình: {e}")
        raise e
        
    yield 
    # Logic dọn dẹp bộ nhớ khi tắt server (nếu cần thiết) có thể viết ở đây
    print("Đang tắt dịch vụ API...")

# Khởi tạo ứng dụng FastAPI với lifespan context
app = FastAPI(title="Text Classification API", lifespan=lifespan)

# Định nghĩa cấu trúc dữ liệu đầu vào (Input)
class TextInput(BaseModel):
    text: str

# Định nghĩa cấu trúc dữ liệu trả về (Output)
class PredictionOutput(BaseModel):
    label: str
    method: str
    confidence: float
    cnn_label: str
    cnn_confidence: float
    processed_text: str

def predict_hybrid_with_score(text: str):
    """Logic dự đoán gốc của bạn được giữ nguyên"""
    processed_text = ViTokenizer.tokenize(text.lower())
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    
    probabilities = model.predict(padded, verbose=0)[0]
    
    cnn_idx = int(np.argmax(probabilities))
    cnn_conf = float(probabilities[cnn_idx])
    cnn_label = index_to_label[cnn_idx]
    
    final_label = cnn_label
    final_conf = cnn_conf
    method = "CNN"
    
    # Apriori Fallback
    if cnn_conf < 0.85 and not rules.empty:
        words = set(processed_text.split())
        best_rule_conf = 0.0
        rule_label = None
        
        for _, row in rules.iterrows():
            antecedents = set(row["antecedents"])
            if antecedents and antecedents.issubset(words):
                rule_conf = float(row["confidence"])
                if rule_conf > best_rule_conf:
                    consequents = list(row["consequents"])
                    if len(consequents) == 1:
                        best_rule_conf = rule_conf
                        rule_label = consequents[0].replace("L__", "")
                        
        if rule_label is not None and (best_rule_conf > 0.9 or best_rule_conf > cnn_conf + 0.2):
            final_label = rule_label
            final_conf = best_rule_conf
            method = "Apriori"
            
    return {
        "label": final_label,
        "method": method,
        "confidence": final_conf,
        "cnn_label": cnn_label,
        "cnn_confidence": cnn_conf,
        "processed_text": processed_text
    }

# Thiết lập Endpoint nhận yêu cầu POST
@app.post("/predict", response_model=PredictionOutput)
async def predict_text(request: TextInput):
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Văn bản truyền vào không được để trống")
    
    try:
        result = predict_hybrid_with_score(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyvi import ViTokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Các hằng số cấu hình
MAX_SEQ_LEN = 300
MODEL_DIR = "model_artifacts"

# Khai báo các biến toàn cục
model = None
tokenizer = None
index_to_label = None
rules = None


def build_cnn_model():
    """
    Dựng lại chính xác 100% kiến trúc mô hình CNN đã huấn luyện
    dựa trên cấu hình layer xuất ra từ log hệ thống.
    """
    cnn_model = models.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN,)),
        layers.Embedding(input_dim=10000, output_dim=100),
        layers.Conv1D(filters=128, kernel_size=5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(units=64, activation="relu"),
        layers.Dropout(rate=0.5),
        layers.Dense(units=27, activation="softmax"),
    ])
    return cnn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo và load mô hình khi server bắt đầu chạy
    global model, tokenizer, index_to_label, rules
    print("Đang khởi tạo kiến trúc mô hình và load trọng số...")

    try:
        # 1. Dựng lại kiến trúc và nạp trọng số (weights)
        model = build_cnn_model()
        weights_path = os.path.join(MODEL_DIR, "cnn_weights.weights.h5")
        model.load_weights(weights_path)

        # 2. Load Tokenizer
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pickle")
        with open(tokenizer_path, "rb") as file:
            tokenizer = pickle.load(file)

        # 3. Load Label Map
        label_map_path = os.path.join(MODEL_DIR, "label_map.pickle")
        with open(label_map_path, "rb") as file:
            index_to_label = pickle.load(file)

        # 4. Load Apriori Rules
        rules_path = os.path.join(MODEL_DIR, "apriori_rules.pkl")
        rules = pd.read_pickle(rules_path)

        print("Load tất cả các artifacts thành công! Sẵn sàng nhận API requests.")

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo: {e}")
        raise e

    yield
    print("Đang dừng dịch vụ API...")


# Khởi tạo FastAPI app
app = FastAPI(title="Text Classification API", lifespan=lifespan)


# Schema dữ liệu đầu vào và đầu ra
class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    method: str
    confidence: float
    cnn_label: str
    cnn_confidence: float
    processed_text: str


def predict_hybrid_with_score(text: str):
    """
    Logic dự đoán kết hợp giữa CNN và Luật Apriori
    """
    # 1. Tiền xử lý văn bản
    processed_text = ViTokenizer.tokenize(text.lower())
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(
        sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )

    # 2. Dự đoán bằng CNN
    probabilities = model.predict(padded, verbose=0)[0]

    cnn_idx = int(np.argmax(probabilities))
    cnn_conf = float(probabilities[cnn_idx])
    cnn_label = index_to_label[cnn_idx]

    final_label = cnn_label
    final_conf = cnn_conf
    method = "CNN"

    # 3. Apriori Fallback (Nếu độ tin cậy CNN < 0.85)
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

        if rule_label is not None and (
            best_rule_conf > 0.9 or best_rule_conf > cnn_conf + 0.2
        ):
            final_label = rule_label
            final_conf = best_rule_conf
            method = "Apriori"

    return {
        "label": final_label,
        "method": method,
        "confidence": final_conf,
        "cnn_label": cnn_label,
        "cnn_confidence": cnn_conf,
        "processed_text": processed_text,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_text(request: TextInput):
    if not request.text or request.text.strip() == "":
        raise HTTPException(
            status_code=400, detail="Văn bản truyền vào không được để trống"
        )

    try:
        result = predict_hybrid_with_score(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

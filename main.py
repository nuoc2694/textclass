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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        with open(tokenizer_path, "rb") as file:
            tokenizer = pickle.load(file)

        # 3. Load Label Map
        label_map_path = os.path.join(MODEL_DIR, "label_map.pkl")
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

@app.get("/", response_class=HTMLResponse)
async def home_page():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Demo Phân Loại Văn Bản</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light p-4">
        <div class="container card shadow p-4" style="max-width: 650px; margin-top: 50px;">
            <h3 class="card-title text-primary text-center mb-4">Hệ Thống Phân Loại Văn Bản</h3>
            <div class="mb-3">
                <label class="form-label font-weight-bold">Nhập đoạn văn bản cần phân loại:</label>
                <textarea id="inputText" class="form-control" rows="4" placeholder="Ví dụ: Trường Đại học Bách khoa công bố điểm chuẩn..."></textarea>
            </div>
            <button onclick="sendPrediction()" class="btn btn-primary w-100">Phân Loại Ngay</button>
            
            <div id="resultBox" class="mt-4 p-3 bg-white rounded border d-none">
                <h5 class="text-success mb-3">Kết quả dự đoán:</h5>
                <p><strong>Nhãn dự đoán:</strong> <span id="resLabel" class="badge bg-danger fs-6"></span></p>
                <p><strong>Phương pháp sử dụng:</strong> <span id="resMethod" class="badge bg-secondary"></span></p>
                <p><strong>Độ tin cậy:</strong> <span id="resConf"></span>%</p>
                <p><strong>Văn bản đã tách từ:</strong> <em id="resProcessed" class="text-muted"></em></p>
            </div>
        </div>

        <script>
            async function sendPrediction() {
                const text = document.getElementById('inputText').value;
                if (!text.trim()) return alert('Vui lòng nhập văn bản!');

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('resLabel').innerText = data.label;
                    document.getElementById('resMethod').innerText = data.method;
                    document.getElementById('resConf').innerText = (data.confidence * 100).toFixed(2);
                    document.getElementById('resProcessed').innerText = data.processed_text;
                    document.getElementById('resultBox').classList.remove('d-none');
                } else {
                    alert('Có lỗi xảy ra khi xử lý dữ liệu.');
                }
            }
        </script>
    </body>
    </html>
    """

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

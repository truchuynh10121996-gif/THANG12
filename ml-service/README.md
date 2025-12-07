# ML Fraud Detection Service

Hệ thống phát hiện giao dịch lừa đảo sử dụng Machine Learning cho Agribank Digital Guard.

## 🚀 Tính năng chính

- **Layer 1 - Global Fraud Detection**:
  - Isolation Forest: Phát hiện anomaly
  - LightGBM: Phân loại fraud

- **Layer 2 - User Profile (Advanced)**:
  - Autoencoder: User embedding và anomaly detection
  - LSTM: Phân tích chuỗi giao dịch
  - GNN: Phát hiện fraud communities

## 📁 Cấu trúc thư mục

```
ml-service/
├── app.py                 # Flask app chính
├── config.py              # Cấu hình
├── requirements.txt       # Dependencies
│
├── data/
│   ├── raw/               # Dữ liệu thô
│   ├── processed/         # Dữ liệu đã xử lý
│   └── synthetic/         # Script tạo dữ liệu giả lập
│
├── models/
│   ├── layer1/            # Isolation Forest, LightGBM
│   ├── layer2/            # Autoencoder, LSTM, GNN
│   └── ensemble/          # Final predictor
│
├── preprocessing/         # Xử lý và tạo features
├── training/              # Scripts training
├── evaluation/            # Đánh giá models
├── api/                   # API endpoints
├── saved_models/          # Models đã train
└── docs/                  # Documentation
```

## 🔧 Cài đặt

### 1. Tạo virtual environment

```bash
cd ml-service
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env theo nhu cầu
```

## 🏃 Chạy ứng dụng

### 1. Tạo dữ liệu giả lập (lần đầu)

```bash
python data/synthetic/generate_data.py
```

### 2. Xử lý dữ liệu

```bash
python preprocessing/data_cleaner.py
python preprocessing/feature_engineering.py
```

### 3. Chạy API server

```bash
python app.py
```

Server sẽ chạy tại: `http://localhost:5001`

## 📡 API Endpoints

### Prediction

```http
POST /api/predict
Content-Type: application/json

{
    "transaction_id": "TXN001",
    "user_id": "USR001",
    "amount": 5000000,
    "transaction_type": "transfer",
    "timestamp": "2024-01-15 14:30:00"
}
```

Response:
```json
{
    "success": true,
    "prediction": {
        "fraud_probability": 0.85,
        "prediction": "fraud",
        "risk_level": "high",
        "should_block": true
    }
}
```

### Training

```http
POST /api/train/layer1
POST /api/train/layer2
POST /api/train/all
```

### Metrics

```http
GET /api/metrics
GET /api/models/status
GET /api/dashboard/stats
```

## 🧠 Models

### Layer 1: Global Fraud Detection

| Model | Mô tả | Use case |
|-------|-------|----------|
| Isolation Forest | Unsupervised anomaly detection | Phát hiện giao dịch bất thường |
| LightGBM | Gradient boosting classifier | Phân loại fraud với dữ liệu có nhãn |

### Layer 2: User Profile (Advanced)

| Model | Mô tả | Use case |
|-------|-------|----------|
| Autoencoder | Neural network tạo embeddings | User profiling, anomaly detection |
| LSTM | Recurrent neural network | Phát hiện anomaly trong sequences |
| GNN | Graph neural network | Phát hiện fraud communities |

## 📊 Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 🔒 Bảo mật

- Không lưu dữ liệu nhạy cảm
- Validate tất cả inputs
- Rate limiting cho API
- Authentication token (nếu cần)

## 📚 Documentation

- [Kiến trúc chi tiết](docs/ARCHITECTURE.md)
- [Giải thích Models](docs/MODEL_EXPLANATION.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [API Documentation](docs/API_DOCUMENTATION.md)

## 🤝 Tích hợp

Service này tích hợp với:
- **Backend Node.js**: Qua proxy endpoints `/api/ml/*`
- **ML Demo Dashboard**: Frontend React để visualize

## 📝 License

Proprietary - Agribank Digital Guard

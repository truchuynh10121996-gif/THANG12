# API Documentation

## Base URL

```
http://localhost:5001/api
```

## Authentication

Hiện tại API không yêu cầu authentication. Trong production, nên thêm API key hoặc JWT token.

---

## Endpoints

### 1. Health Check

Kiểm tra trạng thái service.

```http
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "ML Fraud Detection",
    "version": "1.0.0",
    "timestamp": "2024-01-15T14:30:00"
}
```

---

### 2. Prediction

#### 2.1 Dự đoán đơn lẻ

```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
    "transaction_id": "TXN001",
    "user_id": "USR001",
    "amount": 5000000,
    "transaction_type": "transfer",
    "channel": "mobile_app",
    "device_type": "android",
    "timestamp": "2024-01-15 14:30:00",
    "location_country": "VN",
    "is_international": false,
    "receiving_bank": "Vietcombank"
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "transaction_id": "TXN001",
        "fraud_probability": 0.85,
        "prediction": "fraud",
        "risk_level": "high",
        "should_block": true,
        "should_review": false,
        "confidence": 0.70
    },
    "timestamp": "2024-01-15T14:30:01"
}
```

**Risk Levels:**
| Level | Probability Range | Action |
|-------|------------------|--------|
| low | < 30% | Allow |
| medium | 30% - 60% | Alert |
| high | 60% - 80% | Review + OTP |
| critical | > 80% | Block |

---

#### 2.2 Dự đoán Batch

```http
POST /api/predict/batch
Content-Type: application/json
```

**Request Body:**
```json
{
    "transactions": [
        {"transaction_id": "TXN001", "amount": 5000000, ...},
        {"transaction_id": "TXN002", "amount": 1000000, ...}
    ]
}
```

**Response:**
```json
{
    "success": true,
    "predictions": [
        {"transaction_id": "TXN001", "fraud_probability": 0.85, ...},
        {"transaction_id": "TXN002", "fraud_probability": 0.15, ...}
    ],
    "summary": {
        "total": 2,
        "fraud_count": 1,
        "fraud_ratio": 0.5
    },
    "timestamp": "2024-01-15T14:30:01"
}
```

**Giới hạn:** Tối đa 1000 giao dịch mỗi request.

---

### 3. Model Management

#### 3.1 Trạng thái Models

```http
GET /api/models/status
```

**Response:**
```json
{
    "success": true,
    "status": {
        "layer1": {
            "fitted": true,
            "models": ["isolation_forest", "lightgbm"]
        },
        "layer2": {
            "fitted": true,
            "models": {
                "autoencoder": true,
                "lstm": true,
                "gnn": false
            }
        },
        "weights": {
            "layer1": 0.4,
            "layer2": 0.6
        }
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

#### 3.2 Train Layer 1

```http
POST /api/train/layer1
Content-Type: application/json
```

**Request Body (optional):**
```json
{
    "data_path": "path/to/features.csv"
}
```

**Response:**
```json
{
    "success": true,
    "result": {
        "status": "success",
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "roc_auc": 0.92
        }
    },
    "message": "Layer 1 training hoàn tất",
    "timestamp": "2024-01-15T14:35:00"
}
```

---

#### 3.3 Train Layer 2

```http
POST /api/train/layer2
Content-Type: application/json
```

**Response:**
```json
{
    "success": true,
    "result": {
        "status": "success",
        "models_trained": ["autoencoder", "lstm"]
    },
    "message": "Layer 2 training hoàn tất",
    "timestamp": "2024-01-15T14:40:00"
}
```

---

#### 3.4 Train All

```http
POST /api/train/all
Content-Type: application/json
```

**Response:**
```json
{
    "success": true,
    "result": {
        "layer1": {...},
        "layer2": {...},
        "status": "success"
    },
    "message": "Tất cả models đã được train",
    "timestamp": "2024-01-15T14:45:00"
}
```

---

### 4. Metrics

#### 4.1 Metrics hiện tại

```http
GET /api/metrics
```

**Response:**
```json
{
    "success": true,
    "metrics": {
        "isolation_forest": {
            "accuracy": 0.90,
            "precision": 0.75,
            "recall": 0.80,
            "f1_score": 0.77,
            "roc_auc": 0.88
        },
        "lightgbm": {
            "accuracy": 0.95,
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "roc_auc": 0.92
        },
        "ensemble": {
            "accuracy": 0.94,
            "precision": 0.83,
            "recall": 0.80,
            "f1_score": 0.81,
            "roc_auc": 0.93
        }
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

#### 4.2 Lịch sử Metrics

```http
GET /api/metrics/history
```

**Response:**
```json
{
    "success": true,
    "history": [
        {
            "date": "2024-01-14",
            "f1_score": 0.80,
            "roc_auc": 0.91
        },
        {
            "date": "2024-01-15",
            "f1_score": 0.81,
            "roc_auc": 0.93
        }
    ],
    "timestamp": "2024-01-15T14:30:00"
}
```

---

### 5. Explainability

```http
POST /api/explain
Content-Type: application/json
```

**Request Body:**
```json
{
    "transaction": {
        "transaction_id": "TXN001",
        "amount": 50000000,
        "hour": 3,
        "is_international": true
    }
}
```

**Response:**
```json
{
    "success": true,
    "explanation": {
        "summary": "Giao dịch có 3 yếu tố rủi ro: Số tiền lớn, Thời gian bất thường, Giao dịch quốc tế",
        "risk_factors": [
            {
                "factor": "Số tiền lớn",
                "description": "Giao dịch 50,000,000 VND vượt ngưỡng 50 triệu",
                "importance": "high"
            },
            {
                "factor": "Thời gian bất thường",
                "description": "Giao dịch ngoài giờ làm việc (22h-6h)",
                "importance": "medium"
            },
            {
                "factor": "Giao dịch quốc tế",
                "description": "Giao dịch từ/đến nước ngoài",
                "importance": "high"
            }
        ],
        "safe_factors": [],
        "recommendations": [
            "Xác nhận lại với khách hàng qua OTP",
            "Yêu cầu xác thực sinh trắc học",
            "Tạm thời chặn giao dịch để review"
        ]
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

### 6. Dashboard

#### 6.1 Thống kê Dashboard

```http
GET /api/dashboard/stats
```

**Response:**
```json
{
    "success": true,
    "stats": {
        "total_predictions": 10000,
        "fraud_detected": 500,
        "fraud_rate": 0.05,
        "high_risk_alerts": 50,
        "total_transactions": 500000,
        "total_fraud": 25000,
        "data_fraud_rate": 0.05,
        "last_updated": "2024-01-15T14:30:00"
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

#### 6.2 Graph Data

```http
GET /api/graph/community
```

**Response:**
```json
{
    "success": true,
    "graph": {
        "nodes": [
            {"id": 0, "label": "Node 0"},
            {"id": 1, "label": "Node 1"}
        ],
        "edges": [
            {"source": 0, "target": 1, "amount": 5000000}
        ],
        "total_nodes": 10000,
        "total_edges": 50000
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

### 7. User Profile

#### 7.1 User Profile

```http
GET /api/user/{user_id}/profile
```

**Response:**
```json
{
    "success": true,
    "user_id": "USR001",
    "profile": {
        "risk_score": 0.3,
        "transaction_count": 150,
        "avg_amount": 2500000,
        "embedding": [0.1, -0.2, 0.3, ...]
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

#### 7.2 User Sequence

```http
GET /api/user/{user_id}/sequence
```

**Response:**
```json
{
    "success": true,
    "user_id": "USR001",
    "sequence": {
        "transactions": [...],
        "sequence_length": 10
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

### 8. Data Generation

```http
POST /api/data/generate
Content-Type: application/json
```

**Request Body:**
```json
{
    "num_users": 1000,
    "num_transactions": 10000
}
```

**Response:**
```json
{
    "success": true,
    "message": "Đã tạo dữ liệu giả lập",
    "stats": {
        "num_users": 1000,
        "num_transactions": 10000,
        "num_fraud": 500
    },
    "timestamp": "2024-01-15T14:30:00"
}
```

---

## Error Handling

### Error Response Format

```json
{
    "success": false,
    "error": "Mô tả lỗi",
    "status_code": 400
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limiting

- Hiện tại: Không có giới hạn
- Production: 100 requests/minute/IP

---

## Examples

### cURL

```bash
# Predict single
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 5000000, "channel": "mobile_app"}'

# Get status
curl http://localhost:5001/api/models/status
```

### Python

```python
import requests

# Predict
response = requests.post(
    "http://localhost:5001/api/predict",
    json={"amount": 5000000, "channel": "mobile_app"}
)
print(response.json())
```

### JavaScript

```javascript
// Predict
const response = await fetch('http://localhost:5001/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({amount: 5000000, channel: 'mobile_app'})
});
const data = await response.json();
```

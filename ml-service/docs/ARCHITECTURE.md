# Kiến trúc ML Fraud Detection System

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRAUD DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Web App    │    │ Mobile App   │    │  Web Admin   │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             │                                       │
│                             ▼                                       │
│                    ┌────────────────┐                               │
│                    │ Backend Node.js │                              │
│                    │   Port: 5000    │                              │
│                    └────────┬───────┘                               │
│                             │                                       │
│              ┌──────────────┼──────────────┐                        │
│              │              │              │                        │
│              ▼              ▼              ▼                        │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│     │ Chatbot    │  │    TTS     │  │ ML Service │                 │
│     │  Service   │  │  Service   │  │ Port: 5001 │                 │
│     └────────────┘  └────────────┘  └─────┬──────┘                 │
│                                           │                         │
└───────────────────────────────────────────┼─────────────────────────┘
                                            │
                           ┌────────────────┴────────────────┐
                           │                                 │
                           ▼                                 ▼
                    ┌─────────────┐                  ┌─────────────┐
                    │   LAYER 1   │                  │   LAYER 2   │
                    │   Global    │                  │ User Profile│
                    │  Detection  │                  │  (Advanced) │
                    └──────┬──────┘                  └──────┬──────┘
                           │                                │
           ┌───────────────┴───────────────┐               │
           │                               │               │
           ▼                               ▼               ▼
    ┌─────────────┐               ┌─────────────┐   ┌─────────────┐
    │  Isolation  │               │  LightGBM   │   │ Autoencoder │
    │   Forest    │               │             │   │    LSTM     │
    │  (Anomaly)  │               │ (Classify)  │   │    GNN      │
    └─────────────┘               └─────────────┘   └─────────────┘
```

## Flow xử lý giao dịch

```
Giao dịch mới
      │
      ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    LAYER 1                          │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │ Isolation Forest│    │    LightGBM     │        │
│  │  Score: 0.7     │    │   Score: 0.8    │        │
│  └────────┬────────┘    └────────┬────────┘        │
│           └──────────┬───────────┘                 │
│                      │                              │
│                      ▼                              │
│           Layer 1 Score: 0.75                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    LAYER 2                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │Autoencoder│  │   LSTM   │   │   GNN    │        │
│  │Score: 0.6│   │Score: 0.8│   │Score: 0.7│        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       └──────────────┼──────────────┘              │
│                      │                              │
│                      ▼                              │
│           Layer 2 Score: 0.70                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    ENSEMBLE     │
              │  Final Score:   │
              │     0.72        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Risk Level:     │
              │    HIGH         │
              │ Action: BLOCK   │
              └─────────────────┘
```

## Chi tiết các Models

### Layer 1: Global Fraud Detection

#### 1. Isolation Forest

**Mục đích**: Phát hiện anomaly (unsupervised)

**Ý tưởng cốt lõi**:
- Anomalies dễ bị "isolate" hơn các điểm bình thường
- Xây dựng random trees và đếm số bước cần để isolate mỗi điểm
- Điểm cần ít bước hơn = anomaly

**Hyperparameters**:
```python
{
    'n_estimators': 100,      # Số trees
    'contamination': 0.05,    # Tỷ lệ anomaly dự kiến
    'random_state': 42
}
```

**Ưu điểm**:
- Không cần labels
- Nhanh và hiệu quả
- Scale tốt với data lớn

**Nhược điểm**:
- Khó tune threshold
- Không giải thích được features

#### 2. LightGBM

**Mục đích**: Phân loại fraud (supervised)

**Ý tưởng**:
- Gradient boosting với leaf-wise tree growth
- Học từ dữ liệu có nhãn

**Hyperparameters**:
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 200
}
```

**Ưu điểm**:
- Rất nhanh
- Có feature importance
- Xử lý tốt imbalanced data

**Nhược điểm**:
- Cần labels
- Có thể overfit

### Layer 2: User Profile (Advanced)

#### 1. Autoencoder

**Mục đích**: Tạo user embedding và phát hiện anomaly

**Architecture**:
```
Input (64) → 32 → 16 → 8 (embedding) → 16 → 32 → Output (64)
```

**Ý tưởng**:
- Nén thông tin user thành embedding nhỏ
- Tái tạo lại input từ embedding
- Reconstruction error cao = bất thường

#### 2. LSTM

**Mục đích**: Phân tích chuỗi giao dịch

**Architecture**:
```
Sequence (10 txns) → LSTM (2 layers, 64 hidden) → Attention → Dense → Prediction
```

**Ý tưởng**:
- Học "nhịp điệu" chi tiêu của user
- Phát hiện khi pattern bị phá vỡ

#### 3. GNN (Graph Neural Network)

**Mục đích**: Phát hiện fraud communities

**Architecture**:
```
Nodes (users/merchants) → GraphSAGE (3 layers) → Node embeddings → Classification
```

**Ý tưởng**:
- Xây dựng graph từ giao dịch (user → merchant)
- Truyền thông tin qua các kết nối
- Phát hiện các nhóm đáng ngờ

## Cách kết hợp Layer 1 và Layer 2

### Weighted Ensemble

```python
final_score = layer1_weight * layer1_score + layer2_weight * layer2_score

# Mặc định:
# layer1_weight = 0.4 (40%)
# layer2_weight = 0.6 (60%)
```

### Logic quyết định

```
if final_score >= 0.8:
    risk_level = "critical"
    action = "BLOCK"
elif final_score >= 0.6:
    risk_level = "high"
    action = "REVIEW + OTP"
elif final_score >= 0.3:
    risk_level = "medium"
    action = "ALERT"
else:
    risk_level = "low"
    action = "ALLOW"
```

## Tại sao chọn kiến trúc này?

### 1. Two-Layer Design

- **Layer 1** xử lý tất cả giao dịch với tốc độ cao
- **Layer 2** phân tích sâu hơn với context user
- Kết hợp để tận dụng cả global patterns và individual behavior

### 2. Ensemble của nhiều models

- Mỗi model có điểm mạnh riêng
- Isolation Forest tốt cho anomaly detection
- LightGBM tốt cho classification với labels
- Autoencoder capture user behavior
- LSTM hiểu sequences
- GNN phát hiện communities

### 3. Modular và Scalable

- Có thể bật/tắt từng model
- Dễ dàng thêm models mới
- Có thể train riêng từng layer

## Dữ liệu và Features

### Input Features

```
Temporal:
- hour, day_of_week, is_weekend, is_night

Amount:
- log_amount, amount_vs_user_avg, amount_zscore

Behavioral:
- is_international, channel_risk
- time_since_last_txn, is_rapid_txn

User Profile:
- credit_score, account_age
- avg_monthly_transactions
```

### Tổng số features: ~64

## Performance Targets

| Metric | Target | Production |
|--------|--------|------------|
| Precision | > 0.80 | 0.85 |
| Recall | > 0.75 | 0.78 |
| F1-Score | > 0.75 | 0.81 |
| ROC-AUC | > 0.90 | 0.92 |
| Latency | < 100ms | 50ms |

# 📖 HƯỚNG DẪN SỬ DỤNG ML FRAUD DETECTION

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Cài Đặt](#2-cài-đặt)
3. [Cấu Trúc Dữ Liệu](#3-cấu-trúc-dữ-liệu)
4. [Tạo Dữ Liệu Huấn Luyện](#4-tạo-dữ-liệu-huấn-luyện)
5. [Huấn Luyện Model](#5-huấn-luyện-model)
6. [Sử Dụng API](#6-sử-dụng-api)
7. [Dashboard Demo](#7-dashboard-demo)
8. [Tùy Chỉnh Model](#8-tùy-chỉnh-model)

---

## 1. Tổng Quan

ML Fraud Detection là hệ thống phát hiện gian lận đa tầng sử dụng Machine Learning:

### Kiến trúc 2 Layer:

**Layer 1 - Global Fraud Detection:**
- 🌲 **Isolation Forest**: Phát hiện anomaly không giám sát
- 🚀 **LightGBM**: Phân loại có giám sát với gradient boosting

**Layer 2 - User Profile (nâng cao):**
- 🔮 **Autoencoder**: Học embedding người dùng
- 📈 **LSTM**: Phân tích chuỗi giao dịch
- 🕸️ **GNN**: Phát hiện cộng đồng gian lận

---

## 2. Cài Đặt

### Yêu Cầu Hệ Thống:
- Python 3.8+
- 4GB RAM (khuyến nghị 8GB+)
- 2GB ổ cứng

### Cài Đặt Dependencies:

```bash
cd ml-service

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Kiểm Tra Cài Đặt:

```bash
python -c "import sklearn, lightgbm, torch; print('✅ Cài đặt thành công!')"
```

---

## 3. Cấu Trúc Dữ Liệu

### 3.1 File Users (users.csv)

| Cột | Kiểu | Mô Tả | Ví Dụ |
|-----|------|-------|-------|
| `user_id` | string | ID người dùng duy nhất | USR000001 |
| `age` | int | Tuổi | 28 |
| `gender` | string | Giới tính (M/F) | M |
| `occupation` | string | Nghề nghiệp | engineer |
| `income_level` | string | Mức thu nhập (low/medium/high) | high |
| `account_age_days` | int | Số ngày từ khi tạo tài khoản | 730 |
| `city` | string | Thành phố | Hanoi |
| `region` | string | Vùng miền (North/Central/South) | North |
| `phone_verified` | int | Đã xác minh SĐT (0/1) | 1 |
| `email_verified` | int | Đã xác minh email (0/1) | 1 |
| `kyc_level` | int | Cấp độ KYC (1-3) | 3 |
| `avg_monthly_transactions` | int | Số giao dịch TB/tháng | 45 |
| `avg_transaction_amount` | int | Số tiền giao dịch TB (VND) | 5500000 |
| `preferred_channel` | string | Kênh ưa thích (mobile/web) | mobile |
| `device_count` | int | Số thiết bị đã dùng | 2 |
| `login_frequency` | string | Tần suất đăng nhập | daily |
| `last_login_days_ago` | int | Số ngày từ lần đăng nhập cuối | 0 |
| `risk_score_historical` | float | Điểm rủi ro lịch sử (0-1) | 0.12 |
| `is_premium` | int | Tài khoản premium (0/1) | 1 |
| `created_at` | date | Ngày tạo tài khoản | 2022-12-07 |

### 3.2 File Transactions (transactions.csv)

| Cột | Kiểu | Mô Tả | Ví Dụ |
|-----|------|-------|-------|
| `transaction_id` | string | ID giao dịch duy nhất | TXN0000000001 |
| `user_id` | string | ID người thực hiện | USR000001 |
| `timestamp` | datetime | Thời gian giao dịch | 2024-12-07 09:15:23 |
| `amount` | int | Số tiền (VND) | 2500000 |
| `transaction_type` | string | Loại GD | transfer, payment, withdrawal |
| `channel` | string | Kênh thực hiện | mobile, web, atm |
| `recipient_id` | string | ID người nhận | USR000002 hoặc MER001 |
| `recipient_type` | string | Loại người nhận | individual, merchant |
| `device_id` | string | ID thiết bị | DEV000001 |
| `device_type` | string | Loại thiết bị | android, ios, windows |
| `ip_address` | string | Địa chỉ IP | 113.161.72.45 |
| `location_city` | string | Thành phố | Hanoi |
| `location_country` | string | Quốc gia | Vietnam |
| `merchant_category` | string | Danh mục merchant | food_delivery, shopping |
| `is_international` | int | GD quốc tế (0/1) | 0 |
| `session_duration_sec` | int | Thời gian session (giây) | 245 |
| `login_attempts` | int | Số lần thử đăng nhập | 1 |
| `time_since_last_transaction_min` | int | Phút từ GD trước | 1440 |
| `is_new_recipient` | int | Người nhận mới (0/1) | 0 |
| `is_new_device` | int | Thiết bị mới (0/1) | 0 |
| `is_new_location` | int | Địa điểm mới (0/1) | 0 |
| `hour_of_day` | int | Giờ trong ngày (0-23) | 9 |
| `day_of_week` | int | Ngày trong tuần (0-6) | 6 |
| `is_weekend` | int | Cuối tuần (0/1) | 1 |
| `velocity_1h` | int | Số GD trong 1h gần nhất | 1 |
| `velocity_24h` | int | Số GD trong 24h gần nhất | 5 |
| `amount_deviation_ratio` | float | Tỷ lệ so với TB | 0.45 |
| `is_fraud` | int | Nhãn gian lận (0/1) | 0 |
| `fraud_type` | string | Loại gian lận | normal, unusual_amount, etc. |

### 3.3 Các Loại Fraud:

| Fraud Type | Mô Tả | Đặc Điểm |
|------------|-------|----------|
| `normal` | Giao dịch hợp lệ | Không có dấu hiệu bất thường |
| `unusual_amount` | Số tiền bất thường | Gấp 3-10 lần mức trung bình |
| `unusual_time` | Thời gian bất thường | 1-5 giờ sáng |
| `new_recipient` | Người nhận mới + số tiền lớn | is_new_recipient=1, amount cao |
| `rapid_succession` | Giao dịch liên tiếp | velocity_1h > 4, time_since_last < 5 |
| `foreign_location` | Địa điểm đáng ngờ | Quốc gia Nigeria, Russia, etc. |
| `device_change` | Thiết bị mới + bất thường | is_new_device=1, hành vi lạ |
| `velocity_abuse` | Vượt tốc độ cho phép | velocity_1h > 5 |
| `account_takeover` | Chiếm đoạt tài khoản | login_attempts > 3, nhiều yếu tố bất thường |

---

## 4. Tạo Dữ Liệu Huấn Luyện

### 4.1 Sử Dụng File Mẫu

Hệ thống cung cấp sẵn file mẫu trong `data/samples/`:

```bash
# Xem file mẫu
head -5 data/samples/users.csv
head -5 data/samples/transactions.csv
```

### 4.2 Tạo Dữ Liệu Tự Động

Sử dụng script `quick_generate.py`:

```bash
# Tạo 1,000 users và 10,000 transactions (5% fraud)
python scripts/quick_generate.py --users 1000 --transactions 10000 --fraud_rate 0.05

# Tạo dữ liệu lớn hơn
python scripts/quick_generate.py --users 50000 --transactions 500000 --fraud_rate 0.05

# Tùy chỉnh thư mục output
python scripts/quick_generate.py --users 5000 --transactions 50000 --output_dir data/my_dataset
```

### 4.3 Tạo Dữ Liệu Từ Nguồn Thực

Nếu bạn có dữ liệu thực, đảm bảo format theo cấu trúc ở mục 3:

```python
import pandas as pd

# Load dữ liệu thực của bạn
my_users = pd.read_csv('path/to/your/users.csv')
my_transactions = pd.read_csv('path/to/your/transactions.csv')

# Đảm bảo có đầy đủ các cột cần thiết
required_user_cols = ['user_id', 'age', 'income_level', 'account_age_days', ...]
required_txn_cols = ['transaction_id', 'user_id', 'amount', 'is_fraud', ...]

# Rename và transform nếu cần
my_transactions['is_fraud'] = my_transactions['fraud_label'].map({'yes': 1, 'no': 0})

# Lưu theo format chuẩn
my_users.to_csv('data/real/users.csv', index=False)
my_transactions.to_csv('data/real/transactions.csv', index=False)
```

---

## 5. Huấn Luyện Model

### 5.1 Training Cơ Bản

```bash
# Sử dụng dữ liệu đã tạo
python scripts/train_model.py --data_dir data/generated

# Tùy chỉnh test size và output
python scripts/train_model.py \
    --data_dir data/generated \
    --output_dir models/v1 \
    --test_size 0.3 \
    --random_state 123
```

### 5.2 Kết Quả Training

Sau khi training hoàn tất, bạn sẽ có:

```
models/trained/
├── isolation_forest.pkl    # Isolation Forest model
├── lightgbm.pkl           # LightGBM model
├── scaler.pkl             # StandardScaler đã fit
├── label_encoders.pkl     # Label encoders cho categorical
├── feature_names.json     # Tên các features
└── training_report.json   # Báo cáo training
```

### 5.3 Xem Báo Cáo Training

```bash
cat models/trained/training_report.json
```

```json
{
  "training_time": "2024-12-07T10:30:45",
  "training_info": {
    "total_samples": 10000,
    "training_samples": 8000,
    "test_samples": 2000,
    "num_features": 20,
    "fraud_rate": 0.05
  },
  "models": {
    "isolation_forest": {
      "accuracy": 0.92,
      "precision": 0.85,
      "recall": 0.78,
      "f1": 0.81,
      "auc_roc": 0.89
    },
    "lightgbm": {
      "accuracy": 0.96,
      "precision": 0.91,
      "recall": 0.88,
      "f1": 0.89,
      "auc_roc": 0.95
    }
  }
}
```

---

## 6. Sử Dụng API

### 6.1 Khởi Động Server

```bash
cd ml-service
python app.py
```

Server chạy tại: `http://localhost:5001`

### 6.2 API Endpoints

#### Health Check
```bash
curl http://localhost:5001/api/health
```

#### Predict Single Transaction
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR000001",
    "amount": 5000000,
    "transaction_type": "transfer",
    "channel": "mobile",
    "recipient_id": "USR000002",
    "device_id": "DEV001",
    "ip_address": "113.161.72.45",
    "location_city": "Hanoi",
    "location_country": "Vietnam",
    "is_new_recipient": false,
    "is_new_device": false
  }'
```

Response:
```json
{
  "transaction_id": "TXN_1733567890",
  "risk_score": 0.15,
  "risk_level": "low",
  "is_fraud": false,
  "model_scores": {
    "isolation_forest": 0.12,
    "lightgbm": 0.18
  },
  "risk_factors": [],
  "recommendation": "APPROVE"
}
```

#### Predict Batch
```bash
curl -X POST http://localhost:5001/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"user_id": "USR001", "amount": 500000, ...},
      {"user_id": "USR002", "amount": 1000000, ...}
    ]
  }'
```

#### Get Model Status
```bash
curl http://localhost:5001/api/models/status
```

#### Train Models
```bash
curl -X POST http://localhost:5001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/generated",
    "model_types": ["isolation_forest", "lightgbm"]
  }'
```

#### Get Explanation
```bash
curl -X POST http://localhost:5001/api/explain \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_1733567890",
    "features": {...}
  }'
```

---

## 7. Dashboard Demo

### 7.1 Khởi Động Dashboard

```bash
cd ml-demo
npm install
npm start
```

Dashboard chạy tại: `http://localhost:3001`

### 7.2 Các Trang Chính

| Trang | Đường dẫn | Mô Tả |
|-------|-----------|-------|
| Dashboard | `/` | Tổng quan, biểu đồ, thống kê |
| Transaction Test | `/test` | Test giao dịch đơn lẻ |
| Batch Analysis | `/batch` | Phân tích hàng loạt |
| Real-time Monitor | `/monitor` | Giám sát real-time |
| Model Training | `/training` | Train và đánh giá model |
| Data Explorer | `/data` | Khám phá và tạo dữ liệu |
| Reports | `/reports` | Báo cáo chi tiết |

### 7.3 Test Giao Dịch

1. Vào trang **Transaction Test**
2. Điền thông tin giao dịch
3. Nhấn **Analyze Transaction**
4. Xem kết quả risk score và recommendations

---

## 8. Tùy Chỉnh Model

### 8.1 Điều Chỉnh Hyperparameters

Edit file `config.py`:

```python
# Isolation Forest
ISOLATION_FOREST_CONFIG = {
    'n_estimators': 200,      # Tăng để chính xác hơn
    'contamination': 0.05,    # Tỷ lệ fraud ước tính
    'max_samples': 'auto',
    'random_state': 42
}

# LightGBM
LIGHTGBM_CONFIG = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,         # Tăng để phức tạp hơn
    'learning_rate': 0.05,    # Giảm để ổn định hơn
    'n_estimators': 500,      # Tăng số trees
    'class_weight': 'balanced'
}
```

### 8.2 Thêm Features Mới

Edit `preprocessing/feature_engineering.py`:

```python
def engineer_features(df):
    # Features có sẵn
    df['amount_log'] = np.log1p(df['amount'])
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)

    # Thêm feature mới
    df['is_high_risk_time'] = df['hour_of_day'].between(1, 5).astype(int)
    df['amount_velocity_ratio'] = df['amount'] / (df['velocity_1h'] + 1)

    return df
```

### 8.3 Điều Chỉnh Ngưỡng Risk

Edit `models/ensemble/final_predictor.py`:

```python
RISK_THRESHOLDS = {
    'low': 0.3,       # < 0.3 = low risk
    'medium': 0.6,    # 0.3-0.6 = medium risk
    'high': 0.8,      # 0.6-0.8 = high risk
    'critical': 1.0   # > 0.8 = critical
}
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:

1. **Logs**: `ml-service/logs/app.log`
2. **API Health**: `curl http://localhost:5001/api/health`
3. **Dependencies**: `pip list | grep -E "sklearn|lightgbm|torch"`

---

*Cập nhật lần cuối: 2024-12-07*

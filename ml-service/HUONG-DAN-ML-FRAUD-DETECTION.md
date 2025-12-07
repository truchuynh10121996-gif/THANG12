# Hướng dẫn sử dụng ML Fraud Detection System v2.0

## Mục lục
1. [Tổng quan về cập nhật](#1-tổng-quan-về-cập-nhật)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt và khởi chạy](#3-cài-đặt-và-khởi-chạy)
4. [Hướng dẫn sử dụng giao diện](#4-hướng-dẫn-sử-dụng-giao-diện)
5. [API Reference](#5-api-reference)
6. [Flow hoạt động](#6-flow-hoạt-động)
7. [Giải thích cấu trúc mới](#7-giải-thích-cấu-trúc-mới)

---

## 1. Tổng quan về cập nhật

### 1.1 Giao diện đã được đồng bộ với dự án
- **Màu chủ đạo**: #FF8DAD (hồng), #FF6B99 (hồng đậm)
- **Gradient header**: `linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)`
- **Gradient sidebar**: `linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)`
- **Nền trang**: `linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)`

### 1.2 Nhãn tiếng Việt
| Tiếng Anh | Tiếng Việt |
|-----------|------------|
| Dashboard | Bảng điều khiển |
| Transaction Test | Kiểm tra giao dịch |
| Batch Analysis | Phân tích hàng loạt |
| Real-time Monitor | Giám sát thời gian thực |
| Model Training | Huấn luyện mô hình |
| Data Explorer | Khám phá dữ liệu |
| Reports | Báo cáo |

### 1.3 Tính năng mới
- **Chọn khách hàng từ danh sách**: Dropdown/Autocomplete cho phép tìm kiếm user
- **Hiển thị profile khách hàng**: Tên, tuổi, nghề nghiệp, mức thu nhập, KYC, điểm rủi ro
- **Lịch sử giao dịch**: Bảng 5-10 giao dịch gần nhất của user
- **Tự động tính features**: velocity_1h, velocity_24h, time_since_last_transaction
- **Tích hợp MongoDB**: Lưu trữ users, transactions, predictions

---

## 2. Cấu trúc thư mục

### 2.1 ML Service (Backend Python)
```
ml-service/
├── api/
│   ├── __init__.py
│   ├── routes.py          # API endpoints (ĐÃ CẬP NHẬT)
│   ├── predict.py         # Prediction logic
│   └── explain.py         # Explainability service
├── database/              # THƯ MỤC MỚI
│   ├── __init__.py
│   └── mongodb.py         # MongoDB connection & CRUD operations
├── models/
│   ├── layer1/            # Isolation Forest, LightGBM
│   └── layer2/            # Autoencoder, LSTM, GNN
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── app.py                 # Flask application
├── config.py              # Configuration
└── requirements.txt
```

### 2.2 ML Demo (Frontend React)
```
ml-demo/
├── src/
│   ├── components/
│   │   ├── Sidebar.js     # ĐÃ CẬP NHẬT - Style mới, nhãn tiếng Việt
│   │   ├── Header.js      # ĐÃ CẬP NHẬT - Gradient header
│   │   └── Cards/
│   ├── pages/
│   │   ├── Dashboard.js
│   │   ├── TransactionTest.js  # ĐÃ CẬP NHẬT - Đầy đủ features
│   │   ├── BatchAnalysis.js
│   │   ├── RealtimeMonitor.js
│   │   ├── ModelTraining.js
│   │   ├── DataExplorer.js
│   │   └── Reports.js
│   ├── services/
│   │   └── api.js         # ĐÃ CẬP NHẬT - Thêm API mới
│   ├── styles/
│   │   └── theme.js       # ĐÃ CẬP NHẬT - Màu sắc mới
│   └── App.js             # ĐÃ CẬP NHẬT - Nền gradient
└── package.json
```

---

## 3. Cài đặt và khởi chạy

### 3.1 Khởi chạy ML Service
```bash
# Di chuyển vào thư mục ml-service
cd ml-service

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt pymongo nếu muốn kết nối MongoDB (optional)
pip install pymongo

# Khởi chạy service
python app.py
```

Service sẽ chạy tại: `http://localhost:5001`

### 3.2 Khởi chạy ML Demo
```bash
# Di chuyển vào thư mục ml-demo
cd ml-demo

# Cài đặt dependencies
npm install

# Khởi chạy demo
npm start
```

Demo sẽ chạy tại: `http://localhost:3001`

### 3.3 Cấu hình MongoDB (Optional)
```bash
# Biến môi trường
export MONGODB_URI=mongodb://localhost:27017/agribank-digital-guard

# Hoặc trong file .env
MONGODB_URI=mongodb://localhost:27017/agribank-digital-guard
```

**Lưu ý**: Nếu không có MongoDB, hệ thống sẽ tự động sử dụng mock data.

---

## 4. Hướng dẫn sử dụng giao diện

### 4.1 Trang Kiểm tra giao dịch (TransactionTest)

#### Bước 1: Chọn khách hàng
1. Nhấp vào ô tìm kiếm "Chọn khách hàng"
2. Gõ ID hoặc tên để tìm kiếm
3. Chọn khách hàng từ danh sách dropdown

#### Bước 2: Xem thông tin khách hàng
Sau khi chọn, hệ thống sẽ hiển thị:
- **Thông tin cơ bản**: Họ tên, tuổi, nghề nghiệp, mức thu nhập
- **Thông tin tài khoản**: Số ngày sử dụng, cấp độ KYC, GD trung bình
- **Chỉ số hành vi**:
  - `velocity_1h`: Số giao dịch trong 1 giờ qua
  - `velocity_24h`: Số giao dịch trong 24 giờ qua
  - `time_since_last_transaction`: Thời gian từ giao dịch cuối (giờ)
  - `amount_deviation_ratio`: Tỷ lệ số tiền so với trung bình

#### Bước 3: Xem lịch sử giao dịch
- Bảng hiển thị 10 giao dịch gần nhất
- Các cột: Thời gian, Số tiền, Loại GD, Người nhận, Trạng thái

#### Bước 4: Nhập thông tin giao dịch mới
1. Nhập số tiền (VND)
2. Chọn loại giao dịch (Chuyển khoản, Thanh toán, Rút tiền, Nạp tiền)
3. Chọn kênh giao dịch (Mobile App, Web Banking, ATM, Chi nhánh)
4. Nhập ID người nhận (nếu có)
5. Điều chỉnh giờ giao dịch, loại thiết bị, giao dịch quốc tế

#### Bước 5: Phân tích giao dịch
1. Nhấn nút "Phân tích giao dịch"
2. Hệ thống sẽ:
   - Tính toán behavioral features từ lịch sử
   - Chạy qua các model ML (Isolation Forest, LightGBM)
   - Trả về kết quả với risk score và giải thích

#### Bước 6: Xem kết quả
- **Xác suất Fraud**: Phần trăm khả năng gian lận
- **Mức độ rủi ro**: Thấp, Trung bình, Cao, Nghiêm trọng
- **Dự đoán**: Bình thường hoặc Nghi ngờ lừa đảo
- **Hành động đề xuất**: Cho phép hoặc Nên chặn
- **Giải thích chi tiết**: Yếu tố rủi ro và khuyến nghị

---

## 5. API Reference

### 5.1 API Endpoints mới

#### GET /api/users
Lấy danh sách users
```json
// Response
{
  "success": true,
  "users": [
    {
      "user_id": "USR_001",
      "name": "Nguyễn Văn An",
      "age": 35,
      "occupation": "Kỹ sư phần mềm",
      "income_level": "high",
      "account_age_days": 730,
      "kyc_level": 3,
      "avg_transaction_amount": 8500000,
      "historical_risk_score": 0.12
    }
  ],
  "total": 8
}
```

#### GET /api/users/:id
Lấy chi tiết user + behavioral features
```json
// Response
{
  "success": true,
  "user_id": "USR_001",
  "profile": { ... },
  "behavioral_features": {
    "velocity_1h": 2,
    "velocity_24h": 8,
    "time_since_last_transaction": 4.5,
    "amount_deviation_ratio": 1.2
  }
}
```

#### GET /api/users/:id/transactions
Lấy lịch sử giao dịch của user
```json
// Response
{
  "success": true,
  "user_id": "USR_001",
  "transactions": [
    {
      "transaction_id": "TXN_USR_001_001",
      "amount": 5000000,
      "transaction_type": "transfer",
      "recipient_id": "RCP_001",
      "status": "completed",
      "timestamp": "2024-01-15T14:30:00"
    }
  ],
  "total": 10
}
```

#### POST /api/predict (Đã cập nhật)
Dự đoán fraud với behavioral features
```json
// Request
{
  "transaction_id": "TXN_TEST_001",
  "user_id": "USR_001",
  "amount": 15000000,
  "transaction_type": "transfer",
  "channel": "mobile_app",
  "recipient_id": "RCP_NEW"
}

// Response
{
  "success": true,
  "prediction": {
    "fraud_probability": 0.75,
    "prediction": "fraud",
    "risk_level": "high",
    "should_block": true,
    "confidence": 0.70
  },
  "behavioral_features": {
    "velocity_1h": 3,
    "velocity_24h": 12,
    "time_since_last_transaction": 0.5,
    "amount_deviation_ratio": 1.76,
    "is_new_recipient": true
  }
}
```

#### GET /api/transactions/recent
Lấy giao dịch gần đây của hệ thống

#### GET /api/health (Đã cập nhật)
```json
{
  "status": "healthy",
  "service": "ML Fraud Detection",
  "version": "2.0.0",
  "database_connected": true
}
```

---

## 6. Flow hoạt động

### 6.1 Flow khi test giao dịch

```
┌─────────────────────────────────────────────────────────────────┐
│                      NGƯỜI DÙNG                                  │
│  1. Chọn user_id từ dropdown                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND (ml-demo)                          │
│  2. Gọi API: GET /api/users/:id                                 │
│  3. Gọi API: GET /api/users/:id/transactions                    │
│  4. Hiển thị profile + lịch sử cho người dùng                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NGƯỜI DÙNG                                  │
│  5. Nhập thông tin giao dịch mới                                │
│  6. Nhấn "Phân tích"                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND                                    │
│  7. Gọi API: POST /api/predict                                  │
│  8. Gọi API: POST /api/explain                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ML SERVICE (Backend)                           │
│  9.  Nhận request với user_id                                   │
│  10. Query MongoDB lấy profile user                             │
│  11. Query MongoDB lấy lịch sử giao dịch                        │
│  12. Tính toán behavioral features:                             │
│      - velocity_1h, velocity_24h                                │
│      - time_since_last_transaction                              │
│      - amount_deviation_ratio                                   │
│      - is_new_recipient                                         │
│  13. Merge thành feature vector đầy đủ                          │
│  14. Chạy qua Isolation Forest + LightGBM                       │
│  15. Tính risk score và phân loại                               │
│  16. Lưu giao dịch + prediction vào MongoDB                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND                                    │
│  17. Nhận response với prediction + behavioral_features         │
│  18. Hiển thị kết quả và giải thích chi tiết                    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Tính toán Behavioral Features

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `velocity_1h` | COUNT(GD trong 1h qua) | Tần suất giao dịch ngắn hạn |
| `velocity_24h` | COUNT(GD trong 24h qua) | Tần suất giao dịch dài hạn |
| `time_since_last` | NOW - timestamp của GD cuối | Thời gian nghỉ |
| `amount_deviation` | amount_mới / avg_amount_user | Độ lệch so với bình thường |
| `is_new_recipient` | recipient_id NOT IN danh sách cũ | Người nhận mới |

---

## 7. Giải thích cấu trúc mới

### 7.1 File database/mongodb.py

**Chức năng**:
- Kết nối MongoDB (fallback to mock data nếu không có)
- CRUD operations cho users, transactions, predictions
- Tính toán behavioral features từ lịch sử

**Class MongoDBConnection**:
```python
# Singleton pattern - chỉ 1 instance
db = MongoDBConnection()

# Các methods chính
db.get_all_users(limit=100)
db.get_user_by_id(user_id)
db.get_user_transactions(user_id, limit=10)
db.calculate_behavioral_features(user_id, amount)
db.save_transaction(data)
db.save_prediction(data)
```

**Mock data**:
- 8 users mẫu với đầy đủ thông tin
- Tự động generate transactions nếu không có MongoDB

### 7.2 File api/routes.py (Đã cập nhật)

**Endpoints mới**:
- `GET /api/users` - Danh sách users
- `GET /api/users/:id` - Chi tiết user + features
- `GET /api/users/:id/transactions` - Lịch sử giao dịch
- `POST /api/users` - Tạo user mới
- `GET /api/transactions/recent` - GD gần đây hệ thống

**POST /api/predict** đã được cập nhật:
1. Nhận user_id từ request
2. Tính behavioral features từ database
3. Merge vào feature vector
4. Lưu giao dịch và prediction
5. Trả về kết quả kèm behavioral_features

### 7.3 File services/api.js (Đã cập nhật)

**Các hàm API mới**:
```javascript
getUsers(limit)           // Lấy danh sách users
getUserDetail(userId)     // Lấy chi tiết user
getUserTransactions(userId, limit)  // Lấy lịch sử GD
createUser(userData)      // Tạo user mới
getRecentTransactions(limit)  // GD gần đây
```

**Utility functions**:
```javascript
formatCurrency(amount)    // Format tiền VND
formatDateTime(dateString) // Format ngày giờ
getRiskColor(riskLevel)   // Lấy màu theo risk
getRiskLabel(riskLevel)   // Nhãn tiếng Việt
getTransactionTypeLabel(type)  // Loại GD tiếng Việt
getIncomeLevelLabel(level)     // Thu nhập tiếng Việt
```

### 7.4 File styles/theme.js (Đã cập nhật)

**Màu sắc mới**:
```javascript
// Primary colors
primary: '#FF8DAD'    // Hồng
secondary: '#FF6B99'  // Hồng đậm

// Gradients
gradients: {
  header: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
  sidebar: 'linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)',
  background: 'linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)',
  cardPink: 'linear-gradient(135deg, #FF8DAD 0%, #FF6B99 100%)',
  cardRed: 'linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%)',
  cardBlue: 'linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)',
  cardPurple: 'linear-gradient(135deg, #7B1FA2 0%, #4A148C 100%)'
}
```

### 7.5 File TransactionTest.js (Đã cải tiến)

**Cấu trúc component**:
```javascript
// States
users              // Danh sách users
selectedUser       // User được chọn
userProfile        // Profile chi tiết
userTransactions   // Lịch sử GD
behavioralFeatures // Chỉ số hành vi

// Render functions
renderUserSelector()      // Dropdown chọn user
renderUserProfile()       // Hiển thị profile
renderTransactionHistory() // Bảng lịch sử GD
renderTransactionForm()   // Form nhập GD mới
renderResults()           // Kết quả phân tích
```

---

## Lưu ý quan trọng

1. **Tương thích ngược**: API cũ vẫn hoạt động bình thường
2. **Fallback**: Nếu không có MongoDB, sử dụng mock data tự động
3. **Mock data**: 8 users mẫu với thông tin đa dạng để test
4. **Behavioral features**: Được tính toán real-time từ database
5. **Giao dịch được lưu**: Mỗi lần test sẽ lưu vào database cho việc học liên tục

---

*Version 2.0.0 - Cập nhật ngày 07/12/2024*

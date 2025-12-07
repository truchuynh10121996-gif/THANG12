# Giải thích chi tiết các Models

## Layer 1: Phát hiện Lừa đảo Toàn cục

### 1. Isolation Forest (Rừng Cô lập)

#### Giải thích đơn giản

Hãy tưởng tượng bạn có một bãi đỗ xe với 100 chiếc xe. 99 chiếc là xe máy bình thường, 1 chiếc là xe tải lớn. Nếu bạn muốn "cô lập" một chiếc xe bất kỳ:
- Xe tải lớn sẽ rất dễ chỉ ra (chỉ cần nói "chiếc xe lớn nhất")
- Xe máy thì khó hơn nhiều (cần mô tả chi tiết hơn)

**Isolation Forest hoạt động tương tự**: Những giao dịch bất thường (anomaly) sẽ dễ bị "cô lập" hơn, cần ít bước hơn để phân biệt với phần còn lại.

#### Cách hoạt động

```
Bước 1: Chọn ngẫu nhiên 1 feature (ví dụ: số tiền)
Bước 2: Chọn ngẫu nhiên 1 ngưỡng (ví dụ: 10 triệu)
Bước 3: Chia dữ liệu thành 2 nhóm (< 10tr và >= 10tr)
Bước 4: Lặp lại cho đến khi mỗi điểm được cô lập

Điểm cần ít bước = Bất thường
Điểm cần nhiều bước = Bình thường
```

#### Ví dụ cụ thể

```
Giao dịch A: 500 triệu lúc 3h sáng → Cần 2 bước để cô lập → ANOMALY
Giao dịch B: 500k lúc 10h sáng → Cần 8 bước để cô lập → NORMAL
```

#### Ưu điểm
- ✅ Không cần dữ liệu có nhãn (unsupervised)
- ✅ Rất nhanh với dữ liệu lớn
- ✅ Phát hiện được các loại fraud mới

#### Nhược điểm
- ❌ Khó giải thích tại sao đánh dấu là fraud
- ❌ Cần tune tham số contamination (tỷ lệ anomaly)

#### Hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| n_estimators | 100 | Số lượng cây (nhiều hơn = ổn định hơn) |
| contamination | 0.05 | Tỷ lệ anomaly dự kiến (5%) |
| max_samples | 'auto' | Số mẫu cho mỗi cây |

---

### 2. LightGBM (Gradient Boosting nhẹ)

#### Giải thích đơn giản

LightGBM giống như một nhóm chuyên gia đưa ra quyết định:
- Chuyên gia 1 nhìn vào số tiền
- Chuyên gia 2 nhìn vào thời gian
- Chuyên gia 3 nhìn vào lịch sử user
- ...

Mỗi chuyên gia học từ SAI LẦM của chuyên gia trước đó. Kết hợp ý kiến của tất cả → Quyết định cuối cùng.

#### Cách hoạt động

```
Cây 1: Dự đoán → Sai lầm: [0.1, -0.2, 0.3, ...]
Cây 2: Học từ sai lầm của Cây 1 → Sai lầm mới: [0.05, -0.1, ...]
Cây 3: Học từ sai lầm của Cây 2 → ...
...
Cây 200: Sai lầm rất nhỏ → KẾT THÚC
```

#### Ví dụ cụ thể

```
Input: Giao dịch 50 triệu lúc 2h sáng từ thiết bị mới

Cây 1: "Số tiền lớn" → Score: 0.3
Cây 2: "Thời gian đêm" → Score: +0.2
Cây 3: "Thiết bị mới" → Score: +0.3
...
Final Score: 0.8 → FRAUD
```

#### Feature Importance

LightGBM cho biết feature nào quan trọng nhất:

```
1. amount              (30%)
2. hour               (15%)
3. is_new_device      (12%)
4. time_since_last    (10%)
5. is_international   (8%)
```

#### Ưu điểm
- ✅ Rất chính xác khi có dữ liệu tốt
- ✅ Cho biết feature nào quan trọng
- ✅ Xử lý tốt dữ liệu không cân bằng

#### Nhược điểm
- ❌ Cần dữ liệu có nhãn (is_fraud)
- ❌ Có thể overfit nếu không cẩn thận

---

## Layer 2: Hồ sơ User (Nâng cao)

### 1. Autoencoder (Bộ mã hóa tự động)

#### Giải thích đơn giản

Hãy tưởng tượng Autoencoder như một họa sĩ vẽ chân dung:
1. Nhìn người thật (64 đặc điểm)
2. Ghi nhớ những đặc điểm quan trọng nhất (8 đặc điểm)
3. Vẽ lại từ trí nhớ (64 đặc điểm)

Nếu bức vẽ khác xa người thật → Có gì đó BẤT THƯỜNG!

#### Kiến trúc

```
Input (64) → 32 → 16 → [8 - Embedding] → 16 → 32 → Output (64)
        ENCODER              ↑           DECODER
                        "Bộ nhớ"
```

#### Cách phát hiện fraud

```
User bình thường:
  Input: [1.2, 0.5, 0.8, ...]
  Output: [1.1, 0.6, 0.7, ...]  ← Giống nhau
  Error: 0.02 ← Thấp = NORMAL

User bất thường:
  Input: [1.2, 5.0, 0.8, ...]  ← Có giá trị lạ
  Output: [1.1, 0.6, 0.7, ...]  ← Không tái tạo được giá trị lạ
  Error: 0.85 ← Cao = ANOMALY
```

#### Ưu điểm
- ✅ Tạo được "hình ảnh" (embedding) của mỗi user
- ✅ Phát hiện hành vi khác thường so với chính user đó

#### Nhược điểm
- ❌ Cần đủ dữ liệu lịch sử của user
- ❌ Khó giải thích kết quả

---

### 2. LSTM (Mạng bộ nhớ dài-ngắn hạn)

#### Giải thích đơn giản

LSTM giống như một điều tra viên theo dõi lịch sử giao dịch:

```
Giao dịch 1: 500k siêu thị
Giao dịch 2: 200k xăng
Giao dịch 3: 1tr nhà hàng
Giao dịch 4: 50k cafe
Giao dịch 5: 500 TRIỆU chuyển khoản ← BẤT THƯỜNG!
```

LSTM nhớ "nhịp điệu" chi tiêu thông thường và phát hiện khi nhịp điệu bị phá vỡ.

#### Cách hoạt động

```
Sequence: [txn1, txn2, txn3, txn4, txn5, txn6, txn7, txn8, txn9, txn10]
                                                                    ↓
LSTM đọc từ trái qua phải, nhớ các patterns:
- "User hay mua hàng vào cuối tuần"
- "Số tiền thường < 2 triệu"
- "Thường giao dịch 10-20h"

Giao dịch tiếp theo:
- 500 triệu lúc 3h sáng → KHÔNG KHỚP PATTERN → FRAUD
```

#### Attention Mechanism

LSTM với attention biết giao dịch nào quan trọng:

```
[0.05, 0.03, 0.02, 0.04, 0.80, 0.03, 0.02, 0.01]
                        ↑
        Giao dịch này quan trọng nhất cho dự đoán!
```

#### Ưu điểm
- ✅ Hiểu được thứ tự thời gian
- ✅ Phát hiện patterns phức tạp

#### Nhược điểm
- ❌ Cần sequence đủ dài (10+ giao dịch)
- ❌ Training chậm hơn các models khác

---

### 3. GNN (Mạng Nơ-ron Đồ thị)

#### Giải thích đơn giản

GNN nhìn vào MỐI QUAN HỆ giữa các tài khoản:

```
     User A ←──── chuyển tiền ────→ User B
        ↓                              ↓
    User C ←── cùng merchant ──→ User D
        ↓                              ↓
     User E ←──── cùng IP ────→ User F
```

Nếu User A là fraud, và User B nhận tiền từ A → B cũng đáng ngờ!

#### Cách xây dựng Graph

```
Nodes (đỉnh):
- Users (50,000)
- Merchants (1,000)
- Banks (50)

Edges (cạnh):
- User → Merchant (mua hàng)
- User → User (chuyển khoản)
- User → Bank (giao dịch)
```

#### Cách hoạt động

```
Bước 1: Mỗi node có features riêng
        User A: [income=10tr, age=25, ...]

Bước 2: Mỗi node học từ neighbors
        User A nhận thông tin từ B, C, D
        A_new = A + average(B, C, D)

Bước 3: Lặp lại 3 lần
        Thông tin lan truyền trong graph

Bước 4: Phân loại mỗi node
        User A: fraud/normal
```

#### Phát hiện Fraud Community

```
    ┌───────────────────────────────┐
    │   FRAUD RING (Vòng lừa đảo)  │
    │                               │
    │  User1 ←──→ User2 ←──→ User3 │
    │    ↑           ↓           ↓  │
    │  User4 ←──→ User5 ←──→ User6 │
    │                               │
    └───────────────────────────────┘

GNN phát hiện: Nhóm này có kết nối chặt chẽ bất thường!
```

#### Ưu điểm
- ✅ Phát hiện fraud rings/communities
- ✅ Học được từ cấu trúc mạng lưới

#### Nhược điểm
- ❌ Phức tạp để implement
- ❌ Cần xây dựng graph từ dữ liệu

---

## So sánh các Models

| Model | Input | Supervised | Điểm mạnh | Điểm yếu |
|-------|-------|------------|-----------|----------|
| Isolation Forest | Features | ❌ | Phát hiện anomaly | Khó giải thích |
| LightGBM | Features | ✅ | Chính xác, nhanh | Cần labels |
| Autoencoder | Features | ❌ | User profiling | Cần history |
| LSTM | Sequences | ✅ | Temporal patterns | Chậm |
| GNN | Graph | ✅ | Communities | Phức tạp |

## Cách kết hợp

### Ensemble Weights

```python
# Layer 1 (40%)
isolation_forest: 15%
lightgbm: 25%

# Layer 2 (60%)
autoencoder: 20%
lstm: 20%
gnn: 20%
```

### Ví dụ kết hợp

```
Giao dịch: 100 triệu lúc 2h sáng

Isolation Forest: 0.7 (bất thường về số tiền + thời gian)
LightGBM: 0.8 (patterns giống fraud đã biết)
Autoencoder: 0.6 (khác với hành vi thường ngày của user)
LSTM: 0.9 (phá vỡ sequence bình thường)
GNN: 0.5 (người nhận không đáng ngờ)

Final Score: 0.15*0.7 + 0.25*0.8 + 0.20*0.6 + 0.20*0.9 + 0.20*0.5
           = 0.105 + 0.20 + 0.12 + 0.18 + 0.10
           = 0.705

Risk Level: HIGH → Cần xác nhận OTP
```

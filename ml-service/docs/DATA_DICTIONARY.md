# Data Dictionary - Từ điển dữ liệu

## Dữ liệu thô (Raw Data)

### 1. users_raw.csv

| Trường | Kiểu | Mô tả | Ví dụ |
|--------|------|-------|-------|
| user_id | string | ID định danh user | USR00000001 |
| age | int | Tuổi | 35 |
| gender | string | Giới tính (M/F) | M |
| province | string | Tỉnh/Thành phố | Hà Nội |
| occupation | string | Nghề nghiệp | employee |
| monthly_income | int | Thu nhập hàng tháng (VND) | 15000000 |
| account_balance | int | Số dư tài khoản (VND) | 50000000 |
| account_age_years | float | Số năm sử dụng | 3.5 |
| credit_score | int | Điểm tín dụng (300-850) | 720 |
| avg_monthly_transactions | int | Số giao dịch TB/tháng | 25 |
| risk_profile | string | Mức độ rủi ro (low/medium/high) | low |
| registration_date | date | Ngày đăng ký | 2020-06-15 |
| is_verified | bool | Đã xác minh | true |
| has_2fa | bool | Có 2FA | true |

### 2. transactions_raw.csv

| Trường | Kiểu | Mô tả | Ví dụ |
|--------|------|-------|-------|
| transaction_id | string | ID giao dịch | TXN0000000001 |
| user_id | string | ID user thực hiện | USR00000001 |
| timestamp | datetime | Thời gian giao dịch | 2024-01-15 14:30:00 |
| amount | int | Số tiền (VND) | 5000000 |
| transaction_type | string | Loại giao dịch | transfer |
| channel | string | Kênh thực hiện | mobile_app |
| device_type | string | Loại thiết bị | android |
| merchant_category | string | Danh mục merchant | supermarket |
| location_country | string | Quốc gia | VN |
| receiving_bank | string | Ngân hàng nhận | Vietcombank |
| balance_before | int | Số dư trước | 55000000 |
| balance_after | int | Số dư sau | 50000000 |
| ip_address | string | Địa chỉ IP | 14.170.xxx.xxx |
| device_id | string | ID thiết bị | DEV00000001_01 |
| session_id | string | ID phiên | SES0000000001 |
| is_international | bool | Giao dịch quốc tế | false |
| is_recurring | bool | Giao dịch định kỳ | false |
| is_fraud | int | Nhãn lừa đảo (0/1) | 0 |
| fraud_type | string | Loại lừa đảo (nếu có) | null |

### 3. fraud_reports_raw.csv

| Trường | Kiểu | Mô tả | Ví dụ |
|--------|------|-------|-------|
| report_id | string | ID báo cáo | RPT00000001 |
| transaction_id | string | ID giao dịch liên quan | TXN0000000001 |
| user_id | string | ID user | USR00000001 |
| report_date | datetime | Ngày báo cáo | 2024-01-15 18:00:00 |
| fraud_type | string | Loại lừa đảo | unusual_amount |
| amount | int | Số tiền bị lừa | 50000000 |
| source | string | Nguồn báo cáo | customer_complaint |
| description | string | Mô tả chi tiết | Số tiền giao dịch lớn bất thường |
| status | string | Trạng thái xử lý | confirmed |
| recovered_amount | int | Số tiền thu hồi | 40000000 |

---

## Giá trị các trường

### transaction_type (Loại giao dịch)

| Giá trị | Mô tả tiếng Việt |
|---------|------------------|
| transfer | Chuyển khoản |
| payment | Thanh toán |
| withdrawal | Rút tiền |
| deposit | Nạp tiền |
| bill_payment | Thanh toán hóa đơn |
| card_purchase | Mua hàng bằng thẻ |
| online_purchase | Mua hàng online |

### channel (Kênh giao dịch)

| Giá trị | Mô tả tiếng Việt |
|---------|------------------|
| mobile_app | Ứng dụng di động |
| web_banking | Internet banking |
| atm | Máy ATM |
| pos | Điểm bán hàng (POS) |
| branch | Chi nhánh ngân hàng |

### device_type (Loại thiết bị)

| Giá trị | Mô tả |
|---------|-------|
| ios | iPhone/iPad |
| android | Điện thoại Android |
| web | Trình duyệt web |
| desktop | Ứng dụng desktop |
| other | Khác |

### fraud_type (Loại lừa đảo)

| Giá trị | Mô tả tiếng Việt | Đặc điểm |
|---------|------------------|----------|
| unusual_amount | Số tiền bất thường | Số tiền lớn hơn 5x thu nhập |
| unusual_time | Thời gian bất thường | Giao dịch 1-5h sáng |
| new_recipient | Người nhận mới | Chuyển tiền cho người chưa giao dịch |
| rapid_succession | Giao dịch liên tiếp | Nhiều giao dịch < 1 giờ |
| foreign_location | Địa điểm nước ngoài | IP từ quốc gia đáng ngờ |
| device_change | Đổi thiết bị | Thiết bị mới chưa đăng ký |
| velocity_abuse | Vượt giới hạn | Vượt số lượng/số tiền cho phép |
| account_takeover | Chiếm đoạt tài khoản | Thay đổi hoàn toàn hành vi |

---

## Features (Đặc trưng) được tạo

### Temporal Features (Thời gian)

| Feature | Công thức | Mô tả |
|---------|-----------|-------|
| hour | timestamp.hour | Giờ giao dịch (0-23) |
| day_of_week | timestamp.dayofweek | Ngày trong tuần (0-6) |
| day_of_month | timestamp.day | Ngày trong tháng (1-31) |
| month | timestamp.month | Tháng (1-12) |
| is_weekend | day_of_week >= 5 | Cuối tuần (1/0) |
| is_night_time | hour < 6 or hour >= 22 | Ban đêm (1/0) |
| is_business_hours | 9 <= hour <= 17 and weekday | Giờ làm việc (1/0) |
| is_salary_day | day_of_month <= 5 | Ngày lương (1/0) |
| hour_sin | sin(2π * hour / 24) | Encoding cyclical |
| hour_cos | cos(2π * hour / 24) | Encoding cyclical |

### Amount Features (Số tiền)

| Feature | Công thức | Mô tả |
|---------|-----------|-------|
| log_amount | log(1 + amount) | Log của số tiền |
| amount_zscore | (amount - mean) / std | Z-score của số tiền |
| balance_change | balance_after - balance_before | Thay đổi số dư |
| balance_change_ratio | balance_change / balance_before | Tỷ lệ thay đổi |
| amount_bin | qcut(amount, 10) | Phân nhóm số tiền |

### User Aggregation Features

| Feature | Công thức | Mô tả |
|---------|-----------|-------|
| user_avg_amount | mean(user.amounts) | Số tiền TB của user |
| user_std_amount | std(user.amounts) | Độ lệch chuẩn |
| user_max_amount | max(user.amounts) | Số tiền max |
| user_txn_count | count(user.txns) | Tổng số giao dịch |
| amount_vs_user_avg | amount / user_avg_amount | So với TB user |
| is_unusual_amount | \|amount - avg\| > 2*std | Số tiền bất thường |

### Velocity Features (Tốc độ)

| Feature | Công thức | Mô tả |
|---------|-----------|-------|
| time_since_last_txn | current - prev timestamp (hours) | Thời gian từ giao dịch trước |
| is_rapid_txn | time_since_last < 1 hour | Giao dịch nhanh |
| is_very_rapid | time_since_last < 6 mins | Giao dịch rất nhanh |
| user_txn_sequence | cumcount per user | Thứ tự giao dịch |

### Behavioral Features (Hành vi)

| Feature | Công thức | Mô tả |
|---------|-----------|-------|
| is_international | location_country != 'VN' | Quốc tế |
| is_domestic | location_country == 'VN' | Trong nước |
| is_recurring | is_recurring flag | Giao dịch định kỳ |
| is_transfer | txn_type == 'transfer' | Chuyển khoản |
| channel_risk | map(channel → risk) | Độ rủi ro kênh |

---

## Quy trình làm sạch dữ liệu

### Step 1: Loại bỏ duplicates

```python
# Tiêu chí trùng lặp
duplicate_cols = ['user_id', 'timestamp', 'amount', 'transaction_type']
df = df.drop_duplicates(subset=duplicate_cols, keep='first')
```

### Step 2: Xử lý Missing Values

```python
# Critical fields → Xóa record
df = df.dropna(subset=['user_id', 'amount', 'timestamp'])

# Numerical → Điền median
df['amount'] = df['amount'].fillna(df['amount'].median())

# Categorical → Điền 'unknown'
df['channel'] = df['channel'].fillna('unknown')
```

### Step 3: Xử lý Outliers

```python
# IQR method
Q1 = df['amount'].quantile(0.01)
Q3 = df['amount'].quantile(0.99)
IQR = Q3 - Q1

# Giữ dữ liệu trong khoảng hợp lý
# KHÔNG xóa fraud transactions
df['amount'] = df['amount'].clip(lower=10_000)  # Min 10k VND
```

### Step 4: Chuẩn hóa Format

```python
# Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Amount thành integer
df['amount'] = df['amount'].astype(int)

# Text lowercase
df['channel'] = df['channel'].str.lower().str.strip()
```

### Step 5: Loại bỏ Noise

```python
# Giao dịch tương lai
df = df[df['timestamp'] <= datetime.now()]

# Số tiền = 0
df = df[df['amount'] > 0]

# User ID không hợp lệ
df = df[df['user_id'].str.len() > 0]
```

---

## Thống kê dữ liệu mẫu

```
Tổng số users: 50,000
Tổng số giao dịch: 500,000
Tỷ lệ fraud: 5% (25,000 giao dịch)

Phân bố fraud types:
- unusual_amount:     25%
- unusual_time:       15%
- new_recipient:      15%
- rapid_succession:   12%
- foreign_location:   10%
- device_change:      10%
- velocity_abuse:      8%
- account_takeover:    5%
```

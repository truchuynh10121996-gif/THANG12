"""
LSTM Data Pipeline - Pipeline xử lý dữ liệu cho LSTM Fraud Detection
=====================================================================
Chuyển đổi dữ liệu giao dịch raw thành sequences 3D cho LSTM training.

Input: Raw CSV với các cột:
    transaction_id, user_id, timestamp, amount, recipient_id, balance,
    channel, device_id, location, recipient_bank, is_fraud, ...

Output:
    X_train: shape (n_train_sequences, seq_length, 40)  # 40 features
    y_train: shape (n_train_sequences,)
    X_test:  shape (n_test_sequences, seq_length, 40)
    y_test:  shape (n_test_sequences,)

Đặc điểm:
- Sequence length = 7: Mỗi sequence gồm 7 giao dịch liên tiếp của 1 user
- Label: is_fraud của giao dịch CUỐI trong sequence
- Padding: Users có < 7 tx được pad zeros ở đầu
- Split by user: 80% users cho train, 20% users cho test (không mix)
- Feature tính tại thời điểm giao dịch: Không data leakage từ tương lai

Author: ML Team
Created: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DANH SÁCH 40 FEATURES CHO LSTM
# ============================================================================
LSTM_FEATURES_VN = {
    # === AMOUNT (6) ===
    'amount_log': 'log(amount + 1)',
    'amount_to_balance_ratio': 'amount / balance',
    'z_score_amount': '(amount - mean) / std của user',
    'is_round_amount': 'amount % 1000000 == 0',
    'amount_std_7d': 'std của amount trong 7 ngày',
    'amount_vs_user_avg_ratio': 'amount / avg_amount của user',

    # === TEMPORAL (6) ===
    'hour_sin': 'sin(2π × hour / 24)',
    'hour_cos': 'cos(2π × hour / 24)',
    'day_of_week_sin': 'sin(2π × day / 7)',
    'day_of_week_cos': 'cos(2π × day / 7)',
    'is_night_transaction': 'hour in [0, 5]',
    'is_salary_period': 'day in [1, 5] or [25, 31]',

    # === VELOCITY (7) ===
    'tx_count_1h': 'Số tx trong 1 giờ trước',
    'tx_count_24h': 'Số tx trong 24 giờ trước',
    'minutes_since_last_tx': 'Phút từ tx trước',
    'velocity_change': 'tx_count_1h / tx_count_24h',
    'amount_acceleration': '(amount - prev_amount) / time_diff',
    'unique_recipients_24h': 'Số người nhận khác nhau trong 24h',
    'cumulative_amount_24h': 'Tổng tiền chuyển trong 24h',

    # === RECIPIENT (7) ===
    'is_new_recipient': 'Chưa từng chuyển cho người này',
    'recipient_tx_count': 'Số lần đã chuyển cho người này',
    'is_same_bank': 'Cùng ngân hàng',
    'tx_count_to_same_recipient_24h': 'Số tx cho cùng người trong 24h',
    'recipient_account_age_days': 'Tuổi tài khoản người nhận',
    'time_since_last_tx_to_recipient': 'Thời gian từ lần chuyển trước cho người này',
    'cumulative_amount_to_recipient_24h': 'Tổng tiền cho người này trong 24h',

    # === DEVICE/CHANNEL (4) ===
    'is_new_device': 'Device mới',
    'channel_encoded': 'MOBILE=0, WEB=1, ATM=2',
    'is_usual_location': 'Vị trí thường dùng',
    'device_tx_count_24h': 'Số tx từ device này trong 24h',

    # === ACCOUNT HEALTH (4) ===
    'account_age_days': 'Tuổi tài khoản',
    'avg_balance_7d': 'Số dư trung bình 7 ngày',
    'balance_after_tx_ratio': '(balance - amount) / balance',
    'is_near_zero_balance_after': 'balance_after < 100000',

    # === SCAM-SPECIFIC VN (6) ===
    'is_incremental_amount': 'amount > prev_amount * 1.5',
    'is_during_business_hours': 'hour in [8, 17] và weekday',
    'recipient_total_received_24h': 'Tổng tiền người nhận nhận được trong 24h',
    'is_transfer_to_ewallet': "recipient_id starts with 'MOMO', 'ZALO', 'VNPAY'",
    'is_first_large_tx_to_new_recipient': 'is_new_recipient AND amount > 5000000',
    'small_tx_count_before_large': 'Số tx < 500k trước tx hiện tại (trong 24h)',
}

# Danh sách tên features theo thứ tự
FEATURE_NAMES = [
    # Amount (6)
    'amount_log', 'amount_to_balance_ratio', 'z_score_amount',
    'is_round_amount', 'amount_std_7d', 'amount_vs_user_avg_ratio',
    # Temporal (6)
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'is_night_transaction', 'is_salary_period',
    # Velocity (7)
    'tx_count_1h', 'tx_count_24h', 'minutes_since_last_tx',
    'velocity_change', 'amount_acceleration', 'unique_recipients_24h',
    'cumulative_amount_24h',
    # Recipient (7)
    'is_new_recipient', 'recipient_tx_count', 'is_same_bank',
    'tx_count_to_same_recipient_24h', 'recipient_account_age_days',
    'time_since_last_tx_to_recipient', 'cumulative_amount_to_recipient_24h',
    # Device/Channel (4)
    'is_new_device', 'channel_encoded', 'is_usual_location', 'device_tx_count_24h',
    # Account Health (4)
    'account_age_days', 'avg_balance_7d', 'balance_after_tx_ratio',
    'is_near_zero_balance_after',
    # Scam-Specific VN (6)
    'is_incremental_amount', 'is_during_business_hours',
    'recipient_total_received_24h', 'is_transfer_to_ewallet',
    'is_first_large_tx_to_new_recipient', 'small_tx_count_before_large',
]


def load_and_sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Đọc và sắp xếp dữ liệu theo user_id và timestamp

    Args:
        df: DataFrame với dữ liệu giao dịch raw

    Returns:
        DataFrame đã được sắp xếp theo user_id và timestamp

    Note:
        - Chuyển đổi timestamp sang datetime format
        - Sort theo user_id trước, sau đó theo timestamp
        - Reset index để có index liên tục
    """
    print("[PIPELINE] Bước 1: Đọc và sắp xếp dữ liệu...")

    df = df.copy()

    # Kiểm tra cột bắt buộc
    required_cols = ['user_id', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    # Chuyển đổi timestamp sang datetime
    if df['timestamp'].dtype == 'object':
        # Thử nhiều format khác nhau
        for fmt in [
            '%m/%d/%Y %I:%M:%S %p',  # 2/13/2025 8:18:38 PM
            '%Y-%m-%d %H:%M:%S',      # 2025-02-13 20:18:38
            '%d/%m/%Y %H:%M:%S',      # 13/02/2025 20:18:38
            '%Y-%m-%dT%H:%M:%S',      # ISO format
            None                       # Let pandas infer
        ]:
            try:
                if fmt:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                break
            except:
                continue

    # Đảm bảo timestamp là datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sắp xếp theo user_id và timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    print(f"  - Số giao dịch: {len(df):,}")
    print(f"  - Số users: {df['user_id'].nunique():,}")
    print(f"  - Khoảng thời gian: {df['timestamp'].min()} đến {df['timestamp'].max()}")

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 2: Tính toán 40 features cho mỗi giao dịch

    QUAN TRỌNG: Tất cả features được tính TẠI THỜI ĐIỂM giao dịch,
    chỉ sử dụng thông tin từ quá khứ để tránh data leakage.

    Args:
        df: DataFrame đã được sort theo user_id và timestamp

    Returns:
        DataFrame với 40 features được thêm vào
    """
    print("[PIPELINE] Bước 2: Tính toán 40 features...")

    df = df.copy()

    # =========================================================================
    # CHUẨN BỊ DỮ LIỆU
    # =========================================================================

    # Đảm bảo các cột số học
    numeric_cols = ['amount', 'balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Thêm cột amount nếu chưa có
    if 'amount' not in df.columns:
        df['amount'] = 0

    # Thêm cột balance nếu chưa có
    if 'balance' not in df.columns:
        df['balance'] = df['amount'] * 10  # Giả định

    # Trích xuất thông tin thời gian
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0 = Monday
    df['day_of_month'] = df['timestamp'].dt.day

    # =========================================================================
    # TÍNH FEATURES THEO NHÓM (USER-LEVEL)
    # =========================================================================

    print("  - Đang tính features theo từng user (có thể mất vài phút)...")

    # Khởi tạo các cột features với giá trị mặc định
    for feat in FEATURE_NAMES:
        df[feat] = 0.0

    # Tính features cho từng user
    users = df['user_id'].unique()
    total_users = len(users)

    for idx, user_id in enumerate(users):
        if (idx + 1) % 500 == 0:
            print(f"    Đã xử lý {idx + 1}/{total_users} users...")

        # Lấy giao dịch của user (đã được sort theo timestamp)
        user_mask = df['user_id'] == user_id
        user_indices = df[user_mask].index.tolist()

        if len(user_indices) == 0:
            continue

        # Lấy dữ liệu user
        user_df = df.loc[user_indices].copy()

        # Tính statistics cho user (chỉ dùng dữ liệu đã có)
        user_amounts = []
        user_recipients = set()
        user_devices = set()
        user_locations = []
        user_balances = []

        for i, row_idx in enumerate(user_indices):
            row = df.loc[row_idx]
            current_time = row['timestamp']
            amount = row['amount']
            balance = row.get('balance', amount * 10)

            # Cập nhật running statistics (chỉ từ dữ liệu trước đó)
            if i > 0:
                past_amounts = user_amounts.copy()
                user_mean_amount = np.mean(past_amounts) if past_amounts else amount
                user_std_amount = np.std(past_amounts) if len(past_amounts) > 1 else 1
            else:
                user_mean_amount = amount
                user_std_amount = 1

            # ================================================================
            # === AMOUNT FEATURES (6) ===
            # ================================================================

            # 1. amount_log: log(amount + 1)
            df.loc[row_idx, 'amount_log'] = np.log1p(amount)

            # 2. amount_to_balance_ratio: amount / balance
            df.loc[row_idx, 'amount_to_balance_ratio'] = (
                amount / balance if balance > 0 else 0
            )

            # 3. z_score_amount: (amount - mean) / std của user
            df.loc[row_idx, 'z_score_amount'] = (
                (amount - user_mean_amount) / user_std_amount if user_std_amount > 0 else 0
            )

            # 4. is_round_amount: amount % 1000000 == 0
            df.loc[row_idx, 'is_round_amount'] = int(amount % 1000000 == 0 and amount > 0)

            # 5. amount_std_7d: std của amount trong 7 ngày trước
            seven_days_ago = current_time - timedelta(days=7)
            amounts_7d = [
                user_amounts[j] for j in range(i)
                if df.loc[user_indices[j], 'timestamp'] >= seven_days_ago
            ]
            df.loc[row_idx, 'amount_std_7d'] = np.std(amounts_7d) if len(amounts_7d) > 1 else 0

            # 6. amount_vs_user_avg_ratio: amount / avg_amount của user
            df.loc[row_idx, 'amount_vs_user_avg_ratio'] = (
                amount / user_mean_amount if user_mean_amount > 0 else 1
            )

            # ================================================================
            # === TEMPORAL FEATURES (6) ===
            # ================================================================

            hour = row['hour']
            day_of_week = row['day_of_week']
            day_of_month = row['day_of_month']

            # 7. hour_sin: sin(2π × hour / 24)
            df.loc[row_idx, 'hour_sin'] = np.sin(2 * np.pi * hour / 24)

            # 8. hour_cos: cos(2π × hour / 24)
            df.loc[row_idx, 'hour_cos'] = np.cos(2 * np.pi * hour / 24)

            # 9. day_of_week_sin: sin(2π × day / 7)
            df.loc[row_idx, 'day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)

            # 10. day_of_week_cos: cos(2π × day / 7)
            df.loc[row_idx, 'day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)

            # 11. is_night_transaction: hour in [0, 5]
            df.loc[row_idx, 'is_night_transaction'] = int(hour in range(0, 6))

            # 12. is_salary_period: day in [1, 5] or [25, 31]
            df.loc[row_idx, 'is_salary_period'] = int(
                day_of_month in range(1, 6) or day_of_month in range(25, 32)
            )

            # ================================================================
            # === VELOCITY FEATURES (7) ===
            # ================================================================

            # Đếm giao dịch trong 1h và 24h trước
            one_hour_ago = current_time - timedelta(hours=1)
            twenty_four_hours_ago = current_time - timedelta(hours=24)

            tx_count_1h = sum(
                1 for j in range(i)
                if df.loc[user_indices[j], 'timestamp'] >= one_hour_ago
            )
            tx_count_24h = sum(
                1 for j in range(i)
                if df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago
            )

            # 13. tx_count_1h
            df.loc[row_idx, 'tx_count_1h'] = tx_count_1h

            # 14. tx_count_24h
            df.loc[row_idx, 'tx_count_24h'] = tx_count_24h

            # 15. minutes_since_last_tx
            if i > 0:
                prev_time = df.loc[user_indices[i-1], 'timestamp']
                minutes_diff = (current_time - prev_time).total_seconds() / 60
                df.loc[row_idx, 'minutes_since_last_tx'] = min(minutes_diff, 10080)  # Max 7 ngày
            else:
                df.loc[row_idx, 'minutes_since_last_tx'] = 10080  # 7 ngày nếu là tx đầu tiên

            # 16. velocity_change: tx_count_1h / tx_count_24h
            df.loc[row_idx, 'velocity_change'] = (
                tx_count_1h / tx_count_24h if tx_count_24h > 0 else 0
            )

            # 17. amount_acceleration: (amount - prev_amount) / time_diff
            if i > 0:
                prev_amount = user_amounts[-1]
                time_diff_hours = minutes_diff / 60 if minutes_diff > 0 else 1
                df.loc[row_idx, 'amount_acceleration'] = (
                    (amount - prev_amount) / time_diff_hours / 1000000  # Scale
                )
            else:
                df.loc[row_idx, 'amount_acceleration'] = 0

            # 18. unique_recipients_24h
            recipients_24h = set()
            for j in range(i):
                if df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago:
                    rcp = df.loc[user_indices[j]].get('recipient_id', '')
                    if rcp:
                        recipients_24h.add(rcp)
            df.loc[row_idx, 'unique_recipients_24h'] = len(recipients_24h)

            # 19. cumulative_amount_24h
            cum_amount_24h = sum(
                df.loc[user_indices[j], 'amount'] for j in range(i)
                if df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago
            )
            df.loc[row_idx, 'cumulative_amount_24h'] = cum_amount_24h / 1000000  # Scale to millions

            # ================================================================
            # === RECIPIENT FEATURES (7) ===
            # ================================================================

            recipient_id = row.get('recipient_id', '')
            recipient_bank = row.get('recipient_bank', '')

            # 20. is_new_recipient: Chưa từng chuyển cho người này
            df.loc[row_idx, 'is_new_recipient'] = int(recipient_id not in user_recipients)

            # 21. recipient_tx_count: Số lần đã chuyển cho người này
            recipient_count = sum(
                1 for j in range(i)
                if df.loc[user_indices[j]].get('recipient_id', '') == recipient_id
            )
            df.loc[row_idx, 'recipient_tx_count'] = recipient_count

            # 22. is_same_bank: Cùng ngân hàng (giả định user bank từ user_id prefix)
            user_bank = user_id[:3] if len(user_id) >= 3 else ''
            recipient_bank_prefix = recipient_bank[:3] if recipient_bank else ''
            df.loc[row_idx, 'is_same_bank'] = int(user_bank == recipient_bank_prefix)

            # 23. tx_count_to_same_recipient_24h
            tx_to_recipient_24h = sum(
                1 for j in range(i)
                if (df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago and
                    df.loc[user_indices[j]].get('recipient_id', '') == recipient_id)
            )
            df.loc[row_idx, 'tx_count_to_same_recipient_24h'] = tx_to_recipient_24h

            # 24. recipient_account_age_days (giả định từ recipient_id hoặc random)
            if 'recipient_account_age_days' in row:
                df.loc[row_idx, 'recipient_account_age_days'] = row['recipient_account_age_days']
            else:
                # Giả định: hash recipient_id để tạo giá trị consistent
                hash_val = hash(recipient_id) % 1000
                df.loc[row_idx, 'recipient_account_age_days'] = hash_val

            # 25. time_since_last_tx_to_recipient (phút)
            last_tx_to_recipient = None
            for j in range(i-1, -1, -1):
                if df.loc[user_indices[j]].get('recipient_id', '') == recipient_id:
                    last_tx_to_recipient = df.loc[user_indices[j], 'timestamp']
                    break
            if last_tx_to_recipient:
                time_since = (current_time - last_tx_to_recipient).total_seconds() / 60
                df.loc[row_idx, 'time_since_last_tx_to_recipient'] = min(time_since, 10080)
            else:
                df.loc[row_idx, 'time_since_last_tx_to_recipient'] = 10080  # 7 ngày

            # 26. cumulative_amount_to_recipient_24h
            cum_to_recipient_24h = sum(
                df.loc[user_indices[j], 'amount'] for j in range(i)
                if (df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago and
                    df.loc[user_indices[j]].get('recipient_id', '') == recipient_id)
            )
            df.loc[row_idx, 'cumulative_amount_to_recipient_24h'] = cum_to_recipient_24h / 1000000

            # ================================================================
            # === DEVICE/CHANNEL FEATURES (4) ===
            # ================================================================

            device_id = row.get('device_id', '')
            channel = row.get('channel', 'MOBILE')
            location = row.get('location', '')

            # 27. is_new_device
            df.loc[row_idx, 'is_new_device'] = int(device_id not in user_devices)

            # 28. channel_encoded: MOBILE=0, WEB=1, ATM=2
            channel_map = {'MOBILE': 0, 'WEB': 1, 'ATM': 2, 'API': 3}
            channel_upper = str(channel).upper()
            df.loc[row_idx, 'channel_encoded'] = channel_map.get(channel_upper, 0)

            # 29. is_usual_location
            # Nếu location xuất hiện nhiều lần trước đó => usual
            location_count = user_locations.count(location) if location else 0
            df.loc[row_idx, 'is_usual_location'] = int(location_count >= 2)

            # 30. device_tx_count_24h
            device_tx_24h = sum(
                1 for j in range(i)
                if (df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago and
                    df.loc[user_indices[j]].get('device_id', '') == device_id)
            )
            df.loc[row_idx, 'device_tx_count_24h'] = device_tx_24h

            # ================================================================
            # === ACCOUNT HEALTH FEATURES (4) ===
            # ================================================================

            # 31. account_age_days
            if 'account_age_days' in row:
                df.loc[row_idx, 'account_age_days'] = row['account_age_days']
            else:
                # Giả định từ số ngày từ giao dịch đầu tiên
                first_tx_time = df.loc[user_indices[0], 'timestamp']
                account_age = (current_time - first_tx_time).days
                df.loc[row_idx, 'account_age_days'] = max(account_age, 1)

            # 32. avg_balance_7d
            balances_7d = [
                df.loc[user_indices[j]].get('balance', 0) for j in range(i)
                if df.loc[user_indices[j], 'timestamp'] >= seven_days_ago
            ]
            df.loc[row_idx, 'avg_balance_7d'] = (
                np.mean(balances_7d) / 1000000 if balances_7d else balance / 1000000
            )

            # 33. balance_after_tx_ratio: (balance - amount) / balance
            balance_after = balance - amount
            df.loc[row_idx, 'balance_after_tx_ratio'] = (
                balance_after / balance if balance > 0 else 0
            )

            # 34. is_near_zero_balance_after: balance_after < 100000
            df.loc[row_idx, 'is_near_zero_balance_after'] = int(balance_after < 100000)

            # ================================================================
            # === SCAM-SPECIFIC VN FEATURES (6) ===
            # ================================================================

            # 35. is_incremental_amount: amount > prev_amount * 1.5
            if i > 0:
                prev_amount = user_amounts[-1]
                df.loc[row_idx, 'is_incremental_amount'] = int(
                    amount > prev_amount * 1.5 if prev_amount > 0 else 0
                )
            else:
                df.loc[row_idx, 'is_incremental_amount'] = 0

            # 36. is_during_business_hours: hour in [8, 17] và weekday (Mon-Fri)
            is_business_hour = 8 <= hour <= 17
            is_weekday = day_of_week < 5  # 0-4 = Mon-Fri
            df.loc[row_idx, 'is_during_business_hours'] = int(is_business_hour and is_weekday)

            # 37. recipient_total_received_24h
            # (Đây là tổng tiền NGƯỜI NHẬN đã nhận trong 24h từ TẤT CẢ users)
            # Trong thực tế cần query từ database, ở đây giả định = cumulative_amount_to_recipient_24h * 2
            df.loc[row_idx, 'recipient_total_received_24h'] = cum_to_recipient_24h * 2 / 1000000

            # 38. is_transfer_to_ewallet: recipient_id starts with 'MOMO', 'ZALO', 'VNPAY'
            ewallet_prefixes = ['MOMO', 'ZALO', 'VNPAY', 'VIETTEL', 'ZALOPAY', 'MOCA']
            recipient_upper = str(recipient_id).upper()
            df.loc[row_idx, 'is_transfer_to_ewallet'] = int(
                any(recipient_upper.startswith(prefix) for prefix in ewallet_prefixes)
            )

            # 39. is_first_large_tx_to_new_recipient: is_new_recipient AND amount > 5000000
            is_new = recipient_id not in user_recipients
            df.loc[row_idx, 'is_first_large_tx_to_new_recipient'] = int(
                is_new and amount > 5000000
            )

            # 40. small_tx_count_before_large: Số tx < 500k trước tx hiện tại (trong 24h)
            small_tx_count = sum(
                1 for j in range(i)
                if (df.loc[user_indices[j], 'timestamp'] >= twenty_four_hours_ago and
                    df.loc[user_indices[j], 'amount'] < 500000)
            )
            df.loc[row_idx, 'small_tx_count_before_large'] = (
                small_tx_count if amount >= 5000000 else 0
            )

            # ================================================================
            # CẬP NHẬT RUNNING STATE
            # ================================================================
            user_amounts.append(amount)
            if recipient_id:
                user_recipients.add(recipient_id)
            if device_id:
                user_devices.add(device_id)
            if location:
                user_locations.append(location)
            user_balances.append(balance)

    print(f"  - Đã tính xong {len(FEATURE_NAMES)} features")

    return df


def create_sequences(
    df: pd.DataFrame,
    seq_length: int = 7,
    feature_names: List[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List]]:
    """
    Bước 3: Tạo sequences cho mỗi user với sliding window

    Args:
        df: DataFrame với features đã tính
        seq_length: Độ dài sequence (mặc định 7)
        feature_names: Danh sách tên features

    Returns:
        Tuple chứa:
        - user_sequences: Dict {user_id: array shape (num_sequences, seq_length, num_features)}
        - user_labels: Dict {user_id: array shape (num_sequences,)}
        - user_tx_ids: Dict {user_id: list of transaction_ids cho mỗi sequence}

    Note:
        - Sử dụng sliding window với step = 1
        - Label = is_fraud của giao dịch CUỐI cùng trong sequence
        - Users có ít hơn seq_length giao dịch sẽ có ít sequences hơn
    """
    print(f"[PIPELINE] Bước 3: Tạo sequences (seq_length={seq_length})...")

    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Kiểm tra cột is_fraud
    if 'is_fraud' not in df.columns:
        raise ValueError("Cần có cột 'is_fraud' trong DataFrame")

    user_sequences = {}
    user_labels = {}
    user_tx_ids = {}

    users = df['user_id'].unique()

    for user_id in users:
        user_df = df[df['user_id'] == user_id].copy()

        if len(user_df) == 0:
            continue

        # Lấy features và labels
        X_user = user_df[feature_names].values
        y_user = user_df['is_fraud'].values
        tx_ids = user_df.get('transaction_id', user_df.index).tolist()

        sequences = []
        labels = []
        seq_tx_ids = []

        # Sliding window
        for i in range(len(X_user)):
            if i >= seq_length - 1:
                # Đủ giao dịch để tạo sequence đầy đủ
                seq = X_user[i - seq_length + 1:i + 1]
                sequences.append(seq)
                labels.append(y_user[i])  # Label của giao dịch cuối
                seq_tx_ids.append(tx_ids[i - seq_length + 1:i + 1])
            else:
                # Chưa đủ giao dịch - sẽ xử lý padding sau
                pass

        if sequences:
            user_sequences[user_id] = np.array(sequences)
            user_labels[user_id] = np.array(labels)
            user_tx_ids[user_id] = seq_tx_ids

    total_sequences = sum(len(v) for v in user_sequences.values())
    print(f"  - Số users có sequences: {len(user_sequences):,}")
    print(f"  - Tổng số sequences: {total_sequences:,}")

    return user_sequences, user_labels, user_tx_ids


def pad_sequences(
    df: pd.DataFrame,
    seq_length: int = 7,
    feature_names: List[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Bước 3b: Padding cho users có ít giao dịch (< seq_length)

    Tạo sequences với zero padding ở đầu cho users có ít hơn seq_length giao dịch.

    Args:
        df: DataFrame với features đã tính
        seq_length: Độ dài sequence (mặc định 7)
        feature_names: Danh sách tên features

    Returns:
        Tuple chứa:
        - padded_sequences: Dict {user_id: array shape (num_sequences, seq_length, num_features)}
        - padded_labels: Dict {user_id: array shape (num_sequences,)}

    Note:
        - Users có >= seq_length giao dịch: sliding window bình thường
        - Users có < seq_length giao dịch: pad zeros ở đầu mỗi sequence
    """
    print(f"[PIPELINE] Bước 3b: Padding sequences...")

    if feature_names is None:
        feature_names = FEATURE_NAMES

    num_features = len(feature_names)

    padded_sequences = {}
    padded_labels = {}

    users = df['user_id'].unique()
    users_need_padding = 0

    for user_id in users:
        user_df = df[df['user_id'] == user_id].copy()

        if len(user_df) == 0:
            continue

        X_user = user_df[feature_names].values
        y_user = user_df['is_fraud'].values
        n_tx = len(X_user)

        sequences = []
        labels = []

        if n_tx >= seq_length:
            # Đủ giao dịch - sliding window bình thường
            for i in range(seq_length - 1, n_tx):
                seq = X_user[i - seq_length + 1:i + 1]
                sequences.append(seq)
                labels.append(y_user[i])
        else:
            # Không đủ giao dịch - pad zeros ở đầu
            users_need_padding += 1
            for i in range(n_tx):
                # Tạo sequence với padding
                seq = np.zeros((seq_length, num_features))
                # Copy giao dịch vào cuối sequence
                start_idx = seq_length - i - 1
                seq[start_idx:] = X_user[:i + 1]
                sequences.append(seq)
                labels.append(y_user[i])

        if sequences:
            padded_sequences[user_id] = np.array(sequences)
            padded_labels[user_id] = np.array(labels)

    total_sequences = sum(len(v) for v in padded_sequences.values())
    print(f"  - Số users cần padding: {users_need_padding:,}")
    print(f"  - Tổng số sequences (sau padding): {total_sequences:,}")

    return padded_sequences, padded_labels


def split_by_user(
    user_sequences: Dict[str, np.ndarray],
    user_labels: Dict[str, np.ndarray],
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Bước 4: Split train/test theo user (không random)

    QUAN TRỌNG: Split theo USER, không theo sequence.
    Điều này đảm bảo không có data leakage giữa train và test.

    Args:
        user_sequences: Dict {user_id: sequences array}
        user_labels: Dict {user_id: labels array}
        test_ratio: Tỷ lệ users cho test set (mặc định 0.2 = 20%)
        random_state: Seed cho reproducibility

    Returns:
        Tuple chứa:
        - X_train: array shape (n_train_sequences, seq_length, num_features)
        - y_train: array shape (n_train_sequences,)
        - X_test: array shape (n_test_sequences, seq_length, num_features)
        - y_test: array shape (n_test_sequences,)
        - train_users: List user_ids trong train set
        - test_users: List user_ids trong test set
    """
    print(f"[PIPELINE] Bước 4: Split train/test theo user (test_ratio={test_ratio})...")

    # Lấy danh sách users
    users = list(user_sequences.keys())
    n_users = len(users)
    n_test_users = int(n_users * test_ratio)

    # Set seed và shuffle
    np.random.seed(random_state)
    shuffled_users = np.random.permutation(users).tolist()

    # Split users
    test_users = shuffled_users[:n_test_users]
    train_users = shuffled_users[n_test_users:]

    # Gom sequences theo split
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for user_id in train_users:
        X_train_list.append(user_sequences[user_id])
        y_train_list.append(user_labels[user_id])

    for user_id in test_users:
        X_test_list.append(user_sequences[user_id])
        y_test_list.append(user_labels[user_id])

    # Concatenate
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])

    print(f"  - Train users: {len(train_users):,}")
    print(f"  - Test users: {len(test_users):,}")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - Train fraud ratio: {y_train.mean()*100:.2f}%")
    print(f"  - Test fraud ratio: {y_test.mean()*100:.2f}%")

    return X_train, y_train, X_test, y_test, train_users, test_users


def prepare_lstm_data(
    df_raw: pd.DataFrame,
    seq_length: int = 7,
    test_ratio: float = 0.2,
    include_padding: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Function chính: Pipeline hoàn chỉnh từ raw data đến LSTM-ready sequences

    Args:
        df_raw: DataFrame với dữ liệu giao dịch raw
        seq_length: Độ dài sequence (mặc định 7)
        test_ratio: Tỷ lệ test set (mặc định 0.2)
        include_padding: Có pad cho users ít giao dịch không (mặc định True)
        verbose: In thông tin chi tiết

    Returns:
        Dict chứa:
        {
            'X_train': np.ndarray,     # shape (n_train, seq_length, 40)
            'y_train': np.ndarray,     # shape (n_train,)
            'X_test': np.ndarray,      # shape (n_test, seq_length, 40)
            'y_test': np.ndarray,      # shape (n_test,)
            'feature_names': List[str], # 40 feature names
            'train_users': List[str],   # User IDs trong train
            'test_users': List[str],    # User IDs trong test
            'metadata': Dict            # Thống kê và thông tin
        }

    Example:
        ```python
        import pandas as pd
        from utils.lstm_data_pipeline import prepare_lstm_data

        df = pd.read_csv('transactions.csv')
        result = prepare_lstm_data(df, seq_length=7, test_ratio=0.2)

        X_train = result['X_train']  # (n_train, 7, 40)
        y_train = result['y_train']  # (n_train,)
        X_test = result['X_test']    # (n_test, 7, 40)
        y_test = result['y_test']    # (n_test,)
        ```
    """
    if verbose:
        print("=" * 70)
        print("LSTM DATA PIPELINE - Bắt đầu xử lý dữ liệu")
        print("=" * 70)
        print(f"  - Sequence length: {seq_length}")
        print(f"  - Test ratio: {test_ratio}")
        print(f"  - Include padding: {include_padding}")
        print(f"  - Số features: {len(FEATURE_NAMES)}")
        print()

    # Bước 1: Load và sort
    df = load_and_sort_data(df_raw)

    # Bước 2: Calculate features
    df = calculate_features(df)

    # Bước 3: Create sequences (với hoặc không có padding)
    if include_padding:
        user_sequences, user_labels = pad_sequences(df, seq_length, FEATURE_NAMES)
    else:
        user_sequences, user_labels, _ = create_sequences(df, seq_length, FEATURE_NAMES)

    # Bước 4: Split by user
    X_train, y_train, X_test, y_test, train_users, test_users = split_by_user(
        user_sequences, user_labels, test_ratio
    )

    # Chuẩn bị metadata
    metadata = {
        'total_transactions': len(df),
        'total_users': df['user_id'].nunique(),
        'seq_length': seq_length,
        'num_features': len(FEATURE_NAMES),
        'train_users_count': len(train_users),
        'test_users_count': len(test_users),
        'train_sequences_count': len(X_train),
        'test_sequences_count': len(X_test),
        'train_fraud_count': int(y_train.sum()) if len(y_train) > 0 else 0,
        'test_fraud_count': int(y_test.sum()) if len(y_test) > 0 else 0,
        'train_fraud_ratio': float(y_train.mean()) if len(y_train) > 0 else 0,
        'test_fraud_ratio': float(y_test.mean()) if len(y_test) > 0 else 0,
        'include_padding': include_padding,
    }

    if verbose:
        print()
        print("=" * 70)
        print("HOÀN TẤT PIPELINE!")
        print("=" * 70)
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - X_test shape: {X_test.shape}")
        print(f"  - y_test shape: {y_test.shape}")
        print(f"  - Train fraud ratio: {metadata['train_fraud_ratio']*100:.2f}%")
        print(f"  - Test fraud ratio: {metadata['test_fraud_ratio']*100:.2f}%")
        print()

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': FEATURE_NAMES.copy(),
        'train_users': train_users,
        'test_users': test_users,
        'metadata': metadata,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_raw_data(df: pd.DataFrame) -> Dict:
    """
    Kiểm tra và validate dữ liệu raw trước khi xử lý

    Args:
        df: DataFrame raw

    Returns:
        Dict với thông tin validation
    """
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # Kiểm tra cột bắt buộc
    required_cols = ['user_id', 'timestamp', 'amount', 'is_fraud']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        result['is_valid'] = False
        result['errors'].append(f"Thiếu cột bắt buộc: {missing_required}")

    # Kiểm tra cột khuyến nghị
    recommended_cols = [
        'recipient_id', 'balance', 'channel', 'device_id',
        'location', 'recipient_bank'
    ]
    missing_recommended = [col for col in recommended_cols if col not in df.columns]
    if missing_recommended:
        result['warnings'].append(f"Thiếu cột khuyến nghị (sẽ dùng giá trị mặc định): {missing_recommended}")

    # Thống kê
    result['info'] = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
        'fraud_ratio': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
    }

    return result


def get_feature_importance_template() -> Dict[str, str]:
    """
    Trả về template giải thích cho từng feature

    Returns:
        Dict mapping feature name -> explanation
    """
    return LSTM_FEATURES_VN.copy()


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == '__main__':
    import sys

    print("LSTM Data Pipeline - Test Mode")
    print("-" * 50)

    # Tạo dữ liệu test
    np.random.seed(42)
    n_users = 100
    n_tx_per_user = 20

    data = []
    base_time = datetime(2025, 1, 1)

    for u in range(n_users):
        user_id = f"USR_{u:04d}"
        balance = np.random.uniform(10_000_000, 100_000_000)

        for t in range(n_tx_per_user):
            amount = np.random.uniform(100_000, 50_000_000)
            is_fraud = 1 if np.random.random() < 0.05 else 0  # 5% fraud

            data.append({
                'transaction_id': f"TX_{u:04d}_{t:04d}",
                'user_id': user_id,
                'timestamp': base_time + timedelta(hours=t * np.random.uniform(1, 24)),
                'amount': amount,
                'balance': balance,
                'recipient_id': f"RCP_{np.random.randint(0, 50):04d}",
                'channel': np.random.choice(['MOBILE', 'WEB', 'ATM']),
                'device_id': f"DEV_{np.random.randint(0, 10):04d}",
                'location': np.random.choice(['HCM', 'HN', 'DN', 'CT']),
                'recipient_bank': np.random.choice(['VCB', 'TCB', 'ACB', 'MBB']),
                'is_fraud': is_fraud,
            })

            balance -= amount * 0.1  # Simulate balance change

    df_test = pd.DataFrame(data)

    print(f"Generated test data: {len(df_test)} transactions, {n_users} users")
    print()

    # Chạy pipeline
    result = prepare_lstm_data(df_test, seq_length=7, test_ratio=0.2)

    print("\nKết quả cuối cùng:")
    print(f"  X_train: {result['X_train'].shape}")
    print(f"  y_train: {result['y_train'].shape}")
    print(f"  X_test: {result['X_test'].shape}")
    print(f"  y_test: {result['y_test'].shape}")

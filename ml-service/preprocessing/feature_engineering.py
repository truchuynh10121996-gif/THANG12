"""
Feature Engineering - Tạo features cho ML Fraud Detection
==========================================================
Module này tạo các features từ dữ liệu gốc:
- Temporal features (thời gian)
- Aggregation features (tổng hợp theo user)
- Statistical features (thống kê)
- Behavioral features (hành vi)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class FeatureEngineer:
    """
    Class tạo features từ dữ liệu giao dịch
    Bao gồm: temporal, aggregation, statistical, behavioral features
    """

    def __init__(self, verbose: bool = True):
        """
        Khởi tạo FeatureEngineer

        Args:
            verbose: In thông tin chi tiết
        """
        self.verbose = verbose
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []

    def log(self, message: str):
        """In log nếu verbose mode"""
        if self.verbose:
            print(f"[FeatureEngineer] {message}")

    def create_features(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Pipeline tạo đầy đủ features

        Args:
            transactions_df: DataFrame giao dịch
            users_df: DataFrame users (optional)

        Returns:
            DataFrame với đầy đủ features
        """
        self.log("Bắt đầu tạo features...")

        # Copy để không modify original
        df = transactions_df.copy()

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort theo thời gian
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Step 1: Temporal features
        df = self._create_temporal_features(df)

        # Step 2: Amount features
        df = self._create_amount_features(df)

        # Step 3: User aggregation features
        df = self._create_user_aggregation_features(df)

        # Step 4: Velocity features (tốc độ giao dịch)
        df = self._create_velocity_features(df)

        # Step 5: Behavioral features
        df = self._create_behavioral_features(df)

        # Step 6: Encode categorical features
        df = self._encode_categorical_features(df)

        # Step 7: Merge user features if available
        if users_df is not None:
            df = self._merge_user_features(df, users_df)

        # Step 8: Handle infinite and NaN values
        df = self._handle_inf_nan(df)

        self.log(f"Hoàn tất! Tổng số features: {len(self.feature_names)}")
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features liên quan đến thời gian

        Features:
        - hour, day_of_week, day_of_month, month
        - is_weekend, is_night_time, is_business_hours
        - is_holiday, is_salary_day
        """
        self.log("  Tạo temporal features...")

        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

        # Derived time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) &
                                   (df['day_of_week'] < 5)).astype(int)

        # Ngày đầu tháng (thường nhận lương)
        df['is_salary_day'] = (df['day_of_month'] <= 5).astype(int)

        # Cuối tháng
        df['is_end_of_month'] = (df['day_of_month'] >= 25).astype(int)

        # Time sin/cos encoding (cyclical)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        temporal_features = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_night_time', 'is_business_hours',
            'is_salary_day', 'is_end_of_month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos'
        ]
        self.feature_names.extend(temporal_features)

        return df

    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features liên quan đến số tiền

        Features:
        - log_amount
        - amount_deviation_from_mean
        - amount_percentile
        """
        self.log("  Tạo amount features...")

        # Log transform để giảm skewness
        df['log_amount'] = np.log1p(df['amount'])

        # Amount deviation
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - mean_amount) / (std_amount + 1e-8)

        # Balance change
        if 'balance_before' in df.columns and 'balance_after' in df.columns:
            df['balance_change'] = df['balance_after'] - df['balance_before']
            df['balance_change_ratio'] = df['balance_change'] / (df['balance_before'] + 1)
        else:
            df['balance_change'] = 0
            df['balance_change_ratio'] = 0

        # Amount bins
        df['amount_bin'] = pd.qcut(
            df['amount'].rank(method='first'),
            q=10,
            labels=list(range(10))
        ).astype(int)

        amount_features = [
            'log_amount', 'amount_zscore', 'balance_change',
            'balance_change_ratio', 'amount_bin'
        ]
        self.feature_names.extend(amount_features)

        return df

    def _create_user_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features tổng hợp theo user

        Features:
        - Thống kê giao dịch 7/30/90 ngày gần nhất
        - So sánh với lịch sử user
        """
        self.log("  Tạo user aggregation features...")

        # Sort by user and time
        df = df.sort_values(['user_id', 'timestamp'])

        # Group by user
        user_groups = df.groupby('user_id')

        # User historical stats (all time)
        user_stats = user_groups['amount'].agg(['mean', 'std', 'max', 'min', 'count'])
        user_stats.columns = ['user_avg_amount', 'user_std_amount',
                              'user_max_amount', 'user_min_amount', 'user_txn_count']

        # Merge back
        df = df.merge(user_stats, on='user_id', how='left')

        # Amount deviation from user's average
        df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1)
        df['amount_vs_user_max'] = df['amount'] / (df['user_max_amount'] + 1)

        # Is this amount unusual for user? (> 2 std)
        df['is_unusual_amount'] = (
            np.abs(df['amount'] - df['user_avg_amount']) > 2 * df['user_std_amount']
        ).astype(int)

        # Rolling features (last 7, 30 days)
        # Tính toán time-based rolling (complex, simplified version)
        df['user_rank'] = df.groupby('user_id').cumcount()

        agg_features = [
            'user_avg_amount', 'user_std_amount', 'user_max_amount',
            'user_min_amount', 'user_txn_count',
            'amount_vs_user_avg', 'amount_vs_user_max', 'is_unusual_amount'
        ]
        self.feature_names.extend(agg_features)

        return df

    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features về tốc độ giao dịch

        Features:
        - time_since_last_txn
        - txn_count_last_hour
        - txn_count_last_24h
        """
        self.log("  Tạo velocity features...")

        # Sort
        df = df.sort_values(['user_id', 'timestamp'])

        # Time since last transaction (in hours)
        df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
        df['time_since_last_txn'] = (
            df['timestamp'] - df['prev_timestamp']
        ).dt.total_seconds() / 3600  # Convert to hours

        # Fill NaN for first transaction
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(720)  # 30 days default

        # Is rapid transaction? (< 1 hour since last)
        df['is_rapid_txn'] = (df['time_since_last_txn'] < 1).astype(int)

        # Transaction sequence number for user
        df['user_txn_sequence'] = df.groupby('user_id').cumcount() + 1

        # Approximate count in last hour (simplified)
        df['is_very_rapid'] = (df['time_since_last_txn'] < 0.1).astype(int)  # < 6 mins

        # Clean up
        df = df.drop(columns=['prev_timestamp'], errors='ignore')

        velocity_features = [
            'time_since_last_txn', 'is_rapid_txn',
            'user_txn_sequence', 'is_very_rapid'
        ]
        self.feature_names.extend(velocity_features)

        return df

    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features về hành vi

        Features:
        - new_merchant, new_location
        - channel_change
        - is_international
        """
        self.log("  Tạo behavioral features...")

        # Is international transaction
        if 'is_international' in df.columns:
            df['is_international'] = df['is_international'].astype(int)
        else:
            df['is_international'] = 0

        if 'location_country' in df.columns:
            df['is_domestic'] = (df['location_country'].str.lower() == 'vn').astype(int)
        else:
            df['is_domestic'] = 1

        # Is recurring transaction
        if 'is_recurring' in df.columns:
            df['is_recurring'] = df['is_recurring'].astype(int)
        else:
            df['is_recurring'] = 0

        # Transaction type one-hot encoded (top types)
        if 'transaction_type' in df.columns:
            for txn_type in ['transfer', 'payment', 'withdrawal', 'deposit']:
                df[f'is_{txn_type}'] = (
                    df['transaction_type'].str.lower() == txn_type
                ).astype(int)

        # Channel risk score (simplified)
        channel_risk = {
            'mobile_app': 1,
            'web_banking': 2,
            'atm': 2,
            'pos': 1,
            'branch': 0,
            'unknown': 3
        }
        if 'channel' in df.columns:
            df['channel_risk'] = df['channel'].str.lower().map(channel_risk).fillna(2)

        behavioral_features = [
            'is_international', 'is_domestic', 'is_recurring',
            'is_transfer', 'is_payment', 'is_withdrawal', 'is_deposit',
            'channel_risk'
        ]
        # Only add features that exist
        self.feature_names.extend([f for f in behavioral_features if f in df.columns])

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features

        Sử dụng LabelEncoder cho các cột categorical
        """
        self.log("  Encoding categorical features...")

        categorical_cols = ['transaction_type', 'channel', 'device_type',
                           'merchant_category', 'location_country', 'receiving_bank']

        for col in categorical_cols:
            if col in df.columns:
                # Handle NaN
                df[col] = df[col].fillna('unknown')

                # Label encode
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    # Transform using existing encoder
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else -1
                    )

                self.feature_names.append(f'{col}_encoded')

        return df

    def _merge_user_features(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge features từ user profile

        Args:
            transactions_df: DataFrame giao dịch
            users_df: DataFrame users

        Returns:
            DataFrame đã merge
        """
        self.log("  Merging user features...")

        # Select relevant user columns
        user_cols = ['user_id', 'age', 'monthly_income', 'account_balance',
                     'account_age_years', 'credit_score', 'avg_monthly_transactions',
                     'is_verified', 'has_2fa']

        existing_cols = [col for col in user_cols if col in users_df.columns]
        users_subset = users_df[existing_cols].copy()

        # Encode boolean columns
        for col in ['is_verified', 'has_2fa']:
            if col in users_subset.columns:
                users_subset[col] = users_subset[col].astype(int)

        # Merge
        df = transactions_df.merge(users_subset, on='user_id', how='left')

        # Create derived features
        if 'monthly_income' in df.columns:
            df['amount_vs_income'] = df['amount'] / (df['monthly_income'] + 1)

        if 'account_balance' in df.columns:
            df['amount_vs_balance'] = df['amount'] / (df['account_balance'] + 1)

        user_features = [col for col in existing_cols if col != 'user_id']
        user_features.extend(['amount_vs_income', 'amount_vs_balance'])
        self.feature_names.extend([f for f in user_features if f in df.columns])

        return df

    def _handle_inf_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý giá trị inf và NaN

        Thay thế inf bằng max/min hợp lý
        Thay thế NaN bằng 0 hoặc median
        """
        self.log("  Xử lý inf và NaN values...")

        # Replace inf with large numbers
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median() if df[col].median() == df[col].median() else 0)

        return df

    def get_feature_names(self) -> List[str]:
        """Trả về danh sách tên features"""
        return self.feature_names

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        include_target: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy feature matrix và target

        Args:
            df: DataFrame đã có features
            include_target: Có lấy target (is_fraud) không

        Returns:
            Tuple (X, y) hoặc chỉ X
        """
        # Lấy các cột feature
        feature_cols = [col for col in self.feature_names if col in df.columns]

        X = df[feature_cols].values

        if include_target and 'is_fraud' in df.columns:
            y = df['is_fraud'].values
            return X, y

        return X


def create_features_from_raw(
    transactions_path: str = None,
    users_path: str = None,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Hàm tiện ích tạo features từ dữ liệu đã làm sạch

    Args:
        transactions_path: Đường dẫn file giao dịch
        users_path: Đường dẫn file users
        output_dir: Thư mục lưu output

    Returns:
        DataFrame với đầy đủ features
    """
    # Đường dẫn mặc định (dữ liệu đã làm sạch)
    if transactions_path is None:
        transactions_path = os.path.join(config.DATA_PROCESSED_DIR, 'transactions_clean.csv')
    if users_path is None:
        users_path = os.path.join(config.DATA_PROCESSED_DIR, 'users_clean.csv')
    if output_dir is None:
        output_dir = config.DATA_PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("TẠO FEATURES CHO ML MODELS")
    print("=" * 50)

    # Đọc dữ liệu
    if not os.path.exists(transactions_path):
        print(f"[ERROR] Không tìm thấy file: {transactions_path}")
        return None

    transactions_df = pd.read_csv(transactions_path)

    users_df = None
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)

    # Tạo features
    engineer = FeatureEngineer(verbose=True)
    features_df = engineer.create_features(transactions_df, users_df)

    # Lưu
    output_path = os.path.join(output_dir, 'features.csv')
    features_df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")

    # Lưu danh sách feature names
    import json
    feature_names_path = os.path.join(output_dir, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(engineer.get_feature_names(), f, indent=2)
    print(f"[SAVED] {feature_names_path}")

    print(f"\n📊 Tổng số features: {len(engineer.get_feature_names())}")

    return features_df


if __name__ == '__main__':
    create_features_from_raw()

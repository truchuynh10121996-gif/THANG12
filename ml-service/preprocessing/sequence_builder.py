"""
Sequence Builder - Tạo sequences cho LSTM
==========================================
Module này tạo chuỗi giao dịch theo thời gian cho mỗi user
Dùng cho mô hình LSTM phát hiện anomaly trong "nhịp điệu" chi tiêu
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class SequenceBuilder:
    """
    Class tạo sequences từ dữ liệu giao dịch
    Mỗi sequence là chuỗi N giao dịch liên tiếp của một user
    """

    def __init__(
        self,
        sequence_length: int = None,
        verbose: bool = True
    ):
        """
        Khởi tạo SequenceBuilder

        Args:
            sequence_length: Độ dài chuỗi (số giao dịch)
            verbose: In thông tin chi tiết
        """
        self.sequence_length = sequence_length or config.LSTM_CONFIG['sequence_length']
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.feature_columns = []

    def log(self, message: str):
        """In log nếu verbose mode"""
        if self.verbose:
            print(f"[SequenceBuilder] {message}")

    def build_sequences(
        self,
        transactions_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Xây dựng sequences từ dữ liệu giao dịch

        Args:
            transactions_df: DataFrame giao dịch đã có features

        Returns:
            Tuple (sequences, labels, user_ids)
            - sequences: shape (num_sequences, sequence_length, num_features)
            - labels: shape (num_sequences,) - 1 nếu sequence chứa fraud
            - user_ids: shape (num_sequences,) - user_id cho mỗi sequence
        """
        self.log(f"Bắt đầu xây dựng sequences (length={self.sequence_length})...")

        # Chuẩn bị features
        df = self._prepare_features(transactions_df)

        # Sort theo user và thời gian
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Xây dựng sequences cho từng user
        sequences = []
        labels = []
        user_ids = []

        users = df['user_id'].unique()
        self.log(f"  Xử lý {len(users):,} users...")

        for user_id in users:
            user_df = df[df['user_id'] == user_id]

            if len(user_df) < self.sequence_length:
                continue

            # Sliding window để tạo sequences
            user_sequences, user_labels = self._create_user_sequences(user_df)

            sequences.extend(user_sequences)
            labels.extend(user_labels)
            user_ids.extend([user_id] * len(user_sequences))

        # Convert to numpy arrays
        sequences = np.array(sequences)
        labels = np.array(labels)
        user_ids = np.array(user_ids)

        self.log(f"Hoàn tất! Tạo được {len(sequences):,} sequences")
        self.log(f"  - Shape: {sequences.shape}")
        self.log(f"  - Fraud sequences: {labels.sum():,} ({labels.mean()*100:.2f}%)")

        return sequences, labels, user_ids

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị features cho sequence building

        Chọn và chuẩn hóa các features phù hợp cho LSTM
        """
        self.log("  Chuẩn bị features...")

        df = df.copy()

        # Các features cần cho LSTM
        self.feature_columns = [
            'log_amount',
            'hour', 'day_of_week',
            'is_weekend', 'is_night_time',
            'time_since_last_txn',
            'amount_vs_user_avg',
            'is_rapid_txn',
            'is_international',
            'channel_risk'
        ]

        # Chỉ giữ các cột có sẵn
        available_features = [col for col in self.feature_columns if col in df.columns]

        # Tạo log_amount nếu chưa có
        if 'log_amount' not in df.columns and 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'])
            if 'log_amount' not in available_features:
                available_features.append('log_amount')

        # Tạo các features cơ bản nếu chưa có
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'hour' not in df.columns:
                df['hour'] = df['timestamp'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Cập nhật danh sách features
        self.feature_columns = available_features

        # Điền NaN
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Chuẩn hóa features
        if len(self.feature_columns) > 0:
            feature_data = df[self.feature_columns].values
            df[self.feature_columns] = self.scaler.fit_transform(feature_data)

        return df

    def _create_user_sequences(
        self,
        user_df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Tạo sequences cho một user

        Args:
            user_df: DataFrame giao dịch của một user (đã sort theo thời gian)

        Returns:
            Tuple (sequences, labels)
        """
        sequences = []
        labels = []

        n = len(user_df)
        feature_data = user_df[self.feature_columns].values
        fraud_labels = user_df['is_fraud'].values if 'is_fraud' in user_df.columns else np.zeros(n)

        # Sliding window
        for i in range(n - self.sequence_length + 1):
            # Sequence gồm các giao dịch từ i đến i+sequence_length-1
            seq = feature_data[i:i + self.sequence_length]
            sequences.append(seq)

            # Label: 1 nếu giao dịch cuối sequence là fraud
            # hoặc có bất kỳ fraud nào trong sequence
            seq_labels = fraud_labels[i:i + self.sequence_length]
            label = 1 if seq_labels[-1] == 1 else 0  # Dự đoán giao dịch tiếp theo
            labels.append(label)

        return sequences, labels

    def build_prediction_sequence(
        self,
        user_history: pd.DataFrame,
        new_transaction: Dict
    ) -> np.ndarray:
        """
        Xây dựng sequence để dự đoán cho một giao dịch mới

        Args:
            user_history: Lịch sử giao dịch của user (sequence_length - 1 giao dịch)
            new_transaction: Giao dịch mới cần dự đoán

        Returns:
            np.ndarray shape (1, sequence_length, num_features)
        """
        # Kết hợp lịch sử với giao dịch mới
        history_df = user_history.copy()

        # Thêm giao dịch mới
        new_df = pd.DataFrame([new_transaction])
        combined = pd.concat([history_df, new_df], ignore_index=True)

        # Lấy sequence_length giao dịch gần nhất
        if len(combined) < self.sequence_length:
            # Pad với zeros
            padding_needed = self.sequence_length - len(combined)
            padding = np.zeros((padding_needed, len(self.feature_columns)))

            features = combined[self.feature_columns].values
            features = self.scaler.transform(features)

            sequence = np.vstack([padding, features])
        else:
            combined = combined.tail(self.sequence_length)
            features = combined[self.feature_columns].values
            features = self.scaler.transform(features)
            sequence = features

        return sequence.reshape(1, self.sequence_length, -1)

    def save_sequences(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        user_ids: np.ndarray,
        output_dir: str = None
    ):
        """
        Lưu sequences ra files

        Args:
            sequences: np.ndarray sequences
            labels: np.ndarray labels
            user_ids: np.ndarray user ids
            output_dir: Thư mục lưu
        """
        if output_dir is None:
            output_dir = config.DATA_PROCESSED_DIR

        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'sequences.npy'), sequences)
        np.save(os.path.join(output_dir, 'sequence_labels.npy'), labels)
        np.save(os.path.join(output_dir, 'sequence_user_ids.npy'), user_ids)

        # Lưu thông tin scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'sequence_scaler.pkl'))

        self.log(f"[SAVED] Sequences to {output_dir}")

    def get_feature_dim(self) -> int:
        """Trả về số chiều features"""
        return len(self.feature_columns)


def build_sequences_from_data(
    features_path: str = None,
    output_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm tiện ích xây dựng sequences từ dữ liệu

    Args:
        features_path: Đường dẫn file features
        output_dir: Thư mục lưu output

    Returns:
        Tuple (sequences, labels, user_ids)
    """
    if features_path is None:
        features_path = os.path.join(config.DATA_PROCESSED_DIR, 'features.csv')
    if output_dir is None:
        output_dir = config.DATA_PROCESSED_DIR

    print("\n" + "=" * 50)
    print("XÂY DỰNG SEQUENCES CHO LSTM")
    print("=" * 50)

    # Đọc dữ liệu
    if not os.path.exists(features_path):
        print(f"[ERROR] Không tìm thấy file: {features_path}")
        return None, None, None

    features_df = pd.read_csv(features_path)

    # Xây dựng sequences
    builder = SequenceBuilder(verbose=True)
    sequences, labels, user_ids = builder.build_sequences(features_df)

    # Lưu sequences
    builder.save_sequences(sequences, labels, user_ids, output_dir)

    return sequences, labels, user_ids


if __name__ == '__main__':
    build_sequences_from_data()

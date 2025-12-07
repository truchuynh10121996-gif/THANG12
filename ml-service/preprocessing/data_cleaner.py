"""
Data Cleaner - Làm sạch dữ liệu cho ML Fraud Detection
======================================================
Module này xử lý các vấn đề phổ biến trong dữ liệu:
- Giá trị null/missing
- Dữ liệu trùng lặp
- Outliers cực đoan
- Format không nhất quán
- Dữ liệu nhiễu
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class DataCleaner:
    """
    Class làm sạch dữ liệu giao dịch
    Xử lý: missing values, duplicates, outliers, format issues
    """

    def __init__(self, verbose: bool = True):
        """
        Khởi tạo DataCleaner

        Args:
            verbose: In thông tin chi tiết trong quá trình xử lý
        """
        self.verbose = verbose
        self.cleaning_report = {}

    def log(self, message: str):
        """In log nếu verbose mode được bật"""
        if self.verbose:
            print(f"[DataCleaner] {message}")

    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline làm sạch dữ liệu giao dịch

        Args:
            df: DataFrame giao dịch gốc

        Returns:
            DataFrame đã được làm sạch
        """
        self.log(f"Bắt đầu làm sạch {len(df):,} giao dịch...")
        original_count = len(df)

        # Step 1: Loại bỏ duplicates
        df = self._remove_duplicates(df)

        # Step 2: Xử lý missing values
        df = self._handle_missing_values(df)

        # Step 3: Xử lý outliers
        df = self._handle_outliers(df)

        # Step 4: Chuẩn hóa format
        df = self._normalize_formats(df)

        # Step 5: Validate data types
        df = self._validate_data_types(df)

        # Step 6: Xóa dữ liệu nhiễu
        df = self._remove_noise(df)

        # Thống kê kết quả
        final_count = len(df)
        removed = original_count - final_count
        self.log(f"Hoàn tất! Đã loại bỏ {removed:,} records ({removed/original_count*100:.2f}%)")
        self.log(f"Còn lại {final_count:,} giao dịch hợp lệ")

        self.cleaning_report['original_count'] = original_count
        self.cleaning_report['final_count'] = final_count
        self.cleaning_report['removed_count'] = removed

        return df

    def clean_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu users

        Args:
            df: DataFrame users gốc

        Returns:
            DataFrame đã được làm sạch
        """
        self.log(f"Làm sạch {len(df):,} users...")

        # Loại bỏ duplicates theo user_id
        df = df.drop_duplicates(subset=['user_id'], keep='first')

        # Xử lý missing values
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
            # Giới hạn tuổi hợp lý
            df['age'] = df['age'].clip(18, 100)

        if 'monthly_income' in df.columns:
            df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
            # Giới hạn thu nhập hợp lý
            df['monthly_income'] = df['monthly_income'].clip(1_000_000, 1_000_000_000)

        if 'credit_score' in df.columns:
            df['credit_score'] = df['credit_score'].fillna(650)
            df['credit_score'] = df['credit_score'].clip(300, 850)

        # Chuẩn hóa gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.upper().replace({
                'MALE': 'M', 'FEMALE': 'F', 'NAM': 'M', 'NU': 'F', 'NỮ': 'F'
            })
            df['gender'] = df['gender'].fillna('M')

        self.log(f"Đã làm sạch {len(df):,} users")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ các giao dịch trùng lặp

        Tiêu chí trùng lặp:
        - Cùng user_id, timestamp, amount, transaction_type
        """
        before = len(df)

        # Xác định các cột để kiểm tra trùng lặp
        duplicate_cols = ['user_id', 'timestamp', 'amount', 'transaction_type']
        existing_cols = [col for col in duplicate_cols if col in df.columns]

        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols, keep='first')

        after = len(df)
        removed = before - after

        if removed > 0:
            self.log(f"  - Đã loại bỏ {removed:,} giao dịch trùng lặp")
            self.cleaning_report['duplicates_removed'] = removed

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý giá trị missing

        Chiến lược:
        - Numerical: Điền bằng median
        - Categorical: Điền bằng mode hoặc 'unknown'
        - Critical fields (amount, user_id): Xóa record
        """
        # Đếm missing trước khi xử lý
        missing_before = df.isnull().sum().sum()

        # Xóa records thiếu thông tin quan trọng
        critical_cols = ['user_id', 'amount', 'timestamp']
        existing_critical = [col for col in critical_cols if col in df.columns]
        df = df.dropna(subset=existing_critical)

        # Xử lý các cột số
        numerical_cols = ['amount', 'balance_before', 'balance_after']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Xử lý các cột categorical
        categorical_cols = ['transaction_type', 'channel', 'device_type',
                           'merchant_category', 'location_country']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        # Xử lý boolean
        bool_cols = ['is_international', 'is_recurring']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False)

        missing_after = df.isnull().sum().sum()
        self.log(f"  - Đã xử lý {missing_before - missing_after:,} giá trị missing")
        self.cleaning_report['missing_handled'] = missing_before - missing_after

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý outliers bằng phương pháp IQR

        Các trường được kiểm tra:
        - amount: Số tiền giao dịch
        - balance_before, balance_after: Số dư
        """
        outliers_removed = 0

        if 'amount' in df.columns:
            before = len(df)

            # Sử dụng IQR để xác định outliers
            Q1 = df['amount'].quantile(0.01)
            Q3 = df['amount'].quantile(0.99)
            IQR = Q3 - Q1

            # Giữ lại dữ liệu trong khoảng hợp lý
            # Nhưng KHÔNG loại bỏ fraud transactions
            lower_bound = max(0, Q1 - 3 * IQR)
            upper_bound = Q3 + 3 * IQR

            # Chỉ cắt bớt giá trị, không xóa record
            df['amount'] = df['amount'].clip(lower=10_000)  # Tối thiểu 10,000 VND

            # Loại bỏ số tiền âm
            df = df[df['amount'] > 0]

            after = len(df)
            outliers_removed = before - after

        if outliers_removed > 0:
            self.log(f"  - Đã xử lý {outliers_removed:,} outliers")
            self.cleaning_report['outliers_handled'] = outliers_removed

        return df

    def _normalize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa định dạng dữ liệu

        - Timestamp: ISO format
        - Amount: Integer
        - Text fields: Lowercase, strip whitespace
        """
        # Chuẩn hóa timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Loại bỏ records với timestamp không hợp lệ
            df = df.dropna(subset=['timestamp'])
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Chuẩn hóa amount thành integer
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(int)

        # Chuẩn hóa text fields
        text_cols = ['transaction_type', 'channel', 'device_type',
                     'merchant_category', 'location_country']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

        # Chuẩn hóa user_id
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str).str.upper().str.strip()

        self.log("  - Đã chuẩn hóa định dạng dữ liệu")
        return df

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate và convert data types

        Đảm bảo các cột có đúng kiểu dữ liệu
        """
        # Định nghĩa data types mong muốn
        dtype_mapping = {
            'amount': 'int64',
            'balance_before': 'int64',
            'balance_after': 'int64',
            'is_fraud': 'int64',
            'is_international': 'bool',
            'is_recurring': 'bool'
        }

        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                try:
                    if dtype == 'bool':
                        df[col] = df[col].astype(bool)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                except Exception as e:
                    self.log(f"  - Warning: Không thể convert {col} sang {dtype}: {e}")

        return df

    def _remove_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ dữ liệu nhiễu

        - Giao dịch với timestamp trong tương lai
        - Giao dịch với số tiền = 0
        - User_id không hợp lệ
        """
        before = len(df)

        # Loại bỏ giao dịch tương lai
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] <= datetime.now()]

        # Loại bỏ giao dịch số tiền = 0 hoặc âm
        if 'amount' in df.columns:
            df = df[df['amount'] > 0]

        # Loại bỏ user_id không hợp lệ
        if 'user_id' in df.columns:
            df = df[df['user_id'].str.len() > 0]
            df = df[df['user_id'] != 'nan']
            df = df[df['user_id'] != 'None']

        after = len(df)
        noise_removed = before - after

        if noise_removed > 0:
            self.log(f"  - Đã loại bỏ {noise_removed:,} records nhiễu")
            self.cleaning_report['noise_removed'] = noise_removed

        return df

    def get_cleaning_report(self) -> Dict:
        """Trả về báo cáo làm sạch dữ liệu"""
        return self.cleaning_report


def clean_raw_data(
    transactions_path: str = None,
    users_path: str = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hàm tiện ích để làm sạch dữ liệu thô

    Args:
        transactions_path: Đường dẫn file giao dịch
        users_path: Đường dẫn file users
        output_dir: Thư mục lưu output

    Returns:
        Tuple (transactions_clean, users_clean)
    """
    # Đường dẫn mặc định
    if transactions_path is None:
        transactions_path = os.path.join(config.DATA_RAW_DIR, 'transactions_raw.csv')
    if users_path is None:
        users_path = os.path.join(config.DATA_RAW_DIR, 'users_raw.csv')
    if output_dir is None:
        output_dir = config.DATA_PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo cleaner
    cleaner = DataCleaner(verbose=True)

    # Đọc và làm sạch dữ liệu
    print("\n" + "=" * 50)
    print("LÀM SẠCH DỮ LIỆU GIAO DỊCH")
    print("=" * 50)

    if os.path.exists(transactions_path):
        transactions_df = pd.read_csv(transactions_path)
        transactions_clean = cleaner.clean_transactions(transactions_df)

        # Lưu dữ liệu đã làm sạch
        output_path = os.path.join(output_dir, 'transactions_clean.csv')
        transactions_clean.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
    else:
        print(f"[WARNING] Không tìm thấy file: {transactions_path}")
        transactions_clean = None

    print("\n" + "=" * 50)
    print("LÀM SẠCH DỮ LIỆU USERS")
    print("=" * 50)

    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)
        users_clean = cleaner.clean_users(users_df)

        # Lưu dữ liệu đã làm sạch
        output_path = os.path.join(output_dir, 'users_clean.csv')
        users_clean.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
    else:
        print(f"[WARNING] Không tìm thấy file: {users_path}")
        users_clean = None

    return transactions_clean, users_clean


if __name__ == '__main__':
    clean_raw_data()

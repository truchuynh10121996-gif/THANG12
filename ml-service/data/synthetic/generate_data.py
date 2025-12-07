"""
Generate Synthetic Data - Tạo dữ liệu giả lập cho hệ thống ML Fraud Detection
=============================================================================
Script này tạo ra dữ liệu giả lập bao gồm:
- 50,000 users
- 500,000 giao dịch
- 5% giao dịch lừa đảo (có nhãn)
- Patterns thực tế: lương hàng tháng, thanh toán định kỳ, chi tiêu cuối tuần
- Các kiểu lừa đảo: số tiền lớn bất thường, thời gian lạ, người nhận mới
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

# Thêm đường dẫn để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()

# Seed cho reproducibility
np.random.seed(config.RANDOM_STATE)
random.seed(config.RANDOM_STATE)


class DataGenerator:
    """Class tạo dữ liệu giả lập cho hệ thống fraud detection"""

    def __init__(self, num_users: int = None, num_transactions: int = None, fraud_ratio: float = None):
        """
        Khởi tạo generator với các tham số

        Args:
            num_users: Số lượng users
            num_transactions: Số lượng giao dịch
            fraud_ratio: Tỷ lệ giao dịch lừa đảo (0-1)
        """
        self.num_users = num_users or config.NUM_USERS
        self.num_transactions = num_transactions or config.NUM_TRANSACTIONS
        self.fraud_ratio = fraud_ratio or config.FRAUD_RATIO

        # Định nghĩa các loại giao dịch
        self.transaction_types = [
            'transfer',       # Chuyển khoản
            'payment',        # Thanh toán
            'withdrawal',     # Rút tiền
            'deposit',        # Nạp tiền
            'bill_payment',   # Thanh toán hóa đơn
            'card_purchase',  # Mua hàng bằng thẻ
            'online_purchase' # Mua hàng online
        ]

        # Kênh giao dịch
        self.channels = [
            'mobile_app',     # Ứng dụng di động
            'web_banking',    # Internet banking
            'atm',            # Máy ATM
            'pos',            # Điểm bán hàng
            'branch'          # Chi nhánh
        ]

        # Loại thiết bị
        self.device_types = ['ios', 'android', 'web', 'desktop', 'other']

        # Danh mục merchant
        self.merchant_categories = [
            'supermarket',    # Siêu thị
            'restaurant',     # Nhà hàng
            'gas_station',    # Cây xăng
            'electronics',    # Điện tử
            'fashion',        # Thời trang
            'travel',         # Du lịch
            'entertainment',  # Giải trí
            'healthcare',     # Y tế
            'education',      # Giáo dục
            'utilities',      # Tiện ích
            'other'           # Khác
        ]

        # Quốc gia
        self.countries = [
            'VN', 'US', 'CN', 'JP', 'KR', 'SG', 'TH', 'MY', 'ID', 'PH'
        ]

        # Tên ngân hàng người nhận
        self.receiving_banks = [
            'Agribank', 'Vietcombank', 'BIDV', 'Techcombank',
            'VPBank', 'MBBank', 'ACB', 'Sacombank', 'VietinBank',
            'TPBank', 'OCB', 'HDBank', 'MSB', 'SeABank'
        ]

        # Ngày lễ Việt Nam (tháng-ngày)
        self.holidays = [
            (1, 1),   # Tết Dương lịch
            (4, 30),  # Giải phóng miền Nam
            (5, 1),   # Quốc tế Lao động
            (9, 2),   # Quốc khánh
        ]

    def generate_users(self) -> pd.DataFrame:
        """
        Tạo dữ liệu users

        Returns:
            DataFrame chứa thông tin users
        """
        print(f"[INFO] Đang tạo {self.num_users:,} users...")

        users = []
        for i in range(self.num_users):
            user_id = f"USR{i+1:08d}"

            # Phân phối tuổi theo thực tế
            age = np.random.choice(
                range(18, 80),
                p=self._age_distribution()
            )

            # Thu nhập hàng tháng (VND) - phân phối log-normal
            monthly_income = int(np.random.lognormal(mean=16.5, sigma=0.5))
            monthly_income = max(5_000_000, min(500_000_000, monthly_income))

            # Số năm sử dụng dịch vụ
            account_age_years = np.random.exponential(scale=3)
            account_age_years = min(20, account_age_years)

            # Số dư tài khoản
            account_balance = int(monthly_income * np.random.uniform(0.5, 5))

            # Điểm tín dụng (300-850)
            credit_score = int(np.random.normal(650, 80))
            credit_score = max(300, min(850, credit_score))

            # Số giao dịch trung bình mỗi tháng
            avg_monthly_transactions = int(np.random.lognormal(mean=2.5, sigma=0.8))
            avg_monthly_transactions = max(1, min(200, avg_monthly_transactions))

            # Tỉnh/Thành phố
            province = np.random.choice([
                'Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ',
                'Bình Dương', 'Đồng Nai', 'Khánh Hòa', 'Nghệ An', 'Thanh Hóa'
            ], p=[0.25, 0.35, 0.08, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04])

            # Nghề nghiệp
            occupation = np.random.choice([
                'employee', 'self_employed', 'student', 'retired',
                'freelancer', 'business_owner', 'civil_servant', 'other'
            ], p=[0.35, 0.15, 0.10, 0.08, 0.12, 0.08, 0.07, 0.05])

            # Risk profile (dựa trên các yếu tố)
            risk_profile = self._calculate_user_risk_profile(
                age, monthly_income, account_age_years, credit_score
            )

            # Ngày đăng ký
            registration_date = datetime.now() - timedelta(
                days=int(account_age_years * 365)
            )

            users.append({
                'user_id': user_id,
                'age': age,
                'gender': np.random.choice(['M', 'F'], p=[0.52, 0.48]),
                'province': province,
                'occupation': occupation,
                'monthly_income': monthly_income,
                'account_balance': account_balance,
                'account_age_years': round(account_age_years, 2),
                'credit_score': credit_score,
                'avg_monthly_transactions': avg_monthly_transactions,
                'risk_profile': risk_profile,
                'registration_date': registration_date.strftime('%Y-%m-%d'),
                'is_verified': np.random.choice([True, False], p=[0.85, 0.15]),
                'has_2fa': np.random.choice([True, False], p=[0.65, 0.35])
            })

        df = pd.DataFrame(users)
        print(f"[SUCCESS] Đã tạo {len(df):,} users")
        return df

    def generate_transactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo dữ liệu giao dịch với patterns thực tế

        Args:
            users_df: DataFrame chứa thông tin users

        Returns:
            DataFrame chứa thông tin giao dịch
        """
        print(f"[INFO] Đang tạo {self.num_transactions:,} giao dịch...")

        transactions = []
        num_fraud = int(self.num_transactions * self.fraud_ratio)
        num_normal = self.num_transactions - num_fraud

        # Tạo mapping user_id -> user info
        user_info = users_df.set_index('user_id').to_dict('index')
        user_ids = users_df['user_id'].tolist()

        # Thời gian bắt đầu: 2 năm trước
        start_date = datetime.now() - timedelta(days=730)

        # Tạo giao dịch bình thường
        print("[INFO] Tạo giao dịch bình thường...")
        for i in range(num_normal):
            user_id = random.choice(user_ids)
            user = user_info[user_id]

            # Tạo timestamp theo pattern thực tế
            timestamp = self._generate_realistic_timestamp(start_date, user)

            # Tạo giao dịch
            txn = self._create_normal_transaction(user_id, user, timestamp, i)
            txn['is_fraud'] = 0
            txn['fraud_type'] = None
            transactions.append(txn)

            if (i + 1) % 100000 == 0:
                print(f"  - Đã tạo {i + 1:,}/{num_normal:,} giao dịch bình thường")

        # Tạo giao dịch lừa đảo
        print("[INFO] Tạo giao dịch lừa đảo...")
        fraud_types = [
            'unusual_amount',       # Số tiền bất thường
            'unusual_time',         # Thời gian bất thường
            'new_recipient',        # Người nhận mới lạ
            'rapid_succession',     # Nhiều giao dịch liên tiếp
            'foreign_location',     # Địa điểm nước ngoài
            'device_change',        # Đổi thiết bị đột ngột
            'velocity_abuse',       # Lạm dụng tốc độ giao dịch
            'account_takeover'      # Chiếm đoạt tài khoản
        ]

        for i in range(num_fraud):
            user_id = random.choice(user_ids)
            user = user_info[user_id]

            fraud_type = random.choice(fraud_types)
            timestamp = self._generate_realistic_timestamp(start_date, user)

            # Tạo giao dịch lừa đảo theo loại
            txn = self._create_fraud_transaction(
                user_id, user, timestamp, i + num_normal, fraud_type
            )
            txn['is_fraud'] = 1
            txn['fraud_type'] = fraud_type
            transactions.append(txn)

            if (i + 1) % 5000 == 0:
                print(f"  - Đã tạo {i + 1:,}/{num_fraud:,} giao dịch lừa đảo")

        # Shuffle và tạo DataFrame
        random.shuffle(transactions)
        df = pd.DataFrame(transactions)

        # Sắp xếp theo thời gian
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Thêm transaction_id
        df['transaction_id'] = [f"TXN{i+1:010d}" for i in range(len(df))]

        print(f"[SUCCESS] Đã tạo {len(df):,} giao dịch")
        print(f"  - Giao dịch bình thường: {(df['is_fraud'] == 0).sum():,} ({(1-self.fraud_ratio)*100:.1f}%)")
        print(f"  - Giao dịch lừa đảo: {(df['is_fraud'] == 1).sum():,} ({self.fraud_ratio*100:.1f}%)")

        return df

    def generate_fraud_reports(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo báo cáo lừa đảo từ các giao dịch fraud

        Args:
            transactions_df: DataFrame chứa giao dịch

        Returns:
            DataFrame chứa báo cáo lừa đảo
        """
        print("[INFO] Đang tạo báo cáo lừa đảo...")

        fraud_txns = transactions_df[transactions_df['is_fraud'] == 1]
        reports = []

        # 80% giao dịch fraud được báo cáo
        reported_fraud = fraud_txns.sample(frac=0.8, random_state=config.RANDOM_STATE)

        for _, txn in reported_fraud.iterrows():
            # Thời gian báo cáo: 1-72 giờ sau giao dịch
            report_delay_hours = np.random.exponential(scale=12)
            report_delay_hours = max(0.5, min(72, report_delay_hours))

            report_date = pd.to_datetime(txn['timestamp']) + timedelta(hours=report_delay_hours)

            # Nguồn báo cáo
            report_source = np.random.choice([
                'customer_complaint',  # Khách hàng khiếu nại
                'system_alert',        # Cảnh báo hệ thống
                'bank_review',         # Rà soát ngân hàng
                'third_party'          # Bên thứ ba
            ], p=[0.50, 0.30, 0.15, 0.05])

            # Mô tả
            descriptions = {
                'unusual_amount': 'Số tiền giao dịch lớn bất thường so với lịch sử',
                'unusual_time': 'Giao dịch vào thời điểm bất thường (đêm khuya)',
                'new_recipient': 'Chuyển tiền cho người nhận chưa từng giao dịch',
                'rapid_succession': 'Nhiều giao dịch liên tiếp trong thời gian ngắn',
                'foreign_location': 'Giao dịch từ địa điểm nước ngoài bất thường',
                'device_change': 'Giao dịch từ thiết bị mới chưa đăng ký',
                'velocity_abuse': 'Vượt quá giới hạn giao dịch cho phép',
                'account_takeover': 'Nghi ngờ tài khoản bị chiếm đoạt'
            }

            reports.append({
                'report_id': f"RPT{len(reports)+1:08d}",
                'transaction_id': txn['transaction_id'],
                'user_id': txn['user_id'],
                'report_date': report_date.strftime('%Y-%m-%d %H:%M:%S'),
                'fraud_type': txn['fraud_type'],
                'amount': txn['amount'],
                'source': report_source,
                'description': descriptions.get(txn['fraud_type'], 'Giao dịch đáng ngờ'),
                'status': np.random.choice(['confirmed', 'investigating', 'resolved'],
                                           p=[0.6, 0.25, 0.15]),
                'recovered_amount': int(txn['amount'] * np.random.uniform(0, 0.8))
            })

        df = pd.DataFrame(reports)
        print(f"[SUCCESS] Đã tạo {len(df):,} báo cáo lừa đảo")
        return df

    def _age_distribution(self) -> List[float]:
        """Phân phối tuổi thực tế của người dùng ngân hàng số"""
        ages = list(range(18, 80))
        probs = []
        for age in ages:
            if 18 <= age <= 25:
                probs.append(0.15)
            elif 26 <= age <= 35:
                probs.append(0.30)
            elif 36 <= age <= 45:
                probs.append(0.25)
            elif 46 <= age <= 55:
                probs.append(0.15)
            elif 56 <= age <= 65:
                probs.append(0.10)
            else:
                probs.append(0.05)

        # Normalize
        total = sum(probs)
        return [p / total for p in probs]

    def _calculate_user_risk_profile(
        self, age: int, income: int, account_age: float, credit_score: int
    ) -> str:
        """Tính toán risk profile của user"""
        risk_score = 0

        # Tuổi trẻ hoặc già có rủi ro cao hơn
        if age < 25 or age > 65:
            risk_score += 1

        # Thu nhập thấp có rủi ro cao hơn
        if income < 10_000_000:
            risk_score += 1

        # Tài khoản mới có rủi ro cao hơn
        if account_age < 1:
            risk_score += 2
        elif account_age < 2:
            risk_score += 1

        # Điểm tín dụng thấp
        if credit_score < 500:
            risk_score += 2
        elif credit_score < 600:
            risk_score += 1

        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        return 'low'

    def _generate_realistic_timestamp(
        self, start_date: datetime, user: Dict
    ) -> str:
        """Tạo timestamp thực tế dựa trên hành vi user"""
        # Random ngày trong khoảng thời gian
        days_offset = random.randint(0, 729)
        date = start_date + timedelta(days=days_offset)

        # Phân phối giờ giao dịch theo thực tế
        hour_probs = self._get_hour_distribution(date.weekday())
        hour = np.random.choice(24, p=hour_probs)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        timestamp = date.replace(hour=hour, minute=minute, second=second)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def _get_hour_distribution(self, day_of_week: int) -> List[float]:
        """Phân phối giờ giao dịch trong ngày"""
        # Cuối tuần có pattern khác
        is_weekend = day_of_week >= 5

        probs = []
        for hour in range(24):
            if 0 <= hour <= 5:
                # Đêm khuya - ít giao dịch
                prob = 0.01 if not is_weekend else 0.02
            elif 6 <= hour <= 8:
                # Sáng sớm
                prob = 0.04 if not is_weekend else 0.02
            elif 9 <= hour <= 11:
                # Sáng
                prob = 0.08 if not is_weekend else 0.06
            elif 12 <= hour <= 13:
                # Trưa
                prob = 0.07
            elif 14 <= hour <= 17:
                # Chiều
                prob = 0.10 if not is_weekend else 0.08
            elif 18 <= hour <= 21:
                # Tối
                prob = 0.09 if not is_weekend else 0.10
            else:
                # Đêm
                prob = 0.03

            probs.append(prob)

        # Normalize
        total = sum(probs)
        return [p / total for p in probs]

    def _create_normal_transaction(
        self, user_id: str, user: Dict, timestamp: str, idx: int
    ) -> Dict:
        """Tạo giao dịch bình thường"""
        # Xác định loại giao dịch dựa trên thời gian
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        day_of_month = dt.day
        day_of_week = dt.weekday()

        # Patterns đặc biệt
        if day_of_month <= 5 and random.random() < 0.3:
            # Đầu tháng: Nhận lương
            txn_type = 'deposit'
            amount = int(user['monthly_income'] * np.random.uniform(0.9, 1.1))
        elif day_of_month >= 1 and day_of_month <= 10 and random.random() < 0.2:
            # Đầu tháng: Thanh toán hóa đơn
            txn_type = 'bill_payment'
            amount = int(np.random.uniform(100_000, 5_000_000))
        elif day_of_week >= 5 and random.random() < 0.3:
            # Cuối tuần: Mua sắm, giải trí
            txn_type = np.random.choice(['card_purchase', 'online_purchase', 'payment'])
            amount = int(np.random.lognormal(mean=12, sigma=1))
        else:
            # Giao dịch thông thường
            txn_type = np.random.choice(
                self.transaction_types,
                p=[0.25, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10]
            )
            amount = int(np.random.lognormal(mean=13, sigma=1.5))

        # Giới hạn số tiền theo thu nhập
        max_amount = user['monthly_income'] * 3
        amount = min(amount, max_amount)
        amount = max(10_000, amount)  # Tối thiểu 10,000 VND

        # Balance
        balance_before = user['account_balance']
        if txn_type in ['deposit']:
            balance_after = balance_before + amount
        else:
            balance_after = max(0, balance_before - amount)

        return {
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': amount,
            'transaction_type': txn_type,
            'channel': np.random.choice(self.channels, p=[0.40, 0.25, 0.15, 0.10, 0.10]),
            'device_type': np.random.choice(self.device_types, p=[0.30, 0.35, 0.20, 0.10, 0.05]),
            'merchant_category': np.random.choice(self.merchant_categories) if txn_type in ['payment', 'card_purchase', 'online_purchase'] else None,
            'location_country': 'VN',  # Giao dịch bình thường: trong nước
            'receiving_bank': np.random.choice(self.receiving_banks) if txn_type == 'transfer' else None,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'ip_address': self._generate_ip_address(is_domestic=True),
            'device_id': f"DEV{user_id[3:]}_{random.randint(1, 3):02d}",
            'session_id': f"SES{idx:010d}",
            'is_international': False,
            'is_recurring': random.random() < 0.15  # 15% là giao dịch định kỳ
        }

    def _create_fraud_transaction(
        self, user_id: str, user: Dict, timestamp: str, idx: int, fraud_type: str
    ) -> Dict:
        """Tạo giao dịch lừa đảo theo loại"""
        # Bắt đầu với giao dịch bình thường
        txn = self._create_normal_transaction(user_id, user, timestamp, idx)

        if fraud_type == 'unusual_amount':
            # Số tiền lớn bất thường (5-20 lần thu nhập)
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(5, 20))
            txn['transaction_type'] = 'transfer'

        elif fraud_type == 'unusual_time':
            # Giao dịch vào lúc 1-5 giờ sáng
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            new_hour = random.randint(1, 5)
            dt = dt.replace(hour=new_hour)
            txn['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(1, 5))

        elif fraud_type == 'new_recipient':
            # Chuyển tiền cho người nhận mới với số tiền lớn
            txn['transaction_type'] = 'transfer'
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(2, 8))
            txn['receiving_bank'] = np.random.choice(['Unknown Bank', 'Offshore Bank', 'Crypto Exchange'])

        elif fraud_type == 'rapid_succession':
            # Đánh dấu là giao dịch nhanh liên tiếp
            txn['is_rapid'] = True
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(0.5, 2))
            txn['transaction_type'] = 'transfer'

        elif fraud_type == 'foreign_location':
            # Giao dịch từ nước ngoài
            txn['location_country'] = np.random.choice(['RU', 'NG', 'GH', 'UA', 'IN'])
            txn['is_international'] = True
            txn['ip_address'] = self._generate_ip_address(is_domestic=False)
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(1, 5))

        elif fraud_type == 'device_change':
            # Thiết bị mới
            txn['device_id'] = f"NEW_DEV_{random.randint(10000, 99999)}"
            txn['device_type'] = 'other'
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(2, 10))

        elif fraud_type == 'velocity_abuse':
            # Nhiều giao dịch vượt giới hạn
            txn['amount'] = int(user['monthly_income'] * np.random.uniform(0.8, 1.5))
            txn['is_velocity_violation'] = True

        elif fraud_type == 'account_takeover':
            # Thay đổi hoàn toàn hành vi
            txn['device_id'] = f"UNKNOWN_{random.randint(10000, 99999)}"
            txn['ip_address'] = self._generate_ip_address(is_domestic=False)
            txn['location_country'] = np.random.choice(['RU', 'CN', 'NG'])
            txn['amount'] = int(user['account_balance'] * np.random.uniform(0.5, 0.95))
            txn['transaction_type'] = 'transfer'
            txn['is_international'] = True

        return txn

    def _generate_ip_address(self, is_domestic: bool = True) -> str:
        """Tạo địa chỉ IP"""
        if is_domestic:
            # IP Việt Nam (giả lập)
            return f"14.{random.randint(160, 180)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            # IP nước ngoài
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def save_data(
        self,
        users_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        fraud_reports_df: pd.DataFrame,
        output_dir: str = None
    ):
        """Lưu dữ liệu ra file CSV"""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(output_dir), 'raw')

        os.makedirs(output_dir, exist_ok=True)

        # Lưu users
        users_path = os.path.join(output_dir, 'users_raw.csv')
        users_df.to_csv(users_path, index=False, encoding='utf-8')
        print(f"[SAVED] {users_path}")

        # Lưu transactions
        txn_path = os.path.join(output_dir, 'transactions_raw.csv')
        transactions_df.to_csv(txn_path, index=False, encoding='utf-8')
        print(f"[SAVED] {txn_path}")

        # Lưu fraud reports
        reports_path = os.path.join(output_dir, 'fraud_reports_raw.csv')
        fraud_reports_df.to_csv(reports_path, index=False, encoding='utf-8')
        print(f"[SAVED] {reports_path}")

        # Lưu thống kê
        stats = {
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_users': len(users_df),
            'num_transactions': len(transactions_df),
            'num_fraud_transactions': int((transactions_df['is_fraud'] == 1).sum()),
            'fraud_ratio': float(self.fraud_ratio),
            'num_fraud_reports': len(fraud_reports_df),
            'transaction_types': transactions_df['transaction_type'].value_counts().to_dict(),
            'fraud_types': transactions_df[transactions_df['is_fraud'] == 1]['fraud_type'].value_counts().to_dict()
        }

        stats_path = os.path.join(output_dir, 'data_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {stats_path}")


def main():
    """Hàm chính để chạy data generation"""
    print("=" * 60)
    print("ML FRAUD DETECTION - DATA GENERATOR")
    print("=" * 60)

    # Khởi tạo generator
    generator = DataGenerator()

    # Tạo dữ liệu
    users_df = generator.generate_users()
    transactions_df = generator.generate_transactions(users_df)
    fraud_reports_df = generator.generate_fraud_reports(transactions_df)

    # Lưu dữ liệu
    generator.save_data(users_df, transactions_df, fraud_reports_df)

    print("\n" + "=" * 60)
    print("HOÀN TẤT TẠO DỮ LIỆU!")
    print("=" * 60)

    # Hiển thị thống kê
    print("\n📊 THỐNG KÊ DỮ LIỆU:")
    print(f"  - Số users: {len(users_df):,}")
    print(f"  - Số giao dịch: {len(transactions_df):,}")
    print(f"  - Giao dịch lừa đảo: {(transactions_df['is_fraud'] == 1).sum():,} ({generator.fraud_ratio * 100:.1f}%)")
    print(f"  - Số báo cáo: {len(fraud_reports_df):,}")

    print("\n📁 FILES ĐÃ TẠO:")
    print("  - data/raw/users_raw.csv")
    print("  - data/raw/transactions_raw.csv")
    print("  - data/raw/fraud_reports_raw.csv")
    print("  - data/raw/data_stats.json")


if __name__ == '__main__':
    main()

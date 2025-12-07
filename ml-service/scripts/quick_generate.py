#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo dữ liệu huấn luyện nhanh
Sử dụng: python scripts/quick_generate.py --users 1000 --transactions 10000 --fraud_rate 0.05
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_user_id(index):
    """Tạo user_id duy nhất"""
    return f"USR{str(index).zfill(6)}"


def generate_device_id(index):
    """Tạo device_id duy nhất"""
    return f"DEV{str(index).zfill(6)}"


def generate_transaction_id(index):
    """Tạo transaction_id duy nhất"""
    return f"TXN{str(index).zfill(10)}"


def generate_users(num_users, output_path):
    """
    Tạo dữ liệu người dùng

    Args:
        num_users: Số lượng người dùng cần tạo
        output_path: Đường dẫn file CSV đầu ra

    Returns:
        DataFrame chứa dữ liệu người dùng
    """
    print(f"🔄 Đang tạo {num_users} người dùng...")

    # Các giá trị có thể có
    genders = ['M', 'F']
    occupations = ['engineer', 'teacher', 'doctor', 'business_owner', 'student',
                   'retired', 'freelancer', 'accountant', 'manager', 'designer',
                   'developer', 'marketing', 'sales', 'nurse', 'lawyer', 'ceo', 'intern']
    income_levels = ['low', 'medium', 'high']
    cities = ['Hanoi', 'HCMC', 'Danang', 'Haiphong', 'Cantho', 'Nhatrang', 'Hue', 'Dalat']
    regions = {'Hanoi': 'North', 'HCMC': 'South', 'Danang': 'Central',
               'Haiphong': 'North', 'Cantho': 'South', 'Nhatrang': 'Central',
               'Hue': 'Central', 'Dalat': 'Central'}
    channels = ['mobile', 'web']
    login_frequencies = ['daily', 'weekly', 'monthly']

    users = []
    base_date = datetime.now()

    for i in range(num_users):
        user_id = generate_user_id(i + 1)
        city = random.choice(cities)
        income = random.choice(income_levels)
        occupation = random.choice(occupations)

        # Tuổi phụ thuộc vào nghề nghiệp
        if occupation == 'student':
            age = random.randint(18, 25)
        elif occupation == 'intern':
            age = random.randint(20, 26)
        elif occupation == 'retired':
            age = random.randint(55, 75)
        elif occupation == 'ceo':
            age = random.randint(40, 65)
        else:
            age = random.randint(22, 60)

        # Account age (ngày)
        account_age = random.randint(30, 3650)
        created_at = base_date - timedelta(days=account_age)

        # Số giao dịch trung bình phụ thuộc vào income
        if income == 'high':
            avg_transactions = random.randint(30, 100)
            avg_amount = random.randint(5000000, 50000000)
        elif income == 'medium':
            avg_transactions = random.randint(15, 40)
            avg_amount = random.randint(2000000, 10000000)
        else:
            avg_transactions = random.randint(5, 20)
            avg_amount = random.randint(300000, 3000000)

        # Xác minh
        phone_verified = 1 if random.random() > 0.1 else 0
        email_verified = 1 if random.random() > 0.2 else 0

        # KYC level
        if account_age > 365 and phone_verified and email_verified:
            kyc_level = 3
        elif account_age > 180 and phone_verified:
            kyc_level = 2
        else:
            kyc_level = 1

        # Risk score lịch sử (0-1)
        risk_score = round(random.uniform(0.01, 0.25), 2)

        # Premium user
        is_premium = 1 if income == 'high' and random.random() > 0.5 else 0

        user = {
            'user_id': user_id,
            'age': age,
            'gender': random.choice(genders),
            'occupation': occupation,
            'income_level': income,
            'account_age_days': account_age,
            'city': city,
            'region': regions[city],
            'phone_verified': phone_verified,
            'email_verified': email_verified,
            'kyc_level': kyc_level,
            'avg_monthly_transactions': avg_transactions,
            'avg_transaction_amount': avg_amount,
            'preferred_channel': random.choice(channels),
            'device_count': random.randint(1, 4),
            'login_frequency': random.choice(login_frequencies),
            'last_login_days_ago': random.randint(0, 30),
            'risk_score_historical': risk_score,
            'is_premium': is_premium,
            'created_at': created_at.strftime('%Y-%m-%d')
        }
        users.append(user)

    df = pd.DataFrame(users)
    df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu {num_users} người dùng vào {output_path}")
    return df


def generate_transactions(num_transactions, users_df, fraud_rate, output_path):
    """
    Tạo dữ liệu giao dịch với tỷ lệ fraud xác định

    Args:
        num_transactions: Số lượng giao dịch cần tạo
        users_df: DataFrame chứa dữ liệu người dùng
        fraud_rate: Tỷ lệ giao dịch gian lận (0-1)
        output_path: Đường dẫn file CSV đầu ra

    Returns:
        DataFrame chứa dữ liệu giao dịch
    """
    print(f"🔄 Đang tạo {num_transactions} giao dịch (fraud rate: {fraud_rate*100}%)...")

    # Các loại fraud
    fraud_types = [
        'unusual_amount',      # Số tiền bất thường (quá lớn so với bình thường)
        'unusual_time',        # Thời gian bất thường (2-5 giờ sáng)
        'new_recipient',       # Người nhận mới + số tiền lớn
        'rapid_succession',    # Nhiều giao dịch liên tiếp trong thời gian ngắn
        'foreign_location',    # Địa điểm nước ngoài đáng ngờ
        'device_change',       # Thiết bị mới + hành vi bất thường
        'velocity_abuse',      # Vượt quá tốc độ giao dịch bình thường
        'account_takeover'     # Chiếm đoạt tài khoản (kết hợp nhiều yếu tố)
    ]

    # Các loại giao dịch
    transaction_types = ['transfer', 'payment', 'withdrawal', 'deposit']
    channels = ['mobile', 'web', 'atm']
    device_types = ['android', 'ios', 'windows', 'macos']
    merchant_categories = ['peer_transfer', 'food_delivery', 'shopping', 'electronics',
                           'fashion', 'travel', 'gaming', 'groceries', 'dining',
                           'entertainment', 'beauty', 'atm_withdrawal']

    # Các quốc gia đáng ngờ cho fraud
    suspicious_countries = ['Nigeria', 'Russia', 'Ukraine', 'China', 'Netherlands']
    suspicious_cities = ['Unknown', 'Lagos', 'Moscow', 'Beijing', 'Amsterdam', 'Paris']

    user_ids = users_df['user_id'].tolist()
    user_data = users_df.set_index('user_id').to_dict('index')

    # Theo dõi lịch sử giao dịch của mỗi user
    user_transaction_history = {uid: [] for uid in user_ids}

    transactions = []
    base_date = datetime.now()
    num_fraud = int(num_transactions * fraud_rate)
    fraud_indices = set(random.sample(range(num_transactions), num_fraud))

    for i in range(num_transactions):
        txn_id = generate_transaction_id(i + 1)
        user_id = random.choice(user_ids)
        user_info = user_data[user_id]

        is_fraud = i in fraud_indices
        fraud_type = 'normal'

        # Thời gian giao dịch
        days_ago = random.randint(0, 90)
        if is_fraud and random.random() > 0.5:
            # Fraud thường xảy ra vào ban đêm
            hour = random.randint(1, 5)
            fraud_type = 'unusual_time'
        else:
            hour = random.randint(6, 23)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        timestamp = base_date - timedelta(days=days_ago, hours=random.randint(0, 23)-hour,
                                          minutes=random.randint(0, 59)-minute)
        timestamp = timestamp.replace(hour=hour, minute=minute, second=second)

        # Số tiền giao dịch
        avg_amount = user_info['avg_transaction_amount']
        if is_fraud and fraud_type == 'normal':
            # Số tiền bất thường (gấp 3-10 lần bình thường)
            amount = int(avg_amount * random.uniform(3, 10))
            fraud_type = 'unusual_amount'
        else:
            # Số tiền bình thường (dao động quanh mức trung bình)
            amount = int(avg_amount * random.uniform(0.1, 1.5))

        # Loại giao dịch
        txn_type = random.choice(transaction_types)
        channel = random.choice(['mobile', 'web']) if txn_type != 'withdrawal' else 'atm'

        # Device
        device_id = generate_device_id(random.randint(1, len(user_ids) * 2))
        device_type = random.choice(device_types)
        is_new_device = random.random() > 0.9

        if is_fraud and fraud_type == 'normal' and is_new_device:
            fraud_type = 'device_change'

        # IP và địa điểm
        if is_fraud and fraud_type == 'normal' and random.random() > 0.6:
            # Fraud từ nước ngoài
            country = random.choice(suspicious_countries)
            city = random.choice(suspicious_cities)
            ip = f"{random.randint(100, 200)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            is_international = 1
            fraud_type = 'foreign_location'
        else:
            country = 'Vietnam'
            city = user_info['city']
            ip = f"{random.choice([113, 27, 42])}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            is_international = 0

        is_new_location = 1 if city != user_info['city'] else 0

        # Người nhận
        recipient_id = random.choice(user_ids + [f'MER{str(random.randint(1, 1000)).zfill(3)}'])
        is_new_recipient = random.random() > 0.8

        if is_fraud and fraud_type == 'normal' and is_new_recipient:
            fraud_type = 'new_recipient'

        recipient_type = 'individual' if recipient_id.startswith('USR') else 'merchant'
        merchant_category = random.choice(merchant_categories) if recipient_type == 'merchant' else 'peer_transfer'

        # Session và login
        session_duration = random.randint(30, 600)
        login_attempts = 1 if not is_fraud else random.randint(1, 8)

        if is_fraud and login_attempts > 3 and fraud_type == 'normal':
            fraud_type = 'account_takeover'

        # Velocity features
        history = user_transaction_history[user_id]
        recent_1h = len([t for t in history if (timestamp - t).total_seconds() < 3600])
        recent_24h = len([t for t in history if (timestamp - t).total_seconds() < 86400])

        if is_fraud and fraud_type == 'normal' and recent_1h > 3:
            fraud_type = 'velocity_abuse'
        elif is_fraud and fraud_type == 'normal' and random.random() > 0.7:
            fraud_type = 'rapid_succession'
            # Thêm nhiều giao dịch giả trong lịch sử gần
            recent_1h = random.randint(4, 8)
            recent_24h = random.randint(8, 15)

        # Thời gian từ giao dịch trước
        if history:
            time_since_last = int((timestamp - history[-1]).total_seconds() / 60)
        else:
            time_since_last = random.randint(60, 10080)  # 1h đến 1 tuần

        # Tính độ lệch số tiền
        amount_deviation = amount / avg_amount if avg_amount > 0 else 1.0

        # Cập nhật lịch sử
        user_transaction_history[user_id].append(timestamp)

        # Day of week và hour features
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        transaction = {
            'transaction_id': txn_id,
            'user_id': user_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': amount,
            'transaction_type': txn_type,
            'channel': channel,
            'recipient_id': recipient_id,
            'recipient_type': recipient_type,
            'device_id': device_id,
            'device_type': device_type,
            'ip_address': ip,
            'location_city': city,
            'location_country': country,
            'merchant_category': merchant_category,
            'is_international': is_international,
            'session_duration_sec': session_duration,
            'login_attempts': login_attempts,
            'time_since_last_transaction_min': time_since_last,
            'is_new_recipient': 1 if is_new_recipient else 0,
            'is_new_device': 1 if is_new_device else 0,
            'is_new_location': is_new_location,
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'velocity_1h': recent_1h + 1,
            'velocity_24h': recent_24h + 1,
            'amount_deviation_ratio': round(amount_deviation, 2),
            'is_fraud': 1 if is_fraud else 0,
            'fraud_type': fraud_type if is_fraud else 'normal'
        }
        transactions.append(transaction)

    df = pd.DataFrame(transactions)

    # Sắp xếp theo thời gian
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Cập nhật lại transaction_id theo thứ tự
    df['transaction_id'] = [generate_transaction_id(i + 1) for i in range(len(df))]

    df.to_csv(output_path, index=False)

    # Thống kê
    fraud_count = df['is_fraud'].sum()
    fraud_stats = df[df['is_fraud'] == 1]['fraud_type'].value_counts()

    print(f"✅ Đã lưu {num_transactions} giao dịch vào {output_path}")
    print(f"   - Giao dịch hợp lệ: {num_transactions - fraud_count}")
    print(f"   - Giao dịch gian lận: {fraud_count} ({fraud_count/num_transactions*100:.1f}%)")
    print(f"   - Phân bố fraud types:")
    for ftype, count in fraud_stats.items():
        print(f"     + {ftype}: {count}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Tạo dữ liệu huấn luyện cho ML Fraud Detection')
    parser.add_argument('--users', type=int, default=1000,
                        help='Số lượng người dùng (default: 1000)')
    parser.add_argument('--transactions', type=int, default=10000,
                        help='Số lượng giao dịch (default: 10000)')
    parser.add_argument('--fraud_rate', type=float, default=0.05,
                        help='Tỷ lệ giao dịch gian lận (default: 0.05)')
    parser.add_argument('--output_dir', type=str, default='data/generated',
                        help='Thư mục đầu ra (default: data/generated)')

    args = parser.parse_args()

    # Tạo thư mục output nếu chưa tồn tại
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 ML FRAUD DETECTION - DATA GENERATOR")
    print("=" * 60)
    print(f"📊 Cấu hình:")
    print(f"   - Số người dùng: {args.users:,}")
    print(f"   - Số giao dịch: {args.transactions:,}")
    print(f"   - Tỷ lệ fraud: {args.fraud_rate * 100}%")
    print(f"   - Thư mục output: {output_dir}")
    print("=" * 60)

    # Tạo users
    users_path = os.path.join(output_dir, 'users.csv')
    users_df = generate_users(args.users, users_path)

    # Tạo transactions
    transactions_path = os.path.join(output_dir, 'transactions.csv')
    transactions_df = generate_transactions(args.transactions, users_df,
                                            args.fraud_rate, transactions_path)

    print("=" * 60)
    print("✅ HOÀN THÀNH!")
    print(f"📁 File đã tạo:")
    print(f"   - {users_path}")
    print(f"   - {transactions_path}")
    print("=" * 60)

    return users_df, transactions_df


if __name__ == '__main__':
    main()

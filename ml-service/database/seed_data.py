"""
Seed Data Script - Tạo dữ liệu đủ lớn cho ML Models
====================================================
Script này tạo dữ liệu giả lập đủ lớn để train các models Layer 2:
- LSTM: Cần chuỗi giao dịch theo thời gian
- Autoencoder: Cần nhiều samples để học representation
- GNN: Cần graph với nhiều nodes và edges

Chạy script:
    python -m database.seed_data

Hoặc:
    python database/seed_data.py --users 10000 --transactions 500000
"""

import os
import sys
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid

# Thêm path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flag kiểm tra có pymongo hay không
try:
    from pymongo import MongoClient
    from bson import ObjectId
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("⚠️  PyMongo chưa được cài đặt. Chạy: pip install pymongo")

# Seed cho random để reproducible
random.seed(42)


# ==================== CẤU HÌNH DỮ LIỆU ====================

# Tên người Việt Nam
FIRST_NAMES = [
    'An', 'Bình', 'Cường', 'Dung', 'Em', 'Phương', 'Giang', 'Hải', 'Inh', 'Khoa',
    'Lan', 'Minh', 'Nam', 'Oanh', 'Phúc', 'Quân', 'Sơn', 'Tâm', 'Uyên', 'Vân',
    'Xuân', 'Yến', 'Anh', 'Bảo', 'Chi', 'Đức', 'Hà', 'Hùng', 'Khánh', 'Linh',
    'Long', 'Mai', 'Nga', 'Nhung', 'Phong', 'Quỳnh', 'Thảo', 'Thủy', 'Trang', 'Tuấn'
]

LAST_NAMES = [
    'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan', 'Vũ', 'Võ', 'Đặng',
    'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý', 'Đinh', 'Trương', 'Lương', 'Cao'
]

MIDDLE_NAMES = ['Văn', 'Thị', 'Đức', 'Minh', 'Thanh', 'Hoàng', 'Ngọc', 'Kim', 'Hữu', 'Quốc']

OCCUPATIONS = [
    ('Kỹ sư phần mềm', 'high'),
    ('Bác sĩ', 'high'),
    ('Giáo viên', 'medium'),
    ('Nhân viên văn phòng', 'medium'),
    ('Sinh viên', 'low'),
    ('Nông dân', 'low'),
    ('Doanh nhân', 'very_high'),
    ('Giám đốc', 'very_high'),
    ('Kế toán', 'medium'),
    ('Luật sư', 'high'),
    ('Kiến trúc sư', 'high'),
    ('Y tá', 'medium'),
    ('Công nhân', 'low'),
    ('Tài xế', 'low'),
    ('Freelancer', 'medium'),
    ('Hưu trí', 'medium'),
    ('Nội trợ', 'low'),
    ('Quản lý', 'high'),
    ('Nhà báo', 'medium'),
    ('Nghệ sĩ', 'medium')
]

TRANSACTION_TYPES = ['transfer', 'payment', 'withdrawal', 'deposit']
CHANNELS = ['mobile_app', 'web_banking', 'atm', 'branch']
DEVICE_TYPES = ['android', 'ios', 'web']

# Phân phối số tiền theo income level
AMOUNT_RANGES = {
    'low': (100000, 5000000),
    'medium': (500000, 15000000),
    'high': (1000000, 50000000),
    'very_high': (5000000, 200000000)
}

# Fraud patterns - các mẫu giao dịch fraud
FRAUD_PATTERNS = [
    'velocity_burst',      # Nhiều GD trong thời gian ngắn
    'large_amount',        # Số tiền lớn bất thường
    'new_recipient',       # Chuyển cho người lạ
    'night_transaction',   # GD ban đêm
    'international',       # GD quốc tế bất thường
    'account_takeover',    # Chiếm đoạt tài khoản
]


class DataGenerator:
    """
    Class tạo dữ liệu giả lập cho ML Fraud Detection

    Tạo dữ liệu với các đặc điểm:
    - Users với profile đa dạng
    - Transactions với pattern thực tế
    - Fraud cases với các pattern khác nhau
    - Relationships giữa users (cho GNN)
    """

    def __init__(self, num_users: int = 10000, num_transactions: int = 500000, fraud_ratio: float = 0.05):
        """
        Khởi tạo generator

        Args:
            num_users: Số lượng users
            num_transactions: Số lượng giao dịch
            fraud_ratio: Tỷ lệ giao dịch fraud (mặc định 5%)
        """
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.fraud_ratio = fraud_ratio
        self.users = []
        self.transactions = []
        self.relationships = []  # Cho GNN

    def generate_user_id(self, idx: int) -> str:
        """Tạo user_id"""
        return f"USR_{idx:06d}"

    def generate_name(self) -> str:
        """Tạo tên người Việt"""
        last = random.choice(LAST_NAMES)
        middle = random.choice(MIDDLE_NAMES)
        first = random.choice(FIRST_NAMES)
        return f"{last} {middle} {first}"

    def generate_users(self) -> List[Dict]:
        """
        Tạo danh sách users với profile đa dạng

        Returns:
            List các user documents
        """
        print(f"🔄 Đang tạo {self.num_users:,} users...")

        users = []
        for i in range(self.num_users):
            occupation, income_level = random.choice(OCCUPATIONS)

            # Tuổi phù hợp với nghề nghiệp
            if occupation == 'Sinh viên':
                age = random.randint(18, 25)
            elif occupation == 'Hưu trí':
                age = random.randint(55, 80)
            else:
                age = random.randint(22, 60)

            # KYC level dựa trên thời gian sử dụng
            account_age_days = random.randint(1, 3650)  # 0-10 năm
            if account_age_days > 730:  # > 2 năm
                kyc_level = 3
            elif account_age_days > 180:  # > 6 tháng
                kyc_level = random.choice([2, 3])
            else:
                kyc_level = random.choice([1, 2])

            # Số tiền GD trung bình theo income level
            min_amt, max_amt = AMOUNT_RANGES[income_level]
            avg_amount = random.randint(min_amt, max_amt)

            # Risk score - đa số thấp, một số cao
            if random.random() < 0.1:  # 10% high risk
                risk_score = random.uniform(0.3, 0.8)
            else:
                risk_score = random.uniform(0.01, 0.3)

            user = {
                'user_id': self.generate_user_id(i),
                'name': self.generate_name(),
                'age': age,
                'occupation': occupation,
                'income_level': income_level,
                'account_age_days': account_age_days,
                'kyc_level': kyc_level,
                'avg_transaction_amount': avg_amount,
                'historical_risk_score': round(risk_score, 3),
                'total_transactions': 0,  # Sẽ cập nhật sau
                'is_verified': kyc_level >= 2,
                'created_at': datetime.now() - timedelta(days=account_age_days),
                'updated_at': datetime.now()
            }
            users.append(user)

            if (i + 1) % 1000 == 0:
                print(f"   Đã tạo {i + 1:,} users...")

        self.users = users
        print(f"✅ Đã tạo {len(users):,} users")
        return users

    def generate_transactions(self) -> List[Dict]:
        """
        Tạo giao dịch với pattern thực tế

        Đảm bảo:
        - Mỗi user có ít nhất 10 GD (cho LSTM sequence)
        - Có chuỗi thời gian liên tục
        - Fraud patterns đa dạng

        Returns:
            List các transaction documents
        """
        if not self.users:
            raise ValueError("Cần generate users trước")

        print(f"🔄 Đang tạo {self.num_transactions:,} transactions...")

        transactions = []
        num_frauds = int(self.num_transactions * self.fraud_ratio)
        fraud_indices = set(random.sample(range(self.num_transactions), num_frauds))

        # Đảm bảo mỗi user có ít nhất 10 GD
        min_tx_per_user = 10
        guaranteed_tx = self.num_users * min_tx_per_user
        remaining_tx = self.num_transactions - guaranteed_tx

        # Tạo recipients pool (một số users sẽ là người nhận)
        recipient_pool = [u['user_id'] for u in random.sample(self.users, min(5000, len(self.users)))]

        tx_idx = 0
        user_tx_counts = {u['user_id']: 0 for u in self.users}

        # Phase 1: Đảm bảo mỗi user có min_tx_per_user GD
        for user in self.users:
            user_id = user['user_id']
            income_level = user['income_level']
            min_amt, max_amt = AMOUNT_RANGES[income_level]

            # Tạo chuỗi GD trong khoảng thời gian
            base_time = datetime.now() - timedelta(days=random.randint(30, 365))

            for j in range(min_tx_per_user):
                is_fraud = tx_idx in fraud_indices

                tx = self._create_transaction(
                    tx_idx, user_id, min_amt, max_amt,
                    base_time + timedelta(hours=j * random.randint(1, 72)),
                    recipient_pool, is_fraud
                )
                transactions.append(tx)
                user_tx_counts[user_id] += 1
                tx_idx += 1

            if (tx_idx) % 10000 == 0:
                print(f"   Đã tạo {tx_idx:,} transactions...")

        # Phase 2: Phân phối remaining transactions ngẫu nhiên
        for _ in range(remaining_tx):
            user = random.choice(self.users)
            user_id = user['user_id']
            income_level = user['income_level']
            min_amt, max_amt = AMOUNT_RANGES[income_level]

            is_fraud = tx_idx in fraud_indices

            tx = self._create_transaction(
                tx_idx, user_id, min_amt, max_amt,
                datetime.now() - timedelta(hours=random.randint(1, 8760)),
                recipient_pool, is_fraud
            )
            transactions.append(tx)
            user_tx_counts[user_id] += 1
            tx_idx += 1

            if tx_idx % 10000 == 0:
                print(f"   Đã tạo {tx_idx:,} transactions...")

        # Cập nhật total_transactions cho users
        for user in self.users:
            user['total_transactions'] = user_tx_counts[user['user_id']]

        # Sắp xếp theo thời gian
        transactions.sort(key=lambda x: x['timestamp'])

        self.transactions = transactions
        actual_frauds = sum(1 for t in transactions if t['is_fraud'])
        print(f"✅ Đã tạo {len(transactions):,} transactions ({actual_frauds:,} fraud - {actual_frauds/len(transactions)*100:.1f}%)")

        return transactions

    def _create_transaction(self, idx: int, user_id: str, min_amt: int, max_amt: int,
                           timestamp: datetime, recipient_pool: List[str],
                           is_fraud: bool) -> Dict:
        """Tạo 1 transaction"""

        tx_type = random.choice(TRANSACTION_TYPES)
        channel = random.choice(CHANNELS)
        device = random.choice(DEVICE_TYPES)
        hour = timestamp.hour

        # Fraud transactions có pattern đặc biệt
        if is_fraud:
            fraud_pattern = random.choice(FRAUD_PATTERNS)

            if fraud_pattern == 'large_amount':
                amount = random.randint(max_amt * 2, max_amt * 10)
            elif fraud_pattern == 'night_transaction':
                hour = random.randint(0, 5)
                amount = random.randint(min_amt, max_amt)
            elif fraud_pattern == 'velocity_burst':
                amount = random.randint(min_amt, max_amt)
            else:
                amount = random.randint(max_amt, max_amt * 3)

            is_international = fraud_pattern == 'international' or random.random() < 0.3
        else:
            amount = random.randint(min_amt, max_amt)
            is_international = random.random() < 0.05

        # Chọn recipient
        if tx_type in ['transfer', 'payment']:
            recipient_id = random.choice(recipient_pool)
            if recipient_id == user_id:
                recipient_id = random.choice([r for r in recipient_pool if r != user_id])
        else:
            recipient_id = None

        return {
            'transaction_id': f"TXN_{idx:08d}",
            'user_id': user_id,
            'amount': amount,
            'transaction_type': tx_type,
            'channel': channel,
            'device_type': device,
            'hour': hour,
            'day_of_week': timestamp.weekday(),
            'is_international': is_international,
            'recipient_id': recipient_id,
            'is_fraud': is_fraud,
            'timestamp': timestamp,
            'status': 'completed' if random.random() > 0.05 else random.choice(['pending', 'failed']),
            'created_at': timestamp
        }

    def generate_relationships(self) -> List[Dict]:
        """
        Tạo relationships giữa users (cho GNN)

        Các loại relationship:
        - Chuyển tiền thường xuyên
        - Cùng merchant
        - Cùng location

        Returns:
            List các relationship documents
        """
        print("🔄 Đang tạo relationships cho GNN...")

        relationships = []

        # Tạo relationships từ transactions
        transfer_counts = {}
        for tx in self.transactions:
            if tx['recipient_id']:
                key = (tx['user_id'], tx['recipient_id'])
                transfer_counts[key] = transfer_counts.get(key, 0) + 1

        # Chỉ giữ các cặp có >= 3 transactions
        for (sender, receiver), count in transfer_counts.items():
            if count >= 3:
                relationships.append({
                    'source': sender,
                    'target': receiver,
                    'weight': count,
                    'type': 'transfer',
                    'created_at': datetime.now()
                })

        self.relationships = relationships
        print(f"✅ Đã tạo {len(relationships):,} relationships")

        return relationships

    def get_statistics(self) -> Dict:
        """Thống kê dữ liệu đã tạo"""
        if not self.transactions:
            return {}

        fraud_count = sum(1 for t in self.transactions if t['is_fraud'])

        return {
            'total_users': len(self.users),
            'total_transactions': len(self.transactions),
            'total_relationships': len(self.relationships),
            'fraud_count': fraud_count,
            'fraud_ratio': fraud_count / len(self.transactions),
            'avg_tx_per_user': len(self.transactions) / len(self.users),
            'min_tx_per_user': min(u['total_transactions'] for u in self.users),
            'max_tx_per_user': max(u['total_transactions'] for u in self.users),
        }


def seed_to_mongodb(generator: DataGenerator, mongo_uri: str = None):
    """
    Seed dữ liệu vào MongoDB

    Args:
        generator: DataGenerator đã generate dữ liệu
        mongo_uri: MongoDB URI
    """
    if not PYMONGO_AVAILABLE:
        print("❌ PyMongo chưa được cài đặt")
        return

    uri = mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/agribank-digital-guard')

    print(f"\n🔄 Đang kết nối MongoDB: {uri}")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')

        db_name = uri.split('/')[-1].split('?')[0]
        db = client[db_name]

        print(f"✅ Đã kết nối database: {db_name}")

        # Xóa dữ liệu cũ
        print("\n🔄 Đang xóa dữ liệu cũ...")
        db.ml_users.delete_many({})
        db.ml_transactions.delete_many({})
        db.ml_relationships.delete_many({})

        # Insert users
        print(f"🔄 Đang insert {len(generator.users):,} users...")
        if generator.users:
            db.ml_users.insert_many(generator.users)
            db.ml_users.create_index('user_id', unique=True)

        # Insert transactions theo batch
        print(f"🔄 Đang insert {len(generator.transactions):,} transactions...")
        batch_size = 10000
        for i in range(0, len(generator.transactions), batch_size):
            batch = generator.transactions[i:i + batch_size]
            db.ml_transactions.insert_many(batch)
            print(f"   Đã insert {min(i + batch_size, len(generator.transactions)):,} transactions...")

        db.ml_transactions.create_index('transaction_id', unique=True)
        db.ml_transactions.create_index('user_id')
        db.ml_transactions.create_index('timestamp')

        # Insert relationships
        print(f"🔄 Đang insert {len(generator.relationships):,} relationships...")
        if generator.relationships:
            db.ml_relationships.insert_many(generator.relationships)
            db.ml_relationships.create_index([('source', 1), ('target', 1)])

        print("\n✅ Seed dữ liệu hoàn tất!")

        # In thống kê
        stats = generator.get_statistics()
        print("\n📊 THỐNG KÊ:")
        print(f"   - Users: {stats['total_users']:,}")
        print(f"   - Transactions: {stats['total_transactions']:,}")
        print(f"   - Relationships: {stats['total_relationships']:,}")
        print(f"   - Fraud ratio: {stats['fraud_ratio']*100:.1f}%")
        print(f"   - Avg TX/user: {stats['avg_tx_per_user']:.1f}")
        print(f"   - Min TX/user: {stats['min_tx_per_user']}")
        print(f"   - Max TX/user: {stats['max_tx_per_user']}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise


def save_to_csv(generator: DataGenerator, output_dir: str = None):
    """
    Lưu dữ liệu ra file CSV (nếu không có MongoDB)
    """
    import csv

    output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'seed')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔄 Đang lưu dữ liệu vào {output_dir}...")

    # Save users
    users_file = os.path.join(output_dir, 'users.csv')
    with open(users_file, 'w', newline='', encoding='utf-8') as f:
        if generator.users:
            writer = csv.DictWriter(f, fieldnames=generator.users[0].keys())
            writer.writeheader()
            writer.writerows(generator.users)
    print(f"   ✅ Saved {len(generator.users):,} users to users.csv")

    # Save transactions
    tx_file = os.path.join(output_dir, 'transactions.csv')
    with open(tx_file, 'w', newline='', encoding='utf-8') as f:
        if generator.transactions:
            writer = csv.DictWriter(f, fieldnames=generator.transactions[0].keys())
            writer.writeheader()
            writer.writerows(generator.transactions)
    print(f"   ✅ Saved {len(generator.transactions):,} transactions to transactions.csv")

    # Save relationships
    rel_file = os.path.join(output_dir, 'relationships.csv')
    with open(rel_file, 'w', newline='', encoding='utf-8') as f:
        if generator.relationships:
            writer = csv.DictWriter(f, fieldnames=generator.relationships[0].keys())
            writer.writeheader()
            writer.writerows(generator.relationships)
    print(f"   ✅ Saved {len(generator.relationships):,} relationships to relationships.csv")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Seed dữ liệu cho ML Fraud Detection')
    parser.add_argument('--users', type=int, default=10000, help='Số lượng users (default: 10000)')
    parser.add_argument('--transactions', type=int, default=500000, help='Số lượng transactions (default: 500000)')
    parser.add_argument('--fraud-ratio', type=float, default=0.05, help='Tỷ lệ fraud (default: 0.05)')
    parser.add_argument('--mongo-uri', type=str, help='MongoDB URI')
    parser.add_argument('--csv', action='store_true', help='Lưu ra CSV thay vì MongoDB')

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ML FRAUD DETECTION - DATA SEEDER")
    print("=" * 60)
    print(f"\nCấu hình:")
    print(f"   - Users: {args.users:,}")
    print(f"   - Transactions: {args.transactions:,}")
    print(f"   - Fraud ratio: {args.fraud_ratio * 100:.1f}%")
    print()

    # Generate data
    generator = DataGenerator(
        num_users=args.users,
        num_transactions=args.transactions,
        fraud_ratio=args.fraud_ratio
    )

    generator.generate_users()
    generator.generate_transactions()
    generator.generate_relationships()

    # Save data
    if args.csv:
        save_to_csv(generator)
    else:
        seed_to_mongodb(generator, args.mongo_uri)

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT!")
    print("=" * 60)


if __name__ == '__main__':
    main()

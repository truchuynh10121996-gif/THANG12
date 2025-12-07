"""
MongoDB Connection - Kết nối với MongoDB
=========================================
File này cung cấp:
- Kết nối MongoDB
- Các collection: ml_users, ml_transactions, ml_predictions
- Các hàm CRUD cho users và transactions
- Mock data khi chưa có database

Sử dụng cùng database với backend (agribank-digital-guard)
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger
import random
import uuid

# Flag kiểm tra có pymongo hay không
try:
    from pymongo import MongoClient
    from bson import ObjectId
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("PyMongo không được cài đặt. Sử dụng mock data.")


class MongoDBConnection:
    """
    Class quản lý kết nối MongoDB

    Sử dụng:
    ```python
    db = MongoDBConnection()
    users = db.get_all_users()
    user = db.get_user_by_id('USR_001')
    ```
    """

    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        """Singleton pattern - Chỉ tạo 1 instance duy nhất"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Khởi tạo kết nối MongoDB"""
        if self._client is None:
            self._connect()

    def _connect(self):
        """Kết nối đến MongoDB"""
        if not PYMONGO_AVAILABLE:
            logger.warning("Không thể kết nối MongoDB - PyMongo chưa được cài đặt")
            return

        try:
            # Lấy URI từ biến môi trường hoặc dùng mặc định
            mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/agribank-digital-guard')

            self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

            # Test kết nối
            self._client.admin.command('ping')

            # Lấy database name từ URI hoặc dùng mặc định
            db_name = os.getenv('MONGODB_DB', 'agribank-digital-guard')
            self._db = self._client[db_name]

            logger.info(f"✅ Đã kết nối MongoDB: {db_name}")

            # Tạo indexes nếu chưa có
            self._create_indexes()

        except Exception as e:
            logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
            self._client = None
            self._db = None

    def _create_indexes(self):
        """Tạo indexes cho các collection"""
        if self._db is None:
            return

        try:
            # Index cho ml_users
            self._db.ml_users.create_index('user_id', unique=True)

            # Index cho ml_transactions
            self._db.ml_transactions.create_index('transaction_id', unique=True)
            self._db.ml_transactions.create_index('user_id')
            self._db.ml_transactions.create_index('timestamp')

            # Index cho ml_predictions
            self._db.ml_predictions.create_index('transaction_id')
            self._db.ml_predictions.create_index('created_at')

            logger.info("✅ Đã tạo indexes cho các collection")
        except Exception as e:
            logger.error(f"Lỗi tạo indexes: {e}")

    @property
    def is_connected(self) -> bool:
        """Kiểm tra trạng thái kết nối"""
        return self._client is not None and self._db is not None

    @property
    def db(self):
        """Lấy database instance"""
        return self._db

    # ==================== USER OPERATIONS ====================

    def get_all_users(self, limit: int = 100) -> List[Dict]:
        """
        Lấy danh sách tất cả users

        Args:
            limit: Số lượng user tối đa

        Returns:
            List các user documents
        """
        if not self.is_connected:
            return self._get_mock_users()

        try:
            users = list(self._db.ml_users.find().limit(limit))
            # Convert ObjectId to string
            for user in users:
                user['_id'] = str(user['_id'])
            return users if users else self._get_mock_users()
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách users: {e}")
            return self._get_mock_users()

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """
        Lấy thông tin user theo user_id

        Args:
            user_id: ID của user

        Returns:
            User document hoặc None
        """
        if not self.is_connected:
            return self._get_mock_user(user_id)

        try:
            user = self._db.ml_users.find_one({'user_id': user_id})
            if user:
                user['_id'] = str(user['_id'])
                return user
            return self._get_mock_user(user_id)
        except Exception as e:
            logger.error(f"Lỗi lấy user {user_id}: {e}")
            return self._get_mock_user(user_id)

    def create_user(self, user_data: Dict) -> Dict:
        """
        Tạo user mới

        Args:
            user_data: Dữ liệu user

        Returns:
            User document đã tạo
        """
        if not self.is_connected:
            return self._create_mock_user(user_data)

        try:
            # Thêm timestamps
            user_data['created_at'] = datetime.now()
            user_data['updated_at'] = datetime.now()

            result = self._db.ml_users.insert_one(user_data)
            user_data['_id'] = str(result.inserted_id)
            return user_data
        except Exception as e:
            logger.error(f"Lỗi tạo user: {e}")
            return self._create_mock_user(user_data)

    def update_user(self, user_id: str, update_data: Dict) -> bool:
        """Cập nhật thông tin user"""
        if not self.is_connected:
            return True

        try:
            update_data['updated_at'] = datetime.now()
            result = self._db.ml_users.update_one(
                {'user_id': user_id},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Lỗi cập nhật user {user_id}: {e}")
            return False

    # ==================== TRANSACTION OPERATIONS ====================

    def get_user_transactions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Lấy lịch sử giao dịch của user

        Args:
            user_id: ID của user
            limit: Số giao dịch tối đa

        Returns:
            List các transaction documents
        """
        if not self.is_connected:
            return self._get_mock_transactions(user_id, limit)

        try:
            transactions = list(
                self._db.ml_transactions
                .find({'user_id': user_id})
                .sort('timestamp', -1)
                .limit(limit)
            )
            for tx in transactions:
                tx['_id'] = str(tx['_id'])
            return transactions if transactions else self._get_mock_transactions(user_id, limit)
        except Exception as e:
            logger.error(f"Lỗi lấy giao dịch của user {user_id}: {e}")
            return self._get_mock_transactions(user_id, limit)

    def get_recent_transactions(self, limit: int = 50) -> List[Dict]:
        """
        Lấy giao dịch gần đây của hệ thống

        Args:
            limit: Số giao dịch tối đa

        Returns:
            List các transaction documents
        """
        if not self.is_connected:
            return self._get_mock_recent_transactions(limit)

        try:
            transactions = list(
                self._db.ml_transactions
                .find()
                .sort('timestamp', -1)
                .limit(limit)
            )
            for tx in transactions:
                tx['_id'] = str(tx['_id'])
            return transactions if transactions else self._get_mock_recent_transactions(limit)
        except Exception as e:
            logger.error(f"Lỗi lấy giao dịch gần đây: {e}")
            return self._get_mock_recent_transactions(limit)

    def save_transaction(self, transaction_data: Dict) -> Dict:
        """
        Lưu giao dịch mới

        Args:
            transaction_data: Dữ liệu giao dịch

        Returns:
            Transaction document đã lưu
        """
        if not self.is_connected:
            return transaction_data

        try:
            if 'timestamp' not in transaction_data:
                transaction_data['timestamp'] = datetime.now()
            transaction_data['created_at'] = datetime.now()

            result = self._db.ml_transactions.insert_one(transaction_data)
            transaction_data['_id'] = str(result.inserted_id)
            return transaction_data
        except Exception as e:
            logger.error(f"Lỗi lưu giao dịch: {e}")
            return transaction_data

    def count_user_transactions_in_period(self, user_id: str, hours: int) -> int:
        """
        Đếm số giao dịch của user trong khoảng thời gian

        Args:
            user_id: ID của user
            hours: Số giờ

        Returns:
            Số giao dịch
        """
        if not self.is_connected:
            return random.randint(0, 5) if hours == 1 else random.randint(5, 20)

        try:
            start_time = datetime.now() - timedelta(hours=hours)
            count = self._db.ml_transactions.count_documents({
                'user_id': user_id,
                'timestamp': {'$gte': start_time}
            })
            return count
        except Exception as e:
            logger.error(f"Lỗi đếm giao dịch: {e}")
            return 0

    def get_user_last_transaction(self, user_id: str) -> Optional[Dict]:
        """Lấy giao dịch cuối cùng của user"""
        if not self.is_connected:
            return None

        try:
            tx = self._db.ml_transactions.find_one(
                {'user_id': user_id},
                sort=[('timestamp', -1)]
            )
            if tx:
                tx['_id'] = str(tx['_id'])
            return tx
        except Exception as e:
            logger.error(f"Lỗi lấy GD cuối của {user_id}: {e}")
            return None

    def get_user_recipients(self, user_id: str) -> List[str]:
        """Lấy danh sách người nhận của user"""
        if not self.is_connected:
            return ['RCP_001', 'RCP_002', 'RCP_003']

        try:
            recipients = self._db.ml_transactions.distinct(
                'recipient_id',
                {'user_id': user_id}
            )
            return recipients
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách người nhận: {e}")
            return []

    # ==================== PREDICTION OPERATIONS ====================

    def save_prediction(self, prediction_data: Dict) -> Dict:
        """Lưu kết quả prediction"""
        if not self.is_connected:
            return prediction_data

        try:
            prediction_data['created_at'] = datetime.now()
            result = self._db.ml_predictions.insert_one(prediction_data)
            prediction_data['_id'] = str(result.inserted_id)
            return prediction_data
        except Exception as e:
            logger.error(f"Lỗi lưu prediction: {e}")
            return prediction_data

    # ==================== BEHAVIORAL FEATURES ====================

    def calculate_behavioral_features(self, user_id: str, current_amount: float = 0) -> Dict:
        """
        Tính toán các behavioral features từ lịch sử giao dịch

        Args:
            user_id: ID của user
            current_amount: Số tiền giao dịch hiện tại

        Returns:
            Dict chứa các features đã tính
        """
        user = self.get_user_by_id(user_id)
        transactions = self.get_user_transactions(user_id, limit=100)

        # Tính velocity (số giao dịch trong 1h và 24h)
        velocity_1h = self.count_user_transactions_in_period(user_id, 1)
        velocity_24h = self.count_user_transactions_in_period(user_id, 24)

        # Tính time_since_last_transaction
        last_tx = self.get_user_last_transaction(user_id)
        if last_tx and 'timestamp' in last_tx:
            time_since_last = (datetime.now() - last_tx['timestamp']).total_seconds() / 3600
        else:
            time_since_last = 24.0  # Mặc định 24h nếu không có GD trước

        # Tính amount_deviation_ratio
        avg_amount = user.get('avg_transaction_amount', 5000000) if user else 5000000
        if avg_amount > 0:
            amount_deviation_ratio = current_amount / avg_amount
        else:
            amount_deviation_ratio = 1.0

        return {
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'time_since_last_transaction': round(time_since_last, 2),
            'amount_deviation_ratio': round(amount_deviation_ratio, 2),
            'user_profile': user,
            'recent_transactions': transactions[:10]
        }

    # ==================== MOCK DATA ====================

    def _get_mock_users(self) -> List[Dict]:
        """Mock data danh sách users"""
        return [
            {
                '_id': '1',
                'user_id': 'USR_001',
                'name': 'Nguyễn Văn An',
                'age': 35,
                'occupation': 'Kỹ sư phần mềm',
                'income_level': 'high',
                'account_age_days': 730,
                'kyc_level': 3,
                'avg_transaction_amount': 8500000,
                'historical_risk_score': 0.12,
                'total_transactions': 256,
                'is_verified': True
            },
            {
                '_id': '2',
                'user_id': 'USR_002',
                'name': 'Trần Thị Bình',
                'age': 28,
                'occupation': 'Kế toán',
                'income_level': 'medium',
                'account_age_days': 365,
                'kyc_level': 2,
                'avg_transaction_amount': 3500000,
                'historical_risk_score': 0.08,
                'total_transactions': 128,
                'is_verified': True
            },
            {
                '_id': '3',
                'user_id': 'USR_003',
                'name': 'Lê Minh Cường',
                'age': 45,
                'occupation': 'Giám đốc',
                'income_level': 'very_high',
                'account_age_days': 1825,
                'kyc_level': 3,
                'avg_transaction_amount': 25000000,
                'historical_risk_score': 0.05,
                'total_transactions': 512,
                'is_verified': True
            },
            {
                '_id': '4',
                'user_id': 'USR_004',
                'name': 'Phạm Thị Dung',
                'age': 22,
                'occupation': 'Sinh viên',
                'income_level': 'low',
                'account_age_days': 90,
                'kyc_level': 1,
                'avg_transaction_amount': 1500000,
                'historical_risk_score': 0.25,
                'total_transactions': 32,
                'is_verified': False
            },
            {
                '_id': '5',
                'user_id': 'USR_005',
                'name': 'Hoàng Văn Em',
                'age': 55,
                'occupation': 'Hưu trí',
                'income_level': 'medium',
                'account_age_days': 3650,
                'kyc_level': 3,
                'avg_transaction_amount': 5000000,
                'historical_risk_score': 0.03,
                'total_transactions': 890,
                'is_verified': True
            },
            {
                '_id': '6',
                'user_id': 'USR_006',
                'name': 'Võ Thị Phương',
                'age': 30,
                'occupation': 'Bác sĩ',
                'income_level': 'high',
                'account_age_days': 540,
                'kyc_level': 3,
                'avg_transaction_amount': 12000000,
                'historical_risk_score': 0.07,
                'total_transactions': 189,
                'is_verified': True
            },
            {
                '_id': '7',
                'user_id': 'USR_007',
                'name': 'Đặng Minh Quang',
                'age': 40,
                'occupation': 'Doanh nhân',
                'income_level': 'very_high',
                'account_age_days': 1460,
                'kyc_level': 3,
                'avg_transaction_amount': 50000000,
                'historical_risk_score': 0.15,
                'total_transactions': 425,
                'is_verified': True
            },
            {
                '_id': '8',
                'user_id': 'USR_008',
                'name': 'Bùi Thị Hoa',
                'age': 25,
                'occupation': 'Nhân viên văn phòng',
                'income_level': 'medium',
                'account_age_days': 180,
                'kyc_level': 2,
                'avg_transaction_amount': 4000000,
                'historical_risk_score': 0.18,
                'total_transactions': 67,
                'is_verified': True
            }
        ]

    def _get_mock_user(self, user_id: str) -> Optional[Dict]:
        """Mock data cho 1 user"""
        users = self._get_mock_users()
        for user in users:
            if user['user_id'] == user_id:
                return user
        # Trả về user mặc định nếu không tìm thấy
        return {
            '_id': 'default',
            'user_id': user_id,
            'name': f'User {user_id}',
            'age': 30,
            'occupation': 'Không xác định',
            'income_level': 'medium',
            'account_age_days': 365,
            'kyc_level': 2,
            'avg_transaction_amount': 5000000,
            'historical_risk_score': 0.15,
            'total_transactions': 100,
            'is_verified': False
        }

    def _create_mock_user(self, user_data: Dict) -> Dict:
        """Tạo mock user"""
        user_data['_id'] = str(uuid.uuid4())
        user_data['created_at'] = datetime.now().isoformat()
        return user_data

    def _get_mock_transactions(self, user_id: str, limit: int) -> List[Dict]:
        """Mock data giao dịch của user"""
        tx_types = ['transfer', 'payment', 'withdrawal', 'deposit']
        statuses = ['completed', 'completed', 'completed', 'pending', 'failed']
        recipients = ['RCP_001', 'RCP_002', 'RCP_003', 'RCP_004', 'RCP_005']

        transactions = []
        for i in range(min(limit, 10)):
            transactions.append({
                '_id': str(i),
                'transaction_id': f'TXN_{user_id}_{i:03d}',
                'user_id': user_id,
                'amount': random.randint(500000, 20000000),
                'transaction_type': random.choice(tx_types),
                'recipient_id': random.choice(recipients),
                'status': random.choice(statuses),
                'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 720))).isoformat(),
                'channel': random.choice(['mobile_app', 'web_banking', 'atm']),
                'description': f'Giao dịch {i+1}'
            })
        return transactions

    def _get_mock_recent_transactions(self, limit: int) -> List[Dict]:
        """Mock data giao dịch gần đây"""
        user_ids = ['USR_001', 'USR_002', 'USR_003', 'USR_004', 'USR_005']
        tx_types = ['transfer', 'payment', 'withdrawal', 'deposit']
        statuses = ['completed', 'pending', 'failed']

        transactions = []
        for i in range(min(limit, 20)):
            user_id = random.choice(user_ids)
            transactions.append({
                '_id': str(i),
                'transaction_id': f'TXN_SYS_{i:03d}',
                'user_id': user_id,
                'amount': random.randint(500000, 50000000),
                'transaction_type': random.choice(tx_types),
                'status': random.choice(statuses),
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat(),
                'channel': random.choice(['mobile_app', 'web_banking', 'atm']),
                'fraud_probability': round(random.random() * 0.5, 2),
                'risk_level': random.choice(['low', 'medium', 'high'])
            })
        return transactions


# Singleton instance
_db_instance = None

def get_db() -> MongoDBConnection:
    """
    Lấy singleton instance của MongoDBConnection

    Returns:
        MongoDBConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MongoDBConnection()
    return _db_instance

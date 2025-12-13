"""
Customer Data Module - Đọc dữ liệu khách hàng từ file Excel
===========================================================
Module này đọc dữ liệu từ 3 file Excel khách hàng:
- USR_000001.xlsx, USR_000002.xlsx, USR_000003.xlsx

Mỗi file có 2 sheets:
- Profile_KhachHang: Thông tin cá nhân
- LichSu_GiaoDich: Lịch sử giao dịch 12 tháng

Các chức năng:
1. Lấy danh sách khách hàng
2. Lấy thông tin profile khách hàng
3. Lấy lịch sử giao dịch
4. Tính toán các chỉ số hành vi (velocity, deviation, ...)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger('customer_data')

# Đường dẫn đến thư mục chứa file Excel
CUSTOMERS_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'customers'
)

# Danh sách các customer files
CUSTOMER_FILES = ['USR_000001.xlsx', 'USR_000002.xlsx', 'USR_000003.xlsx']


class CustomerDataService:
    """Service để đọc và xử lý dữ liệu khách hàng từ file Excel"""

    def __init__(self):
        self.data_dir = CUSTOMERS_DATA_DIR
        self._cache = {}  # Cache dữ liệu để tránh đọc file nhiều lần
        self._load_all_customers()

    def _load_all_customers(self):
        """Load tất cả dữ liệu khách hàng vào cache"""
        logger.info(f"Loading customer data from: {self.data_dir}")

        for filename in CUSTOMER_FILES:
            filepath = os.path.join(self.data_dir, filename)
            user_id = filename.replace('.xlsx', '')

            if os.path.exists(filepath):
                try:
                    # Đọc cả 2 sheets
                    profile_df = pd.read_excel(filepath, sheet_name='Profile_KhachHang')
                    transactions_df = pd.read_excel(filepath, sheet_name='LichSu_GiaoDich')

                    # Parse timestamp
                    if 'timestamp' in transactions_df.columns:
                        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

                    # Sắp xếp theo thời gian giảm dần (mới nhất trước)
                    transactions_df = transactions_df.sort_values('timestamp', ascending=False)

                    self._cache[user_id] = {
                        'profile': profile_df.iloc[0].to_dict(),
                        'transactions': transactions_df
                    }
                    logger.info(f"Loaded {user_id}: {len(transactions_df)} transactions")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
            else:
                logger.warning(f"File not found: {filepath}")

    def get_all_customers(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả khách hàng

        Returns:
            List[Dict]: Danh sách thông tin cơ bản của khách hàng
        """
        customers = []

        for user_id, data in self._cache.items():
            profile = data['profile']
            transactions = data['transactions']

            customers.append({
                'user_id': user_id,
                'ho_ten': profile.get('ho_ten', 'N/A'),
                'tuoi': profile.get('tuoi', 0),
                'nghe_nghiep': profile.get('nghe_nghiep', 'N/A'),
                'thu_nhap_hang_thang': profile.get('thu_nhap_hang_thang', 0),
                'so_ngay_mo_tai_khoan': profile.get('so_ngay_mo_tai_khoan', 0),
                'ngan_hang': profile.get('ngan_hang', 'N/A'),
                'total_transactions': len(transactions)
            })

        return customers

    def get_customer_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin profile chi tiết của khách hàng

        Args:
            user_id: ID của khách hàng (VD: USR_000001)

        Returns:
            Dict: Thông tin profile hoặc None nếu không tìm thấy
        """
        if user_id not in self._cache:
            return None

        profile = self._cache[user_id]['profile']
        transactions = self._cache[user_id]['transactions']

        # Tính số tiền giao dịch trung bình
        avg_amount = transactions['so_tien'].mean() if len(transactions) > 0 else 0

        return {
            'user_id': user_id,
            'ho_ten': profile.get('ho_ten', 'N/A'),
            'tuoi': profile.get('tuoi', 0),
            'gioi_tinh': profile.get('gioi_tinh', 'N/A'),
            'nghe_nghiep': profile.get('nghe_nghiep', 'N/A'),
            'thu_nhap_hang_thang': profile.get('thu_nhap_hang_thang', 0),
            'dia_chi': profile.get('dia_chi', 'N/A'),
            'tinh_thanh': profile.get('tinh_thanh', 'N/A'),
            'so_ngay_mo_tai_khoan': profile.get('so_ngay_mo_tai_khoan', 0),
            'so_du_tai_khoan': profile.get('so_du_tai_khoan', 0),
            'ngan_hang': profile.get('ngan_hang', 'N/A'),
            'so_tai_khoan': profile.get('so_tai_khoan', 'N/A'),
            'thiet_bi_chinh': profile.get('thiet_bi_chinh', 'N/A'),
            'loai_thiet_bi': profile.get('loai_thiet_bi', 'N/A'),
            'vi_dien_tu_lien_ket': profile.get('vi_dien_tu_lien_ket', 'N/A'),
            'han_muc_giao_dich_ngay': profile.get('han_muc_giao_dich_ngay', 0),
            'han_muc_moi_giao_dich': profile.get('han_muc_moi_giao_dich', 0),
            'muc_do_hoat_dong': profile.get('muc_do_hoat_dong', 'N/A'),
            'avg_transaction_amount': avg_amount,
            'total_transactions': len(transactions)
        }

    def get_customer_transactions(self, user_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử giao dịch của khách hàng

        Args:
            user_id: ID của khách hàng
            limit: Số giao dịch tối đa

        Returns:
            List[Dict]: Danh sách giao dịch
        """
        if user_id not in self._cache:
            return []

        transactions = self._cache[user_id]['transactions']

        # Lấy N giao dịch gần nhất
        recent_txs = transactions.head(limit)

        result = []
        for _, tx in recent_txs.iterrows():
            result.append({
                'transaction_id': tx.get('transaction_id', ''),
                'timestamp': str(tx.get('timestamp', '')),
                'ngay_giao_dich': tx.get('ngay_giao_dich', ''),
                'gio_giao_dich': tx.get('gio_giao_dich', ''),
                'loai_giao_dich': tx.get('loai_giao_dich', ''),
                'so_tien': float(tx.get('so_tien', 0)),
                'so_tien_formatted': tx.get('so_tien_formatted', ''),
                'ten_nguoi_nhan': tx.get('ten_nguoi_nhan', ''),
                'ngan_hang_nguoi_nhan': tx.get('ngan_hang_nguoi_nhan', ''),
                'channel': tx.get('channel', ''),
                'noi_dung_giao_dich': tx.get('noi_dung_giao_dich', ''),
                'trang_thai': tx.get('trang_thai', ''),
                'is_new_recipient': bool(tx.get('is_new_recipient', False)),
                'is_night_transaction': bool(tx.get('is_night_transaction', False)),
            })

        return result

    def calculate_behavioral_features(self, user_id: str, new_amount: float = 0) -> Dict[str, Any]:
        """
        Tính toán các chỉ số hành vi của khách hàng

        Args:
            user_id: ID của khách hàng
            new_amount: Số tiền giao dịch mới (để tính deviation)

        Returns:
            Dict: Các chỉ số hành vi
        """
        if user_id not in self._cache:
            return {
                'velocity_1h': 0,
                'velocity_24h': 0,
                'time_since_last_transaction': 999,
                'amount_deviation_ratio': 1.0,
                'avg_amount': 0,
                'std_amount': 0,
            }

        transactions = self._cache[user_id]['transactions']

        if len(transactions) == 0:
            return {
                'velocity_1h': 0,
                'velocity_24h': 0,
                'time_since_last_transaction': 999,
                'amount_deviation_ratio': 1.0,
                'avg_amount': 0,
                'std_amount': 0,
            }

        # Lấy thời gian hiện tại (hoặc giả lập từ giao dịch cuối)
        now = datetime.now()

        # Đảm bảo timestamp là datetime
        txs = transactions.copy()
        txs['timestamp'] = pd.to_datetime(txs['timestamp'])

        # Giao dịch trong 1 giờ
        one_hour_ago = now - timedelta(hours=1)
        velocity_1h = len(txs[txs['timestamp'] >= one_hour_ago])

        # Giao dịch trong 24 giờ
        twenty_four_hours_ago = now - timedelta(hours=24)
        velocity_24h = len(txs[txs['timestamp'] >= twenty_four_hours_ago])

        # Thời gian từ giao dịch cuối
        last_tx_time = txs['timestamp'].max()
        time_since_last = (now - last_tx_time).total_seconds() / 3600  # Đổi ra giờ

        # Tính trung bình và độ lệch chuẩn số tiền
        avg_amount = txs['so_tien'].mean()
        std_amount = txs['so_tien'].std()
        if pd.isna(std_amount) or std_amount == 0:
            std_amount = avg_amount * 0.5  # Fallback

        # Tính độ lệch số tiền
        if new_amount > 0 and avg_amount > 0:
            deviation = abs(new_amount - avg_amount) / std_amount if std_amount > 0 else 1
            deviation_ratio = round(deviation, 2)
        else:
            deviation_ratio = 1.0

        return {
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'time_since_last_transaction': round(time_since_last, 1),
            'amount_deviation_ratio': deviation_ratio,
            'avg_amount': round(avg_amount, 0),
            'std_amount': round(std_amount, 0),
        }

    def prepare_transaction_for_analysis(self, user_id: str, transaction_data: Dict) -> Dict[str, Any]:
        """
        Chuẩn bị dữ liệu giao dịch để đưa vào model phân tích

        Args:
            user_id: ID khách hàng
            transaction_data: Dữ liệu giao dịch từ frontend

        Returns:
            Dict: Dữ liệu đã được chuẩn bị với đầy đủ features
        """
        profile = self.get_customer_profile(user_id)
        transactions = self._cache.get(user_id, {}).get('transactions', pd.DataFrame())
        amount = float(transaction_data.get('amount', 0))

        # Tính behavioral features
        behavioral = self.calculate_behavioral_features(user_id, amount)

        # Lấy thông tin từ profile
        avg_amount = profile.get('avg_transaction_amount', 0) if profile else 0
        account_age = profile.get('so_ngay_mo_tai_khoan', 0) if profile else 0
        income = profile.get('thu_nhap_hang_thang', 0) if profile else 0

        # Tính các features bổ sung
        hour = int(transaction_data.get('hour', datetime.now().hour))
        is_night = 1 if hour < 6 or hour >= 22 else 0
        is_business_hours = 1 if 9 <= hour < 17 else 0

        # Amount features
        amount_log = np.log1p(amount)
        amount_to_avg_ratio = amount / avg_amount if avg_amount > 0 else 1
        amount_to_income_ratio = amount / income if income > 0 else 1

        # Thông tin người nhận từ form mới
        recipient_name = transaction_data.get('recipient_name', '')
        recipient_account = transaction_data.get('recipient_account', '')
        recipient_bank = transaction_data.get('recipient_bank', '')

        # Tạo recipient_id từ số tài khoản + ngân hàng (để kiểm tra người nhận mới)
        recipient_id = f"{recipient_account}_{recipient_bank}" if recipient_account and recipient_bank else ''

        # Kiểm tra người nhận mới dựa vào tên hoặc số tài khoản
        is_new_recipient = 1
        if len(transactions) > 0:
            # Kiểm tra tên người nhận đã giao dịch trước đó
            if 'ten_nguoi_nhan' in transactions.columns and recipient_name:
                known_names = transactions['ten_nguoi_nhan'].dropna().unique().tolist()
                if recipient_name in known_names:
                    is_new_recipient = 0

            # Kiểm tra số tài khoản + ngân hàng đã giao dịch trước đó
            if is_new_recipient == 1 and 'recipient_id' in transactions.columns and recipient_id:
                known_recipients = transactions['recipient_id'].dropna().unique().tolist()
                if recipient_id in known_recipients:
                    is_new_recipient = 0

        # Bank risk encoding - một số ngân hàng/ví có rủi ro cao hơn
        bank_risk_map = {
            # Ngân hàng lớn - rủi ro thấp
            'VCB': 0, 'BIDV': 0, 'VTB': 0, 'AGRB': 0,
            # Ngân hàng trung bình
            'TCB': 1, 'ACB': 1, 'MBB': 1, 'VPB': 1, 'STB': 1, 'SHB': 1,
            'TPB': 1, 'HDB': 1, 'MSB': 1, 'OCB': 1, 'LPB': 1,
            # Ngân hàng nhỏ - rủi ro cao hơn một chút
            'SCB': 2, 'NAB': 2, 'BAB': 2,
            # Ví điện tử - cần đánh giá kỹ hơn
            'MOMO': 2, 'ZALO': 2, 'SHOPEEPAY': 2, 'GRABPAY': 2,
        }
        recipient_bank_risk = bank_risk_map.get(recipient_bank, 1)

        # Kiểm tra số tài khoản bất thường (độ dài, format)
        account_length = len(recipient_account) if recipient_account else 0
        is_unusual_account = 1 if account_length > 0 and (account_length < 6 or account_length > 20) else 0

        # Transaction type encoding
        tx_type_map = {
            'transfer': 0,
            'payment': 1,
            'withdrawal': 2,
            'deposit': 3,
            'Chuyen khoan': 0,
            'Thanh toan': 1,
            'Rut tien': 2,
            'Nap tien': 3
        }
        tx_type = transaction_data.get('transaction_type', 'transfer')
        tx_type_encoded = tx_type_map.get(tx_type, 0)

        # Channel encoding
        channel_map = {
            'mobile_app': 0,
            'Mobile App': 0,
            'web_banking': 1,
            'Web': 1,
            'atm': 2,
            'ATM': 2,
            'branch': 3,
            'POS': 4
        }
        channel = transaction_data.get('channel', 'mobile_app')
        channel_encoded = channel_map.get(channel, 0)

        # Trả về dict với tất cả features
        prepared_data = {
            # Identifiers
            'user_id': user_id,
            'transaction_id': transaction_data.get('transaction_id', f'TX_{user_id}_{datetime.now().timestamp()}'),

            # Amount features
            'amount': amount,
            'amount_log': amount_log,
            'amount_to_avg_ratio': amount_to_avg_ratio,
            'amount_to_income_ratio': amount_to_income_ratio,

            # Time features
            'hour': hour,
            'is_night_transaction': is_night,
            'is_during_business_hours': is_business_hours,
            'day_of_week': datetime.now().weekday(),

            # Behavioral features
            'velocity_1h': behavioral['velocity_1h'],
            'velocity_24h': behavioral['velocity_24h'],
            'time_since_last_transaction': behavioral['time_since_last_transaction'],
            'amount_deviation_ratio': behavioral['amount_deviation_ratio'],

            # User features
            'account_age_days': account_age,
            'avg_transaction_amount': avg_amount,

            # Transaction type features
            'transaction_type': tx_type,
            'transaction_type_encoded': tx_type_encoded,
            'channel': channel,
            'channel_encoded': channel_encoded,

            # Recipient features
            'recipient_id': recipient_id,
            'recipient_name': recipient_name,
            'recipient_account': recipient_account,
            'recipient_bank': recipient_bank,
            'recipient_bank_risk': recipient_bank_risk,
            'is_new_recipient': is_new_recipient,
            'is_unusual_account': is_unusual_account,

            # Other flags
            'is_international': int(transaction_data.get('is_international', False)),

            # Raw behavioral for response
            '_behavioral_features': behavioral,
            '_profile': profile,
            '_recipient_info': {
                'name': recipient_name,
                'account': recipient_account,
                'bank': recipient_bank,
                'bank_risk': recipient_bank_risk,
                'is_new': is_new_recipient,
            }
        }

        return prepared_data

    def get_recipients_history(self, user_id: str) -> List[str]:
        """Lấy danh sách người nhận đã giao dịch"""
        if user_id not in self._cache:
            return []

        transactions = self._cache[user_id]['transactions']
        return transactions['recipient_id'].unique().tolist()


# Singleton instance
_customer_service = None

def get_customer_service() -> CustomerDataService:
    """Lấy instance của CustomerDataService (singleton)"""
    global _customer_service
    if _customer_service is None:
        _customer_service = CustomerDataService()
    return _customer_service

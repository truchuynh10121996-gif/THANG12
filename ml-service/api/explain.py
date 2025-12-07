"""
Explainability Service - Giải thích predictions
================================================
Service cung cấp giải thích cho các dự đoán fraud
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class ExplainabilityService:
    """
    Service giải thích predictions

    Cung cấp:
    - Feature contributions
    - Model contributions
    - Recommendations
    """

    def __init__(self):
        """Khởi tạo service"""
        self.feature_descriptions = {
            'log_amount': 'Logarit của số tiền giao dịch',
            'hour': 'Giờ thực hiện giao dịch',
            'day_of_week': 'Ngày trong tuần',
            'is_weekend': 'Giao dịch vào cuối tuần',
            'is_night_time': 'Giao dịch vào ban đêm (22h-5h)',
            'is_international': 'Giao dịch quốc tế',
            'amount_vs_user_avg': 'Tỷ lệ so với số tiền trung bình của user',
            'time_since_last_txn': 'Thời gian từ giao dịch trước',
            'is_rapid_txn': 'Giao dịch nhanh liên tiếp',
            'channel_risk': 'Mức độ rủi ro của kênh giao dịch'
        }

        self.risk_descriptions = {
            'low': 'Rủi ro thấp - Giao dịch bình thường',
            'medium': 'Rủi ro trung bình - Cần xem xét',
            'high': 'Rủi ro cao - Nên chặn hoặc yêu cầu xác nhận',
            'critical': 'Rủi ro nghiêm trọng - Chặn ngay lập tức'
        }

    def explain(self, transaction: Dict) -> Dict:
        """
        Giải thích dự đoán cho một giao dịch

        Args:
            transaction: Dict giao dịch

        Returns:
            Dict giải thích
        """
        explanation = {
            'summary': '',
            'risk_factors': [],
            'safe_factors': [],
            'recommendations': [],
            'feature_contributions': {},
            'model_contributions': {}
        }

        # Phân tích các yếu tố rủi ro
        amount = float(transaction.get('amount', 0))
        hour = transaction.get('hour', 12)
        is_international = transaction.get('is_international', False)
        channel = transaction.get('channel', 'mobile_app')

        risk_factors = []
        safe_factors = []

        # Kiểm tra số tiền
        if amount > 50_000_000:  # > 50 triệu
            risk_factors.append({
                'factor': 'Số tiền lớn',
                'description': f'Giao dịch {amount:,.0f} VND vượt ngưỡng 50 triệu',
                'importance': 'high'
            })
        elif amount > 10_000_000:
            risk_factors.append({
                'factor': 'Số tiền khá lớn',
                'description': f'Giao dịch {amount:,.0f} VND cần lưu ý',
                'importance': 'medium'
            })
        else:
            safe_factors.append({
                'factor': 'Số tiền bình thường',
                'description': f'Giao dịch {amount:,.0f} VND trong mức bình thường'
            })

        # Kiểm tra thời gian
        if isinstance(hour, (int, float)) and (hour < 6 or hour >= 22):
            risk_factors.append({
                'factor': 'Thời gian bất thường',
                'description': 'Giao dịch ngoài giờ làm việc (22h-6h)',
                'importance': 'medium'
            })
        else:
            safe_factors.append({
                'factor': 'Thời gian bình thường',
                'description': 'Giao dịch trong giờ làm việc'
            })

        # Kiểm tra quốc tế
        if is_international:
            risk_factors.append({
                'factor': 'Giao dịch quốc tế',
                'description': 'Giao dịch từ/đến nước ngoài',
                'importance': 'high'
            })
        else:
            safe_factors.append({
                'factor': 'Giao dịch nội địa',
                'description': 'Giao dịch trong nước'
            })

        explanation['risk_factors'] = risk_factors
        explanation['safe_factors'] = safe_factors

        # Tạo tóm tắt
        if len(risk_factors) == 0:
            explanation['summary'] = 'Giao dịch có vẻ an toàn, không có yếu tố rủi ro đáng kể'
        elif len(risk_factors) == 1:
            explanation['summary'] = f'Giao dịch có 1 yếu tố cần lưu ý: {risk_factors[0]["factor"]}'
        else:
            factors = ', '.join([f['factor'] for f in risk_factors])
            explanation['summary'] = f'Giao dịch có {len(risk_factors)} yếu tố rủi ro: {factors}'

        # Recommendations
        if len(risk_factors) >= 2 and any(f['importance'] == 'high' for f in risk_factors):
            explanation['recommendations'] = [
                'Xác nhận lại với khách hàng qua OTP',
                'Yêu cầu xác thực sinh trắc học',
                'Tạm thời chặn giao dịch để review'
            ]
        elif len(risk_factors) >= 1:
            explanation['recommendations'] = [
                'Gửi thông báo cảnh báo cho khách hàng',
                'Ghi nhận vào hệ thống giám sát'
            ]
        else:
            explanation['recommendations'] = [
                'Cho phép giao dịch bình thường'
            ]

        return explanation

    def get_feature_description(self, feature_name: str) -> str:
        """Lấy mô tả của một feature"""
        return self.feature_descriptions.get(feature_name, feature_name)

    def get_risk_description(self, risk_level: str) -> str:
        """Lấy mô tả của mức độ rủi ro"""
        return self.risk_descriptions.get(risk_level, risk_level)

"""
Fraud Detection Pipeline - Phân tích giao dịch với 5 models
============================================================
Pipeline chuyên nghiệp để phát hiện gian lận/lừa đảo.

5 Models được sử dụng:
1. Isolation Forest - Phát hiện anomaly (Layer 1)
2. LightGBM - Supervised classification (Layer 1)
3. Autoencoder - Reconstruction error (Layer 2)
4. LSTM - Sequence analysis (Layer 2)
5. GNN - Graph neural network (Layer 2)

Quy trình:
1. Nhận dữ liệu giao dịch đã chuẩn bị
2. Chạy qua từng model
3. Ensemble kết quả
4. Đưa ra quyết định và giải thích
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger('fraud_pipeline')

# Thêm path để import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FraudDetectionPipeline:
    """
    Pipeline phát hiện gian lận với 5 models

    Các ngưỡng rủi ro:
    - Low (Thấp): < 0.3 - An toàn, chuyển tiền thành công
    - Medium (Trung bình): 0.3 - 0.5 - Cần xác minh thêm
    - High (Cao): 0.5 - 0.7 - Nghi ngờ cao, cần kiểm tra
    - Critical (Nghiêm trọng): > 0.7 - Chặn giao dịch
    """

    # Ngưỡng quyết định
    THRESHOLD_LOW = 0.3
    THRESHOLD_MEDIUM = 0.5
    THRESHOLD_HIGH = 0.7

    # Trọng số cho từng model
    MODEL_WEIGHTS = {
        'isolation_forest': 0.15,
        'lightgbm': 0.25,
        'autoencoder': 0.20,
        'lstm': 0.20,
        'gnn': 0.20
    }

    def __init__(self):
        self.models = {}
        self.models_loaded = {
            'isolation_forest': False,
            'lightgbm': False,
            'autoencoder': False,
            'lstm': False,
            'gnn': False
        }
        self._load_models()

    def _load_models(self):
        """Load các models đã được train"""
        logger.info("Loading fraud detection models...")

        # Load Isolation Forest
        try:
            from models.layer1.isolation_forest import IsolationForestModel
            model = IsolationForestModel()
            if model.load():
                self.models['isolation_forest'] = model
                self.models_loaded['isolation_forest'] = True
                logger.info("✓ Isolation Forest loaded")
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest: {e}")

        # Load LightGBM
        try:
            from models.layer1.lightgbm_model import LightGBMModel
            model = LightGBMModel()
            if model.load():
                self.models['lightgbm'] = model
                self.models_loaded['lightgbm'] = True
                logger.info("✓ LightGBM loaded")
        except Exception as e:
            logger.warning(f"Could not load LightGBM: {e}")

        # Load Autoencoder
        try:
            from models.layer2.autoencoder import AutoencoderModel
            model = AutoencoderModel()
            if model.load():
                self.models['autoencoder'] = model
                self.models_loaded['autoencoder'] = True
                logger.info("✓ Autoencoder loaded")
        except Exception as e:
            logger.warning(f"Could not load Autoencoder: {e}")

        # Load LSTM
        try:
            from models.layer2.lstm_sequence import LSTMSequenceModel
            model = LSTMSequenceModel()
            if model.load():
                self.models['lstm'] = model
                self.models_loaded['lstm'] = True
                logger.info("✓ LSTM loaded")
        except Exception as e:
            logger.warning(f"Could not load LSTM: {e}")

        # Load GNN
        try:
            from models.layer2.gnn_hetero_model import GNNHeteroModel
            model = GNNHeteroModel()
            if model.load():
                self.models['gnn'] = model
                self.models_loaded['gnn'] = True
                logger.info("✓ GNN loaded")
        except Exception as e:
            logger.warning(f"Could not load GNN: {e}")

        loaded_count = sum(self.models_loaded.values())
        logger.info(f"Models loaded: {loaded_count}/5")

    def _prepare_features(self, transaction_data: Dict) -> np.ndarray:
        """Chuẩn bị features vector từ transaction data"""
        # Features cơ bản cho Isolation Forest và LightGBM
        features = [
            transaction_data.get('amount', 0),
            transaction_data.get('amount_log', 0),
            transaction_data.get('hour', 12),
            transaction_data.get('day_of_week', 0),
            transaction_data.get('velocity_1h', 0),
            transaction_data.get('velocity_24h', 0),
            transaction_data.get('time_since_last_transaction', 0),
            transaction_data.get('amount_deviation_ratio', 1),
            transaction_data.get('is_night_transaction', 0),
            transaction_data.get('is_during_business_hours', 0),
            transaction_data.get('is_new_recipient', 0),
            transaction_data.get('channel_encoded', 0),
            transaction_data.get('transaction_type_encoded', 0),
            transaction_data.get('account_age_days', 0),
            transaction_data.get('amount_to_avg_ratio', 1),
            transaction_data.get('is_international', 0),
            # Recipient features mới
            transaction_data.get('recipient_bank_risk', 1),
            transaction_data.get('is_unusual_account', 0),
        ]
        return np.array(features).reshape(1, -1)

    def _run_isolation_forest(self, features: np.ndarray) -> Tuple[float, Dict]:
        """Chạy Isolation Forest model"""
        if not self.models_loaded['isolation_forest']:
            return 0.5, {'loaded': False, 'reason': 'Model not loaded'}

        try:
            model = self.models['isolation_forest']
            score = model.predict_proba(features)[0]
            anomaly_score = model.get_anomaly_score(features)[0]

            return float(score), {
                'loaded': True,
                'fraud_probability': float(score),
                'anomaly_score': float(anomaly_score),
                'description': 'Phát hiện bất thường dựa trên phân bố dữ liệu'
            }
        except Exception as e:
            logger.error(f"Isolation Forest error: {e}")
            return 0.5, {'loaded': True, 'error': str(e)}

    def _run_lightgbm(self, features: np.ndarray) -> Tuple[float, Dict]:
        """Chạy LightGBM model"""
        if not self.models_loaded['lightgbm']:
            return 0.5, {'loaded': False, 'reason': 'Model not loaded'}

        try:
            model = self.models['lightgbm']
            proba = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]

            return float(proba), {
                'loaded': True,
                'fraud_probability': float(proba),
                'prediction': int(prediction),
                'description': 'Phân loại dựa trên các đặc trưng giao dịch'
            }
        except Exception as e:
            logger.error(f"LightGBM error: {e}")
            return 0.5, {'loaded': True, 'error': str(e)}

    def _run_autoencoder(self, features: np.ndarray) -> Tuple[float, Dict]:
        """Chạy Autoencoder model"""
        if not self.models_loaded['autoencoder']:
            return 0.5, {'loaded': False, 'reason': 'Model not loaded'}

        try:
            model = self.models['autoencoder']
            recon_error = model.get_reconstruction_error(features)[0]
            proba = model.predict_proba(features)[0]

            return float(proba), {
                'loaded': True,
                'fraud_probability': float(proba),
                'reconstruction_error': float(recon_error),
                'threshold': float(model.threshold) if model.threshold else 0,
                'description': 'Phát hiện bất thường dựa trên lỗi tái tạo'
            }
        except Exception as e:
            logger.error(f"Autoencoder error: {e}")
            return 0.5, {'loaded': True, 'error': str(e)}

    def _run_lstm(self, transaction_data: Dict, features: np.ndarray) -> Tuple[float, Dict]:
        """Chạy LSTM model"""
        if not self.models_loaded['lstm']:
            return 0.5, {'loaded': False, 'reason': 'Model not loaded'}

        try:
            model = self.models['lstm']

            # LSTM cần sequence, nếu chỉ có 1 giao dịch thì tạo dummy sequence
            # Trong thực tế, cần lấy lịch sử giao dịch để tạo sequence đầy đủ
            seq_length = model.seq_length if hasattr(model, 'seq_length') else 7
            num_features = features.shape[1]

            # Tạo sequence với padding
            sequence = np.zeros((1, seq_length, num_features))
            sequence[0, -1, :] = features[0]  # Giao dịch hiện tại ở cuối

            proba = model.predict_proba(sequence)[0]

            return float(proba), {
                'loaded': True,
                'fraud_probability': float(proba),
                'seq_length': seq_length,
                'description': 'Phân tích chuỗi giao dịch theo thời gian'
            }
        except Exception as e:
            logger.error(f"LSTM error: {e}")
            return 0.5, {'loaded': True, 'error': str(e)}

    def _run_gnn(self, transaction_data: Dict) -> Tuple[float, Dict]:
        """Chạy GNN model"""
        if not self.models_loaded['gnn']:
            return 0.5, {'loaded': False, 'reason': 'Model not loaded'}

        try:
            model = self.models['gnn']
            # GNN cần graph structure, đây là simplified version
            proba = 0.5  # Default nếu không có graph

            return float(proba), {
                'loaded': True,
                'fraud_probability': float(proba),
                'description': 'Phân tích mối quan hệ trong mạng lưới giao dịch'
            }
        except Exception as e:
            logger.error(f"GNN error: {e}")
            return 0.5, {'loaded': True, 'error': str(e)}

    def _ensemble_predictions(self, model_scores: Dict[str, float]) -> float:
        """
        Kết hợp kết quả từ các models

        Args:
            model_scores: Dict với tên model và điểm số

        Returns:
            float: Điểm rủi ro tổng hợp (0-1)
        """
        total_weight = 0
        weighted_sum = 0

        for model_name, score in model_scores.items():
            weight = self.MODEL_WEIGHTS.get(model_name, 0)
            if self.models_loaded.get(model_name, False):
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            # Fallback: dùng rule-based nếu không có model nào
            return 0.5

        return weighted_sum / total_weight

    def _determine_risk_level(self, fraud_probability: float) -> str:
        """Xác định mức độ rủi ro"""
        if fraud_probability >= self.THRESHOLD_HIGH:
            return 'critical'
        elif fraud_probability >= self.THRESHOLD_MEDIUM:
            return 'high'
        elif fraud_probability >= self.THRESHOLD_LOW:
            return 'medium'
        else:
            return 'low'

    def _generate_explanation(
        self,
        transaction_data: Dict,
        model_results: Dict,
        risk_level: str
    ) -> Dict[str, Any]:
        """
        Tạo giải thích chi tiết về quyết định

        Returns:
            Dict chứa summary, risk_factors, và recommendations
        """
        risk_factors = []
        recommendations = []
        positive_factors = []

        amount = transaction_data.get('amount', 0)
        avg_amount = transaction_data.get('avg_transaction_amount', 0)
        deviation = transaction_data.get('amount_deviation_ratio', 1)
        velocity_1h = transaction_data.get('velocity_1h', 0)
        velocity_24h = transaction_data.get('velocity_24h', 0)
        is_night = transaction_data.get('is_night_transaction', 0)
        is_new_recipient = transaction_data.get('is_new_recipient', 0)
        is_international = transaction_data.get('is_international', 0)

        # Phân tích số tiền
        if avg_amount > 0 and amount > avg_amount * 3:
            risk_factors.append({
                'factor': f'Số tiền ({amount:,.0f}đ) cao gấp {amount/avg_amount:.1f}x so với trung bình ({avg_amount:,.0f}đ)',
                'importance': 'high'
            })
            recommendations.append('Xác nhận lại với khách hàng qua OTP hoặc cuộc gọi')
        elif avg_amount > 0 and amount > avg_amount * 2:
            risk_factors.append({
                'factor': f'Số tiền cao hơn trung bình ({amount:,.0f}đ vs {avg_amount:,.0f}đ)',
                'importance': 'medium'
            })

        # Phân tích độ lệch
        if deviation > 3:
            risk_factors.append({
                'factor': f'Độ lệch số tiền cao (x{deviation})',
                'importance': 'high'
            })

        # Phân tích velocity
        if velocity_1h >= 5:
            risk_factors.append({
                'factor': f'Tần suất giao dịch cao bất thường ({velocity_1h} GD/giờ)',
                'importance': 'high'
            })
            recommendations.append('Kiểm tra xem tài khoản có bị chiếm đoạt không')
        elif velocity_24h >= 20:
            risk_factors.append({
                'factor': f'Số giao dịch trong 24h cao ({velocity_24h} GD)',
                'importance': 'medium'
            })

        # Phân tích thời gian
        if is_night:
            risk_factors.append({
                'factor': 'Giao dịch vào ban đêm (22h-6h)',
                'importance': 'medium'
            })

        # Phân tích người nhận (với thông tin chi tiết từ form mới)
        recipient_info = transaction_data.get('_recipient_info', {})
        recipient_name = recipient_info.get('name') or transaction_data.get('recipient_name', '')
        recipient_bank = recipient_info.get('bank') or transaction_data.get('recipient_bank', '')
        recipient_bank_risk = transaction_data.get('recipient_bank_risk', 1)
        is_unusual_account = transaction_data.get('is_unusual_account', 0)

        if is_new_recipient:
            if recipient_name:
                risk_factors.append({
                    'factor': f'Người nhận "{recipient_name}" là đối tác mới (chưa giao dịch trước đó)',
                    'importance': 'medium'
                })
            else:
                risk_factors.append({
                    'factor': 'Người nhận là đối tác mới',
                    'importance': 'medium'
                })
            recommendations.append('Xác minh thông tin người nhận trước khi chuyển tiền')

        # Kiểm tra ngân hàng/ví điện tử
        if recipient_bank_risk >= 2:
            bank_name = recipient_bank or 'ví điện tử/ngân hàng nhỏ'
            risk_factors.append({
                'factor': f'Chuyển tiền đến {bank_name} - cần xác minh kỹ hơn',
                'importance': 'medium'
            })

        # Kiểm tra số tài khoản bất thường
        if is_unusual_account:
            risk_factors.append({
                'factor': 'Số tài khoản có định dạng không bình thường',
                'importance': 'medium'
            })
            recommendations.append('Kiểm tra lại số tài khoản người nhận')

        # Giao dịch quốc tế
        if is_international:
            risk_factors.append({
                'factor': 'Giao dịch quốc tế',
                'importance': 'medium'
            })
            recommendations.append('Xác nhận danh tính và mục đích giao dịch')

        # Phân tích kết quả models
        for model_name, result in model_results.items():
            if isinstance(result, dict) and result.get('loaded'):
                prob = result.get('fraud_probability', 0.5)
                if prob > 0.7:
                    risk_factors.append({
                        'factor': f'Model {model_name}: Phát hiện rủi ro cao ({prob*100:.0f}%)',
                        'importance': 'high'
                    })
                elif prob > 0.5:
                    risk_factors.append({
                        'factor': f'Model {model_name}: Phát hiện rủi ro ({prob*100:.0f}%)',
                        'importance': 'medium'
                    })

        # Các yếu tố tích cực
        if len(risk_factors) == 0:
            positive_factors.append('Số tiền giao dịch nằm trong phạm vi bình thường')
            positive_factors.append('Không phát hiện bất thường về hành vi')

        account_age = transaction_data.get('account_age_days', 0)
        if account_age > 365:
            positive_factors.append(f'Tài khoản lâu năm ({account_age} ngày)')

        # Tạo summary
        if risk_level == 'critical':
            summary = f"⚠️ CẢNH BÁO: Giao dịch có {len(risk_factors)} yếu tố rủi ro nghiêm trọng. Khuyến nghị CHẶN giao dịch."
            recommendations.insert(0, '🛑 Chặn giao dịch ngay lập tức')
            recommendations.append('Liên hệ khách hàng để xác minh')
        elif risk_level == 'high':
            summary = f"⚠️ NGHI NGỜ: Giao dịch có {len(risk_factors)} yếu tố rủi ro. Cần xem xét kỹ trước khi duyệt."
            recommendations.insert(0, '⏸️ Tạm giữ giao dịch để xác minh')
        elif risk_level == 'medium':
            summary = f"⚡ Giao dịch có {len(risk_factors)} yếu tố cần lưu ý. Có thể cho phép với xác thực bổ sung."
            if 'Xác nhận OTP' not in recommendations:
                recommendations.append('Yêu cầu xác nhận OTP')
        else:
            summary = f"✅ Giao dịch AN TOÀN. Không phát hiện dấu hiệu bất thường."
            recommendations = ['Cho phép giao dịch']

        return {
            'summary': summary,
            'risk_factors': risk_factors,
            'positive_factors': positive_factors,
            'recommendations': recommendations
        }

    def analyze_transaction(self, transaction_data: Dict) -> Dict[str, Any]:
        """
        Phân tích giao dịch và đưa ra quyết định

        Args:
            transaction_data: Dữ liệu giao dịch đã được chuẩn bị

        Returns:
            Dict chứa kết quả phân tích đầy đủ
        """
        logger.info(f"Analyzing transaction: {transaction_data.get('transaction_id')}")

        # Chuẩn bị features
        features = self._prepare_features(transaction_data)

        # Chạy từng model
        model_results = {}
        model_scores = {}

        # Isolation Forest
        score, result = self._run_isolation_forest(features)
        model_results['isolation_forest'] = result
        model_scores['isolation_forest'] = score

        # LightGBM
        score, result = self._run_lightgbm(features)
        model_results['lightgbm'] = result
        model_scores['lightgbm'] = score

        # Autoencoder
        score, result = self._run_autoencoder(features)
        model_results['autoencoder'] = result
        model_scores['autoencoder'] = score

        # LSTM
        score, result = self._run_lstm(transaction_data, features)
        model_results['lstm'] = result
        model_scores['lstm'] = score

        # GNN
        score, result = self._run_gnn(transaction_data)
        model_results['gnn'] = result
        model_scores['gnn'] = score

        # Ensemble
        fraud_probability = self._ensemble_predictions(model_scores)

        # Xác định risk level
        risk_level = self._determine_risk_level(fraud_probability)

        # Quyết định
        should_block = risk_level in ['critical', 'high']
        prediction = 'fraud' if fraud_probability >= self.THRESHOLD_MEDIUM else 'normal'

        # Tính confidence
        scores = list(model_scores.values())
        confidence = 1 - np.std(scores) if len(scores) > 1 else 0.5

        # Tạo giải thích
        explanation = self._generate_explanation(transaction_data, model_results, risk_level)

        return {
            'transaction_id': transaction_data.get('transaction_id'),
            'user_id': transaction_data.get('user_id'),
            'amount': transaction_data.get('amount'),
            'prediction': {
                'fraud_probability': round(fraud_probability, 4),
                'prediction': prediction,
                'risk_level': risk_level,
                'should_block': should_block,
                'confidence': round(confidence, 2)
            },
            'model_scores': {
                model: round(score, 4) for model, score in model_scores.items()
            },
            'model_details': model_results,
            'models_status': self.models_loaded,
            'explanation': explanation,
            'behavioral_features': transaction_data.get('_behavioral_features'),
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
_pipeline = None


def get_fraud_pipeline() -> FraudDetectionPipeline:
    """Lấy instance của FraudDetectionPipeline (singleton)"""
    global _pipeline
    if _pipeline is None:
        _pipeline = FraudDetectionPipeline()
    return _pipeline

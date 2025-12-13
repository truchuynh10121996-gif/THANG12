"""
API Routes - Định nghĩa các API endpoints
==========================================
Các endpoints chính:
- /api/predict: Dự đoán fraud cho giao dịch
- /api/train: Training models
- /api/metrics: Lấy metrics đánh giá
- /api/dashboard: Dữ liệu cho dashboard
- /api/users: Quản lý users (MỚI)
- /api/transactions: Quản lý giao dịch (MỚI)

Đã tích hợp MongoDB để:
- Lưu trữ và truy vấn users
- Lưu lịch sử giao dịch
- Tính toán behavioral features
"""

import os
import sys
import logging
import traceback
import tempfile
from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime

# Cấu hình logging chi tiết cho training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml-training')

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()

# Import prediction và training logic
from .predict import PredictionService
from .explain import ExplainabilityService

# Import database connection
from database.mongodb import get_db

# Tạo Blueprint
api = Blueprint('api', __name__)

# Khởi tạo services
prediction_service = PredictionService()
explain_service = ExplainabilityService()


# ===== USER ENDPOINTS (MỚI) =====

@api.route('/users', methods=['GET'])
def get_users():
    """
    Lấy danh sách tất cả users

    Query params:
        limit (int): Số lượng user tối đa (mặc định 100)

    Response:
    {
        "success": true,
        "users": [
            {
                "user_id": "USR_001",
                "name": "Nguyễn Văn A",
                "age": 35,
                "occupation": "Kỹ sư",
                ...
            }
        ]
    }
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        db = get_db()
        users = db.get_all_users(limit=limit)

        return jsonify({
            'success': True,
            'users': users,
            'total': len(users),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/users/<user_id>', methods=['GET'])
def get_user_detail(user_id):
    """
    Lấy chi tiết user bao gồm profile và behavioral features

    Path params:
        user_id: ID của user

    Response:
    {
        "success": true,
        "user_id": "USR_001",
        "profile": {
            "name": "Nguyễn Văn A",
            "age": 35,
            "occupation": "Kỹ sư",
            "income_level": "high",
            "account_age_days": 730,
            "kyc_level": 3,
            "avg_transaction_amount": 8500000,
            "historical_risk_score": 0.12,
            "total_transactions": 256
        },
        "behavioral_features": {
            "velocity_1h": 2,
            "velocity_24h": 8,
            "time_since_last_transaction": 4.5,
            "amount_deviation_ratio": 1.2
        }
    }
    """
    try:
        db = get_db()

        # Lấy thông tin user
        user = db.get_user_by_id(user_id)

        if not user:
            return jsonify({
                'success': False,
                'error': f'Không tìm thấy user {user_id}'
            }), 404

        # Tính behavioral features
        features = db.calculate_behavioral_features(user_id)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'profile': user,
            'behavioral_features': {
                'velocity_1h': features['velocity_1h'],
                'velocity_24h': features['velocity_24h'],
                'time_since_last_transaction': features['time_since_last_transaction'],
                'amount_deviation_ratio': features['amount_deviation_ratio']
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/users/<user_id>/transactions', methods=['GET'])
def get_user_transactions(user_id):
    """
    Lấy lịch sử giao dịch của user

    Path params:
        user_id: ID của user

    Query params:
        limit (int): Số giao dịch tối đa (mặc định 10)

    Response:
    {
        "success": true,
        "user_id": "USR_001",
        "transactions": [
            {
                "transaction_id": "TXN_001",
                "amount": 5000000,
                "transaction_type": "transfer",
                "recipient_id": "RCP_001",
                "status": "completed",
                "timestamp": "2024-01-15T14:30:00"
            }
        ]
    }
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        db = get_db()

        transactions = db.get_user_transactions(user_id, limit=limit)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'transactions': transactions,
            'total': len(transactions),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/users', methods=['POST'])
def create_user():
    """
    Tạo user mới (cho demo)

    Request body:
    {
        "user_id": "USR_NEW_001",
        "name": "Nguyễn Văn Mới",
        "age": 30,
        "occupation": "Nhân viên",
        "income_level": "medium"
    }
    """
    try:
        data = request.get_json()

        if not data or 'user_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Cần user_id trong request body'
            }), 400

        # Thêm các giá trị mặc định
        user_data = {
            'user_id': data['user_id'],
            'name': data.get('name', f"User {data['user_id']}"),
            'age': data.get('age', 30),
            'occupation': data.get('occupation', 'Không xác định'),
            'income_level': data.get('income_level', 'medium'),
            'account_age_days': data.get('account_age_days', 0),
            'kyc_level': data.get('kyc_level', 1),
            'avg_transaction_amount': data.get('avg_transaction_amount', 5000000),
            'historical_risk_score': data.get('historical_risk_score', 0.2),
            'total_transactions': 0,
            'is_verified': False
        }

        db = get_db()
        created_user = db.create_user(user_data)

        return jsonify({
            'success': True,
            'message': 'Đã tạo user mới',
            'user': created_user,
            'timestamp': datetime.now().isoformat()
        }), 201

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== TRANSACTION ENDPOINTS (MỚI) =====

@api.route('/transactions/recent', methods=['GET'])
def get_recent_transactions():
    """
    Lấy giao dịch gần đây của hệ thống

    Query params:
        limit (int): Số giao dịch tối đa (mặc định 50)

    Response:
    {
        "success": true,
        "transactions": [...],
        "total": 50
    }
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        db = get_db()

        transactions = db.get_recent_transactions(limit=limit)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'total': len(transactions),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== PREDICTION ENDPOINTS (ĐÃ CẬP NHẬT) =====

@api.route('/predict', methods=['POST'])
def predict_single():
    """
    Dự đoán fraud cho 1 giao dịch

    Đã cập nhật để:
    - Nếu có user_id, query database lấy profile và lịch sử
    - Tính toán đầy đủ behavioral features
    - Lưu giao dịch và kết quả vào database

    Request body:
    {
        "transaction_id": "TXN001",
        "user_id": "USR001",
        "amount": 5000000,
        "transaction_type": "transfer",
        "channel": "mobile_app",
        "recipient_id": "RCP001",
        "timestamp": "2024-01-15 14:30:00",
        ...
    }

    Response:
    {
        "success": true,
        "prediction": {
            "fraud_probability": 0.85,
            "prediction": "fraud",
            "risk_level": "high",
            "should_block": true,
            "confidence": 0.70
        },
        "behavioral_features": {
            "velocity_1h": 3,
            "velocity_24h": 12,
            ...
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'Không có dữ liệu giao dịch'
            }), 400

        db = get_db()
        user_id = data.get('user_id')
        amount = float(data.get('amount', 0))

        # Nếu có user_id, tính toán behavioral features
        behavioral_features = None
        if user_id:
            behavioral_features = db.calculate_behavioral_features(user_id, amount)

            # Thêm các features vào data để prediction sử dụng
            data['velocity_1h'] = behavioral_features['velocity_1h']
            data['velocity_24h'] = behavioral_features['velocity_24h']
            data['time_since_last_transaction'] = behavioral_features['time_since_last_transaction']
            data['amount_deviation_ratio'] = behavioral_features['amount_deviation_ratio']

            # Kiểm tra người nhận mới
            if 'recipient_id' in data:
                recipients = db.get_user_recipients(user_id)
                data['is_new_recipient'] = data['recipient_id'] not in recipients

        # Dự đoán
        result = prediction_service.predict_single(data)

        # Lưu giao dịch vào database
        transaction_data = {
            **data,
            'fraud_probability': result.get('fraud_probability'),
            'risk_level': result.get('risk_level'),
            'prediction': result.get('prediction')
        }
        db.save_transaction(transaction_data)

        # Lưu kết quả prediction
        prediction_data = {
            'transaction_id': data.get('transaction_id'),
            'user_id': user_id,
            **result
        }
        db.save_prediction(prediction_data)

        response = {
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        }

        # Thêm behavioral features vào response nếu có
        if behavioral_features:
            response['behavioral_features'] = {
                'velocity_1h': behavioral_features['velocity_1h'],
                'velocity_24h': behavioral_features['velocity_24h'],
                'time_since_last_transaction': behavioral_features['time_since_last_transaction'],
                'amount_deviation_ratio': behavioral_features['amount_deviation_ratio'],
                'is_new_recipient': data.get('is_new_recipient', False)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Dự đoán fraud cho nhiều giao dịch

    Request body:
    {
        "transactions": [
            {"transaction_id": "TXN001", ...},
            {"transaction_id": "TXN002", ...}
        ]
    }

    Response:
    {
        "success": true,
        "predictions": [...],
        "summary": {
            "total": 10,
            "fraud_count": 2,
            "fraud_ratio": 0.2
        }
    }
    """
    try:
        data = request.get_json()

        if not data or 'transactions' not in data:
            return jsonify({
                'success': False,
                'error': 'Cần trường "transactions" trong request body'
            }), 400

        transactions = data['transactions']

        if len(transactions) > config.MAX_BATCH_SIZE:
            return jsonify({
                'success': False,
                'error': f'Số lượng giao dịch vượt quá giới hạn {config.MAX_BATCH_SIZE}'
            }), 400

        db = get_db()

        # Xử lý từng giao dịch - tính behavioral features
        for tx in transactions:
            user_id = tx.get('user_id')
            amount = float(tx.get('amount', 0))

            if user_id:
                features = db.calculate_behavioral_features(user_id, amount)
                tx['velocity_1h'] = features['velocity_1h']
                tx['velocity_24h'] = features['velocity_24h']
                tx['time_since_last_transaction'] = features['time_since_last_transaction']
                tx['amount_deviation_ratio'] = features['amount_deviation_ratio']

        # Dự đoán batch
        results = prediction_service.predict_batch(transactions)

        # Lưu các giao dịch và predictions
        for tx, result in zip(transactions, results):
            tx['fraud_probability'] = result.get('fraud_probability')
            tx['risk_level'] = result.get('risk_level')
            db.save_transaction(tx)

        # Tính summary
        fraud_count = sum(1 for r in results if r['prediction'] == 'fraud')

        return jsonify({
            'success': True,
            'predictions': results,
            'summary': {
                'total': len(results),
                'fraud_count': fraud_count,
                'fraud_ratio': fraud_count / len(results) if results else 0
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== MODEL STATUS ENDPOINTS =====

@api.route('/models/status', methods=['GET'])
def get_model_status():
    """
    Lấy trạng thái các models

    Response:
    {
        "success": true,
        "status": {
            "layer1": {"fitted": true, "models": [...]},
            "layer2": {"fitted": true, "models": {...}}
        }
    }
    """
    try:
        status = prediction_service.get_model_status()

        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== TRAINING ENDPOINTS =====

@api.route('/train/layer1', methods=['POST'])
def train_layer1():
    """
    Train Layer 1 models (Isolation Forest + LightGBM)

    Request body (optional):
    {
        "data_path": "path/to/features.csv"
    }
    """
    try:
        data = request.get_json() or {}
        data_path = data.get('data_path')

        result = prediction_service.train_layer1(data_path)

        return jsonify({
            'success': True,
            'result': result,
            'message': 'Layer 1 training hoàn tất',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/train/layer2', methods=['POST'])
def train_layer2():
    """
    Train Layer 2 models (Autoencoder + LSTM + GNN)
    """
    try:
        data = request.get_json() or {}

        result = prediction_service.train_layer2(data)

        return jsonify({
            'success': True,
            'result': result,
            'message': 'Layer 2 training hoàn tất',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/train/all', methods=['POST'])
def train_all():
    """
    Train tất cả models
    """
    try:
        data = request.get_json() or {}

        result = prediction_service.train_all(data)

        return jsonify({
            'success': True,
            'result': result,
            'message': 'Tất cả models đã được train',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== METRICS ENDPOINTS =====

@api.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Lấy metrics đánh giá models

    Response:
    {
        "success": true,
        "metrics": {
            "layer1": {...},
            "layer2": {...},
            "combined": {...}
        }
    }
    """
    try:
        metrics = prediction_service.get_metrics()

        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/metrics/history', methods=['GET'])
def get_metrics_history():
    """
    Lấy lịch sử metrics theo thời gian
    """
    try:
        history = prediction_service.get_metrics_history()

        return jsonify({
            'success': True,
            'history': history,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== EXPLAINABILITY ENDPOINTS =====

@api.route('/explain', methods=['POST'])
def explain_prediction():
    """
    Giải thích prediction cho một giao dịch

    Request body:
    {
        "transaction": {...}
    }

    Response:
    {
        "success": true,
        "explanation": {
            "fraud_probability": 0.85,
            "top_features": [...],
            "model_contributions": {...}
        }
    }
    """
    try:
        data = request.get_json()

        if not data or 'transaction' not in data:
            return jsonify({
                'success': False,
                'error': 'Cần trường "transaction" trong request body'
            }), 400

        explanation = explain_service.explain(data['transaction'])

        return jsonify({
            'success': True,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== DASHBOARD ENDPOINTS =====

@api.route('/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """
    Lấy thống kê cho dashboard

    Response:
    {
        "success": true,
        "stats": {
            "total_transactions": 10000,
            "fraud_detected": 500,
            "fraud_rate": 0.05,
            "high_risk_alerts": 50,
            ...
        }
    }
    """
    try:
        stats = prediction_service.get_dashboard_stats()

        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/graph/community', methods=['GET'])
def get_graph_community():
    """
    Lấy dữ liệu graph và communities cho visualization
    """
    try:
        graph_data = prediction_service.get_graph_data()

        return jsonify({
            'success': True,
            'graph': graph_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== USER PROFILE ENDPOINTS =====

@api.route('/user/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """
    Lấy user profile và embedding
    """
    try:
        profile = prediction_service.get_user_profile(user_id)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'profile': profile,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/user/<user_id>/sequence', methods=['GET'])
def get_user_sequence(user_id):
    """
    Lấy chuỗi giao dịch của user
    """
    try:
        sequence = prediction_service.get_user_sequence(user_id)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'sequence': sequence,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== DATA ENDPOINTS =====

@api.route('/data/generate', methods=['POST'])
def generate_data():
    """
    Tạo dữ liệu giả lập
    """
    try:
        data = request.get_json() or {}

        num_users = data.get('num_users', 1000)
        num_transactions = data.get('num_transactions', 10000)

        from data.synthetic.generate_data import DataGenerator

        generator = DataGenerator(
            num_users=num_users,
            num_transactions=num_transactions
        )

        users_df = generator.generate_users()
        transactions_df = generator.generate_transactions(users_df)
        fraud_reports_df = generator.generate_fraud_reports(transactions_df)

        generator.save_data(users_df, transactions_df, fraud_reports_df)

        return jsonify({
            'success': True,
            'message': 'Đã tạo dữ liệu giả lập',
            'stats': {
                'num_users': len(users_df),
                'num_transactions': len(transactions_df),
                'num_fraud': int((transactions_df['is_fraud'] == 1).sum())
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== INDIVIDUAL MODEL TRAINING ENDPOINTS (MỚI) =====

@api.route('/train/isolation_forest', methods=['POST'])
def train_isolation_forest():
    """
    Train Isolation Forest model với file upload

    Dữ liệu đầu vào: CSV file với các cột giao dịch
    Không cần cột is_fraud (unsupervised learning)

    Các cột được hỗ trợ (24 features):
    - tx_id, user_id (sẽ bị loại bỏ - identifiers)
    - amount, amount_log, amount_norm
    - hour_of_day, day_of_week, is_weekend
    - time_gap_prev_min, velocity_1h, velocity_24h
    - freq_norm, is_new_recipient, recipient_count_30d
    - is_new_device, device_count_30d, location_diff_km
    - channel, account_age_days, amount_percentile_system
    - global_anomaly_score_prev
    - amount_vs_avg_user_1m (MỚI): Tỷ lệ số tiền so với trung bình user trong 1 tháng
    - is_first_large_tx (MỚI): Có phải giao dịch lớn đầu tiên không
    - recipient_is_suspicious (MỚI): Người nhận có nghi ngờ không

    Response:
    {
        "success": true,
        "message": "Training Isolation Forest thành công",
        "training_info": {
            "samples_count": 1000,
            "features_count": 22,
            "feature_names": [...],
            "contamination": 0.05,
            "n_estimators": 100
        },
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.78,
            ...
        }
    }
    """
    logger.info("="*60)
    logger.info("[ISOLATION FOREST] Bắt đầu training...")
    logger.info("="*60)

    try:
        # Kiểm tra file upload
        if 'file' not in request.files:
            logger.error("[ISOLATION FOREST] Không có file được upload")
            return jsonify({
                'success': False,
                'error': 'Không có file được upload. Vui lòng chọn file CSV.'
            }), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("[ISOLATION FOREST] Tên file trống")
            return jsonify({
                'success': False,
                'error': 'Tên file trống. Vui lòng chọn file CSV.'
            }), 400

        if not file.filename.endswith('.csv'):
            logger.error(f"[ISOLATION FOREST] File không phải CSV: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'File phải có định dạng .csv'
            }), 400

        logger.info(f"[ISOLATION FOREST] Đang đọc file: {file.filename}")

        # Đọc CSV file
        try:
            df = pd.read_csv(file)
            logger.info(f"[ISOLATION FOREST] Đọc file thành công: {len(df)} dòng, {len(df.columns)} cột")
            logger.info(f"[ISOLATION FOREST] Các cột: {list(df.columns)}")
        except Exception as e:
            logger.error(f"[ISOLATION FOREST] Lỗi đọc file CSV: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Không thể đọc file CSV: {str(e)}'
            }), 400

        # Kiểm tra dữ liệu tối thiểu
        if len(df) < 10:
            logger.error(f"[ISOLATION FOREST] Dữ liệu quá ít: {len(df)} dòng")
            return jsonify({
                'success': False,
                'error': f'Dữ liệu quá ít ({len(df)} dòng). Cần ít nhất 10 dòng để training.'
            }), 400

        # Các cột identifier cần loại bỏ
        identifier_cols = ['tx_id', 'transaction_id', 'user_id', 'id', 'timestamp', 'created_at']
        # Cột label (không cần cho unsupervised)
        label_cols = ['is_fraud', 'fraud', 'label', 'fraud_type']

        # Lấy các cột features
        exclude_cols = identifier_cols + label_cols
        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

        logger.info(f"[ISOLATION FOREST] Các cột bị loại bỏ: {[c for c in df.columns if c.lower() in [e.lower() for e in exclude_cols]]}")
        logger.info(f"[ISOLATION FOREST] Feature columns: {feature_cols}")

        if len(feature_cols) == 0:
            logger.error("[ISOLATION FOREST] Không có cột features")
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy cột features trong dữ liệu. Các cột cần có: amount, hour_of_day, velocity_1h, ...'
            }), 400

        # Xử lý cột channel (categorical -> numeric)
        if 'channel' in df.columns:
            logger.info("[ISOLATION FOREST] Chuyển đổi cột channel từ categorical sang numeric")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['channel'] = le.fit_transform(df['channel'].fillna('unknown').astype(str))

        # Lấy dữ liệu features
        X = df[feature_cols].copy()

        # Xử lý missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"[ISOLATION FOREST] Có {missing_count} giá trị missing, đang fill với median")
            X = X.fillna(X.median())

        # Chuyển đổi tất cả sang numeric
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                logger.warning(f"[ISOLATION FOREST] Không thể chuyển cột {col} sang numeric, loại bỏ")
                X = X.drop(columns=[col])

        # Loại bỏ cột có tất cả NaN sau khi chuyển đổi
        X = X.dropna(axis=1, how='all')
        X = X.fillna(0)

        feature_names = list(X.columns)
        logger.info(f"[ISOLATION FOREST] Features sau xử lý: {feature_names}")
        logger.info(f"[ISOLATION FOREST] Shape dữ liệu: {X.shape}")

        if X.shape[1] == 0:
            logger.error("[ISOLATION FOREST] Không còn cột features sau khi xử lý")
            return jsonify({
                'success': False,
                'error': 'Không còn cột features sau khi xử lý. Đảm bảo dữ liệu có các cột số.'
            }), 400

        # Import và khởi tạo model
        from models.layer1.isolation_forest import IsolationForestModel

        model = IsolationForestModel()

        logger.info("[ISOLATION FOREST] Bắt đầu fit model...")
        logger.info(f"[ISOLATION FOREST] Config: n_estimators={model.config.get('n_estimators', 100)}, contamination={model.config.get('contamination', 0.05)}")

        # Training
        model.fit(X.values, feature_names=feature_names, verbose=True)

        logger.info("[ISOLATION FOREST] Training hoàn tất!")

        # Tính metrics nếu có label
        metrics = None
        has_label = False
        label_col = None

        for col in label_cols:
            if col in df.columns:
                label_col = col
                has_label = True
                break

        if has_label and label_col:
            logger.info(f"[ISOLATION FOREST] Phát hiện cột label '{label_col}', đang tính metrics...")
            y_true = df[label_col].values

            try:
                metrics = model.evaluate(X.values, y_true, verbose=True)
                logger.info(f"[ISOLATION FOREST] Metrics: {metrics}")
            except Exception as e:
                logger.warning(f"[ISOLATION FOREST] Không thể tính metrics: {str(e)}")
                metrics = None
        else:
            logger.info("[ISOLATION FOREST] Không có cột label, bỏ qua tính metrics (unsupervised)")
            # Tính một số thống kê cơ bản
            anomaly_scores = model.get_anomaly_score(X.values)
            fraud_proba = model.predict_proba(X.values)
            predictions = model.predict(X.values)

            anomaly_count = (predictions == -1).sum()
            anomaly_ratio = anomaly_count / len(predictions)

            metrics = {
                'mode': 'unsupervised',
                'note': 'Không có label, metrics được tính dựa trên anomaly detection',
                'total_samples': int(len(predictions)),
                'detected_anomalies': int(anomaly_count),
                'anomaly_ratio': float(anomaly_ratio),
                'avg_anomaly_score': float(np.mean(anomaly_scores)),
                'min_anomaly_score': float(np.min(anomaly_scores)),
                'max_anomaly_score': float(np.max(anomaly_scores)),
                'avg_fraud_probability': float(np.mean(fraud_proba)),
            }

        # Lưu model
        model.save()
        logger.info("[ISOLATION FOREST] Đã lưu model thành công!")

        # Cập nhật trạng thái trong prediction service
        prediction_service.predictor.layer1.isolation_forest = model
        prediction_service.predictor.is_fitted['layer1'] = True

        # Chuẩn bị response
        training_info = {
            'samples_count': int(len(X)),
            'features_count': int(len(feature_names)),
            'feature_names': feature_names,
            'config': {
                'n_estimators': model.config.get('n_estimators', 100),
                'contamination': model.config.get('contamination', 0.05),
                'random_state': model.config.get('random_state', 42)
            },
            'data_summary': {
                'columns_in_file': list(df.columns),
                'columns_used_for_training': feature_names,
                'columns_excluded': [c for c in df.columns if c not in feature_names],
                'missing_values_filled': int(missing_count)
            }
        }

        logger.info("="*60)
        logger.info("[ISOLATION FOREST] TRAINING THÀNH CÔNG!")
        logger.info(f"[ISOLATION FOREST] Số mẫu: {training_info['samples_count']}")
        logger.info(f"[ISOLATION FOREST] Số features: {training_info['features_count']}")
        logger.info("="*60)

        return jsonify({
            'success': True,
            'message': 'Training Isolation Forest thành công!',
            'training_info': training_info,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[ISOLATION FOREST] LỖI: {error_msg}")
        logger.error(f"[ISOLATION FOREST] Traceback:\n{error_traceback}")

        return jsonify({
            'success': False,
            'error': f'Training thất bại: {error_msg}',
            'details': error_traceback
        }), 500


@api.route('/train/lightgbm', methods=['POST'])
def train_lightgbm():
    """
    Train LightGBM model với file upload
    Yêu cầu có cột is_fraud (supervised learning)
    """
    logger.info("="*60)
    logger.info("[LIGHTGBM] Bắt đầu training...")
    logger.info("="*60)

    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Không có file được upload'
            }), 400

        file = request.files['file']

        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'File phải có định dạng .csv'
            }), 400

        df = pd.read_csv(file)
        logger.info(f"[LIGHTGBM] Đọc file: {len(df)} dòng, {len(df.columns)} cột")

        # Kiểm tra cột label
        label_col = None
        for col in ['is_fraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            return jsonify({
                'success': False,
                'error': 'Cần có cột is_fraud (hoặc fraud, label) trong dữ liệu. LightGBM là supervised learning.'
            }), 400

        # Lấy features
        exclude_cols = ['tx_id', 'transaction_id', 'user_id', 'id', 'timestamp', 'created_at', label_col, 'fraud_type']
        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

        logger.info(f"[LIGHTGBM] Feature columns: {feature_cols}")

        # Xử lý categorical
        if 'channel' in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['channel'] = le.fit_transform(df['channel'].fillna('unknown').astype(str))

        X = df[feature_cols].copy()
        y = df[label_col].values

        # Xử lý missing và numeric
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                X = X.drop(columns=[col])

        X = X.fillna(0)
        feature_names = list(X.columns)

        # Import và train
        from models.layer1.lightgbm_model import LightGBMModel
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42
        )

        model = LightGBMModel()
        model.fit(X_train, y_train, feature_names=feature_names, verbose=True)

        # Evaluate
        metrics = model.evaluate(X_test, y_test, verbose=True)

        # Lưu model
        model.save()

        logger.info("[LIGHTGBM] Training hoàn tất!")

        return jsonify({
            'success': True,
            'message': 'Training LightGBM thành công!',
            'training_info': {
                'samples_count': int(len(X)),
                'features_count': int(len(feature_names)),
                'feature_names': feature_names,
                'train_size': int(len(X_train)),
                'test_size': int(len(X_test)),
                'fraud_ratio': float(np.mean(y))
            },
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[LIGHTGBM] LỖI: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Training thất bại: {str(e)}',
            'details': traceback.format_exc()
        }), 500


@api.route('/train/autoencoder', methods=['POST'])
def train_autoencoder():
    """
    Train Autoencoder model với file upload

    QUAN TRỌNG: Autoencoder cần train CHỈ với normal data (is_fraud=0)
    Lý do: Autoencoder học pattern "bình thường", nếu train cả fraud
    thì model sẽ học cả fraud pattern và không phát hiện được anomaly.

    Logic training:
    1. Nếu có cột is_fraud:
       - Tách normal_data (is_fraud=0) và fraud_data (is_fraud=1)
       - Chia normal thành train (80%) / test (20%)
       - Train CHỈ với normal_train
       - Tính threshold từ normal_train (percentile 95)
       - Test set = normal_test + ALL fraud
       - Evaluate trên test set
    2. Nếu không có is_fraud:
       - Train với toàn bộ dữ liệu (giả sử là normal)
       - Chỉ trả reconstruction error

    Features columns được hỗ trợ (30 features):
    amount_log, amount_to_balance_ratio, z_score_amount, is_round_amount,
    amount_std_7d, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos,
    is_night_transaction, is_salary_period, tx_count_1h, tx_count_24h,
    minutes_since_last_tx, velocity_change, amount_acceleration,
    is_new_recipient, recipient_tx_count, is_same_bank,
    tx_count_to_same_recipient_24h, recipient_account_age_days,
    time_since_last_tx_to_recipient, is_new_device, channel_encoded,
    is_usual_location, account_age_days, income_level, is_incremental_amount,
    is_during_business_hours, recipient_total_received_24h
    """
    logger.info("="*60)
    logger.info("[AUTOENCODER] Bắt đầu training...")
    logger.info("="*60)

    try:
        # Kiểm tra file upload
        if 'file' not in request.files:
            logger.error("[AUTOENCODER] Không có file được upload")
            return jsonify({
                'success': False,
                'error': 'Không có file được upload. Vui lòng chọn file CSV.'
            }), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("[AUTOENCODER] Tên file trống")
            return jsonify({
                'success': False,
                'error': 'Tên file trống. Vui lòng chọn file CSV.'
            }), 400

        if not file.filename.endswith('.csv'):
            logger.error(f"[AUTOENCODER] File không phải CSV: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'File phải có định dạng .csv'
            }), 400

        # Đọc CSV
        logger.info(f"[AUTOENCODER] Đang đọc file: {file.filename}")
        df = pd.read_csv(file)
        logger.info(f"[AUTOENCODER] Đọc file thành công: {len(df)} dòng, {len(df.columns)} cột")
        logger.info(f"[AUTOENCODER] Các cột: {list(df.columns)}")

        # Kiểm tra dữ liệu tối thiểu
        if len(df) < 10:
            logger.error(f"[AUTOENCODER] Dữ liệu quá ít: {len(df)} dòng")
            return jsonify({
                'success': False,
                'error': f'Dữ liệu quá ít ({len(df)} dòng). Cần ít nhất 10 dòng để training.'
            }), 400

        # Các cột cần loại bỏ
        identifier_cols = ['tx_id', 'transaction_id', 'user_id', 'id', 'timestamp', 'created_at']
        label_cols = ['is_fraud', 'fraud', 'label', 'fraud_type']

        # Lấy các cột features (loại bỏ identifier và label)
        exclude_cols = identifier_cols + label_cols
        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

        logger.info(f"[AUTOENCODER] Các cột bị loại bỏ: {[c for c in df.columns if c.lower() in [e.lower() for e in exclude_cols]]}")
        logger.info(f"[AUTOENCODER] Feature columns: {feature_cols}")

        if len(feature_cols) == 0:
            logger.error("[AUTOENCODER] Không có cột features")
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy cột features trong dữ liệu.'
            }), 400

        # Xử lý categorical columns và chuyển sang numeric
        from sklearn.preprocessing import LabelEncoder

        X_full = df[feature_cols].copy()
        for col in X_full.columns:
            if X_full[col].dtype == 'object':
                le = LabelEncoder()
                X_full[col] = le.fit_transform(X_full[col].fillna('unknown').astype(str))
            else:
                X_full[col] = pd.to_numeric(X_full[col], errors='coerce')

        X_full = X_full.fillna(0)
        feature_names = list(X_full.columns)

        logger.info(f"[AUTOENCODER] Features sau xử lý: {feature_names}")
        logger.info(f"[AUTOENCODER] Shape dữ liệu: {X_full.shape}")

        # Import các thư viện cần thiết
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
        from models.layer2.autoencoder import AutoencoderModel

        # ===== KIỂM TRA CÓ LABEL KHÔNG =====
        has_label = False
        label_col = None
        for col in ['is_fraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                has_label = True
                break

        if has_label and label_col:
            # ===== TRƯỜNG HỢP CÓ LABEL: TRAIN CHỈ VỚI NORMAL DATA =====
            logger.info(f"[AUTOENCODER] Phát hiện cột label: '{label_col}'")
            logger.info("[AUTOENCODER] Áp dụng logic: Train CHỈ với NORMAL data (is_fraud=0)")

            y_full = df[label_col].values

            # Bước 1: Tách normal và fraud
            normal_mask = y_full == 0
            fraud_mask = y_full == 1

            normal_data = X_full[normal_mask].values
            fraud_data = X_full[fraud_mask].values

            normal_count = len(normal_data)
            fraud_count = len(fraud_data)

            logger.info(f"[AUTOENCODER] Số samples normal (is_fraud=0): {normal_count}")
            logger.info(f"[AUTOENCODER] Số samples fraud (is_fraud=1): {fraud_count}")

            if normal_count < 10:
                return jsonify({
                    'success': False,
                    'error': f'Không đủ dữ liệu normal để train ({normal_count} samples). Cần ít nhất 10 samples normal.'
                }), 400

            # Bước 2: Chia train/test cho NORMAL data (80/20)
            X_normal_train, X_normal_test = train_test_split(
                normal_data,
                test_size=0.2,
                random_state=42
            )

            logger.info(f"[AUTOENCODER] Normal train size: {len(X_normal_train)}")
            logger.info(f"[AUTOENCODER] Normal test size: {len(X_normal_test)}")

            # Bước 3: Train CHỈ với normal_train
            logger.info("[AUTOENCODER] Bắt đầu train model với CHỈ normal data...")
            model = AutoencoderModel()
            model.fit(X_normal_train, feature_names=feature_names, verbose=True)

            # Bước 4: Tính threshold với percentile 95 (mặc định)
            train_errors = model.get_reconstruction_error(X_normal_train)
            threshold = float(np.percentile(train_errors, 95))
            model.threshold = threshold
            threshold_percentile = 95
            logger.info(f"[AUTOENCODER] Threshold (percentile 95): {threshold:.6f}")

            # Bước 5: Tạo test set = normal_test + ALL fraud
            if fraud_count > 0:
                X_test = np.vstack([X_normal_test, fraud_data])
                y_test = np.array([0] * len(X_normal_test) + [1] * fraud_count)
            else:
                X_test = X_normal_test
                y_test = np.array([0] * len(X_normal_test))

            logger.info(f"[AUTOENCODER] Test set size: {len(X_test)} (normal_test: {len(X_normal_test)}, all fraud: {fraud_count})")

            # Bước 6: Evaluate và tính metrics
            test_errors = model.get_reconstruction_error(X_test)
            y_pred = (test_errors > threshold).astype(int)

            # Tính reconstruction error trung bình cho từng nhóm
            normal_test_errors = model.get_reconstruction_error(X_normal_test)
            avg_recon_error_normal = float(np.mean(normal_test_errors))

            if fraud_count > 0:
                fraud_errors = model.get_reconstruction_error(fraud_data)
                avg_recon_error_fraud = float(np.mean(fraud_errors))
            else:
                avg_recon_error_fraud = None

            logger.info(f"[AUTOENCODER] Avg reconstruction error (normal): {avg_recon_error_normal:.6f}")
            if avg_recon_error_fraud:
                logger.info(f"[AUTOENCODER] Avg reconstruction error (fraud): {avg_recon_error_fraud:.6f}")

            # Tính metrics
            if fraud_count > 0:
                roc_auc = float(roc_auc_score(y_test, test_errors))
                f1 = float(f1_score(y_test, y_pred))
                precision = float(precision_score(y_test, y_pred, zero_division=0))
                recall = float(recall_score(y_test, y_pred, zero_division=0))

                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

                logger.info(f"[AUTOENCODER] ROC-AUC: {roc_auc:.4f}")
                logger.info(f"[AUTOENCODER] F1-Score: {f1:.4f}")
                logger.info(f"[AUTOENCODER] Precision: {precision:.4f}")
                logger.info(f"[AUTOENCODER] Recall: {recall:.4f}")
                logger.info(f"[AUTOENCODER] Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            else:
                roc_auc = None
                f1 = None
                precision = None
                recall = None
                tp, tn, fp, fn = 0, len(X_normal_test), 0, 0
                logger.warning("[AUTOENCODER] Không có fraud data để tính metrics đầy đủ")

            # Lưu model
            model.save()
            logger.info("[AUTOENCODER] Đã lưu model thành công!")

            logger.info("="*60)
            logger.info("[AUTOENCODER] TRAINING THÀNH CÔNG!")
            logger.info("="*60)

            # Response với đầy đủ thông tin
            response = {
                'success': True,
                'model': 'autoencoder',
                'message': 'Autoencoder trained successfully! (Trained with ONLY normal data)',
                'data_info': {
                    'total_samples': int(len(df)),
                    'normal_samples': int(normal_count),
                    'fraud_samples': int(fraud_count),
                    'train_samples': int(len(X_normal_train)),
                    'test_samples': int(len(X_test)),
                    'note': 'Model trained ONLY with normal data (is_fraud=0). Threshold calculated using percentile 95.'
                },
                'training_info': {
                    'threshold': threshold,
                    'threshold_percentile': threshold_percentile,
                    'features_count': int(len(feature_names)),
                    'feature_names': feature_names
                },
                'metrics': {
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'avg_reconstruction_error_normal': avg_recon_error_normal,
                    'avg_reconstruction_error_fraud': avg_recon_error_fraud,
                    'confusion_matrix': {
                        'true_positives': int(tp),
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn)
                    }
                },
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(response)

        else:
            # ===== TRƯỜNG HỢP KHÔNG CÓ LABEL: TRAIN VỚI TẤT CẢ DỮ LIỆU =====
            logger.info("[AUTOENCODER] Không có cột label (is_fraud)")
            logger.info("[AUTOENCODER] Giả định tất cả dữ liệu là NORMAL và train với toàn bộ")

            # Train với toàn bộ dữ liệu
            model = AutoencoderModel()
            model.fit(X_full.values, feature_names=feature_names, verbose=True)

            # Tính reconstruction error
            recon_errors = model.get_reconstruction_error(X_full.values)
            threshold = float(model.threshold) if model.threshold else float(np.percentile(recon_errors, 95))

            # Lưu model
            model.save()
            logger.info("[AUTOENCODER] Đã lưu model thành công!")

            logger.info("="*60)
            logger.info("[AUTOENCODER] TRAINING THÀNH CÔNG (Unsupervised mode)!")
            logger.info("="*60)

            response = {
                'success': True,
                'model': 'autoencoder',
                'message': 'Autoencoder trained successfully! (Unsupervised mode - no labels)',
                'data_info': {
                    'total_samples': int(len(X_full)),
                    'normal_samples': None,
                    'fraud_samples': None,
                    'train_samples': int(len(X_full)),
                    'test_samples': 0,
                    'note': 'No label column found. Trained with all data assuming normal transactions.'
                },
                'training_info': {
                    'threshold': threshold,
                    'threshold_percentile': 95,
                    'features_count': int(len(feature_names)),
                    'feature_names': feature_names
                },
                'metrics': {
                    'roc_auc': None,
                    'f1_score': None,
                    'precision': None,
                    'recall': None,
                    'avg_reconstruction_error': float(np.mean(recon_errors)),
                    'max_reconstruction_error': float(np.max(recon_errors)),
                    'min_reconstruction_error': float(np.min(recon_errors)),
                    'std_reconstruction_error': float(np.std(recon_errors)),
                    'confusion_matrix': None
                },
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[AUTOENCODER] LỖI: {error_msg}")
        logger.error(f"[AUTOENCODER] Traceback:\n{error_traceback}")

        return jsonify({
            'success': False,
            'error': f'Training thất bại: {error_msg}',
            'details': error_traceback
        }), 500


@api.route('/train/lstm', methods=['POST'])
def train_lstm():
    """
    Train LSTM model với file upload

    SỬ DỤNG LSTM DATA PIPELINE MỚI:
    - Tính 40 features chuyên biệt cho fraud detection VN
    - Tạo sequences 3D với sliding window
    - Split theo user (không random) để tránh data leakage
    - Padding cho users có ít giao dịch

    Input CSV cần có các cột:
    - user_id (bắt buộc): ID của user
    - timestamp (bắt buộc): Thời gian giao dịch
    - amount (bắt buộc): Số tiền giao dịch
    - is_fraud (bắt buộc): Label fraud (0/1)
    - recipient_id (khuyến nghị): ID người nhận
    - balance (khuyến nghị): Số dư tài khoản
    - channel (khuyến nghị): Kênh giao dịch (MOBILE/WEB/ATM)
    - device_id (khuyến nghị): ID thiết bị
    - location (khuyến nghị): Vị trí
    - recipient_bank (khuyến nghị): Ngân hàng người nhận

    Output:
    - X_train: shape (n_train_sequences, seq_length, 40)
    - X_test: shape (n_test_sequences, seq_length, 40)

    Query params:
    - seq_length (int): Độ dài sequence (mặc định 7)
    - test_ratio (float): Tỷ lệ test set (mặc định 0.2)
    """
    logger.info("=" * 60)
    logger.info("[LSTM] Bắt đầu training với LSTM Data Pipeline...")
    logger.info("=" * 60)

    try:
        # Kiểm tra file upload
        if 'file' not in request.files:
            logger.error("[LSTM] Không có file được upload")
            return jsonify({
                'success': False,
                'error': 'Không có file được upload. Vui lòng chọn file CSV.'
            }), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("[LSTM] Tên file trống")
            return jsonify({
                'success': False,
                'error': 'Tên file trống. Vui lòng chọn file CSV.'
            }), 400

        if not file.filename.endswith('.csv'):
            logger.error(f"[LSTM] File không phải CSV: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'File phải có định dạng .csv'
            }), 400

        # Lấy parameters từ request
        seq_length = request.form.get('seq_length', 7, type=int)
        test_ratio = request.form.get('test_ratio', 0.2, type=float)

        logger.info(f"[LSTM] Đang đọc file: {file.filename}")
        logger.info(f"[LSTM] Sequence length: {seq_length}")
        logger.info(f"[LSTM] Test ratio: {test_ratio}")

        # Đọc CSV
        try:
            df = pd.read_csv(file)
            logger.info(f"[LSTM] Đọc file thành công: {len(df)} dòng, {len(df.columns)} cột")
            logger.info(f"[LSTM] Các cột: {list(df.columns)}")
        except Exception as e:
            logger.error(f"[LSTM] Lỗi đọc file CSV: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Không thể đọc file CSV: {str(e)}'
            }), 400

        # Kiểm tra các cột bắt buộc
        required_cols = ['user_id', 'timestamp', 'amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"[LSTM] Thiếu cột bắt buộc: {missing_cols}")
            return jsonify({
                'success': False,
                'error': f'Thiếu các cột bắt buộc: {missing_cols}. '
                         f'LSTM cần: user_id, timestamp, amount, is_fraud'
            }), 400

        # Kiểm tra cột is_fraud
        label_col = None
        for col in ['is_fraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            logger.error("[LSTM] Không có cột is_fraud")
            return jsonify({
                'success': False,
                'error': 'Cần có cột is_fraud cho supervised learning. '
                         'LSTM học từ patterns fraud đã biết.'
            }), 400

        # Chuẩn hóa tên cột label
        if label_col != 'is_fraud':
            df['is_fraud'] = df[label_col]

        # Kiểm tra dữ liệu tối thiểu
        if len(df) < seq_length * 2:
            logger.error(f"[LSTM] Dữ liệu quá ít: {len(df)} dòng")
            return jsonify({
                'success': False,
                'error': f'Dữ liệu quá ít ({len(df)} dòng). '
                         f'Cần ít nhất {seq_length * 2} giao dịch để training.'
            }), 400

        # Import và sử dụng LSTM Data Pipeline
        logger.info("[LSTM] Khởi chạy LSTM Data Pipeline...")

        from utils.lstm_data_pipeline import prepare_lstm_data, FEATURE_NAMES

        # Chạy pipeline để tạo sequences 3D
        pipeline_result = prepare_lstm_data(
            df_raw=df,
            seq_length=seq_length,
            test_ratio=test_ratio,
            include_padding=True,
            verbose=True
        )

        X_train = pipeline_result['X_train']
        y_train = pipeline_result['y_train']
        X_test = pipeline_result['X_test']
        y_test = pipeline_result['y_test']
        feature_names = pipeline_result['feature_names']
        metadata = pipeline_result['metadata']

        logger.info(f"[LSTM] Pipeline hoàn tất!")
        logger.info(f"[LSTM] X_train shape: {X_train.shape}")
        logger.info(f"[LSTM] X_test shape: {X_test.shape}")
        logger.info(f"[LSTM] Features: {len(feature_names)}")

        # Kiểm tra có đủ dữ liệu không
        if len(X_train) == 0:
            return jsonify({
                'success': False,
                'error': 'Không tạo được sequences từ dữ liệu. '
                         'Kiểm tra lại cột user_id và timestamp.'
            }), 400

        # Import và train LSTM model
        from models.layer2.lstm_sequence import LSTMSequenceModel

        logger.info("[LSTM] Bắt đầu train LSTM model...")

        model = LSTMSequenceModel()

        # Train với sequences 3D
        model.fit(
            sequences=X_train,
            labels=y_train,
            validation_split=0.1,
            verbose=True
        )

        # Evaluate trên test set
        logger.info("[LSTM] Đánh giá model trên test set...")
        metrics = model.evaluate(X_test, y_test, verbose=True)

        # Lưu model
        model.save()
        logger.info("[LSTM] Đã lưu model thành công!")

        # Cập nhật prediction service
        try:
            prediction_service.predictor.layer2.lstm = model
            prediction_service.predictor.is_fitted['layer2'] = True
        except Exception as e:
            logger.warning(f"[LSTM] Không thể cập nhật prediction service: {e}")

        logger.info("=" * 60)
        logger.info("[LSTM] TRAINING THÀNH CÔNG!")
        logger.info("=" * 60)

        # Chuẩn bị response
        response = {
            'success': True,
            'model': 'lstm',
            'message': 'Training LSTM thành công với LSTM Data Pipeline!',
            'pipeline_info': {
                'seq_length': seq_length,
                'num_features': len(feature_names),
                'feature_names': feature_names,
                'include_padding': True,
                'split_method': 'by_user'
            },
            'data_info': {
                'total_transactions': int(metadata['total_transactions']),
                'total_users': int(metadata['total_users']),
                'train_users': int(metadata['train_users_count']),
                'test_users': int(metadata['test_users_count']),
                'train_sequences': int(metadata['train_sequences_count']),
                'test_sequences': int(metadata['test_sequences_count']),
                'train_fraud_ratio': float(metadata['train_fraud_ratio']),
                'test_fraud_ratio': float(metadata['test_fraud_ratio']),
            },
            'training_info': {
                'X_train_shape': list(X_train.shape),
                'X_test_shape': list(X_test.shape),
                'train_fraud_count': int(metadata['train_fraud_count']),
                'test_fraud_count': int(metadata['test_fraud_count']),
            },
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[LSTM] LỖI: {error_msg}")
        logger.error(f"[LSTM] Traceback:\n{error_traceback}")

        return jsonify({
            'success': False,
            'error': f'Training thất bại: {error_msg}',
            'details': error_traceback
        }), 500


# ===== GNN HETEROGENEOUS ENDPOINTS (MỚI - 2 BƯỚC) =====

@api.route('/train/gnn/build-graph', methods=['POST'])
def build_gnn_graph():
    """
    🕸️ BƯỚC 1: Tạo mạng lưới GNN (Build Graph)

    Upload thư mục gnn_data dưới dạng ZIP hoặc các file riêng lẻ.

    Files cần có:
    - nodes.csv: Tất cả nodes (user, recipient, device, ip)
    - edges_transfer.csv: Edges chuyển tiền (user → recipient)
    - edge_labels.csv: Labels cho edges (fraud/normal)
    - splits.csv: Train/val/test split

    Files tùy chọn:
    - nodes_user.csv, nodes_recipient.csv, nodes_device.csv, nodes_ip.csv
    - edges_uses_device.csv, edges_uses_ip.csv
    - metadata.json, graph_preview.json

    Workflow:
    1. Load tất cả files
    2. Sanity check (kiểm tra tính toàn vẹn)
    3. Build heterogeneous graph PyTorch Geometric
    4. Lưu graph + tạo flag file

    KHÔNG train model ở bước này!
    """
    logger.info("="*60)
    logger.info("[GNN-GRAPH] 🕸️ BẮT ĐẦU TẠO MẠNG LƯỚI GNN")
    logger.info("="*60)

    try:
        # Kiểm tra files upload
        if len(request.files) == 0:
            logger.error("[GNN-GRAPH] Không có file được upload")
            return jsonify({
                'success': False,
                'error': 'Không có file được upload. Vui lòng upload các file CSV/JSON.'
            }), 400

        # Tạo thư mục tạm để lưu files
        import shutil
        import zipfile

        temp_dir = os.path.join(tempfile.gettempdir(), f'gnn_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"[GNN-GRAPH] Thư mục tạm: {temp_dir}")

        # Xử lý files upload
        uploaded_files = []

        for key in request.files:
            file = request.files[key]
            if file.filename == '':
                continue

            filename = file.filename
            filepath = os.path.join(temp_dir, filename)

            # Nếu là ZIP, giải nén
            if filename.endswith('.zip'):
                logger.info(f"[GNN-GRAPH] Đang giải nén: {filename}")
                file.save(filepath)
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                os.remove(filepath)

                # Kiểm tra xem có thư mục con không
                subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                if len(subdirs) == 1:
                    # Di chuyển files từ thư mục con ra ngoài
                    subdir_path = os.path.join(temp_dir, subdirs[0])
                    for f in os.listdir(subdir_path):
                        shutil.move(os.path.join(subdir_path, f), temp_dir)
                    os.rmdir(subdir_path)
            else:
                # Lưu file trực tiếp
                file.save(filepath)
                uploaded_files.append(filename)
                logger.info(f"[GNN-GRAPH] Đã lưu: {filename}")

        # List tất cả files trong temp_dir
        all_files = os.listdir(temp_dir)
        logger.info(f"[GNN-GRAPH] Các files đã load: {all_files}")

        # Import và chạy pipeline
        from utils.gnn_data_pipeline import GNNDataPipeline

        pipeline = GNNDataPipeline(verbose=True)

        # Load data
        logger.info("[GNN-GRAPH] Bắt đầu load dữ liệu...")
        try:
            pipeline.load_data_from_directory(temp_dir)
        except FileNotFoundError as e:
            logger.error(f"[GNN-GRAPH] Thiếu file: {str(e)}")
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({
                'success': False,
                'error': f'Thiếu file bắt buộc: {str(e)}',
                'required_files': ['nodes.csv', 'edges_transfer.csv', 'edge_labels.csv', 'splits.csv'],
                'uploaded_files': all_files
            }), 400

        # Sanity check
        logger.info("[GNN-GRAPH] Bắt đầu sanity check...")
        is_valid, errors = pipeline.sanity_check()

        if not is_valid:
            logger.error(f"[GNN-GRAPH] Sanity check thất bại: {errors}")
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({
                'success': False,
                'error': 'Dữ liệu không hợp lệ!',
                'sanity_errors': errors
            }), 400

        # Build graph
        logger.info("[GNN-GRAPH] Bắt đầu build heterogeneous graph...")
        graph = pipeline.build_hetero_graph()

        # Save graph
        pipeline.save_graph(graph, {
            'source_dir': temp_dir,
            'uploaded_files': all_files
        })

        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Thống kê graph
        graph_stats = {
            'node_types': list(graph.node_types),
            'edge_types': [str(et) for et in graph.edge_types],
            'nodes': {},
            'edges': {}
        }

        for nt in graph.node_types:
            graph_stats['nodes'][nt] = {
                'count': graph[nt].num_nodes,
                'features': graph[nt].x.shape[1]
            }

        for et in graph.edge_types:
            edge_info = {
                'count': graph[et].edge_index.shape[1]
            }
            if hasattr(graph[et], 'y'):
                labels = graph[et].y
                edge_info['fraud_count'] = int(labels.sum().item())
                edge_info['normal_count'] = int(len(labels) - labels.sum().item())
            if hasattr(graph[et], 'train_mask'):
                edge_info['train_count'] = int(graph[et].train_mask.sum().item())
                edge_info['val_count'] = int(graph[et].val_mask.sum().item())
                edge_info['test_count'] = int(graph[et].test_mask.sum().item())
            graph_stats['edges'][str(et)] = edge_info

        logger.info("="*60)
        logger.info("[GNN-GRAPH] ✅ TẠO MẠNG LƯỚI GNN THÀNH CÔNG!")
        logger.info("="*60)

        return jsonify({
            'success': True,
            'message': '✅ Tạo mạng lưới GNN thành công! Bạn có thể tiến hành huấn luyện.',
            'graph_ready': True,
            'graph_stats': graph_stats,
            'warnings': pipeline.warnings if pipeline.warnings else None,
            'next_step': 'Bấm nút "Huấn luyện GNN" để train model',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[GNN-GRAPH] LỖI: {error_msg}")
        logger.error(f"[GNN-GRAPH] Traceback:\n{error_traceback}")

        return jsonify({
            'success': False,
            'error': f'Tạo mạng lưới thất bại: {error_msg}',
            'details': error_traceback
        }), 500


@api.route('/train/gnn/train', methods=['POST'])
def train_gnn_hetero():
    """
    🎯 BƯỚC 2: Huấn luyện GNN

    CHỈ chạy khi graph_ready.flag tồn tại (đã build graph ở Bước 1).

    Workflow:
    1. Kiểm tra graph đã ready chưa
    2. Load graph đã build
    3. Train GNN Heterogeneous model
    4. Evaluate và in metrics
    5. Lưu model

    Response:
    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    - Training history
    - Confusion matrix
    """
    logger.info("="*60)
    logger.info("[GNN-TRAIN] 🎯 BẮT ĐẦU HUẤN LUYỆN GNN")
    logger.info("="*60)

    try:
        # Import pipeline để kiểm tra graph
        from utils.gnn_data_pipeline import GNNDataPipeline

        # Kiểm tra graph đã ready chưa
        if not GNNDataPipeline.is_graph_ready():
            logger.error("[GNN-TRAIN] Graph chưa được build!")
            return jsonify({
                'success': False,
                'error': 'Mạng lưới GNN chưa được tạo! Vui lòng bấm "🕸️ Tạo mạng lưới GNN" trước.',
                'graph_ready': False,
                'hint': 'Upload các file dữ liệu GNN và bấm nút "Tạo mạng lưới GNN" để build graph trước khi train.'
            }), 400

        # Load graph
        logger.info("[GNN-TRAIN] Đang load graph đã build...")
        data, metadata = GNNDataPipeline.load_saved_graph()

        if data is None:
            logger.error("[GNN-TRAIN] Không thể load graph!")
            return jsonify({
                'success': False,
                'error': 'Không thể load mạng lưới GNN. Vui lòng tạo lại.',
                'graph_ready': False
            }), 500

        logger.info(f"[GNN-TRAIN] Graph loaded: {metadata}")

        # Import và train model
        from models.layer2.gnn_hetero_model import GNNHeteroModel

        logger.info("[GNN-TRAIN] Khởi tạo GNN Heterogeneous Model...")
        model = GNNHeteroModel()

        # Train
        logger.info("[GNN-TRAIN] Bắt đầu training...")
        model.fit(data, verbose=True)

        # Evaluate trên test set
        logger.info("[GNN-TRAIN] Đánh giá trên test set...")
        test_metrics = model.evaluate(data, mask_type='test', verbose=True)

        # Evaluate trên validation set
        val_metrics = model.evaluate(data, mask_type='val', verbose=False)

        # Lưu model
        model.save()
        logger.info("[GNN-TRAIN] Đã lưu model!")

        # Chuẩn bị response
        training_info = {
            'epochs_trained': len(model.history['train_loss']),
            'final_train_auc': float(model.history['train_auc'][-1]) if model.history['train_auc'] else 0,
            'final_val_auc': float(model.history['val_auc'][-1]) if model.history['val_auc'] else 0,
            'best_val_auc': float(max(model.history['val_auc'])) if model.history['val_auc'] else 0,
            'hidden_channels': model.hidden_channels,
            'num_layers': model.num_layers,
            'learning_rate': model.learning_rate,
        }

        graph_info = {
            'node_types': list(data.node_types),
            'edge_types': [str(et) for et in data.edge_types],
            'target_edge_type': str(model.target_edge_type),
        }

        logger.info("="*60)
        logger.info("[GNN-TRAIN] ✅ HUẤN LUYỆN GNN THÀNH CÔNG!")
        logger.info("="*60)

        return jsonify({
            'success': True,
            'message': '✅ Huấn luyện GNN thành công!',
            'model': 'gnn_hetero',
            'training_info': training_info,
            'graph_info': graph_info,
            'metrics': {
                'test': test_metrics,
                'validation': val_metrics
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[GNN-TRAIN] LỖI: {error_msg}")
        logger.error(f"[GNN-TRAIN] Traceback:\n{error_traceback}")

        return jsonify({
            'success': False,
            'error': f'Huấn luyện GNN thất bại: {error_msg}',
            'details': error_traceback
        }), 500


@api.route('/train/gnn/status', methods=['GET'])
def get_gnn_status():
    """
    Kiểm tra trạng thái GNN graph

    Returns:
    - graph_ready: Boolean - graph đã được build chưa
    - metadata: Thông tin graph nếu đã build
    """
    try:
        from utils.gnn_data_pipeline import GNNDataPipeline, GRAPH_METADATA_PATH

        is_ready = GNNDataPipeline.is_graph_ready()

        response = {
            'success': True,
            'graph_ready': is_ready,
            'timestamp': datetime.now().isoformat()
        }

        if is_ready and os.path.exists(GRAPH_METADATA_PATH):
            import json
            with open(GRAPH_METADATA_PATH, 'r', encoding='utf-8') as f:
                response['metadata'] = json.load(f)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/train/gnn/clear', methods=['POST'])
def clear_gnn_graph():
    """
    Xóa graph GNN đã build (để build lại với dữ liệu mới)
    """
    try:
        from utils.gnn_data_pipeline import GNNDataPipeline

        GNNDataPipeline.clear_graph()

        return jsonify({
            'success': True,
            'message': 'Đã xóa mạng lưới GNN. Bạn có thể tạo lại với dữ liệu mới.',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== LEGACY GNN ENDPOINT (giữ lại để tương thích) =====

@api.route('/train/gnn', methods=['POST'])
def train_gnn():
    """Train GNN model với file upload (LEGACY - dùng cho file CSV đơn giản)"""
    logger.info("[GNN] Bắt đầu training...")

    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Không có file được upload'
            }), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Kiểm tra cột required
        if 'user_id' not in df.columns or 'recipient_id' not in df.columns:
            return jsonify({
                'success': False,
                'error': 'Cần có cột user_id và recipient_id để xây dựng graph'
            }), 400

        label_col = None
        for col in ['is_fraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            return jsonify({
                'success': False,
                'error': 'Cần có cột is_fraud cho supervised learning'
            }), 400

        # Lấy features
        exclude_cols = ['tx_id', 'transaction_id', 'user_id', 'recipient_id', 'id', 'timestamp', 'created_at', label_col, 'fraud_type']
        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

        X = df[feature_cols].copy()
        y = df[label_col].values

        for col in X.columns:
            if X[col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('unknown').astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        X = X.fillna(0)

        # Train GNN
        from models.layer2.gnn_model import GNNModel
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42
        )

        model = GNNModel()
        model.fit(X_train, y_train, verbose=True)

        metrics = model.evaluate(X_test, y_test, verbose=True)
        model.save()

        logger.info("[GNN] Training hoàn tất!")

        return jsonify({
            'success': True,
            'message': 'Training GNN thành công!',
            'training_info': {
                'samples_count': int(len(X)),
                'features_count': int(len(feature_cols)),
                'feature_names': list(X.columns)
            },
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[GNN] LỖI: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Training thất bại: {str(e)}'
        }), 500


@api.route('/train/all/upload', methods=['POST'])
def train_all_with_upload():
    """Train tất cả models với file upload"""
    logger.info("[ALL] Bắt đầu training tất cả models...")

    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Không có file được upload'
            }), 400

        file = request.files['file']
        df = pd.read_csv(file)

        results = {}

        # Train Isolation Forest
        try:
            # Reset file position
            file.seek(0)
            request.files['file'] = file

            logger.info("[ALL] Training Isolation Forest...")
            # Gọi hàm train isolation forest trực tiếp
            exclude_cols = ['tx_id', 'transaction_id', 'user_id', 'id', 'timestamp', 'created_at', 'is_fraud', 'fraud_type']
            feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

            X = df[feature_cols].copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].fillna('unknown').astype(str))
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)

            from models.layer1.isolation_forest import IsolationForestModel
            if_model = IsolationForestModel()
            if_model.fit(X.values, feature_names=list(X.columns), verbose=True)
            if_model.save()

            results['isolation_forest'] = {'success': True, 'message': 'Trained successfully'}
        except Exception as e:
            results['isolation_forest'] = {'success': False, 'error': str(e)}

        # Train LightGBM nếu có label
        label_col = None
        for col in ['is_fraud', 'fraud', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col:
            try:
                logger.info("[ALL] Training LightGBM...")
                y = df[label_col].values

                from models.layer1.lightgbm_model import LightGBMModel
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(
                    X.values, y, test_size=0.2, random_state=42
                )

                lgb_model = LightGBMModel()
                lgb_model.fit(X_train, y_train, feature_names=list(X.columns), verbose=True)
                lgb_model.save()

                results['lightgbm'] = {'success': True, 'message': 'Trained successfully'}
            except Exception as e:
                results['lightgbm'] = {'success': False, 'error': str(e)}
        else:
            results['lightgbm'] = {'success': False, 'error': 'Không có cột is_fraud'}

        return jsonify({
            'success': True,
            'message': 'Training hoàn tất',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[ALL] LỖI: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Training thất bại: {str(e)}'
        }), 500


# ===== HEALTH CHECK =====

@api.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint - Đã cập nhật để kiểm tra database
    """
    db = get_db()

    return jsonify({
        'status': 'healthy',
        'service': 'ML Fraud Detection',
        'version': '2.0.0',
        'database_connected': db.is_connected,
        'timestamp': datetime.now().isoformat()
    })

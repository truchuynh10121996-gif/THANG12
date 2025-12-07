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
from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime

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

"""
API Routes - Định nghĩa các API endpoints
==========================================
Các endpoints chính:
- /api/predict: Dự đoán fraud cho giao dịch
- /api/train: Training models
- /api/metrics: Lấy metrics đánh giá
- /api/dashboard: Dữ liệu cho dashboard
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

# Tạo Blueprint
api = Blueprint('api', __name__)

# Khởi tạo services
prediction_service = PredictionService()
explain_service = ExplainabilityService()


# ===== PREDICTION ENDPOINTS =====

@api.route('/predict', methods=['POST'])
def predict_single():
    """
    Dự đoán fraud cho 1 giao dịch

    Request body:
    {
        "transaction_id": "TXN001",
        "user_id": "USR001",
        "amount": 5000000,
        "transaction_type": "transfer",
        "channel": "mobile_app",
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

        # Dự đoán
        result = prediction_service.predict_single(data)

        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })

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

        # Dự đoán batch
        results = prediction_service.predict_batch(transactions)

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
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'service': 'ML Fraud Detection',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

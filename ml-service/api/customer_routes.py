"""
Customer API Routes - API endpoints cho dữ liệu khách hàng demo
===============================================================
Các endpoints:
- GET /api/demo/customers - Danh sách 3 khách hàng demo
- GET /api/demo/customers/:id - Chi tiết khách hàng
- GET /api/demo/customers/:id/transactions - Lịch sử giao dịch
- POST /api/demo/analyze - Phân tích giao dịch với 5 models
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

from utils.customer_data import get_customer_service
from utils.fraud_pipeline import get_fraud_pipeline

logger = logging.getLogger('customer_api')

# Tạo Blueprint
customer_api = Blueprint('customer_api', __name__)


@customer_api.route('/demo/customers', methods=['GET'])
def get_demo_customers():
    """
    Lấy danh sách 3 khách hàng demo

    Response:
    {
        "success": true,
        "customers": [
            {
                "user_id": "USR_000001",
                "ho_ten": "Trần Đức Nam",
                "tuoi": 45,
                "nghe_nghiep": "Giáo viên",
                ...
            }
        ]
    }
    """
    try:
        service = get_customer_service()
        customers = service.get_all_customers()

        return jsonify({
            'success': True,
            'customers': customers,
            'total': len(customers),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting customers: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@customer_api.route('/demo/customers/<user_id>', methods=['GET'])
def get_demo_customer_detail(user_id):
    """
    Lấy chi tiết khách hàng và các chỉ số hành vi

    Path params:
        user_id: ID của khách hàng (VD: USR_000001)

    Response:
    {
        "success": true,
        "user_id": "USR_000001",
        "profile": {
            "ho_ten": "Trần Đức Nam",
            "tuoi": 45,
            "nghe_nghiep": "Giáo viên",
            "thu_nhap_hang_thang": 9400000,
            "so_ngay_mo_tai_khoan": 687,
            ...
        },
        "behavioral_features": {
            "velocity_1h": 0,
            "velocity_24h": 2,
            "time_since_last_transaction": 5.3,
            "amount_deviation_ratio": 1.2,
            ...
        }
    }
    """
    try:
        service = get_customer_service()

        # Lấy profile
        profile = service.get_customer_profile(user_id)

        if not profile:
            return jsonify({
                'success': False,
                'error': f'Không tìm thấy khách hàng {user_id}'
            }), 404

        # Tính behavioral features
        behavioral = service.calculate_behavioral_features(user_id)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'profile': profile,
            'behavioral_features': behavioral,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting customer {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@customer_api.route('/demo/customers/<user_id>/transactions', methods=['GET'])
def get_demo_customer_transactions(user_id):
    """
    Lấy lịch sử giao dịch của khách hàng

    Path params:
        user_id: ID của khách hàng

    Query params:
        limit: Số giao dịch tối đa (mặc định 15)

    Response:
    {
        "success": true,
        "user_id": "USR_000001",
        "transactions": [
            {
                "transaction_id": "TX_USR_000001_000025",
                "timestamp": "2024-12-01 11:47:22",
                "loai_giao_dich": "Thanh toan QR",
                "so_tien": 96000,
                ...
            }
        ]
    }
    """
    try:
        limit = request.args.get('limit', 15, type=int)
        service = get_customer_service()

        transactions = service.get_customer_transactions(user_id, limit)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'transactions': transactions,
            'total': len(transactions),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting transactions for {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@customer_api.route('/demo/analyze', methods=['POST'])
def analyze_demo_transaction():
    """
    Phân tích giao dịch mới với 5 models ML

    Request body:
    {
        "user_id": "USR_000001",
        "amount": 5000000,
        "transaction_type": "transfer",
        "channel": "mobile_app",
        "hour": 14,
        "recipient_id": "RCP_001",
        "is_international": false
    }

    Response:
    {
        "success": true,
        "result": {
            "transaction_id": "TX_...",
            "prediction": {
                "fraud_probability": 0.75,
                "prediction": "fraud",
                "risk_level": "high",
                "should_block": true,
                "confidence": 0.65
            },
            "model_scores": {
                "isolation_forest": 0.72,
                "lightgbm": 0.78,
                ...
            },
            "explanation": {
                "summary": "...",
                "risk_factors": [...],
                "recommendations": [...]
            }
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

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'Cần user_id trong request'
            }), 400

        # Lấy customer service và fraud pipeline
        customer_service = get_customer_service()
        fraud_pipeline = get_fraud_pipeline()

        # Chuẩn bị dữ liệu
        prepared_data = customer_service.prepare_transaction_for_analysis(user_id, data)

        # Phân tích với 5 models
        result = fraud_pipeline.analyze_transaction(prepared_data)

        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}")
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'details': traceback.format_exc()
        }), 500


@customer_api.route('/demo/models/status', methods=['GET'])
def get_demo_models_status():
    """
    Kiểm tra trạng thái các models

    Response:
    {
        "success": true,
        "models_loaded": {
            "isolation_forest": true,
            "lightgbm": true,
            "autoencoder": false,
            "lstm": false,
            "gnn": false
        }
    }
    """
    try:
        pipeline = get_fraud_pipeline()

        return jsonify({
            'success': True,
            'models_loaded': pipeline.models_loaded,
            'total_loaded': sum(pipeline.models_loaded.values()),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

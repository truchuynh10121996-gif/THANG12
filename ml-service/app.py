"""
ML Fraud Detection Service - Flask Application
===============================================
API service cho hệ thống phát hiện giao dịch lừa đảo sử dụng Machine Learning

Author: Agribank Digital Guard Team
Version: 1.0.0
"""

import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger

# Import config
from config import get_config

config = get_config()

# Import API routes
from api.routes import api
from api.customer_routes import customer_api


def create_app():
    """
    Factory function tạo Flask app

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG

    # CORS configuration - cho phép frontend kết nối
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",      # Web app
                "http://localhost:3001",      # ML demo
                "http://localhost:3002",      # Web admin
                "http://localhost:5000",      # Backend
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://127.0.0.1:3002",
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(customer_api, url_prefix='/api')

    # Root endpoint
    @app.route('/')
    def index():
        """Root endpoint với thông tin service"""
        return jsonify({
            'name': 'ML Fraud Detection Service',
            'version': '1.0.0',
            'description': 'Machine Learning service for fraud detection',
            'endpoints': {
                'health': '/api/health',
                'predict': '/api/predict',
                'predict_batch': '/api/predict/batch',
                'models_status': '/api/models/status',
                'train_layer1': '/api/train/layer1',
                'train_layer2': '/api/train/layer2',
                'train_all': '/api/train/all',
                'metrics': '/api/metrics',
                'explain': '/api/explain',
                'dashboard_stats': '/api/dashboard/stats',
                'graph_community': '/api/graph/community',
                'demo_customers': '/api/demo/customers',
                'demo_customer_detail': '/api/demo/customers/:id',
                'demo_transactions': '/api/demo/customers/:id/transactions',
                'demo_analyze': '/api/demo/analyze'
            },
            'models': {
                'layer1': ['Isolation Forest', 'LightGBM'],
                'layer2': ['Autoencoder', 'LSTM', 'GNN']
            }
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint không tồn tại',
            'status_code': 404
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Lỗi server nội bộ',
            'status_code': 500
        }), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 'Request không hợp lệ',
            'status_code': 400
        }), 400

    return app


def setup_logging():
    """Thiết lập logging"""
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stdout,
        format=config.LOG_FORMAT,
        level=config.LOG_LEVEL
    )

    # Add file handler
    log_file = os.path.join(config.BASE_DIR, 'logs', 'app.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


def print_banner():
    """In banner khi khởi động"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     🤖 ML FRAUD DETECTION SERVICE                                  ║
║     Hệ thống phát hiện giao dịch lừa đảo bằng Machine Learning    ║
║                                                                    ║
║     Layer 1: Isolation Forest + LightGBM                          ║
║     Layer 2: Autoencoder + LSTM + GNN                             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point"""
    # Setup logging
    setup_logging()

    # Print banner
    print_banner()

    # Create app
    app = create_app()

    # Run server
    logger.info(f"Starting ML Fraud Detection Service on http://{config.HOST}:{config.PORT}")

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )


if __name__ == '__main__':
    main()

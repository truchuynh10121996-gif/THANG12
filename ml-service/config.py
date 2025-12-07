"""
Config - Cấu hình cho ML Fraud Detection Service
================================================
File này chứa tất cả các cấu hình cho hệ thống ML
"""

import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


class Config:
    """Cấu hình cơ bản cho ứng dụng"""

    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'ml-fraud-detection-secret-key-2024')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

    # Server Configuration
    HOST = os.getenv('ML_SERVICE_HOST', '0.0.0.0')
    PORT = int(os.getenv('ML_SERVICE_PORT', 5001))

    # Backend Node.js Configuration
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')

    # Paths Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    DATA_SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

    # Model Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1

    # Data Generation Configuration
    NUM_USERS = 50000
    NUM_TRANSACTIONS = 500000
    FRAUD_RATIO = 0.05  # 5% giao dịch lừa đảo

    # Layer 1 Model Configuration
    ISOLATION_FOREST_CONFIG = {
        'n_estimators': 100,
        'contamination': 0.05,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    LIGHTGBM_CONFIG = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 200,
        'early_stopping_rounds': 50
    }

    # Layer 2 Model Configuration
    AUTOENCODER_CONFIG = {
        'input_dim': 64,  # Số features đầu vào
        'encoding_dims': [32, 16, 8],  # Kích thước các layer encoder
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 256,
        'threshold_percentile': 95  # Ngưỡng phát hiện bất thường
    }

    LSTM_CONFIG = {
        'sequence_length': 10,  # Số giao dịch trong 1 chuỗi
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 128
    }

    GNN_CONFIG = {
        'hidden_channels': 64,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.01,
        'epochs': 100,
        'batch_size': 64
    }

    # Ensemble Configuration
    ENSEMBLE_WEIGHTS = {
        'isolation_forest': 0.15,
        'lightgbm': 0.25,
        'autoencoder': 0.20,
        'lstm': 0.20,
        'gnn': 0.20
    }

    # Risk Thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,       # Dưới 30%: Rủi ro thấp
        'medium': 0.6,    # 30-60%: Rủi ro trung bình
        'high': 0.8,      # 60-80%: Rủi ro cao
        'critical': 1.0   # Trên 80%: Rủi ro nghiêm trọng
    }

    # Feature Configuration
    NUMERICAL_FEATURES = [
        'amount', 'balance_before', 'balance_after',
        'transaction_count_1h', 'transaction_count_24h',
        'avg_amount_30d', 'max_amount_30d',
        'time_since_last_transaction'
    ]

    CATEGORICAL_FEATURES = [
        'transaction_type', 'channel', 'device_type',
        'merchant_category', 'location_country'
    ]

    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'is_weekend',
        'is_holiday', 'is_night_time'
    ]

    # API Configuration
    MAX_BATCH_SIZE = 1000
    PREDICTION_TIMEOUT = 30  # seconds

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'


class DevelopmentConfig(Config):
    """Cấu hình cho môi trường development"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Cấu hình cho môi trường production"""
    DEBUG = False
    TESTING = False

    # Tăng cường bảo mật
    SECRET_KEY = os.getenv('SECRET_KEY')

    # Production model paths
    ISOLATION_FOREST_CONFIG = {
        **Config.ISOLATION_FOREST_CONFIG,
        'n_estimators': 200
    }


class TestingConfig(Config):
    """Cấu hình cho môi trường testing"""
    DEBUG = True
    TESTING = True

    # Giảm số lượng để test nhanh hơn
    NUM_USERS = 1000
    NUM_TRANSACTIONS = 10000


# Mapping môi trường
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Lấy config dựa trên biến môi trường"""
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)

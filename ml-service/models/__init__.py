"""
Models Module - Module chứa các ML models
=========================================
Layer 1: Global Fraud Detection
- Isolation Forest: Phát hiện anomaly
- LightGBM: Phân loại fraud

Layer 2: User Profile (Advanced)
- Autoencoder: User embedding và anomaly detection
- LSTM: Sequence modeling
- GNN: Graph-based fraud detection

Ensemble: Kết hợp tất cả models
"""

from .layer1.isolation_forest import IsolationForestModel
from .layer1.lightgbm_model import LightGBMModel
from .layer1.layer1_ensemble import Layer1Ensemble

from .layer2.autoencoder import AutoencoderModel
from .layer2.lstm_sequence import LSTMSequenceModel
from .layer2.gnn_model import GNNModel
from .layer2.layer2_ensemble import Layer2Ensemble

from .ensemble.final_predictor import FinalPredictor

__all__ = [
    # Layer 1
    'IsolationForestModel',
    'LightGBMModel',
    'Layer1Ensemble',

    # Layer 2
    'AutoencoderModel',
    'LSTMSequenceModel',
    'GNNModel',
    'Layer2Ensemble',

    # Ensemble
    'FinalPredictor'
]

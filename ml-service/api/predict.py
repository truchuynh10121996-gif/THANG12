"""
Prediction Service - Logic dự đoán fraud
=========================================
Service chứa logic để dự đoán fraud cho giao dịch
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json

# Import config và models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
from models.ensemble.final_predictor import FinalPredictor
from preprocessing.feature_engineering import FeatureEngineer

config = get_config()


class PredictionService:
    """
    Service xử lý predictions và training
    """

    def __init__(self):
        """
        Khởi tạo service

        Tự động load models nếu có sẵn
        """
        self.predictor = FinalPredictor()
        self.feature_engineer = FeatureEngineer(verbose=False)

        # Cache cho data
        self._features_df = None
        self._users_df = None

        # Metrics history
        self.metrics_history = []

        # Dashboard stats
        self.stats = {
            'total_predictions': 0,
            'fraud_detected': 0,
            'high_risk_alerts': 0,
            'last_updated': None
        }

        # Thử load models đã lưu
        self._try_load_models()

    def _try_load_models(self):
        """Thử load models đã lưu"""
        try:
            config_path = os.path.join(config.SAVED_MODELS_DIR, 'final_predictor_config.json')
            if os.path.exists(config_path):
                self.predictor.load()
                print("[PredictionService] Đã load models từ disk")
        except Exception as e:
            print(f"[PredictionService] Không thể load models: {e}")

    def predict_single(self, transaction: Dict) -> Dict:
        """
        Dự đoán fraud cho một giao dịch

        Args:
            transaction: Dict chứa thông tin giao dịch

        Returns:
            Dict kết quả dự đoán
        """
        # Extract features
        features = self._extract_features(transaction)

        # Dự đoán
        if not self.predictor.is_fitted['layer1']:
            # Model chưa được train, trả về kết quả mặc định
            return {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'fraud_probability': 0.0,
                'prediction': 'normal',
                'risk_level': 'low',
                'should_block': False,
                'should_review': False,
                'confidence': 0.0,
                'warning': 'Model chưa được train'
            }

        proba = self.predictor.predict_proba(features)[0]
        prediction = 'fraud' if proba >= 0.5 else 'normal'
        risk_level = self._get_risk_level(proba)

        result = {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'fraud_probability': float(proba),
            'prediction': prediction,
            'risk_level': risk_level,
            'should_block': risk_level in ['high', 'critical'],
            'should_review': risk_level == 'medium',
            'confidence': float(abs(proba - 0.5) * 2)
        }

        # Update stats
        self.stats['total_predictions'] += 1
        if prediction == 'fraud':
            self.stats['fraud_detected'] += 1
        if risk_level in ['high', 'critical']:
            self.stats['high_risk_alerts'] += 1
        self.stats['last_updated'] = datetime.now().isoformat()

        return result

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Dự đoán fraud cho nhiều giao dịch

        Args:
            transactions: List các giao dịch

        Returns:
            List kết quả dự đoán
        """
        results = []
        for txn in transactions:
            result = self.predict_single(txn)
            results.append(result)
        return results

    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract features từ transaction

        Args:
            transaction: Dict giao dịch

        Returns:
            np.ndarray feature vector
        """
        # Tạo DataFrame từ transaction
        df = pd.DataFrame([transaction])

        # Thêm các features cơ bản
        features = []

        # Amount
        amount = float(transaction.get('amount', 0))
        features.append(np.log1p(amount))

        # Time features
        timestamp = transaction.get('timestamp')
        if timestamp:
            try:
                dt = pd.to_datetime(timestamp)
                features.extend([
                    dt.hour,
                    dt.dayofweek,
                    1 if dt.dayofweek >= 5 else 0,  # is_weekend
                    1 if dt.hour < 6 or dt.hour >= 22 else 0,  # is_night
                ])
            except:
                features.extend([12, 0, 0, 0])
        else:
            features.extend([12, 0, 0, 0])

        # Transaction type (one-hot simplified)
        txn_type = transaction.get('transaction_type', '').lower()
        features.extend([
            1 if txn_type == 'transfer' else 0,
            1 if txn_type == 'payment' else 0,
            1 if txn_type == 'withdrawal' else 0,
        ])

        # Channel
        channel = transaction.get('channel', '').lower()
        features.extend([
            1 if channel == 'mobile_app' else 0,
            1 if channel == 'web_banking' else 0,
        ])

        # Is international
        features.append(1 if transaction.get('is_international', False) else 0)

        # Pad to expected size (64 features)
        while len(features) < 64:
            features.append(0)

        return np.array([features[:64]])

    def _get_risk_level(self, proba: float) -> str:
        """Xác định mức độ rủi ro"""
        if proba < config.RISK_THRESHOLDS['low']:
            return 'low'
        elif proba < config.RISK_THRESHOLDS['medium']:
            return 'medium'
        elif proba < config.RISK_THRESHOLDS['high']:
            return 'high'
        else:
            return 'critical'

    def train_layer1(self, data_path: str = None) -> Dict:
        """
        Train Layer 1

        Args:
            data_path: Đường dẫn file features

        Returns:
            Dict kết quả training
        """
        if data_path is None:
            data_path = os.path.join(config.DATA_PROCESSED_DIR, 'features.csv')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

        # Load data
        df = pd.read_csv(data_path)

        # Lấy features và labels
        feature_cols = [col for col in df.columns if col not in [
            'user_id', 'transaction_id', 'timestamp', 'is_fraud', 'fraud_type'
        ]]

        X = df[feature_cols].values
        y = df['is_fraud'].values

        # Train
        self.predictor.fit_layer1(X, y, feature_names=feature_cols)

        # Evaluate
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        metrics = self.predictor.layer1.evaluate(X_test, y_test, verbose=False)

        # Lưu model
        self.predictor.save()

        return {
            'status': 'success',
            'metrics': metrics['ensemble']
        }

    def train_layer2(self, params: Dict = None) -> Dict:
        """
        Train Layer 2

        Returns:
            Dict kết quả training
        """
        params = params or {}

        # Load data
        features_path = os.path.join(config.DATA_PROCESSED_DIR, 'features.csv')
        if not os.path.exists(features_path):
            raise FileNotFoundError("Cần chạy preprocessing trước")

        df = pd.read_csv(features_path)

        # Lấy features và labels
        feature_cols = [col for col in df.columns if col not in [
            'user_id', 'transaction_id', 'timestamp', 'is_fraud', 'fraud_type'
        ]]

        X = df[feature_cols].values
        y = df['is_fraud'].values

        # Train Autoencoder
        self.predictor.layer2.fit_autoencoder(X, feature_names=feature_cols)

        # Load sequences nếu có
        sequences_path = os.path.join(config.DATA_PROCESSED_DIR, 'sequences.npy')
        if os.path.exists(sequences_path):
            sequences = np.load(sequences_path)
            seq_labels = np.load(os.path.join(config.DATA_PROCESSED_DIR, 'sequence_labels.npy'))
            self.predictor.layer2.fit_lstm(sequences, seq_labels)

        # Lưu model
        self.predictor.is_fitted['layer2'] = True
        self.predictor.save()

        return {
            'status': 'success',
            'models_trained': ['autoencoder', 'lstm']
        }

    def train_all(self, params: Dict = None) -> Dict:
        """
        Train tất cả models

        Returns:
            Dict kết quả training
        """
        layer1_result = self.train_layer1()
        layer2_result = self.train_layer2(params)

        return {
            'layer1': layer1_result,
            'layer2': layer2_result,
            'status': 'success'
        }

    def get_model_status(self) -> Dict:
        """Lấy trạng thái models"""
        return self.predictor.get_model_status()

    def get_metrics(self) -> Dict:
        """Lấy metrics hiện tại"""
        if not self.predictor.is_fitted['layer1']:
            return {'warning': 'Model chưa được train'}

        # Load test data
        features_path = os.path.join(config.DATA_PROCESSED_DIR, 'features.csv')
        if not os.path.exists(features_path):
            return {'warning': 'Không có dữ liệu để đánh giá'}

        df = pd.read_csv(features_path)
        feature_cols = [col for col in df.columns if col not in [
            'user_id', 'transaction_id', 'timestamp', 'is_fraud', 'fraud_type'
        ]]

        X = df[feature_cols].values
        y = df['is_fraud'].values

        # Đánh giá
        metrics = self.predictor.layer1.evaluate(X, y, verbose=False)

        return metrics

    def get_metrics_history(self) -> List[Dict]:
        """Lấy lịch sử metrics"""
        return self.metrics_history

    def get_dashboard_stats(self) -> Dict:
        """Lấy thống kê cho dashboard"""
        stats = self.stats.copy()

        if stats['total_predictions'] > 0:
            stats['fraud_rate'] = stats['fraud_detected'] / stats['total_predictions']
        else:
            stats['fraud_rate'] = 0

        # Load thêm thống kê từ dữ liệu
        features_path = os.path.join(config.DATA_PROCESSED_DIR, 'features.csv')
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            stats['total_transactions'] = len(df)
            stats['total_fraud'] = int(df['is_fraud'].sum())
            stats['data_fraud_rate'] = float(df['is_fraud'].mean())

        return stats

    def get_graph_data(self) -> Dict:
        """Lấy dữ liệu graph cho visualization"""
        edges_path = os.path.join(config.DATA_PROCESSED_DIR, 'graph_edges.csv')

        if not os.path.exists(edges_path):
            return {'nodes': [], 'edges': [], 'communities': []}

        edges_df = pd.read_csv(edges_path)

        # Lấy nodes
        nodes = list(set(edges_df['source'].tolist() + edges_df['target'].tolist()))
        nodes_data = [{'id': n, 'label': f'Node {n}'} for n in nodes[:100]]  # Giới hạn 100 nodes

        # Lấy edges
        edges_data = edges_df.head(500).to_dict('records')  # Giới hạn 500 edges

        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'total_nodes': len(nodes),
            'total_edges': len(edges_df)
        }

    def get_user_profile(self, user_id: str) -> Dict:
        """Lấy user profile"""
        return {
            'user_id': user_id,
            'risk_score': 0.3,
            'transaction_count': 0,
            'avg_amount': 0,
            'embedding': []
        }

    def get_user_sequence(self, user_id: str) -> Dict:
        """Lấy sequence giao dịch của user"""
        return {
            'user_id': user_id,
            'sequence': [],
            'sequence_length': 0
        }

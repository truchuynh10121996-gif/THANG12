"""
Final Predictor - Kết hợp Layer 1 và Layer 2
============================================
Kết hợp kết quả từ tất cả các models để đưa ra dự đoán cuối cùng.

Layer 1 (Global): Isolation Forest + LightGBM
Layer 2 (User Profile): Autoencoder + LSTM + GNN

Chiến lược kết hợp:
- Weighted average với weights có thể điều chỉnh
- Layer 2 được ưu tiên khi có đủ dữ liệu user
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import joblib
import json

# Import các ensembles
from ..layer1.layer1_ensemble import Layer1Ensemble
from ..layer2.layer2_ensemble import Layer2Ensemble

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()


class FinalPredictor:
    """
    Final predictor kết hợp tất cả models

    Flow:
    1. Layer 1 chạy với tất cả giao dịch (global detection)
    2. Layer 2 chạy với user-specific data (nếu có)
    3. Kết hợp kết quả theo weights
    """

    def __init__(
        self,
        layer1_weight: float = 0.4,
        layer2_weight: float = 0.6
    ):
        """
        Khởi tạo Final Predictor

        Args:
            layer1_weight: Trọng số cho Layer 1
            layer2_weight: Trọng số cho Layer 2
        """
        # Normalize weights
        total = layer1_weight + layer2_weight
        self.layer1_weight = layer1_weight / total
        self.layer2_weight = layer2_weight / total

        # Initialize ensembles
        self.layer1 = Layer1Ensemble()
        self.layer2 = Layer2Ensemble()

        self.is_fitted = {
            'layer1': False,
            'layer2': False
        }

        # Risk thresholds
        self.risk_thresholds = config.RISK_THRESHOLDS

        # Stats
        self.prediction_stats = {
            'total_predictions': 0,
            'fraud_predictions': 0,
            'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        }

    def fit_layer1(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        verbose: bool = True
    ):
        """
        Train Layer 1 (Isolation Forest + LightGBM)

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Tên features
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING LAYER 1 - GLOBAL FRAUD DETECTION")
            print("=" * 60)

        self.layer1.fit(X, y, feature_names=feature_names, verbose=verbose)
        self.is_fitted['layer1'] = True

    def fit_layer2(
        self,
        X: np.ndarray = None,
        sequences: np.ndarray = None,
        sequence_labels: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None,
        node_labels: np.ndarray = None,
        feature_names: List[str] = None,
        verbose: bool = True
    ):
        """
        Train Layer 2 (Autoencoder + LSTM + GNN)

        Args:
            X: Features cho Autoencoder
            sequences: Sequences cho LSTM
            sequence_labels: Labels cho sequences
            node_features: Node features cho GNN
            edge_index: Graph edges
            node_labels: Node labels cho GNN
            feature_names: Tên features
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING LAYER 2 - USER PROFILE MODELS")
            print("=" * 60)

        if X is not None:
            self.layer2.fit_autoencoder(X, feature_names=feature_names, verbose=verbose)

        if sequences is not None and sequence_labels is not None:
            self.layer2.fit_lstm(sequences, sequence_labels, verbose=verbose)

        if node_features is not None and edge_index is not None and node_labels is not None:
            self.layer2.fit_gnn(node_features, edge_index, node_labels, verbose=verbose)

        self.is_fitted['layer2'] = True

    def predict(
        self,
        X: np.ndarray,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Dự đoán fraud

        Args:
            X: Features cho Layer 1 và Autoencoder
            sequences: Sequences cho LSTM
            node_features: Node features cho GNN
            edge_index: Graph edges
            threshold: Ngưỡng quyết định

        Returns:
            np.ndarray predictions
        """
        proba = self.predict_proba(X, sequences, node_features, edge_index)
        return (proba >= threshold).astype(int)

    def predict_proba(
        self,
        X: np.ndarray,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None
    ) -> np.ndarray:
        """
        Tính xác suất fraud

        Args:
            X: Features
            sequences: Sequences cho LSTM (optional)
            node_features: Node features cho GNN (optional)
            edge_index: Graph edges (optional)

        Returns:
            np.ndarray probabilities
        """
        if not self.is_fitted['layer1']:
            raise ValueError("Layer 1 chưa được train!")

        # Layer 1 predictions
        layer1_proba = self.layer1.predict_proba(X)

        # Layer 2 predictions (nếu có)
        if self.is_fitted['layer2']:
            try:
                layer2_proba = self.layer2.predict_proba(
                    X=X,
                    sequences=sequences,
                    node_features=node_features,
                    edge_index=edge_index
                )

                # Align lengths
                min_len = min(len(layer1_proba), len(layer2_proba))
                layer1_proba = layer1_proba[:min_len]
                layer2_proba = layer2_proba[:min_len]

                # Weighted average
                combined_proba = (
                    self.layer1_weight * layer1_proba +
                    self.layer2_weight * layer2_proba
                )
            except Exception as e:
                # Fallback to Layer 1 only
                combined_proba = layer1_proba
        else:
            combined_proba = layer1_proba

        return combined_proba

    def predict_single(
        self,
        transaction: Dict,
        user_history: pd.DataFrame = None
    ) -> Dict:
        """
        Dự đoán cho một giao dịch đơn lẻ

        Args:
            transaction: Dict chứa thông tin giao dịch
            user_history: DataFrame lịch sử giao dịch của user (optional)

        Returns:
            Dict chứa kết quả dự đoán chi tiết
        """
        # Convert transaction to feature vector
        # (Đây là placeholder - cần implement feature extraction)
        X = self._extract_features(transaction)

        # Predict
        proba = self.predict_proba(X)[0]
        prediction = int(proba >= 0.5)
        risk_level = self._get_risk_level(proba)

        result = {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'fraud_probability': float(proba),
            'prediction': 'fraud' if prediction == 1 else 'normal',
            'risk_level': risk_level,
            'should_block': risk_level in ['high', 'critical'],
            'should_review': risk_level == 'medium',
            'confidence': float(abs(proba - 0.5) * 2)  # 0-1 scale
        }

        # Update stats
        self.prediction_stats['total_predictions'] += 1
        if prediction == 1:
            self.prediction_stats['fraud_predictions'] += 1
        self.prediction_stats['risk_distribution'][risk_level] += 1

        return result

    def predict_batch(
        self,
        transactions: List[Dict]
    ) -> List[Dict]:
        """
        Dự đoán cho nhiều giao dịch

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

    def predict_detail(
        self,
        X: np.ndarray,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None
    ) -> Dict:
        """
        Dự đoán chi tiết với kết quả từ mỗi layer

        Returns:
            Dict với kết quả chi tiết
        """
        result = {
            'layer1': {},
            'layer2': {},
            'combined': {}
        }

        # Layer 1 detail
        if self.is_fitted['layer1']:
            result['layer1'] = self.layer1.predict_detail(X)

        # Layer 2 detail
        if self.is_fitted['layer2']:
            try:
                result['layer2'] = self.layer2.predict_detail(
                    X=X,
                    sequences=sequences,
                    node_features=node_features,
                    edge_index=edge_index
                )
            except:
                result['layer2'] = {'error': 'Layer 2 prediction failed'}

        # Combined
        combined_proba = self.predict_proba(X, sequences, node_features, edge_index)
        result['combined'] = {
            'fraud_probability': combined_proba.tolist(),
            'predictions': self.predict(X, sequences, node_features, edge_index).tolist(),
            'layer1_weight': self.layer1_weight,
            'layer2_weight': self.layer2_weight
        }

        return result

    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract features từ transaction dict

        Args:
            transaction: Dict giao dịch

        Returns:
            np.ndarray feature vector
        """
        # Đây là placeholder - cần implement đầy đủ
        features = []

        # Amount features
        amount = transaction.get('amount', 0)
        features.append(np.log1p(amount))

        # Time features
        hour = transaction.get('hour', 12)
        features.extend([
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ])

        # Pad to expected size (giả sử 64 features)
        while len(features) < 64:
            features.append(0)

        return np.array([features[:64]])

    def _get_risk_level(self, proba: float) -> str:
        """
        Xác định mức độ rủi ro

        Args:
            proba: Xác suất fraud

        Returns:
            str: 'low', 'medium', 'high', 'critical'
        """
        if proba < self.risk_thresholds['low']:
            return 'low'
        elif proba < self.risk_thresholds['medium']:
            return 'medium'
        elif proba < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá toàn bộ pipeline

        Returns:
            Dict metrics cho từng layer và combined
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        results = {}

        if verbose:
            print("\n" + "=" * 60)
            print("ĐÁNH GIÁ FINAL PREDICTOR")
            print("=" * 60)

        # Layer 1
        if verbose:
            print("\n" + "-" * 40)
            print("LAYER 1 EVALUATION")
            print("-" * 40)
        results['layer1'] = self.layer1.evaluate(X, y_true, verbose)

        # Layer 2
        if self.is_fitted['layer2']:
            if verbose:
                print("\n" + "-" * 40)
                print("LAYER 2 EVALUATION")
                print("-" * 40)
            results['layer2'] = self.layer2.evaluate(
                X=X, y_true=y_true,
                sequences=sequences,
                node_features=node_features,
                edge_index=edge_index,
                verbose=verbose
            )

        # Combined
        y_pred = self.predict(X, sequences, node_features, edge_index)
        y_proba = self.predict_proba(X, sequences, node_features, edge_index)

        # Align lengths
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_proba = y_proba[:min_len]
        y_true_aligned = y_true[:min_len]

        combined_metrics = {
            'accuracy': accuracy_score(y_true_aligned, y_pred),
            'precision': precision_score(y_true_aligned, y_pred, zero_division=0),
            'recall': recall_score(y_true_aligned, y_pred, zero_division=0),
            'f1_score': f1_score(y_true_aligned, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true_aligned, y_proba),
            'confusion_matrix': confusion_matrix(y_true_aligned, y_pred).tolist()
        }
        results['combined'] = combined_metrics

        if verbose:
            print("\n" + "-" * 40)
            print("COMBINED EVALUATION")
            print("-" * 40)
            print(f"  - Accuracy:  {combined_metrics['accuracy']:.4f}")
            print(f"  - Precision: {combined_metrics['precision']:.4f}")
            print(f"  - Recall:    {combined_metrics['recall']:.4f}")
            print(f"  - F1-Score:  {combined_metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {combined_metrics['roc_auc']:.4f}")

            # Summary comparison
            print("\n" + "=" * 60)
            print("SUMMARY - F1 SCORES")
            print("=" * 60)
            print(f"  Layer 1 Ensemble:  {results['layer1']['ensemble']['f1_score']:.4f}")
            if 'layer2' in results and 'ensemble' in results['layer2']:
                print(f"  Layer 2 Ensemble:  {results['layer2']['ensemble']['f1_score']:.4f}")
            print(f"  Final Combined:    {combined_metrics['f1_score']:.4f}")

        return results

    def save(self, path: str = None):
        """
        Lưu toàn bộ predictor

        Args:
            path: Thư mục lưu
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        os.makedirs(path, exist_ok=True)

        # Lưu các layers
        self.layer1.save(path)
        if self.is_fitted['layer2']:
            self.layer2.save(path)

        # Lưu config
        predictor_config = {
            'layer1_weight': self.layer1_weight,
            'layer2_weight': self.layer2_weight,
            'is_fitted': self.is_fitted,
            'risk_thresholds': self.risk_thresholds,
            'prediction_stats': self.prediction_stats
        }

        config_path = os.path.join(path, 'final_predictor_config.json')
        with open(config_path, 'w') as f:
            json.dump(predictor_config, f, indent=2)

        print(f"[FinalPredictor] Đã lưu predictor: {path}")

    def load(self, path: str = None):
        """
        Load predictor

        Args:
            path: Thư mục chứa models
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        # Load config
        config_path = os.path.join(path, 'final_predictor_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                predictor_config = json.load(f)

            self.layer1_weight = predictor_config['layer1_weight']
            self.layer2_weight = predictor_config['layer2_weight']
            self.is_fitted = predictor_config['is_fitted']
            self.risk_thresholds = predictor_config['risk_thresholds']
            self.prediction_stats = predictor_config['prediction_stats']

        # Load layers
        self.layer1.load(path)
        if self.is_fitted.get('layer2', False):
            self.layer2.load(path)

        print(f"[FinalPredictor] Đã load predictor: {path}")

    def get_stats(self) -> Dict:
        """Lấy thống kê predictions"""
        return self.prediction_stats

    def get_model_status(self) -> Dict:
        """Lấy trạng thái các models"""
        return {
            'layer1': {
                'fitted': self.is_fitted['layer1'],
                'models': ['isolation_forest', 'lightgbm']
            },
            'layer2': {
                'fitted': self.is_fitted['layer2'],
                'models': {
                    'autoencoder': self.layer2.is_fitted.get('autoencoder', False),
                    'lstm': self.layer2.is_fitted.get('lstm', False),
                    'gnn': self.layer2.is_fitted.get('gnn', False)
                }
            },
            'weights': {
                'layer1': self.layer1_weight,
                'layer2': self.layer2_weight
            }
        }

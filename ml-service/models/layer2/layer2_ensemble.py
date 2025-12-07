"""
Layer 2 Ensemble - Kết hợp Autoencoder, LSTM và GNN
===================================================
Kết hợp kết quả từ 3 advanced models để có dự đoán tốt hơn:
- Autoencoder: User profiling và anomaly detection
- LSTM: Sequence-based fraud detection
- GNN: Graph-based fraud detection
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import joblib

# Import các models
from .autoencoder import AutoencoderModel
from .lstm_sequence import LSTMSequenceModel
from .gnn_model import GNNModel

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()


class Layer2Ensemble:
    """
    Ensemble của Autoencoder, LSTM và GNN

    Kết hợp weighted average với khả năng động điều chỉnh weights
    """

    def __init__(
        self,
        autoencoder_weight: float = 0.3,
        lstm_weight: float = 0.4,
        gnn_weight: float = 0.3
    ):
        """
        Khởi tạo Layer 2 Ensemble

        Args:
            autoencoder_weight: Trọng số cho Autoencoder
            lstm_weight: Trọng số cho LSTM
            gnn_weight: Trọng số cho GNN
        """
        # Normalize weights
        total = autoencoder_weight + lstm_weight + gnn_weight
        self.ae_weight = autoencoder_weight / total
        self.lstm_weight = lstm_weight / total
        self.gnn_weight = gnn_weight / total

        # Initialize models
        self.autoencoder = AutoencoderModel()
        self.lstm = LSTMSequenceModel()
        self.gnn = GNNModel()

        self.is_fitted = {
            'autoencoder': False,
            'lstm': False,
            'gnn': False
        }

    def fit_autoencoder(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        verbose: bool = True
    ):
        """
        Train Autoencoder

        Args:
            X: Feature matrix
            feature_names: Tên features
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING AUTOENCODER")
            print("=" * 50)

        self.autoencoder.fit(X, feature_names=feature_names, verbose=verbose)
        self.is_fitted['autoencoder'] = True

    def fit_lstm(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ):
        """
        Train LSTM

        Args:
            sequences: Sequence data
            labels: Labels
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING LSTM")
            print("=" * 50)

        self.lstm.fit(sequences, labels, verbose=verbose)
        self.is_fitted['lstm'] = True

    def fit_gnn(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ):
        """
        Train GNN

        Args:
            node_features: Node features
            edge_index: Graph edges
            labels: Node labels
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING GNN")
            print("=" * 50)

        self.gnn.fit(node_features, edge_index, labels, verbose=verbose)
        self.is_fitted['gnn'] = True

    def predict(
        self,
        X: np.ndarray = None,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Dự đoán fraud

        Args:
            X: Features cho Autoencoder
            sequences: Sequences cho LSTM
            node_features: Node features cho GNN
            edge_index: Edges cho GNN
            threshold: Ngưỡng quyết định

        Returns:
            np.ndarray predictions
        """
        proba = self.predict_proba(X, sequences, node_features, edge_index)
        return (proba >= threshold).astype(int)

    def predict_proba(
        self,
        X: np.ndarray = None,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None
    ) -> np.ndarray:
        """
        Tính xác suất fraud (weighted average của các models có sẵn)

        Args:
            X: Features cho Autoencoder
            sequences: Sequences cho LSTM
            node_features: Node features cho GNN
            edge_index: Edges cho GNN

        Returns:
            np.ndarray probabilities
        """
        probas = []
        weights = []
        n_samples = None

        # Autoencoder
        if X is not None and self.is_fitted['autoencoder']:
            ae_proba = self.autoencoder.predict_proba(X)
            probas.append(ae_proba)
            weights.append(self.ae_weight)
            n_samples = len(ae_proba)

        # LSTM
        if sequences is not None and self.is_fitted['lstm']:
            lstm_proba = self.lstm.predict_proba(sequences)
            probas.append(lstm_proba)
            weights.append(self.lstm_weight)
            if n_samples is None:
                n_samples = len(lstm_proba)

        # GNN
        if node_features is not None and edge_index is not None and self.is_fitted['gnn']:
            gnn_proba = self.gnn.predict_proba(node_features, edge_index)
            probas.append(gnn_proba)
            weights.append(self.gnn_weight)
            if n_samples is None:
                n_samples = len(gnn_proba)

        if not probas:
            raise ValueError("Không có model nào được fit hoặc không có dữ liệu đầu vào!")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Ensure all probas have same length
        min_len = min(len(p) for p in probas)
        probas = [p[:min_len] for p in probas]

        # Weighted average
        combined = np.zeros(min_len)
        for proba, weight in zip(probas, weights):
            combined += weight * proba

        return combined

    def predict_detail(
        self,
        X: np.ndarray = None,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None
    ) -> Dict:
        """
        Dự đoán chi tiết với kết quả từ mỗi model

        Returns:
            Dict với kết quả chi tiết
        """
        result = {
            'models_used': [],
            'weights': {}
        }

        if X is not None and self.is_fitted['autoencoder']:
            result['autoencoder_score'] = self.autoencoder.predict_proba(X).tolist()
            result['models_used'].append('autoencoder')
            result['weights']['autoencoder'] = self.ae_weight

        if sequences is not None and self.is_fitted['lstm']:
            result['lstm_score'] = self.lstm.predict_proba(sequences).tolist()
            result['models_used'].append('lstm')
            result['weights']['lstm'] = self.lstm_weight

        if node_features is not None and edge_index is not None and self.is_fitted['gnn']:
            result['gnn_score'] = self.gnn.predict_proba(node_features, edge_index).tolist()
            result['models_used'].append('gnn')
            result['weights']['gnn'] = self.gnn_weight

        result['combined_score'] = self.predict_proba(
            X, sequences, node_features, edge_index
        ).tolist()

        result['predictions'] = self.predict(
            X, sequences, node_features, edge_index
        ).tolist()

        return result

    def evaluate(
        self,
        X: np.ndarray = None,
        sequences: np.ndarray = None,
        node_features: np.ndarray = None,
        edge_index: np.ndarray = None,
        y_true: np.ndarray = None,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá ensemble và từng model

        Returns:
            Dict chứa metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        results = {}

        # Đánh giá từng model
        if verbose:
            print("\n" + "=" * 50)
            print("ĐÁNH GIÁ LAYER 2 ENSEMBLE")
            print("=" * 50)

        if X is not None and self.is_fitted['autoencoder']:
            if verbose:
                print("\n--- Autoencoder ---")
            results['autoencoder'] = self.autoencoder.evaluate(X, y_true, verbose)

        if sequences is not None and self.is_fitted['lstm']:
            if verbose:
                print("\n--- LSTM ---")
            # LSTM labels có thể khác size
            lstm_labels = y_true[:len(sequences)] if len(y_true) > len(sequences) else y_true
            results['lstm'] = self.lstm.evaluate(sequences, lstm_labels, verbose)

        if node_features is not None and edge_index is not None and self.is_fitted['gnn']:
            if verbose:
                print("\n--- GNN ---")
            results['gnn'] = self.gnn.evaluate(node_features, edge_index, y_true, verbose=verbose)

        # Ensemble
        y_pred = self.predict(X, sequences, node_features, edge_index)
        y_proba = self.predict_proba(X, sequences, node_features, edge_index)

        # Align lengths
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_proba = y_proba[:min_len]
        y_true_aligned = y_true[:min_len]

        ensemble_metrics = {
            'accuracy': accuracy_score(y_true_aligned, y_pred),
            'precision': precision_score(y_true_aligned, y_pred, zero_division=0),
            'recall': recall_score(y_true_aligned, y_pred, zero_division=0),
            'f1_score': f1_score(y_true_aligned, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true_aligned, y_proba),
            'confusion_matrix': confusion_matrix(y_true_aligned, y_pred).tolist()
        }
        results['ensemble'] = ensemble_metrics

        if verbose:
            print("\n--- Ensemble (Combined) ---")
            print(f"  - Accuracy:  {ensemble_metrics['accuracy']:.4f}")
            print(f"  - Precision: {ensemble_metrics['precision']:.4f}")
            print(f"  - Recall:    {ensemble_metrics['recall']:.4f}")
            print(f"  - F1-Score:  {ensemble_metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {ensemble_metrics['roc_auc']:.4f}")

        return results

    def save(self, path: str = None):
        """
        Lưu ensemble

        Args:
            path: Thư mục lưu
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        os.makedirs(path, exist_ok=True)

        # Lưu từng model nếu đã fit
        if self.is_fitted['autoencoder']:
            self.autoencoder.save(os.path.join(path, 'autoencoder.pth'))

        if self.is_fitted['lstm']:
            self.lstm.save(os.path.join(path, 'lstm.pth'))

        if self.is_fitted['gnn']:
            self.gnn.save(os.path.join(path, 'gnn.pth'))

        # Lưu ensemble config
        ensemble_config = {
            'ae_weight': self.ae_weight,
            'lstm_weight': self.lstm_weight,
            'gnn_weight': self.gnn_weight,
            'is_fitted': self.is_fitted
        }
        joblib.dump(ensemble_config, os.path.join(path, 'layer2_ensemble_config.pkl'))

        print(f"[Layer2Ensemble] Đã lưu ensemble: {path}")

    def load(self, path: str = None):
        """
        Load ensemble

        Args:
            path: Thư mục chứa models
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        # Load ensemble config
        config_path = os.path.join(path, 'layer2_ensemble_config.pkl')
        if os.path.exists(config_path):
            ensemble_config = joblib.load(config_path)
            self.ae_weight = ensemble_config['ae_weight']
            self.lstm_weight = ensemble_config['lstm_weight']
            self.gnn_weight = ensemble_config['gnn_weight']
            self.is_fitted = ensemble_config['is_fitted']

        # Load từng model
        ae_path = os.path.join(path, 'autoencoder.pth')
        if os.path.exists(ae_path):
            self.autoencoder.load(ae_path)

        lstm_path = os.path.join(path, 'lstm.pth')
        if os.path.exists(lstm_path):
            self.lstm.load(lstm_path)

        gnn_path = os.path.join(path, 'gnn.pth')
        if os.path.exists(gnn_path):
            self.gnn.load(gnn_path)

        print(f"[Layer2Ensemble] Đã load ensemble: {path}")

    def get_user_profile(
        self,
        X: np.ndarray,
        sequences: np.ndarray = None,
        sample_idx: int = 0
    ) -> Dict:
        """
        Lấy user profile từ các models

        Args:
            X: Features
            sequences: Sequences (optional)
            sample_idx: Index của sample

        Returns:
            Dict chứa user profile
        """
        profile = {}

        if self.is_fitted['autoencoder']:
            profile['embedding'] = self.autoencoder.get_embedding(X[sample_idx:sample_idx + 1])[0].tolist()
            profile['reconstruction_error'] = float(
                self.autoencoder.get_reconstruction_error(X[sample_idx:sample_idx + 1])[0]
            )

        if sequences is not None and self.is_fitted['lstm']:
            profile['sequence_fraud_prob'] = float(
                self.lstm.predict_proba(sequences[sample_idx:sample_idx + 1])[0]
            )

        profile['combined_fraud_prob'] = float(
            self.predict_proba(X[sample_idx:sample_idx + 1])[0]
        )

        return profile

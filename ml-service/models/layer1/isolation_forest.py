"""
Isolation Forest Model - Phát hiện giao dịch bất thường (Anomaly Detection)
===========================================================================
Isolation Forest là thuật toán unsupervised learning để phát hiện anomaly.
Ý tưởng: Anomalies dễ bị "isolate" hơn các điểm bình thường.

Ưu điểm:
- Không cần dữ liệu có nhãn
- Nhanh và hiệu quả với dữ liệu lớn
- Hoạt động tốt với high-dimensional data

Nhược điểm:
- Khó giải thích kết quả
- Cần tune contamination parameter
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()


class IsolationForestModel:
    """
    Isolation Forest model cho fraud detection

    Model sử dụng các features của giao dịch để phát hiện
    các giao dịch bất thường (anomaly).
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo Isolation Forest model

        Args:
            model_config: Dict cấu hình model, sử dụng config mặc định nếu None
        """
        self.config = model_config or config.ISOLATION_FOREST_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []

        # Threshold cho anomaly score
        self.threshold = -0.1  # Scores < threshold = anomaly

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        verbose: bool = True
    ):
        """
        Train Isolation Forest model

        Args:
            X: Feature matrix (num_samples, num_features)
            feature_names: Tên các features
            verbose: In thông tin training
        """
        if verbose:
            print("[IsolationForest] Bắt đầu training...")
            print(f"  - Số samples: {X.shape[0]:,}")
            print(f"  - Số features: {X.shape[1]}")

        # Lưu feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Chuẩn hóa features
        X_scaled = self.scaler.fit_transform(X)

        # Khởi tạo và train model
        self.model = IsolationForest(
            n_estimators=self.config.get('n_estimators', 100),
            contamination=self.config.get('contamination', 0.05),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1),
            max_samples='auto',
            bootstrap=False
        )

        self.model.fit(X_scaled)
        self.is_fitted = True

        if verbose:
            print("[IsolationForest] Training hoàn tất!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán anomaly cho các samples

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: 1 = normal, -1 = anomaly
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tính xác suất anomaly (fraud score)

        Chuyển đổi anomaly score thành xác suất [0, 1]
        Score cao = nghi ngờ fraud

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Xác suất fraud cho mỗi sample
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)

        # Lấy anomaly scores (-1 đến 0 cho anomaly, 0+ cho normal)
        scores = self.model.decision_function(X_scaled)

        # Chuyển đổi scores thành xác suất
        # Score càng âm = càng anomaly = xác suất fraud cao hơn
        # Sử dụng sigmoid transformation
        proba = 1 / (1 + np.exp(5 * scores))  # Scaling factor 5 để điều chỉnh độ nhạy

        return proba

    def get_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Lấy anomaly score gốc

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá model trên tập test

        Args:
            X: Feature matrix
            y_true: Labels thực tế (0 = normal, 1 = fraud)
            verbose: In kết quả

        Returns:
            Dict chứa các metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        # Dự đoán
        y_pred_labels = self.predict(X)
        # Convert -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0
        y_pred = (y_pred_labels == -1).astype(int)

        # Xác suất fraud
        y_proba = self.predict_proba(X)

        # Tính metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        if verbose:
            print("\n[IsolationForest] Kết quả đánh giá:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  - Confusion Matrix:")
            print(f"    TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}")
            print(f"    FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}")

        return metrics

    def save(self, path: str = None):
        """
        Lưu model ra file

        Args:
            path: Đường dẫn file, mặc định lưu vào saved_models/
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'isolation_forest.pkl')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, path)
        print(f"[IsolationForest] Đã lưu model: {path}")

    def load(self, path: str = None):
        """
        Load model từ file

        Args:
            path: Đường dẫn file
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'isolation_forest.pkl')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.is_fitted = model_data['is_fitted']

        print(f"[IsolationForest] Đã load model: {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Isolation Forest không có feature importance trực tiếp
        Trả về None
        """
        return None

    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Dict:
        """
        Giải thích dự đoán cho một sample

        Args:
            X: Feature matrix
            sample_idx: Index của sample cần giải thích

        Returns:
            Dict chứa giải thích
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        sample = X[sample_idx:sample_idx + 1]

        # Lấy anomaly score
        score = self.get_anomaly_score(sample)[0]
        proba = self.predict_proba(sample)[0]
        prediction = self.predict(sample)[0]

        explanation = {
            'anomaly_score': float(score),
            'fraud_probability': float(proba),
            'prediction': 'fraud' if prediction == -1 else 'normal',
            'threshold': self.threshold,
            'feature_values': {
                name: float(sample[0, i])
                for i, name in enumerate(self.feature_names)
            }
        }

        return explanation


# Alias cho backward compatibility
IsolationForestFraudDetector = IsolationForestModel

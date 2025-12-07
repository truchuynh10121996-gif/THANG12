"""
Layer 1 Ensemble - Kết hợp Isolation Forest và LightGBM
=======================================================
Kết hợp kết quả từ cả 2 models để có dự đoán tốt hơn:
- Isolation Forest: Phát hiện anomaly (unsupervised)
- LightGBM: Phân loại fraud (supervised)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import joblib

# Import các models
from .isolation_forest import IsolationForestModel
from .lightgbm_model import LightGBMModel

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()


class Layer1Ensemble:
    """
    Ensemble của Isolation Forest và LightGBM

    Kết hợp theo cách:
    - Weighted average của xác suất từ 2 models
    - Có thể adjust weights dựa trên performance
    """

    def __init__(
        self,
        isolation_forest_weight: float = 0.3,
        lightgbm_weight: float = 0.7
    ):
        """
        Khởi tạo Layer 1 Ensemble

        Args:
            isolation_forest_weight: Trọng số cho Isolation Forest
            lightgbm_weight: Trọng số cho LightGBM
        """
        # Normalize weights
        total = isolation_forest_weight + lightgbm_weight
        self.if_weight = isolation_forest_weight / total
        self.lgb_weight = lightgbm_weight / total

        # Initialize models
        self.isolation_forest = IsolationForestModel()
        self.lightgbm = LightGBMModel()

        self.is_fitted = False
        self.feature_names = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        verbose: bool = True
    ):
        """
        Train cả 2 models

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Tên features
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING LAYER 1 ENSEMBLE")
            print("=" * 50)
            print(f"Weights: IF={self.if_weight:.2f}, LGB={self.lgb_weight:.2f}")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Train Isolation Forest (unsupervised - không cần y)
        if verbose:
            print("\n--- Isolation Forest ---")
        self.isolation_forest.fit(X, feature_names=self.feature_names, verbose=verbose)

        # Train LightGBM (supervised)
        if verbose:
            print("\n--- LightGBM ---")
        self.lightgbm.fit(X, y, feature_names=self.feature_names, verbose=verbose)

        self.is_fitted = True

        if verbose:
            print("\n[Layer1Ensemble] Training hoàn tất!")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán labels

        Args:
            X: Feature matrix
            threshold: Ngưỡng để quyết định fraud

        Returns:
            np.ndarray: 0 = normal, 1 = fraud
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tính xác suất fraud (weighted average)

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Xác suất fraud
        """
        if not self.is_fitted:
            raise ValueError("Ensemble chưa được train!")

        # Lấy xác suất từ mỗi model
        if_proba = self.isolation_forest.predict_proba(X)
        lgb_proba = self.lightgbm.predict_proba(X)

        # Weighted average
        combined_proba = (
            self.if_weight * if_proba +
            self.lgb_weight * lgb_proba
        )

        return combined_proba

    def predict_detail(self, X: np.ndarray) -> Dict:
        """
        Dự đoán chi tiết với kết quả từ mỗi model

        Args:
            X: Feature matrix

        Returns:
            Dict với kết quả chi tiết
        """
        if not self.is_fitted:
            raise ValueError("Ensemble chưa được train!")

        if_proba = self.isolation_forest.predict_proba(X)
        lgb_proba = self.lightgbm.predict_proba(X)
        combined_proba = self.predict_proba(X)

        return {
            'isolation_forest_score': if_proba.tolist(),
            'lightgbm_score': lgb_proba.tolist(),
            'combined_score': combined_proba.tolist(),
            'predictions': self.predict(X).tolist(),
            'weights': {
                'isolation_forest': self.if_weight,
                'lightgbm': self.lgb_weight
            }
        }

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá ensemble và từng model riêng

        Args:
            X: Feature matrix
            y_true: Labels thực tế
            verbose: In kết quả

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
            print("ĐÁNH GIÁ LAYER 1 ENSEMBLE")
            print("=" * 50)

        # Isolation Forest
        if verbose:
            print("\n--- Isolation Forest ---")
        results['isolation_forest'] = self.isolation_forest.evaluate(X, y_true, verbose)

        # LightGBM
        if verbose:
            print("\n--- LightGBM ---")
        results['lightgbm'] = self.lightgbm.evaluate(X, y_true, verbose)

        # Ensemble
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        ensemble_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        results['ensemble'] = ensemble_metrics

        if verbose:
            print("\n--- Ensemble (Combined) ---")
            print(f"  - Accuracy:  {ensemble_metrics['accuracy']:.4f}")
            print(f"  - Precision: {ensemble_metrics['precision']:.4f}")
            print(f"  - Recall:    {ensemble_metrics['recall']:.4f}")
            print(f"  - F1-Score:  {ensemble_metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {ensemble_metrics['roc_auc']:.4f}")

            # So sánh
            print("\n📊 So sánh F1-Score:")
            print(f"  - Isolation Forest: {results['isolation_forest']['f1_score']:.4f}")
            print(f"  - LightGBM:         {results['lightgbm']['f1_score']:.4f}")
            print(f"  - Ensemble:         {ensemble_metrics['f1_score']:.4f}")

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy feature importance từ LightGBM

        Returns:
            Dict {feature_name: importance}
        """
        return self.lightgbm.get_feature_importance()

    def save(self, path: str = None):
        """
        Lưu ensemble (cả 2 models)

        Args:
            path: Thư mục lưu
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        os.makedirs(path, exist_ok=True)

        # Lưu từng model
        self.isolation_forest.save(os.path.join(path, 'isolation_forest.pkl'))
        self.lightgbm.save(os.path.join(path, 'lightgbm.pkl'))

        # Lưu ensemble config
        ensemble_config = {
            'if_weight': self.if_weight,
            'lgb_weight': self.lgb_weight,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(ensemble_config, os.path.join(path, 'layer1_ensemble_config.pkl'))

        print(f"[Layer1Ensemble] Đã lưu ensemble: {path}")

    def load(self, path: str = None):
        """
        Load ensemble từ files

        Args:
            path: Thư mục chứa models
        """
        if path is None:
            path = config.SAVED_MODELS_DIR

        # Load từng model
        self.isolation_forest.load(os.path.join(path, 'isolation_forest.pkl'))
        self.lightgbm.load(os.path.join(path, 'lightgbm.pkl'))

        # Load ensemble config
        config_path = os.path.join(path, 'layer1_ensemble_config.pkl')
        if os.path.exists(config_path):
            ensemble_config = joblib.load(config_path)
            self.if_weight = ensemble_config['if_weight']
            self.lgb_weight = ensemble_config['lgb_weight']
            self.feature_names = ensemble_config['feature_names']
            self.is_fitted = ensemble_config['is_fitted']

        print(f"[Layer1Ensemble] Đã load ensemble: {path}")

    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Dict:
        """
        Giải thích dự đoán chi tiết

        Args:
            X: Feature matrix
            sample_idx: Index của sample

        Returns:
            Dict chứa giải thích từ cả 2 models
        """
        sample = X[sample_idx:sample_idx + 1]

        # Lấy giải thích từ mỗi model
        if_explanation = self.isolation_forest.explain_prediction(X, sample_idx)
        lgb_explanation = self.lightgbm.explain_prediction(X, sample_idx)

        # Kết hợp
        combined_proba = self.predict_proba(sample)[0]

        return {
            'combined_prediction': {
                'fraud_probability': float(combined_proba),
                'prediction': 'fraud' if combined_proba >= 0.5 else 'normal'
            },
            'isolation_forest': if_explanation,
            'lightgbm': lgb_explanation,
            'model_contributions': {
                'isolation_forest': {
                    'weight': self.if_weight,
                    'score': float(if_explanation['fraud_probability'])
                },
                'lightgbm': {
                    'weight': self.lgb_weight,
                    'score': float(lgb_explanation['fraud_probability'])
                }
            }
        }

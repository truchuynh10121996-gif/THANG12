"""
LightGBM Model - Phân loại giao dịch lừa đảo
==============================================
LightGBM là gradient boosting framework sử dụng tree-based learning.

Ưu điểm:
- Tốc độ training rất nhanh
- Hiệu quả với dữ liệu lớn
- Có feature importance
- Xử lý tốt imbalanced data

Nhược điểm:
- Cần dữ liệu có nhãn để train
- Có thể overfit nếu không tune đúng
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()


class LightGBMModel:
    """
    LightGBM model cho fraud classification

    Model sử dụng supervised learning để phân loại
    giao dịch thành fraud/normal dựa trên dữ liệu có nhãn.
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo LightGBM model

        Args:
            model_config: Dict cấu hình model
        """
        self.config = model_config or config.LIGHTGBM_CONFIG.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.feature_importance = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        validation_split: float = 0.1,
        verbose: bool = True
    ):
        """
        Train LightGBM model

        Args:
            X: Feature matrix
            y: Labels (0 = normal, 1 = fraud)
            feature_names: Tên các features
            validation_split: Tỷ lệ validation set
            verbose: In thông tin training
        """
        if verbose:
            print("[LightGBM] Bắt đầu training...")
            print(f"  - Số samples: {X.shape[0]:,}")
            print(f"  - Số features: {X.shape[1]}")
            print(f"  - Fraud ratio: {y.mean()*100:.2f}%")

        # Lưu feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Chuẩn hóa features
        X_scaled = self.scaler.fit_transform(X)

        # Chia train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=validation_split,
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        # Tính class weight cho imbalanced data
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1

        # Cấu hình LightGBM
        params = {
            'objective': self.config.get('objective', 'binary'),
            'metric': self.config.get('metric', 'auc'),
            'boosting_type': self.config.get('boosting_type', 'gbdt'),
            'num_leaves': self.config.get('num_leaves', 31),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'feature_fraction': self.config.get('feature_fraction', 0.9),
            'bagging_fraction': self.config.get('bagging_fraction', 0.8),
            'bagging_freq': self.config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': self.config.get('random_state', 42),
            'scale_pos_weight': scale_pos_weight,
            'is_unbalance': True
        }

        # Tạo datasets
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_names
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=self.feature_names,
            reference=train_data
        )

        # Training với early stopping
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))
        callbacks.append(lgb.early_stopping(
            stopping_rounds=self.config.get('early_stopping_rounds', 50)
        ))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.get('n_estimators', 200),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        self.is_fitted = True

        # Lưu feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, importance))

        if verbose:
            print("[LightGBM] Training hoàn tất!")
            print(f"  - Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán labels cho các samples

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: 0 = normal, 1 = fraud
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tính xác suất fraud

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Xác suất fraud cho mỗi sample
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, num_iteration=self.model.best_iteration)

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
            y_true: Labels thực tế
            verbose: In kết quả

        Returns:
            Dict chứa các metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
            precision_recall_curve, average_precision_score
        )

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        if verbose:
            print("\n[LightGBM] Kết quả đánh giá:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  - AP Score:  {metrics['average_precision']:.4f}")
            print(f"  - Confusion Matrix:")
            print(f"    TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}")
            print(f"    FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}")

        return metrics

    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_k: int = None
    ) -> Dict[str, float]:
        """
        Lấy feature importance

        Args:
            importance_type: 'gain', 'split', hoặc 'both'
            top_k: Số features quan trọng nhất

        Returns:
            Dict {feature_name: importance}
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        importance = self.model.feature_importance(importance_type=importance_type)
        result = dict(zip(self.feature_names, importance))

        # Sắp xếp theo importance giảm dần
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        if top_k:
            result = dict(list(result.items())[:top_k])

        return result

    def save(self, path: str = None):
        """
        Lưu model ra file

        Args:
            path: Đường dẫn file
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'lightgbm.pkl')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, path)
        print(f"[LightGBM] Đã lưu model: {path}")

    def load(self, path: str = None):
        """
        Load model từ file

        Args:
            path: Đường dẫn file
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'lightgbm.pkl')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']

        print(f"[LightGBM] Đã load model: {path}")

    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Dict:
        """
        Giải thích dự đoán cho một sample

        Sử dụng SHAP values nếu có thể

        Args:
            X: Feature matrix
            sample_idx: Index của sample

        Returns:
            Dict chứa giải thích
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        sample = X[sample_idx:sample_idx + 1]
        X_scaled = self.scaler.transform(sample)

        proba = self.predict_proba(sample)[0]
        prediction = self.predict(sample)[0]

        # Feature contributions (simplified)
        contributions = {}
        for i, name in enumerate(self.feature_names):
            # Giá trị feature * importance (đơn giản hóa)
            importance = self.feature_importance.get(name, 0)
            contributions[name] = {
                'value': float(sample[0, i]),
                'importance': float(importance),
                'contribution': float(X_scaled[0, i] * importance / 1000)  # Normalized
            }

        # Sắp xếp theo contribution
        contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        ))

        explanation = {
            'fraud_probability': float(proba),
            'prediction': 'fraud' if prediction == 1 else 'normal',
            'feature_contributions': contributions,
            'top_features': list(contributions.keys())[:5]
        }

        return explanation


# Alias
LightGBMFraudClassifier = LightGBMModel

"""
Autoencoder Model - User Embedding và Anomaly Detection
=======================================================
Autoencoder học cách nén dữ liệu user thành embedding và tái tạo lại.
Khi reconstruction error cao = user hành động bất thường.

Ưu điểm:
- Tạo được user embedding hữu ích
- Phát hiện anomaly dựa trên reconstruction error
- Không cần labels

Nhược điểm:
- Cần tune architecture và hyperparameters
- Training có thể không ổn định
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoencoderNetwork(nn.Module):
    """
    Neural network architecture cho Autoencoder

    Encoder: input_dim -> 32 -> 16 -> 8 (embedding)
    Decoder: 8 -> 16 -> 32 -> input_dim
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = None
    ):
        """
        Khởi tạo Autoencoder network

        Args:
            input_dim: Số chiều input
            encoding_dims: Kích thước các hidden layers [32, 16, 8]
        """
        super(AutoencoderNetwork, self).__init__()

        encoding_dims = encoding_dims or [32, 16, 8]

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout

        # Decoder layers (reverse)
        decoder_layers = []
        decoder_dims = encoding_dims[::-1][1:] + [input_dim]
        prev_dim = encoding_dims[-1]
        for i, dim in enumerate(decoder_dims):
            if i == len(decoder_dims) - 1:
                # Output layer - không có activation
                decoder_layers.append(nn.Linear(prev_dim, dim))
            else:
                decoder_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            prev_dim = dim

        self.decoder = nn.Sequential(*decoder_layers)

        self.embedding_dim = encoding_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input thành embedding"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embedding thành reconstruction"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            Tuple (reconstruction, embedding)
        """
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding


class AutoencoderModel:
    """
    Autoencoder model cho user profiling và anomaly detection

    Sử dụng reconstruction error để phát hiện giao dịch bất thường
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo Autoencoder model

        Args:
            model_config: Dict cấu hình model
        """
        self.config = model_config or config.AUTOENCODER_CONFIG.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None  # Threshold cho anomaly detection
        self.feature_names = []

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        validation_split: float = 0.1,
        verbose: bool = True
    ):
        """
        Train Autoencoder

        Args:
            X: Feature matrix (chỉ normal samples nếu có thể)
            feature_names: Tên features
            validation_split: Tỷ lệ validation
            verbose: In thông tin
        """
        if verbose:
            print("[Autoencoder] Bắt đầu training...")
            print(f"  - Số samples: {X.shape[0]:,}")
            print(f"  - Số features: {X.shape[1]}")
            print(f"  - Device: {device}")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Chuẩn hóa features
        X_scaled = self.scaler.fit_transform(X)

        # Chia train/validation
        n_samples = len(X_scaled)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        X_train = X_scaled[indices[n_val:]]
        X_val = X_scaled[indices[:n_val]]

        # Convert to tensors
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)

        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=self.config.get('batch_size', 256),
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor, val_tensor),
            batch_size=self.config.get('batch_size', 256)
        )

        # Khởi tạo model
        input_dim = X.shape[1]
        encoding_dims = self.config.get('encoding_dims', [32, 16, 8])

        self.model = AutoencoderNetwork(input_dim, encoding_dims).to(device)

        # Optimizer và loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.MSELoss()

        # Training loop
        epochs = self.config.get('epochs', 100)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                reconstruction, _ = self.model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    reconstruction, _ = self.model(batch_x)
                    loss = criterion(reconstruction, batch_x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Tính threshold từ training data
        self._calculate_threshold(X_scaled)

        self.is_fitted = True
        if verbose:
            print("[Autoencoder] Training hoàn tất!")

    def _calculate_threshold(self, X: np.ndarray, percentile: float = None):
        """
        Tính threshold cho anomaly detection

        Dựa trên percentile của reconstruction error trên training data
        """
        percentile = percentile or self.config.get('threshold_percentile', 95)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            reconstruction, _ = self.model(X_tensor)

            # Tính reconstruction error (MSE per sample)
            errors = ((reconstruction - X_tensor) ** 2).mean(dim=1).cpu().numpy()

        self.threshold = np.percentile(errors, percentile)

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Tính reconstruction error cho mỗi sample

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Reconstruction error cho mỗi sample
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            reconstruction, _ = self.model(X_tensor)

            errors = ((reconstruction - X_tensor) ** 2).mean(dim=1).cpu().numpy()

        return errors

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Lấy embedding cho các samples

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Embedding vectors
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            _, embedding = self.model(X_tensor)

        return embedding.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán anomaly

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: 0 = normal, 1 = anomaly/fraud
        """
        errors = self.get_reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Tính xác suất anomaly/fraud

        Chuyển đổi reconstruction error thành xác suất

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Xác suất fraud
        """
        errors = self.get_reconstruction_error(X)

        # Normalize errors to [0, 1]
        # Sử dụng sigmoid với threshold làm center
        proba = 1 / (1 + np.exp(-(errors - self.threshold) / (self.threshold + 1e-8)))

        return np.clip(proba, 0, 1)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá model

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

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'threshold': float(self.threshold)
        }

        if verbose:
            print("\n[Autoencoder] Kết quả đánh giá:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  - Threshold: {metrics['threshold']:.6f}")

        return metrics

    def save(self, path: str = None):
        """
        Lưu model

        Args:
            path: Đường dẫn file
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'autoencoder.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'history': self.history,
            'is_fitted': self.is_fitted
        }

        torch.save(save_data, path)
        print(f"[Autoencoder] Đã lưu model: {path}")

    def load(self, path: str = None):
        """
        Load model

        Args:
            path: Đường dẫn file
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'autoencoder.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.scaler = save_data['scaler']
        self.config = save_data['config']
        self.threshold = save_data['threshold']
        self.feature_names = save_data['feature_names']
        self.history = save_data['history']
        self.is_fitted = save_data['is_fitted']

        # Rebuild model
        input_dim = len(self.feature_names)
        encoding_dims = self.config.get('encoding_dims', [32, 16, 8])
        self.model = AutoencoderNetwork(input_dim, encoding_dims).to(device)
        self.model.load_state_dict(save_data['model_state_dict'])

        print(f"[Autoencoder] Đã load model: {path}")

    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Dict:
        """
        Giải thích dự đoán

        Args:
            X: Feature matrix
            sample_idx: Index của sample

        Returns:
            Dict giải thích
        """
        sample = X[sample_idx:sample_idx + 1]
        X_scaled = self.scaler.transform(sample)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            reconstruction, embedding = self.model(X_tensor)

            # Error per feature
            feature_errors = ((reconstruction - X_tensor) ** 2).squeeze().cpu().numpy()
            total_error = feature_errors.mean()

        # Top features với error cao nhất
        feature_error_dict = dict(zip(self.feature_names, feature_errors))
        feature_error_dict = dict(sorted(
            feature_error_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return {
            'reconstruction_error': float(total_error),
            'threshold': float(self.threshold),
            'is_anomaly': bool(total_error > self.threshold),
            'fraud_probability': float(self.predict_proba(sample)[0]),
            'embedding': embedding.squeeze().cpu().numpy().tolist(),
            'feature_errors': feature_error_dict,
            'top_anomalous_features': list(feature_error_dict.keys())[:5]
        }

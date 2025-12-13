"""
LSTM Sequence Model - Phát hiện bất thường trong chuỗi giao dịch
================================================================
LSTM học "nhịp điệu" chi tiêu của user và phát hiện khi pattern bị phá vỡ.

Ưu điểm:
- Capture được temporal dependencies
- Phát hiện patterns trong sequences
- Tốt cho time-series anomaly detection

Nhược điểm:
- Cần đủ dữ liệu lịch sử
- Training chậm hơn các models khác

UPDATE 2025: Tích hợp Threshold Optimizer cho ngân hàng Việt Nam
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()

# Import threshold optimizer (lazy import để tránh circular import)
def _get_threshold_optimizer():
    """Lazy import threshold optimizer"""
    try:
        from utils.threshold_optimizer import (
            recommend_threshold,
            log_threshold_analysis,
            print_summary_table,
            compute_metrics_at_threshold,
            compute_roc_curve,
            compute_precision_recall_curve,
            get_tier_distribution,
            classify_transactions,
            FraudThresholdClassifier,
            DEFAULT_THRESHOLDS,
            BUSINESS_CONSTRAINTS
        )
        return {
            'recommend_threshold': recommend_threshold,
            'log_threshold_analysis': log_threshold_analysis,
            'print_summary_table': print_summary_table,
            'compute_metrics_at_threshold': compute_metrics_at_threshold,
            'compute_roc_curve': compute_roc_curve,
            'compute_precision_recall_curve': compute_precision_recall_curve,
            'get_tier_distribution': get_tier_distribution,
            'classify_transactions': classify_transactions,
            'FraudThresholdClassifier': FraudThresholdClassifier,
            'DEFAULT_THRESHOLDS': DEFAULT_THRESHOLDS,
            'BUSINESS_CONSTRAINTS': BUSINESS_CONSTRAINTS
        }
    except ImportError as e:
        logger.warning(f"Could not import threshold_optimizer: {e}")
        return None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMNetwork(nn.Module):
    """
    LSTM network cho sequence classification

    Architecture:
    - LSTM layers với hidden states
    - Fully connected layer cho classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Khởi tạo LSTM network

        Args:
            input_size: Số features mỗi timestep
            hidden_size: Kích thước hidden state
            num_layers: Số LSTM layers
            dropout: Dropout rate
            bidirectional: Sử dụng bi-directional LSTM
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention layer (simplified)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classification layers
        fc_input_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor shape (batch, seq_len, input_size)

        Returns:
            Fraud probability shape (batch, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * directions)

        # Attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)  # (batch, hidden * directions)

        # Classification
        output = self.classifier(context)

        return output.squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """Lấy attention weights để giải thích"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attention_weights = torch.softmax(
                self.attention(lstm_out).squeeze(-1),
                dim=1
            )
        return attention_weights.cpu().numpy()


class LSTMSequenceModel:
    """
    LSTM model cho sequence-based fraud detection

    Phân tích chuỗi giao dịch để phát hiện patterns bất thường
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo LSTM model

        Args:
            model_config: Dict cấu hình
        """
        self.config = model_config or config.LSTM_CONFIG.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.sequence_length = self.config.get('sequence_length', 10)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # Threshold optimization results (NEW)
        self.threshold_config = None
        self.optimal_threshold = 0.5  # Default
        self.threshold_optimizer_result = None

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.1,
        verbose: bool = True,
        optimize_threshold: bool = True,
        threshold_strategy: str = 'balanced'
    ):
        """
        Train LSTM model với tùy chọn tối ưu threshold tự động

        Args:
            sequences: Shape (num_samples, sequence_length, num_features)
            labels: Shape (num_samples,)
            validation_split: Tỷ lệ validation
            verbose: In thông tin
            optimize_threshold: Tự động tối ưu threshold sau khi train (NEW)
            threshold_strategy: Chiến lược tối ưu threshold (NEW)
                - 'balanced': Recall >= 70%, Precision >= 5%
                - 'recall_focused': Recall >= 80%, Precision >= 3%
                - 'precision_focused': Recall >= 50%, Precision >= 10%
                - 'fpr_controlled': FPR <= 30%, Recall >= 60%
        """
        if verbose:
            print("[LSTM] Bắt đầu training...")
            print(f"  - Số sequences: {sequences.shape[0]:,}")
            print(f"  - Sequence length: {sequences.shape[1]}")
            print(f"  - Num features: {sequences.shape[2]}")
            print(f"  - Fraud ratio: {labels.mean()*100:.2f}%")
            print(f"  - Device: {device}")

        # Reshape để scale
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, original_shape[2])
        sequences_scaled = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_scaled.reshape(original_shape)

        # Chia train/validation
        n_samples = len(sequences)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        X_train = sequences[indices[n_val:]]
        y_train = labels[indices[n_val:]]
        X_val = sequences[indices[:n_val]]
        y_val = labels[indices[:n_val]]

        # DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 128),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 128)
        )

        # Model
        input_size = sequences.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config.get('hidden_size', 64),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2)
        ).to(device)

        # Optimizer và loss (với class weights cho imbalanced data)
        pos_weight = torch.tensor([(1 - labels.mean()) / labels.mean()]).to(device)
        criterion = nn.BCELoss()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )

        # Training
        epochs = self.config.get('epochs', 50)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += ((output >= 0.5) == batch_y).sum().item()
                train_total += len(batch_y)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)

                    val_loss += loss.item()
                    val_correct += ((output >= 0.5) == batch_y).sum().item()
                    val_total += len(batch_y)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

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

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        self.is_fitted = True
        if verbose:
            print("[LSTM] Training hoàn tất!")

        # ========================================================
        # THRESHOLD OPTIMIZATION (NEW FEATURE)
        # ========================================================
        if optimize_threshold:
            if verbose:
                print("\n" + "=" * 60)
                print("[LSTM] BẮT ĐẦU THRESHOLD OPTIMIZATION")
                print("=" * 60)

            # Tối ưu threshold trên validation set
            self._run_threshold_optimization(
                X_val=X_val,
                y_val=y_val,
                strategy=threshold_strategy,
                verbose=verbose
            )

    def predict(self, sequences: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán fraud cho sequences

        Args:
            sequences: Shape (num_samples, sequence_length, num_features)
            threshold: Ngưỡng quyết định

        Returns:
            np.ndarray: 0 = normal, 1 = fraud
        """
        proba = self.predict_proba(sequences)
        return (proba >= threshold).astype(int)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """
        Tính xác suất fraud

        Args:
            sequences: Shape (num_samples, sequence_length, num_features)

        Returns:
            np.ndarray: Xác suất fraud
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        # Scale
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, original_shape[2])
        sequences_scaled = self.scaler.transform(sequences_flat)
        sequences = sequences_scaled.reshape(original_shape)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sequences).to(device)
            proba = self.model(X_tensor).cpu().numpy()

        return proba

    def evaluate(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá model

        Args:
            sequences: Test sequences
            labels: Test labels
            verbose: In kết quả

        Returns:
            Dict metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        y_pred = self.predict(sequences)
        y_proba = self.predict_proba(sequences)

        metrics = {
            'accuracy': accuracy_score(labels, y_pred),
            'precision': precision_score(labels, y_pred, zero_division=0),
            'recall': recall_score(labels, y_pred, zero_division=0),
            'f1_score': f1_score(labels, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(labels, y_proba),
            'confusion_matrix': confusion_matrix(labels, y_pred).tolist()
        }

        if verbose:
            print("\n[LSTM] Kết quả đánh giá:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def save(self, path: str = None):
        """Lưu model và threshold config"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'lstm.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history,
            'is_fitted': self.is_fitted,
            # Threshold optimization data (NEW)
            'optimal_threshold': self.optimal_threshold,
            'threshold_config': self.threshold_config
        }

        torch.save(save_data, path)
        print(f"[LSTM] Đã lưu model: {path}")

        # Lưu threshold config riêng (JSON)
        if self.threshold_config is not None:
            threshold_path = path.replace('.pth', '_threshold.json')
            with open(threshold_path, 'w', encoding='utf-8') as f:
                json.dump(self.threshold_config, f, indent=2, ensure_ascii=False)
            print(f"[LSTM] Đã lưu threshold config: {threshold_path}")

    def load(self, path: str = None):
        """Load model và threshold config"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'lstm.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.scaler = save_data['scaler']
        self.config = save_data['config']
        self.history = save_data['history']
        self.is_fitted = save_data['is_fitted']

        # Load threshold config (NEW)
        self.optimal_threshold = save_data.get('optimal_threshold', 0.5)
        self.threshold_config = save_data.get('threshold_config', None)

        # Rebuild model (cần biết input_size)
        # Sẽ được set lại khi sử dụng
        print(f"[LSTM] Đã load model: {path}")
        print(f"[LSTM] Optimal threshold: {self.optimal_threshold:.4f}")

    def explain_prediction(
        self,
        sequences: np.ndarray,
        sample_idx: int = 0
    ) -> Dict:
        """
        Giải thích dự đoán với attention weights

        Args:
            sequences: Input sequences
            sample_idx: Index cần giải thích

        Returns:
            Dict giải thích
        """
        sample = sequences[sample_idx:sample_idx + 1]

        # Scale
        original_shape = sample.shape
        sample_flat = sample.reshape(-1, original_shape[2])
        sample_scaled = self.scaler.transform(sample_flat)
        sample = sample_scaled.reshape(original_shape)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sample).to(device)
            proba = self.model(X_tensor).cpu().numpy()[0]
            attention = self.model.get_attention_weights(X_tensor)[0]

        return {
            'fraud_probability': float(proba),
            'prediction': 'fraud' if proba >= self.optimal_threshold else 'normal',
            'attention_weights': attention.tolist(),
            'most_important_positions': np.argsort(attention)[::-1][:3].tolist(),
            'sequence_length': len(attention)
        }

    # ========================================================================
    # THRESHOLD OPTIMIZATION METHODS (NEW)
    # ========================================================================

    def _run_threshold_optimization(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        strategy: str = 'balanced',
        verbose: bool = True
    ):
        """
        Chạy threshold optimization trên validation set

        Args:
            X_val: Validation sequences (đã scale)
            y_val: Validation labels
            strategy: Chiến lược tối ưu
            verbose: In chi tiết
        """
        optimizer = _get_threshold_optimizer()

        if optimizer is None:
            logger.warning("[LSTM] Threshold optimizer not available. Using default threshold 0.5")
            return

        # Predict probabilities trên validation set
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(device)
            y_pred_prob = self.model(X_tensor).cpu().numpy()

        if verbose:
            print(f"\n[THRESHOLD] Đang tối ưu threshold trên {len(y_val):,} validation samples...")
            print(f"  - Fraud ratio: {y_val.mean()*100:.2f}%")
            print(f"  - Prob range: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")
            print(f"  - Strategy: {strategy}")

        # Tính ROC-AUC và AP
        fpr, tpr, _, auc = optimizer['compute_roc_curve'](y_val, y_pred_prob)
        prec, rec, _, ap = optimizer['compute_precision_recall_curve'](y_val, y_pred_prob)

        if verbose:
            print(f"\n  Model Performance:")
            print(f"    - ROC-AUC: {auc:.4f}")
            print(f"    - Average Precision: {ap:.4f}")

        # Log threshold analysis
        if verbose:
            print("\n" + "-" * 50)
            print("  PHÂN TÍCH THRESHOLD CHI TIẾT")
            print("-" * 50)
            optimizer['log_threshold_analysis'](y_val, y_pred_prob)

        # Recommend threshold
        result = optimizer['recommend_threshold'](y_val, y_pred_prob, strategy=strategy)

        self.threshold_optimizer_result = result
        self.optimal_threshold = result['recommended_threshold']

        if verbose:
            print("\n" + "-" * 50)
            print("  THRESHOLD ĐỀ XUẤT")
            print("-" * 50)
            print(f"\n  ✅ Recommended Threshold: {self.optimal_threshold:.4f}")

            if result['metrics'] is not None:
                m = result['metrics']
                print(f"     - Recall: {m.recall:.2%}")
                print(f"     - Precision: {m.precision:.2%}")
                print(f"     - F1: {m.f1:.4f}")
                print(f"     - FPR: {m.fpr:.2%}")

            if result['warnings']:
                print("\n  ⚠️ Warnings:")
                for warn in result['warnings']:
                    print(f"     - {warn}")

        # Print summary table
        if verbose:
            print("\n" + "-" * 50)
            print("  BẢNG SUMMARY 3 MỨC THRESHOLD")
            print("-" * 50)
            optimizer['print_summary_table'](y_val, y_pred_prob)

        # Tier distribution
        tier_dist = optimizer['get_tier_distribution'](y_pred_prob, y_val)

        if verbose:
            print("\n  PHÂN BỔ THEO TIER:")
            print(tier_dist.to_string(index=False))

        # Lưu threshold config
        self.threshold_config = {
            'optimal_threshold': self.optimal_threshold,
            'strategy': strategy,
            'roc_auc': auc,
            'average_precision': ap,
            'thresholds': optimizer['DEFAULT_THRESHOLDS'],
            'tier_distribution': tier_dist.to_dict('records')
        }

        if result['metrics'] is not None:
            self.threshold_config['metrics'] = result['metrics'].to_dict()

        if verbose:
            print("\n" + "=" * 60)
            print("[LSTM] THRESHOLD OPTIMIZATION HOÀN TẤT!")
            print(f"       Optimal threshold: {self.optimal_threshold:.4f}")
            print("=" * 60)

    def get_threshold_config(self) -> Optional[Dict]:
        """
        Lấy threshold config đã optimize

        Returns:
            Dict threshold config hoặc None nếu chưa optimize
        """
        return self.threshold_config

    def set_threshold(self, threshold: float):
        """
        Set threshold thủ công

        Args:
            threshold: Giá trị threshold mới (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold phải trong khoảng [0, 1], got {threshold}")

        self.optimal_threshold = threshold
        logger.info(f"[LSTM] Threshold set to: {threshold:.4f}")

    def classify_with_tiers(
        self,
        sequences: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Phân loại giao dịch vào các risk tiers

        Args:
            sequences: Input sequences

        Returns:
            Tuple (risk_tiers array, DataFrame chi tiết)
        """
        optimizer = _get_threshold_optimizer()

        if optimizer is None:
            raise RuntimeError("Threshold optimizer not available")

        proba = self.predict_proba(sequences)

        risk_tiers, details = optimizer['classify_transactions'](
            proba,
            return_details=True
        )

        return risk_tiers, details

    def get_production_classifier(self) -> object:
        """
        Lấy FraudThresholdClassifier để dùng trong production

        Returns:
            FraudThresholdClassifier instance
        """
        optimizer = _get_threshold_optimizer()

        if optimizer is None:
            raise RuntimeError("Threshold optimizer not available")

        return optimizer['FraudThresholdClassifier'](
            thresholds=optimizer['DEFAULT_THRESHOLDS'],
            default_threshold=self.optimal_threshold
        )

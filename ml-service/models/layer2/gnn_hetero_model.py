"""
GNN Heterogeneous Model - Graph Neural Network cho Edge-Level Fraud Detection
==============================================================================
Model GNN sử dụng Heterogeneous Graph Transformer (HGT) để phát hiện
giao dịch lừa đảo ở mức edge (edge-level classification).

Kiến trúc:
- HGTConv layers cho message passing trên heterogeneous graph
- Edge-level prediction với link features
- Hỗ trợ multiple node types và edge types

Author: Senior ML Engineer
Version: 2.0.0
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.nn import MessagePassing

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EdgePredictor(nn.Module):
    """
    Edge-level predictor sử dụng node embeddings và edge features

    Dự đoán fraud probability cho mỗi edge dựa trên:
    - Source node embedding
    - Destination node embedding
    - Edge features (nếu có)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Khởi tạo Edge Predictor

        Args:
            node_dim: Kích thước node embedding
            edge_dim: Kích thước edge features (0 nếu không có)
            hidden_dim: Kích thước hidden layer
            dropout: Dropout rate
        """
        super(EdgePredictor, self).__init__()

        # Input: concat(src_emb, dst_emb, edge_features)
        input_dim = node_dim * 2 + edge_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            src_emb: Source node embeddings (num_edges, node_dim)
            dst_emb: Destination node embeddings (num_edges, node_dim)
            edge_attr: Edge features (num_edges, edge_dim) - optional

        Returns:
            Fraud probabilities (num_edges,)
        """
        # Concatenate features
        if edge_attr is not None:
            x = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        else:
            x = torch.cat([src_emb, dst_emb], dim=1)

        return self.predictor(x).squeeze(-1)


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN Encoder sử dụng HeteroConv

    Xử lý multiple node types và edge types để tạo node embeddings
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        input_dims: Dict[str, int],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4
    ):
        """
        Khởi tạo Hetero GNN Encoder

        Args:
            node_types: Danh sách các node types
            edge_types: Danh sách các edge types (src, edge, dst)
            input_dims: Dict mapping node_type -> input dimension
            hidden_dim: Hidden dimension cho GNN layers
            num_layers: Số GNN layers
            dropout: Dropout rate
            heads: Số attention heads (cho GAT)
        """
        super(HeteroGNNEncoder, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Linear projections để map input features sang hidden_dim
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            in_dim = input_dims.get(node_type, 1)
            self.input_projections[node_type] = nn.Linear(in_dim, hidden_dim)

        # HeteroConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleDict()

        for i in range(num_layers):
            # Tạo convolution cho mỗi edge type
            conv_dict = {}
            for edge_type in edge_types:
                # Sử dụng SAGEConv (đơn giản và hiệu quả)
                conv_dict[edge_type] = SAGEConv(
                    hidden_dim if i > 0 else hidden_dim,
                    hidden_dim,
                    aggr='mean'
                )

            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)

            # BatchNorm cho mỗi node type
            for node_type in node_types:
                self.bns[f'{node_type}_{i}'] = nn.BatchNorm1d(hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x_dict: Dict mapping node_type -> node features
            edge_index_dict: Dict mapping edge_type -> edge_index

        Returns:
            Dict mapping node_type -> node embeddings
        """
        # Project input features
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                h_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
            else:
                # Placeholder nếu không có features
                h_dict[node_type] = torch.zeros(1, self.hidden_dim).to(device)

        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)

            # Apply batch norm và activation
            for node_type in h_dict_new:
                bn_key = f'{node_type}_{i}'
                if bn_key in self.bns and h_dict_new[node_type].shape[0] > 1:
                    h_dict_new[node_type] = self.bns[bn_key](h_dict_new[node_type])
                h_dict_new[node_type] = F.relu(h_dict_new[node_type])
                h_dict_new[node_type] = F.dropout(
                    h_dict_new[node_type],
                    p=self.dropout,
                    training=self.training
                )

            # Residual connection
            for node_type in h_dict_new:
                if node_type in h_dict:
                    h_dict_new[node_type] = h_dict_new[node_type] + h_dict[node_type]

            h_dict = h_dict_new

        return h_dict


class HeteroGNNFraudDetector(nn.Module):
    """
    Complete Heterogeneous GNN Model cho Edge-Level Fraud Detection

    Kết hợp:
    - HeteroGNNEncoder: Tạo node embeddings
    - EdgePredictor: Dự đoán fraud cho edges
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        input_dims: Dict[str, int],
        edge_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Khởi tạo model

        Args:
            node_types: Danh sách node types
            edge_types: Danh sách edge types
            input_dims: Dict mapping node_type -> input dimension
            edge_dim: Số edge features
            hidden_dim: Hidden dimension
            num_layers: Số GNN layers
            dropout: Dropout rate
        """
        super(HeteroGNNFraudDetector, self).__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim

        # GNN Encoder
        self.encoder = HeteroGNNEncoder(
            node_types=node_types,
            edge_types=edge_types,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Edge Predictor
        self.edge_predictor = EdgePredictor(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        target_edge_type: Tuple[str, str, str],
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x_dict: Node features dict
            edge_index_dict: Edge indices dict
            target_edge_type: Edge type để predict (e.g., ('user', 'transfer', 'recipient'))
            edge_attr: Edge features cho target edge type

        Returns:
            Fraud probabilities cho target edges
        """
        # Get node embeddings
        node_embeddings = self.encoder(x_dict, edge_index_dict)

        # Get source and destination node types
        src_type, _, dst_type = target_edge_type
        edge_index = edge_index_dict[target_edge_type]

        # Get embeddings for source and destination nodes
        src_emb = node_embeddings[src_type][edge_index[0]]
        dst_emb = node_embeddings[dst_type][edge_index[1]]

        # Predict fraud probability
        predictions = self.edge_predictor(src_emb, dst_emb, edge_attr)

        return predictions

    def get_node_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Lấy node embeddings sau GNN encoding"""
        return self.encoder(x_dict, edge_index_dict)


class GNNHeteroTrainer:
    """
    Trainer class cho Heterogeneous GNN Fraud Detection

    Cung cấp:
    - Training loop với early stopping
    - Validation và evaluation
    - Threshold optimization
    - Model saving/loading
    """

    def __init__(
        self,
        model_config: Dict = None,
        verbose: bool = True
    ):
        """
        Khởi tạo trainer

        Args:
            model_config: Dict cấu hình model
            verbose: In thông tin chi tiết
        """
        self.config = model_config or config.GNN_CONFIG.copy()
        self.verbose = verbose
        self.model = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': []
        }
        self.best_threshold = 0.5
        self.is_fitted = False

    def log(self, message: str):
        """In log message"""
        if self.verbose:
            print(f"[GNN Trainer] {message}")

    def fit(
        self,
        data: HeteroData,
        target_edge_type: Tuple[str, str, str] = None,
        epochs: int = None,
        learning_rate: float = None,
        patience: int = 15
    ) -> Dict[str, Any]:
        """
        Train model

        Args:
            data: HeteroData object
            target_edge_type: Edge type có labels (để predict)
            epochs: Số epochs
            learning_rate: Learning rate
            patience: Patience cho early stopping

        Returns:
            Dict chứa training results
        """
        self.log("Bắt đầu training...")

        # Parse config
        epochs = epochs or self.config.get('epochs', 100)
        learning_rate = learning_rate or self.config.get('learning_rate', 0.01)
        hidden_dim = self.config.get('hidden_channels', 64)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.3)

        # Tìm target edge type nếu không được chỉ định
        if target_edge_type is None:
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'y'):
                    target_edge_type = edge_type
                    break

        if target_edge_type is None:
            raise ValueError("Không tìm thấy edge type có labels!")

        self.log(f"Target edge type: {target_edge_type}")

        # Lấy thông tin về graph
        node_types = list(data.node_types)
        edge_types = list(data.edge_types)

        # Lấy input dimensions cho mỗi node type
        input_dims = {}
        for node_type in node_types:
            if hasattr(data[node_type], 'x'):
                input_dims[node_type] = data[node_type].x.shape[1]
            else:
                input_dims[node_type] = 1

        # Lấy edge feature dimension
        edge_dim = 0
        if hasattr(data[target_edge_type], 'edge_attr') and data[target_edge_type].edge_attr is not None:
            edge_dim = data[target_edge_type].edge_attr.shape[1]

        self.log(f"Node types: {node_types}")
        self.log(f"Edge types: {edge_types}")
        self.log(f"Input dims: {input_dims}")
        self.log(f"Edge dim: {edge_dim}")

        # Tạo model
        self.model = HeteroGNNFraudDetector(
            node_types=node_types,
            edge_types=edge_types,
            input_dims=input_dims,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # Optimizer và scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=5e-4
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Chuẩn bị dữ liệu
        x_dict = {nt: data[nt].x.to(device) for nt in node_types if hasattr(data[nt], 'x')}
        edge_index_dict = {et: data[et].edge_index.to(device) for et in edge_types}

        edge_attr = None
        if hasattr(data[target_edge_type], 'edge_attr') and data[target_edge_type].edge_attr is not None:
            edge_attr = data[target_edge_type].edge_attr.to(device)

        labels = data[target_edge_type].y.float().to(device)

        # Masks
        train_mask = data[target_edge_type].train_mask.to(device) if hasattr(data[target_edge_type], 'train_mask') else None
        val_mask = data[target_edge_type].val_mask.to(device) if hasattr(data[target_edge_type], 'val_mask') else None

        # Nếu không có masks, tạo random split
        if train_mask is None or val_mask is None:
            num_edges = labels.shape[0]
            indices = torch.randperm(num_edges)
            n_val = int(num_edges * 0.15)
            n_test = int(num_edges * 0.15)

            train_mask = torch.zeros(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)

            train_mask[indices[n_val + n_test:]] = True
            val_mask[indices[:n_val]] = True

            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)

        # Class weights cho imbalanced data
        fraud_ratio = labels[train_mask].mean().item()
        pos_weight = torch.tensor([(1 - fraud_ratio) / max(fraud_ratio, 1e-6)]).to(device)
        criterion = nn.BCELoss()

        self.log(f"Fraud ratio: {fraud_ratio*100:.2f}%")
        self.log(f"Train samples: {train_mask.sum().item()}, Val samples: {val_mask.sum().item()}")

        # Training loop
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()

            predictions = self.model(x_dict, edge_index_dict, target_edge_type, edge_attr)
            loss = criterion(predictions[train_mask], labels[train_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(x_dict, edge_index_dict, target_edge_type, edge_attr)

                train_loss = criterion(predictions[train_mask], labels[train_mask]).item()
                val_loss = criterion(predictions[val_mask], labels[val_mask]).item()

                # Metrics
                train_pred = predictions[train_mask].cpu().numpy()
                train_labels = labels[train_mask].cpu().numpy()
                val_pred = predictions[val_mask].cpu().numpy()
                val_labels = labels[val_mask].cpu().numpy()

                train_auc = roc_auc_score(train_labels, train_pred) if len(np.unique(train_labels)) > 1 else 0.5
                val_auc = roc_auc_score(val_labels, val_pred) if len(np.unique(val_labels)) > 1 else 0.5

                train_f1 = f1_score(train_labels, (train_pred >= 0.5).astype(int), zero_division=0)
                val_f1 = f1_score(val_labels, (val_pred >= 0.5).astype(int), zero_division=0)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            # Learning rate scheduling
            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch + 1}")
                    break

            # Logging
            if (epoch + 1) % 10 == 0:
                self.log(f"Epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, "
                         f"auc={train_auc:.4f}, val_auc={val_auc:.4f}, val_f1={val_f1:.4f}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Optimize threshold
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_dict, edge_index_dict, target_edge_type, edge_attr)
            val_pred = predictions[val_mask].cpu().numpy()
            val_labels = labels[val_mask].cpu().numpy()
            self.best_threshold = self._optimize_threshold(val_labels, val_pred)

        self.is_fitted = True
        self.target_edge_type = target_edge_type

        self.log(f"Training hoàn tất! Best val AUC: {best_val_auc:.4f}, Optimal threshold: {self.best_threshold:.4f}")

        return {
            'best_val_auc': best_val_auc,
            'best_threshold': self.best_threshold,
            'epochs_trained': epoch + 1,
            'history': self.history
        }

    def _optimize_threshold(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Tìm threshold tối ưu dựa trên F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)

        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return 0.5

    def evaluate(
        self,
        data: HeteroData,
        mask_type: str = 'test'
    ) -> Dict[str, Any]:
        """
        Đánh giá model

        Args:
            data: HeteroData object
            mask_type: 'train', 'val', hoặc 'test'

        Returns:
            Dict chứa metrics
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        self.model.eval()

        # Chuẩn bị dữ liệu
        node_types = list(data.node_types)
        edge_types = list(data.edge_types)

        x_dict = {nt: data[nt].x.to(device) for nt in node_types if hasattr(data[nt], 'x')}
        edge_index_dict = {et: data[et].edge_index.to(device) for et in edge_types}

        edge_attr = None
        if hasattr(data[self.target_edge_type], 'edge_attr') and data[self.target_edge_type].edge_attr is not None:
            edge_attr = data[self.target_edge_type].edge_attr.to(device)

        labels = data[self.target_edge_type].y.float().to(device)

        # Get mask
        mask_attr = f'{mask_type}_mask'
        if hasattr(data[self.target_edge_type], mask_attr):
            mask = getattr(data[self.target_edge_type], mask_attr).to(device)
        else:
            mask = torch.ones(labels.shape[0], dtype=torch.bool).to(device)

        with torch.no_grad():
            predictions = self.model(x_dict, edge_index_dict, self.target_edge_type, edge_attr)

            y_true = labels[mask].cpu().numpy()
            y_pred_proba = predictions[mask].cpu().numpy()
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'average_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'threshold': self.best_threshold,
            'num_samples': int(mask.sum().item()),
            'fraud_ratio': float(y_true.mean())
        }

        return metrics

    def predict(
        self,
        data: HeteroData,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Dự đoán fraud cho tất cả target edges

        Args:
            data: HeteroData object
            return_proba: Trả về probability thay vì class

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        self.model.eval()

        node_types = list(data.node_types)
        edge_types = list(data.edge_types)

        x_dict = {nt: data[nt].x.to(device) for nt in node_types if hasattr(data[nt], 'x')}
        edge_index_dict = {et: data[et].edge_index.to(device) for et in edge_types}

        edge_attr = None
        if hasattr(data[self.target_edge_type], 'edge_attr') and data[self.target_edge_type].edge_attr is not None:
            edge_attr = data[self.target_edge_type].edge_attr.to(device)

        with torch.no_grad():
            predictions = self.model(x_dict, edge_index_dict, self.target_edge_type, edge_attr)
            predictions = predictions.cpu().numpy()

        if return_proba:
            return predictions
        return (predictions >= self.best_threshold).astype(int)

    def save(self, path: str = None):
        """Lưu model"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'node_types': self.model.node_types,
                'edge_types': self.model.edge_types,
                'hidden_dim': self.model.hidden_dim,
            },
            'config': self.config,
            'history': self.history,
            'best_threshold': self.best_threshold,
            'target_edge_type': self.target_edge_type,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }

        torch.save(save_data, path)
        self.log(f"Đã lưu model: {path}")

        return path

    def load(self, path: str = None, data: HeteroData = None):
        """
        Load model

        Args:
            path: Đường dẫn model
            data: HeteroData để khởi tạo lại model architecture
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.config = save_data['config']
        self.history = save_data['history']
        self.best_threshold = save_data['best_threshold']
        self.target_edge_type = save_data['target_edge_type']
        self.is_fitted = save_data['is_fitted']

        # Rebuild model if data provided
        if data is not None:
            node_types = list(data.node_types)
            edge_types = list(data.edge_types)

            input_dims = {}
            for node_type in node_types:
                if hasattr(data[node_type], 'x'):
                    input_dims[node_type] = data[node_type].x.shape[1]
                else:
                    input_dims[node_type] = 1

            edge_dim = 0
            if hasattr(data[self.target_edge_type], 'edge_attr') and data[self.target_edge_type].edge_attr is not None:
                edge_dim = data[self.target_edge_type].edge_attr.shape[1]

            self.model = HeteroGNNFraudDetector(
                node_types=node_types,
                edge_types=edge_types,
                input_dims=input_dims,
                edge_dim=edge_dim,
                hidden_dim=save_data['model_config']['hidden_dim'],
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.3)
            ).to(device)

            self.model.load_state_dict(save_data['model_state_dict'])

        self.log(f"Đã load model: {path}")


def train_gnn_model(
    data: HeteroData,
    model_config: Dict = None,
    save_path: str = None,
    verbose: bool = True
) -> Tuple[GNNHeteroTrainer, Dict]:
    """
    Hàm tiện ích để train GNN model

    Args:
        data: HeteroData object
        model_config: Config cho model
        save_path: Đường dẫn lưu model
        verbose: In thông tin

    Returns:
        Tuple (trainer, metrics)
    """
    print("\n" + "=" * 60)
    print("🤖 HUẤN LUYỆN GNN FRAUD DETECTION MODEL")
    print("=" * 60 + "\n")

    # Tạo trainer
    trainer = GNNHeteroTrainer(model_config=model_config, verbose=verbose)

    # Train
    train_results = trainer.fit(data)

    # Evaluate
    print("\n📊 ĐÁNH GIÁ MODEL")
    print("-" * 40)

    metrics = {}
    for split in ['train', 'val', 'test']:
        try:
            split_metrics = trainer.evaluate(data, mask_type=split)
            metrics[split] = split_metrics
            print(f"\n{split.upper()} Set:")
            print(f"  Accuracy:  {split_metrics['accuracy']:.4f}")
            print(f"  Precision: {split_metrics['precision']:.4f}")
            print(f"  Recall:    {split_metrics['recall']:.4f}")
            print(f"  F1-Score:  {split_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {split_metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"  {split}: Không có dữ liệu ({e})")

    # Save model
    if save_path is None:
        save_path = os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth')

    trainer.save(save_path)

    print("\n" + "=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)

    return trainer, metrics


if __name__ == '__main__':
    print("GNN Heterogeneous Model for Edge-Level Fraud Detection")
    print("Run with Streamlit app or import as module")

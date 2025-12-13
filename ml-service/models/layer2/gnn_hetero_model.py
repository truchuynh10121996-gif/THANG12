"""
GNN Heterogeneous Model - Model GNN cho Edge-Level Fraud Detection
===================================================================
Model này sử dụng Heterogeneous Graph Neural Networks để phát hiện
fraud trên cạnh (edge-level classification).

Kiến trúc:
- HeteroConv với SAGEConv cho message passing
- Edge classifier với edge features + node embeddings
- Hỗ trợ nhiều node types và edge types

Target: Binary classification (fraud/normal) trên edge type "transfer"
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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN Encoder

    Sử dụng HeteroConv để xử lý nhiều loại nodes và edges.
    Mỗi edge type có một SAGEConv layer riêng.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Khởi tạo encoder

        Args:
            node_types: Danh sách node types
            edge_types: Danh sách edge types (src, rel, dst)
            in_channels_dict: Dict mapping node_type -> input dim
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Số GNN layers
            dropout: Dropout rate
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial projection layers cho mỗi node type
        self.projections = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_channels_dict.get(node_type, 16)
            self.projections[node_type] = nn.Sequential(
                Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # HeteroConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # Mỗi edge type có SAGEConv riêng
                if i == 0:
                    conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)
                elif i == num_layers - 1:
                    conv_dict[edge_type] = SAGEConv(hidden_channels, out_channels)
                else:
                    conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels)

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            # Batch normalization cho mỗi node type
            norm_dict = nn.ModuleDict()
            for node_type in node_types:
                if i == num_layers - 1:
                    norm_dict[node_type] = nn.BatchNorm1d(out_channels)
                else:
                    norm_dict[node_type] = nn.BatchNorm1d(hidden_channels)
            self.norms.append(norm_dict)

        # Output projection layers - đảm bảo tất cả node types có cùng output dimension
        # Dùng cho các node types không nhận messages ở layer cuối
        self.output_projections = nn.ModuleDict()
        for node_type in node_types:
            self.output_projections[node_type] = nn.Linear(hidden_channels, out_channels)

        self.out_channels = out_channels

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x_dict: Dict mapping node_type -> node features
            edge_index_dict: Dict mapping edge_type -> edge indices

        Returns:
            Dict mapping node_type -> node embeddings
        """
        # Project initial features
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type] is not None:
                h_dict[node_type] = self.projections[node_type](x_dict[node_type])

        # Filter edge_index_dict to only include edges where both src and dst have features
        # This prevents NoneType errors during message passing
        valid_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # Only include edge if both source and destination node types have features
            if src_type in h_dict and dst_type in h_dict:
                # Also validate that edge indices are within bounds
                if edge_index is not None and edge_index.numel() > 0:
                    src_num_nodes = h_dict[src_type].shape[0]
                    dst_num_nodes = h_dict[dst_type].shape[0]

                    # Check if indices are valid
                    max_src_idx = edge_index[0].max().item() if edge_index[0].numel() > 0 else -1
                    max_dst_idx = edge_index[1].max().item() if edge_index[1].numel() > 0 else -1

                    if max_src_idx < src_num_nodes and max_dst_idx < dst_num_nodes:
                        valid_edge_index_dict[edge_type] = edge_index

        # GNN layers
        for i, conv in enumerate(self.convs):
            # Save current h_dict to preserve node types that don't receive messages
            h_dict_prev = {k: v.clone() for k, v in h_dict.items()}

            # Run convolution only if there are valid edges
            if valid_edge_index_dict:
                h_dict_new = conv(h_dict, valid_edge_index_dict)
            else:
                h_dict_new = {}

            # Merge: keep node types from previous layer if not updated by conv
            # This ensures all node types remain in h_dict
            is_last_layer = (i == self.num_layers - 1)

            for node_type in self.node_types:
                if node_type in h_dict_new and h_dict_new[node_type] is not None:
                    h_dict[node_type] = h_dict_new[node_type]
                elif node_type in h_dict_prev:
                    # Node type didn't receive any messages
                    # Nếu là layer cuối, cần project về out_channels để đảm bảo dimension khớp
                    if is_last_layer and h_dict_prev[node_type].shape[-1] != self.out_channels:
                        h_dict[node_type] = self.output_projections[node_type](h_dict_prev[node_type])
                    else:
                        h_dict[node_type] = h_dict_prev[node_type]

            # Apply normalization và activation
            for node_type in list(h_dict.keys()):
                if h_dict[node_type] is not None:
                    if node_type in self.norms[i]:
                        # Only apply batch norm if tensor has correct size
                        expected_dim = self.norms[i][node_type].num_features
                        if h_dict[node_type].shape[-1] == expected_dim:
                            h_dict[node_type] = self.norms[i][node_type](h_dict[node_type])
                    h_dict[node_type] = F.relu(h_dict[node_type])
                    h_dict[node_type] = F.dropout(h_dict[node_type], p=self.dropout, training=self.training)

        return h_dict


class EdgeClassifier(nn.Module):
    """
    Edge Classifier

    Dự đoán fraud cho mỗi edge dựa trên:
    - Source node embedding
    - Destination node embedding
    - Edge features (nếu có)
    """

    def __init__(
        self,
        node_embedding_dim: int,
        edge_feature_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Khởi tạo classifier

        Args:
            node_embedding_dim: Dimension của node embeddings
            edge_feature_dim: Dimension của edge features (0 nếu không có)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Input: concat(src_embed, dst_embed, edge_features)
        # Hoặc: concat(src_embed, dst_embed, src_embed * dst_embed, |src_embed - dst_embed|)
        input_dim = node_embedding_dim * 4 + edge_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            src_embeddings: Source node embeddings (num_edges, embed_dim)
            dst_embeddings: Destination node embeddings (num_edges, embed_dim)
            edge_attr: Edge features (num_edges, edge_dim) - optional

        Returns:
            torch.Tensor: Fraud probabilities (num_edges,)
        """
        # Compute interaction features
        element_product = src_embeddings * dst_embeddings
        element_diff = torch.abs(src_embeddings - dst_embeddings)

        # Concatenate
        if edge_attr is not None:
            x = torch.cat([src_embeddings, dst_embeddings, element_product, element_diff, edge_attr], dim=1)
        else:
            x = torch.cat([src_embeddings, dst_embeddings, element_product, element_diff], dim=1)

        return self.classifier(x).squeeze(-1)


class GNNHeteroModel:
    """
    GNN Model cho Edge-Level Fraud Detection

    Workflow:
    1. Load heterogeneous graph đã build
    2. Encode nodes với HeteroGNN
    3. Classify edges với EdgeClassifier
    4. Train với cross-entropy loss + class weighting
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo model

        Args:
            model_config: Dict cấu hình
        """
        self.config = model_config or config.GNN_CONFIG.copy()
        self.encoder = None
        self.classifier = None
        self.is_fitted = False

        # Hyperparameters
        self.hidden_channels = self.config.get('hidden_channels', 64)
        self.out_channels = self.config.get('out_channels', 32)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.3)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        self.patience = self.config.get('patience', 15)

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

        # Target edge type
        self.target_edge_type = None

    def fit(
        self,
        data: HeteroData,
        target_edge_type: Tuple[str, str, str] = None,
        verbose: bool = True
    ):
        """
        Train model

        Args:
            data: HeteroData object với labels và masks
            target_edge_type: Edge type để classify (mặc định: tìm edge có 'transfer')
            verbose: In thông tin
        """
        if verbose:
            print("\n" + "="*60)
            print("[GNN-Hetero] BẮT ĐẦU TRAINING")
            print("="*60)
            print(f"  Device: {device}")

        # Di chuyển data lên device
        data = data.to(device)

        # Xác định target edge type
        if target_edge_type is None:
            for et in data.edge_types:
                if 'transfer' in et[1]:
                    target_edge_type = et
                    break
        self.target_edge_type = target_edge_type

        if target_edge_type is None:
            raise ValueError("Không tìm thấy edge type 'transfer' trong graph")

        if verbose:
            print(f"  Target edge type: {target_edge_type}")

        # Lấy thông tin nodes và edges
        node_types = list(data.node_types)
        edge_types = list(data.edge_types)

        in_channels_dict = {}
        for nt in node_types:
            in_channels_dict[nt] = data[nt].x.shape[1]

        if verbose:
            print(f"  Node types: {node_types}")
            print(f"  Edge types: {edge_types}")
            print(f"  Input channels: {in_channels_dict}")

        # Khởi tạo encoder
        self.encoder = HeteroGNNEncoder(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        # Lấy edge features nếu có
        edge_attr = data[target_edge_type].get('edge_attr', None)
        edge_feature_dim = edge_attr.shape[1] if edge_attr is not None else 0

        # Khởi tạo classifier
        self.classifier = EdgeClassifier(
            node_embedding_dim=self.out_channels,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.hidden_channels,
            dropout=self.dropout
        ).to(device)

        if verbose:
            total_params = sum(p.numel() for p in self.encoder.parameters()) + \
                          sum(p.numel() for p in self.classifier.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Optimizer
        optimizer = Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        # Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Loss với class weighting
        labels = data[target_edge_type].y
        train_mask = data[target_edge_type].train_mask
        val_mask = data[target_edge_type].val_mask

        # Tính class weights
        train_labels = labels[train_mask]
        pos_count = train_labels.sum().item()
        neg_count = len(train_labels) - pos_count

        if pos_count > 0:
            pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        else:
            pos_weight = torch.tensor([1.0]).to(device)

        criterion = nn.BCELoss(reduction='none')

        if verbose:
            print(f"\n  Training samples: {train_mask.sum().item()}")
            print(f"  Validation samples: {val_mask.sum().item()}")
            print(f"  Fraud ratio: {pos_count/(pos_count+neg_count)*100:.2f}%")
            print(f"  Positive weight: {pos_weight.item():.2f}")
            print(f"\n  Training for {self.epochs} epochs (patience={self.patience})...")
            print("-"*60)

        # Training loop
        best_val_auc = 0
        patience_counter = 0

        # Build x_dict - only include node types that have valid features
        x_dict = {}
        for nt in node_types:
            if hasattr(data[nt], 'x') and data[nt].x is not None:
                x_dict[nt] = data[nt].x
            else:
                if verbose:
                    print(f"  ⚠️ Warning: Node type '{nt}' has no features, skipping")

        # Validate that target edge type's source and destination have features
        src_type, _, dst_type = target_edge_type
        if src_type not in x_dict:
            raise ValueError(f"Source node type '{src_type}' has no features in data!")
        if dst_type not in x_dict:
            raise ValueError(f"Destination node type '{dst_type}' has no features in data!")

        # Build edge_index_dict - only include edge types with valid indices
        edge_index_dict = {}
        for et in edge_types:
            if hasattr(data[et], 'edge_index') and data[et].edge_index is not None:
                edge_index_dict[et] = data[et].edge_index

        edge_index = data[target_edge_type].edge_index

        for epoch in range(self.epochs):
            # === Training ===
            self.encoder.train()
            self.classifier.train()

            optimizer.zero_grad()

            # Forward
            h_dict = self.encoder(x_dict, edge_index_dict)

            # Lấy embeddings cho source và destination nodes
            src_type, _, dst_type = target_edge_type
            src_embeddings = h_dict[src_type][edge_index[0]]
            dst_embeddings = h_dict[dst_type][edge_index[1]]

            # Predict
            pred = self.classifier(src_embeddings, dst_embeddings, edge_attr)

            # Loss (chỉ trên train mask)
            train_pred = pred[train_mask]
            train_labels_float = labels[train_mask].float()

            # Weighted loss
            loss = criterion(train_pred, train_labels_float)
            weights = torch.where(train_labels_float == 1, pos_weight, torch.ones_like(pos_weight))
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()

            # === Evaluation ===
            self.encoder.eval()
            self.classifier.eval()

            with torch.no_grad():
                h_dict = self.encoder(x_dict, edge_index_dict)
                src_embeddings = h_dict[src_type][edge_index[0]]
                dst_embeddings = h_dict[dst_type][edge_index[1]]
                pred = self.classifier(src_embeddings, dst_embeddings, edge_attr)

                # Metrics
                train_pred_np = pred[train_mask].cpu().numpy()
                train_labels_np = labels[train_mask].cpu().numpy()

                val_pred_np = pred[val_mask].cpu().numpy()
                val_labels_np = labels[val_mask].cpu().numpy()

                train_loss = loss.item()
                val_loss = criterion(pred[val_mask], labels[val_mask].float()).mean().item()

                # AUC
                if len(np.unique(train_labels_np)) > 1:
                    train_auc = roc_auc_score(train_labels_np, train_pred_np)
                else:
                    train_auc = 0.5

                if len(np.unique(val_labels_np)) > 1:
                    val_auc = roc_auc_score(val_labels_np, val_pred_np)
                else:
                    val_auc = 0.5

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)

            # Scheduler
            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                self._best_encoder_state = self.encoder.state_dict()
                self._best_classifier_state = self.classifier.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Epoch {epoch+1}: Early stopping!")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}: loss={train_loss:.4f}, "
                      f"train_auc={train_auc:.4f}, val_auc={val_auc:.4f}")

        # Load best model
        if hasattr(self, '_best_encoder_state'):
            self.encoder.load_state_dict(self._best_encoder_state)
            self.classifier.load_state_dict(self._best_classifier_state)

        self.is_fitted = True

        if verbose:
            print("-"*60)
            print(f"[GNN-Hetero] Training hoàn tất! Best val AUC: {best_val_auc:.4f}")

    def predict(self, data: HeteroData, threshold: float = 0.5) -> np.ndarray:
        """
        Dự đoán fraud cho edges

        Args:
            data: HeteroData object
            threshold: Ngưỡng phân loại

        Returns:
            np.ndarray: Predictions (0/1)
        """
        proba = self.predict_proba(data)
        return (proba >= threshold).astype(int)

    def predict_proba(self, data: HeteroData) -> np.ndarray:
        """
        Tính xác suất fraud

        Args:
            data: HeteroData object

        Returns:
            np.ndarray: Fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        data = data.to(device)

        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            node_types = list(data.node_types)
            edge_types = list(data.edge_types)

            # Build x_dict - only include node types with valid features
            x_dict = {}
            for nt in node_types:
                if hasattr(data[nt], 'x') and data[nt].x is not None:
                    x_dict[nt] = data[nt].x

            # Build edge_index_dict - only include edge types with valid indices
            edge_index_dict = {}
            for et in edge_types:
                if hasattr(data[et], 'edge_index') and data[et].edge_index is not None:
                    edge_index_dict[et] = data[et].edge_index

            h_dict = self.encoder(x_dict, edge_index_dict)

            src_type, _, dst_type = self.target_edge_type
            edge_index = data[self.target_edge_type].edge_index

            src_embeddings = h_dict[src_type][edge_index[0]]
            dst_embeddings = h_dict[dst_type][edge_index[1]]

            edge_attr = data[self.target_edge_type].get('edge_attr', None)
            pred = self.classifier(src_embeddings, dst_embeddings, edge_attr)

        return pred.cpu().numpy()

    def evaluate(self, data: HeteroData, mask_type: str = 'test', verbose: bool = True) -> Dict:
        """
        Đánh giá model

        Args:
            data: HeteroData object
            mask_type: 'train', 'val', hoặc 'test'
            verbose: In kết quả

        Returns:
            Dict metrics
        """
        proba = self.predict_proba(data)
        pred = (proba >= 0.5).astype(int)

        # Lấy mask
        mask_attr = f'{mask_type}_mask'
        if hasattr(data[self.target_edge_type], mask_attr):
            mask = getattr(data[self.target_edge_type], mask_attr).cpu().numpy()
        else:
            mask = np.ones(len(proba), dtype=bool)

        labels = data[self.target_edge_type].y.cpu().numpy()

        # Filter by mask
        proba = proba[mask]
        pred = pred[mask]
        labels = labels[mask]

        # Compute metrics
        metrics = {
            'accuracy': float(accuracy_score(labels, pred)),
            'precision': float(precision_score(labels, pred, zero_division=0)),
            'recall': float(recall_score(labels, pred, zero_division=0)),
            'f1_score': float(f1_score(labels, pred, zero_division=0)),
        }

        if len(np.unique(labels)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(labels, proba))
        else:
            metrics['roc_auc'] = 0.5

        cm = confusion_matrix(labels, pred)
        metrics['confusion_matrix'] = cm.tolist()

        if verbose:
            print(f"\n[GNN-Hetero] Kết quả đánh giá ({mask_type}):")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        return metrics

    def save(self, path: str = None):
        """Lưu model"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'encoder_state': self.encoder.state_dict() if self.encoder else None,
            'classifier_state': self.classifier.state_dict() if self.classifier else None,
            'config': self.config,
            'history': self.history,
            'target_edge_type': self.target_edge_type,
            'is_fitted': self.is_fitted,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'saved_at': datetime.now().isoformat()
        }

        torch.save(save_data, path)
        print(f"[GNN-Hetero] Đã lưu model: {path}")

    def load(self, path: str = None, data: HeteroData = None):
        """
        Load model

        Args:
            path: Đường dẫn file model
            data: HeteroData để rebuild architecture
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.config = save_data['config']
        self.history = save_data['history']
        self.target_edge_type = save_data['target_edge_type']
        self.is_fitted = save_data['is_fitted']
        self.hidden_channels = save_data.get('hidden_channels', 64)
        self.out_channels = save_data.get('out_channels', 32)
        self.num_layers = save_data.get('num_layers', 2)
        self.dropout = save_data.get('dropout', 0.3)

        # Rebuild và load state nếu có data
        if data is not None and save_data['encoder_state'] is not None:
            node_types = list(data.node_types)
            edge_types = list(data.edge_types)

            in_channels_dict = {}
            for nt in node_types:
                in_channels_dict[nt] = data[nt].x.shape[1]

            self.encoder = HeteroGNNEncoder(
                node_types=node_types,
                edge_types=edge_types,
                in_channels_dict=in_channels_dict,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(device)

            edge_attr = data[self.target_edge_type].get('edge_attr', None)
            edge_feature_dim = edge_attr.shape[1] if edge_attr is not None else 0

            self.classifier = EdgeClassifier(
                node_embedding_dim=self.out_channels,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=self.hidden_channels,
                dropout=self.dropout
            ).to(device)

            self.encoder.load_state_dict(save_data['encoder_state'])
            self.classifier.load_state_dict(save_data['classifier_state'])

        print(f"[GNN-Hetero] Đã load model: {path}")

    def get_node_embeddings(self, data: HeteroData) -> Dict[str, np.ndarray]:
        """
        Lấy node embeddings

        Args:
            data: HeteroData object

        Returns:
            Dict mapping node_type -> embeddings array
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        data = data.to(device)

        self.encoder.eval()
        with torch.no_grad():
            node_types = list(data.node_types)
            edge_types = list(data.edge_types)

            # Build x_dict - only include node types with valid features
            x_dict = {}
            for nt in node_types:
                if hasattr(data[nt], 'x') and data[nt].x is not None:
                    x_dict[nt] = data[nt].x

            # Build edge_index_dict - only include edge types with valid indices
            edge_index_dict = {}
            for et in edge_types:
                if hasattr(data[et], 'edge_index') and data[et].edge_index is not None:
                    edge_index_dict[et] = data[et].edge_index

            h_dict = self.encoder(x_dict, edge_index_dict)

        return {nt: h_dict[nt].cpu().numpy() for nt in h_dict if h_dict[nt] is not None}


def train_gnn_hetero_from_saved_graph(verbose: bool = True) -> Dict:
    """
    Hàm tiện ích: Train GNN từ graph đã lưu

    Returns:
        Dict với kết quả training
    """
    from utils.gnn_data_pipeline import GNNDataPipeline

    # Kiểm tra graph đã ready chưa
    if not GNNDataPipeline.is_graph_ready():
        return {
            'success': False,
            'error': 'Graph chưa được build. Vui lòng chạy "Tạo mạng lưới GNN" trước.'
        }

    # Load graph
    data, metadata = GNNDataPipeline.load_saved_graph()

    if data is None:
        return {
            'success': False,
            'error': 'Không thể load graph đã lưu.'
        }

    # Train model
    model = GNNHeteroModel()
    model.fit(data, verbose=verbose)

    # Evaluate trên test set
    metrics = model.evaluate(data, mask_type='test', verbose=verbose)

    # Lưu model
    model.save()

    return {
        'success': True,
        'message': 'Training GNN thành công!',
        'metrics': metrics,
        'history': {
            'final_train_auc': model.history['train_auc'][-1] if model.history['train_auc'] else 0,
            'final_val_auc': model.history['val_auc'][-1] if model.history['val_auc'] else 0,
            'epochs_trained': len(model.history['train_loss'])
        }
    }


if __name__ == '__main__':
    # Test training
    result = train_gnn_hetero_from_saved_graph(verbose=True)
    print(f"\nKết quả: {result}")

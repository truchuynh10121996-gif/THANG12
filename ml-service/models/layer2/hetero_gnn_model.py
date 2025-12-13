"""
Heterogeneous GNN Model - Edge-Level Fraud Detection
=====================================================
Model GNN dị thể cho phát hiện gian lận ở mức edge (giao dịch).

Kiến trúc:
- Heterogeneous Graph với 4 loại node: user, recipient, device, ip
- 3 loại edge: transfer (có label), uses_device, uses_ip
- Edge-level classification: phân loại edge transfer là fraud hay normal

Ưu điểm:
- Capture được mối quan hệ phức tạp giữa các entity
- Phát hiện fraud patterns dựa trên network topology
- Kết hợp được edge features và node features

Author: ML Team - Agribank Vietnam
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
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
    Heterogeneous GNN Encoder sử dụng HeteroConv

    Mỗi layer HeteroConv chứa các SAGEConv cho từng loại edge.
    Output là node embeddings cho tất cả các loại node.
    """

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Khởi tạo HeteroGNN Encoder

        Args:
            node_feature_dims: Dict mapping node_type -> feature dimension
            hidden_channels: Số hidden channels
            num_layers: Số GNN layers
            dropout: Dropout rate
        """
        super(HeteroGNNEncoder, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Linear projections để chuyển node features về cùng dimension
        self.node_projections = nn.ModuleDict()
        for node_type, feat_dim in node_feature_dims.items():
            self.node_projections[node_type] = Linear(feat_dim, hidden_channels)

        # HeteroConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            # Định nghĩa convolution cho mỗi loại edge
            conv_dict = {
                # user -> recipient (transfer)
                ('user', 'transfer', 'recipient'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # recipient -> user (reverse transfer)
                ('recipient', 'rev_transfer', 'user'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # user -> device
                ('user', 'uses_device', 'device'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # device -> user (reverse)
                ('device', 'rev_uses_device', 'user'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # user -> ip
                ('user', 'uses_ip', 'ip'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # ip -> user (reverse)
                ('ip', 'rev_uses_ip', 'user'): SAGEConv(
                    hidden_channels, hidden_channels
                ),
            }

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            # Batch normalization cho mỗi node type
            bn_dict = nn.ModuleDict({
                'user': nn.BatchNorm1d(hidden_channels),
                'recipient': nn.BatchNorm1d(hidden_channels),
                'device': nn.BatchNorm1d(hidden_channels),
                'ip': nn.BatchNorm1d(hidden_channels),
            })
            self.bns.append(bn_dict)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x_dict: Dict mapping node_type -> node features
            edge_index_dict: Dict mapping edge_type -> edge_index

        Returns:
            Dict mapping node_type -> node embeddings
        """
        # Project node features to hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_projections:
                h_dict[node_type] = self.node_projections[node_type](x)
            else:
                h_dict[node_type] = x

        # Apply HeteroConv layers
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict)

            # Apply batch norm and activation
            h_dict_new = {}
            for node_type, h in h_dict.items():
                if node_type in self.bns[i] and h.shape[0] > 0:
                    h = self.bns[i][node_type](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_dict_new[node_type] = h
            h_dict = h_dict_new

        return h_dict


class EdgeClassifier(nn.Module):
    """
    Edge Classifier - Phân loại edge dựa trên embeddings của 2 node + edge features
    """

    def __init__(
        self,
        node_embed_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Khởi tạo Edge Classifier

        Args:
            node_embed_dim: Dimension của node embedding
            edge_feature_dim: Dimension của edge features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(EdgeClassifier, self).__init__()

        # Input: concat(src_embed, dst_embed, edge_features)
        input_dim = node_embed_dim * 2 + edge_feature_dim

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
        src_embed: torch.Tensor,
        dst_embed: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            src_embed: Source node embeddings (num_edges, embed_dim)
            dst_embed: Destination node embeddings (num_edges, embed_dim)
            edge_attr: Edge features (num_edges, edge_feat_dim)

        Returns:
            Predictions (num_edges,)
        """
        # Concatenate: [src_embed || dst_embed || edge_features]
        edge_repr = torch.cat([src_embed, dst_embed, edge_attr], dim=1)

        # Classify
        output = self.classifier(edge_repr)

        return output.squeeze(-1)


class HeteroGNNModel:
    """
    Heterogeneous GNN Model cho Edge-Level Fraud Detection

    Pipeline:
    1. Encode tất cả nodes với HeteroGNNEncoder
    2. Lấy embeddings của src và dst nodes cho transfer edges
    3. Kết hợp với edge features và classify
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo HeteroGNN Model

        Args:
            model_config: Dict cấu hình model
        """
        self.config = model_config or {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'weight_decay': 5e-4
        }

        self.encoder = None
        self.classifier = None
        self.is_fitted = False

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': []
        }

        self.node_feature_dims = None
        self.edge_feature_dim = None

    def _add_reverse_edges(self, data: HeteroData) -> HeteroData:
        """
        Thêm reverse edges để message passing được đối xứng

        Args:
            data: HeteroData gốc

        Returns:
            HeteroData với reverse edges
        """
        # Transfer: user -> recipient => recipient -> user
        if ('user', 'transfer', 'recipient') in data.edge_types:
            edge_index = data['user', 'transfer', 'recipient'].edge_index
            data['recipient', 'rev_transfer', 'user'].edge_index = edge_index.flip(0)

        # Uses device: user -> device => device -> user
        if ('user', 'uses_device', 'device') in data.edge_types:
            edge_index = data['user', 'uses_device', 'device'].edge_index
            data['device', 'rev_uses_device', 'user'].edge_index = edge_index.flip(0)

        # Uses IP: user -> ip => ip -> user
        if ('user', 'uses_ip', 'ip') in data.edge_types:
            edge_index = data['user', 'uses_ip', 'ip'].edge_index
            data['ip', 'rev_uses_ip', 'user'].edge_index = edge_index.flip(0)

        return data

    def fit(
        self,
        data: HeteroData,
        verbose: bool = True
    ) -> Dict:
        """
        Train HeteroGNN Model

        Args:
            data: HeteroData chứa graph và labels
            verbose: In thông tin training

        Returns:
            Dict metrics
        """
        if verbose:
            print("\n" + "=" * 60)
            print("[HeteroGNN] BẮT ĐẦU TRAINING")
            print("=" * 60)

        # Thêm reverse edges
        data = self._add_reverse_edges(data)

        # Lấy thông tin dimensions
        self.node_feature_dims = {
            node_type: data[node_type].x.shape[1]
            for node_type in data.node_types
            if hasattr(data[node_type], 'x')
        }

        transfer_key = ('user', 'transfer', 'recipient')
        self.edge_feature_dim = data[transfer_key].edge_attr.shape[1]

        if verbose:
            print(f"  Node feature dims: {self.node_feature_dims}")
            print(f"  Edge feature dim: {self.edge_feature_dim}")
            print(f"  Device: {device}")

        # Khởi tạo models
        self.encoder = HeteroGNNEncoder(
            node_feature_dims=self.node_feature_dims,
            hidden_channels=self.config.get('hidden_channels', 64),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.3)
        ).to(device)

        self.classifier = EdgeClassifier(
            node_embed_dim=self.config.get('hidden_channels', 64),
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=self.config.get('hidden_channels', 64),
            dropout=self.config.get('dropout', 0.3)
        ).to(device)

        # Move data to device
        data = data.to(device)

        # Lấy labels và masks
        labels = data[transfer_key].y.float()
        train_mask = data[transfer_key].train_mask
        val_mask = data[transfer_key].val_mask
        edge_attr = data[transfer_key].edge_attr
        edge_index = data[transfer_key].edge_index

        if verbose:
            train_fraud = labels[train_mask].sum().item()
            val_fraud = labels[val_mask].sum().item()
            print(f"  Train edges: {train_mask.sum().item()} ({train_fraud} fraud)")
            print(f"  Val edges: {val_mask.sum().item()} ({val_fraud} fraud)")

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 5e-4)
        )

        # Loss function với class weights (do imbalanced data)
        pos_weight = (1 - labels.mean()) / labels.mean() if labels.mean() > 0 else 1.0
        criterion = nn.BCELoss()

        # Training loop
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 15)
        best_val_auc = 0
        patience_counter = 0

        # Chuẩn bị x_dict và edge_index_dict
        x_dict = {
            node_type: data[node_type].x
            for node_type in data.node_types
            if hasattr(data[node_type], 'x')
        }

        edge_index_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index

        for epoch in range(epochs):
            # ========== TRAINING ==========
            self.encoder.train()
            self.classifier.train()
            optimizer.zero_grad()

            # Forward pass
            h_dict = self.encoder(x_dict, edge_index_dict)

            # Lấy embeddings cho transfer edges
            src_embed = h_dict['user'][edge_index[0]]
            dst_embed = h_dict['recipient'][edge_index[1]]

            # Classify
            output = self.classifier(src_embed, dst_embed, edge_attr)

            # Loss chỉ trên training edges
            loss = criterion(output[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            # ========== VALIDATION ==========
            self.encoder.eval()
            self.classifier.eval()

            with torch.no_grad():
                h_dict = self.encoder(x_dict, edge_index_dict)
                src_embed = h_dict['user'][edge_index[0]]
                dst_embed = h_dict['recipient'][edge_index[1]]
                output = self.classifier(src_embed, dst_embed, edge_attr)

                train_loss = criterion(output[train_mask], labels[train_mask]).item()
                val_loss = criterion(output[val_mask], labels[val_mask]).item()

                # Metrics
                train_pred = output[train_mask].cpu().numpy()
                train_true = labels[train_mask].cpu().numpy()
                val_pred = output[val_mask].cpu().numpy()
                val_true = labels[val_mask].cpu().numpy()

                try:
                    train_auc = roc_auc_score(train_true, train_pred)
                    val_auc = roc_auc_score(val_true, val_pred) if len(np.unique(val_true)) > 1 else 0.5

                    train_f1 = f1_score(train_true, (train_pred >= 0.5).astype(int))
                    val_f1 = f1_score(val_true, (val_pred >= 0.5).astype(int))
                except:
                    train_auc, val_auc = 0.5, 0.5
                    train_f1, val_f1 = 0, 0

            # Lưu history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, "
                      f"AUC={train_auc:.4f}, val_AUC={val_auc:.4f}, val_F1={val_f1:.4f}")

        self.is_fitted = True

        if verbose:
            print("=" * 60)
            print(f"[HeteroGNN] TRAINING HOÀN TẤT!")
            print(f"  Best Validation AUC: {best_val_auc:.4f}")
            print("=" * 60)

        return {
            'best_val_auc': best_val_auc,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'epochs_trained': len(self.history['train_loss'])
        }

    def predict(
        self,
        data: HeteroData,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Dự đoán fraud cho transfer edges

        Args:
            data: HeteroData chứa graph
            threshold: Ngưỡng quyết định

        Returns:
            np.ndarray: Predictions (0/1)
        """
        proba = self.predict_proba(data)
        return (proba >= threshold).astype(int)

    def predict_proba(self, data: HeteroData) -> np.ndarray:
        """
        Tính xác suất fraud cho transfer edges

        Args:
            data: HeteroData chứa graph

        Returns:
            np.ndarray: Fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train! Gọi fit() trước.")

        # Thêm reverse edges
        data = self._add_reverse_edges(data)
        data = data.to(device)

        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            # Chuẩn bị dữ liệu
            x_dict = {
                node_type: data[node_type].x
                for node_type in data.node_types
                if hasattr(data[node_type], 'x')
            }

            edge_index_dict = {}
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_index'):
                    edge_index_dict[edge_type] = data[edge_type].edge_index

            transfer_key = ('user', 'transfer', 'recipient')
            edge_index = data[transfer_key].edge_index
            edge_attr = data[transfer_key].edge_attr

            # Forward pass
            h_dict = self.encoder(x_dict, edge_index_dict)
            src_embed = h_dict['user'][edge_index[0]]
            dst_embed = h_dict['recipient'][edge_index[1]]
            output = self.classifier(src_embed, dst_embed, edge_attr)

            return output.cpu().numpy()

    def evaluate(
        self,
        data: HeteroData,
        test_mask: torch.Tensor = None,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá model trên test set

        Args:
            data: HeteroData chứa graph và labels
            test_mask: Mask cho test edges (nếu không có, lấy từ data)
            verbose: In kết quả

        Returns:
            Dict metrics
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        transfer_key = ('user', 'transfer', 'recipient')

        if test_mask is None:
            test_mask = data[transfer_key].test_mask

        labels = data[transfer_key].y.cpu().numpy()
        proba = self.predict_proba(data)
        pred = (proba >= 0.5).astype(int)

        # Lọc theo test_mask
        test_mask_np = test_mask.cpu().numpy()
        labels_test = labels[test_mask_np]
        proba_test = proba[test_mask_np]
        pred_test = pred[test_mask_np]

        # Tính metrics
        metrics = {
            'accuracy': accuracy_score(labels_test, pred_test),
            'precision': precision_score(labels_test, pred_test, zero_division=0),
            'recall': recall_score(labels_test, pred_test, zero_division=0),
            'f1_score': f1_score(labels_test, pred_test, zero_division=0),
            'roc_auc': roc_auc_score(labels_test, proba_test) if len(np.unique(labels_test)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(labels_test, pred_test).tolist()
        }

        if verbose:
            print("\n" + "=" * 60)
            print("[HeteroGNN] KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
            print("=" * 60)
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"\n  Confusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"    [[{cm[0][0]:5d}  {cm[0][1]:5d}]  <- Normal")
            print(f"     [{cm[1][0]:5d}  {cm[1][1]:5d}]] <- Fraud")
            print("=" * 60)

        return metrics

    def get_edge_embeddings(self, data: HeteroData) -> np.ndarray:
        """
        Lấy embeddings cho tất cả transfer edges

        Args:
            data: HeteroData chứa graph

        Returns:
            np.ndarray: Edge embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        data = self._add_reverse_edges(data)
        data = data.to(device)

        self.encoder.eval()

        with torch.no_grad():
            x_dict = {
                node_type: data[node_type].x
                for node_type in data.node_types
                if hasattr(data[node_type], 'x')
            }

            edge_index_dict = {}
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_index'):
                    edge_index_dict[edge_type] = data[edge_type].edge_index

            transfer_key = ('user', 'transfer', 'recipient')
            edge_index = data[transfer_key].edge_index
            edge_attr = data[transfer_key].edge_attr

            h_dict = self.encoder(x_dict, edge_index_dict)
            src_embed = h_dict['user'][edge_index[0]]
            dst_embed = h_dict['recipient'][edge_index[1]]

            # Concatenate src, dst, edge_attr as edge embedding
            edge_embed = torch.cat([src_embed, dst_embed, edge_attr], dim=1)

            return edge_embed.cpu().numpy()

    def save(self, path: str = None):
        """
        Lưu model

        Args:
            path: Đường dẫn lưu model
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'hetero_gnn.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'encoder_state_dict': self.encoder.state_dict() if self.encoder else None,
            'classifier_state_dict': self.classifier.state_dict() if self.classifier else None,
            'config': self.config,
            'history': self.history,
            'is_fitted': self.is_fitted,
            'node_feature_dims': self.node_feature_dims,
            'edge_feature_dim': self.edge_feature_dim,
            'saved_at': datetime.now().isoformat()
        }

        torch.save(save_data, path)
        print(f"[HeteroGNN] Đã lưu model: {path}")

        # Lưu training report
        report_path = path.replace('.pth', '_report.json')
        report = {
            'model_type': 'HeteroGNN',
            'task': 'edge_classification',
            'config': self.config,
            'final_metrics': {
                'train_auc': self.history['train_auc'][-1] if self.history['train_auc'] else None,
                'val_auc': self.history['val_auc'][-1] if self.history['val_auc'] else None,
                'train_f1': self.history['train_f1'][-1] if self.history['train_f1'] else None,
                'val_f1': self.history['val_f1'][-1] if self.history['val_f1'] else None,
            },
            'epochs_trained': len(self.history['train_loss']),
            'saved_at': datetime.now().isoformat()
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[HeteroGNN] Đã lưu training report: {report_path}")

    def load(self, path: str = None):
        """
        Load model

        Args:
            path: Đường dẫn file model
        """
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'hetero_gnn.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.config = save_data['config']
        self.history = save_data['history']
        self.is_fitted = save_data['is_fitted']
        self.node_feature_dims = save_data['node_feature_dims']
        self.edge_feature_dim = save_data['edge_feature_dim']

        # Khởi tạo lại encoder và classifier
        if self.node_feature_dims and self.edge_feature_dim:
            self.encoder = HeteroGNNEncoder(
                node_feature_dims=self.node_feature_dims,
                hidden_channels=self.config.get('hidden_channels', 64),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.3)
            ).to(device)

            self.classifier = EdgeClassifier(
                node_embed_dim=self.config.get('hidden_channels', 64),
                edge_feature_dim=self.edge_feature_dim,
                hidden_dim=self.config.get('hidden_channels', 64),
                dropout=self.config.get('dropout', 0.3)
            ).to(device)

            if save_data['encoder_state_dict']:
                self.encoder.load_state_dict(save_data['encoder_state_dict'])
            if save_data['classifier_state_dict']:
                self.classifier.load_state_dict(save_data['classifier_state_dict'])

        print(f"[HeteroGNN] Đã load model: {path}")


# ====================================================
# MAIN - Test model
# ====================================================
if __name__ == '__main__':
    print("Testing HeteroGNN Model...")
    print("Hãy chạy train_gnn.py để train model thực tế.")

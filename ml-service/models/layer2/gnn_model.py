"""
GNN Model - Graph Neural Network cho Fraud Detection
=====================================================
GNN phân tích mạng lưới quan hệ giữa các tài khoản để phát hiện
các cộng đồng lừa đảo và các patterns đáng ngờ.

Ưu điểm:
- Phát hiện được fraud rings/communities
- Capture được structural patterns
- Tốt cho relational data

Nhược điểm:
- Phức tạp để implement
- Cần xây dựng graph từ dữ liệu
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_config

config = get_config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNNNetwork(nn.Module):
    """
    Graph Neural Network sử dụng GraphSAGE convolutions

    Architecture:
    - 3 GraphSAGE layers
    - Global pooling
    - Classification layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        """
        Khởi tạo GNN network

        Args:
            input_dim: Số features của mỗi node
            hidden_channels: Số hidden channels
            num_layers: Số GNN layers
            dropout: Dropout rate
        """
        super(GNNNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connections (2, num_edges)
            batch: Batch assignment (num_nodes,)

        Returns:
            Predictions (num_nodes, 1) hoặc (batch_size, 1)
        """
        # GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Node-level predictions
        if batch is None:
            # Node classification
            output = self.classifier(x)
        else:
            # Graph classification (pool nodes)
            x = global_mean_pool(x, batch)
            output = self.classifier(x)

        return output.squeeze(-1)

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Lấy node embeddings sau GNN layers"""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        return x


class GNNModel:
    """
    GNN model cho graph-based fraud detection

    Phân tích cấu trúc graph để phát hiện:
    - Fraud rings (nhóm accounts lừa đảo)
    - Unusual connection patterns
    - Community anomalies
    """

    def __init__(self, model_config: Dict = None):
        """
        Khởi tạo GNN model

        Args:
            model_config: Dict cấu hình
        """
        self.config = model_config or config.GNN_CONFIG.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def fit(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray = None,
        val_mask: np.ndarray = None,
        verbose: bool = True
    ):
        """
        Train GNN model

        Args:
            node_features: Features của nodes (num_nodes, num_features)
            edge_index: Edge connections (2, num_edges)
            labels: Node labels (num_nodes,)
            train_mask: Boolean mask cho training nodes
            val_mask: Boolean mask cho validation nodes
            verbose: In thông tin
        """
        if verbose:
            print("[GNN] Bắt đầu training...")
            print(f"  - Số nodes: {node_features.shape[0]:,}")
            print(f"  - Số edges: {edge_index.shape[1]:,}")
            print(f"  - Num features: {node_features.shape[1]}")
            print(f"  - Fraud ratio: {labels.mean()*100:.2f}%")
            print(f"  - Device: {device}")

        # Chuẩn hóa features
        node_features_scaled = self.scaler.fit_transform(node_features)

        # Tạo masks nếu không có
        if train_mask is None or val_mask is None:
            n_nodes = len(labels)
            indices = np.random.permutation(n_nodes)
            n_val = int(n_nodes * 0.1)

            train_mask = np.zeros(n_nodes, dtype=bool)
            val_mask = np.zeros(n_nodes, dtype=bool)

            val_mask[indices[:n_val]] = True
            train_mask[indices[n_val:]] = True

        # Convert to PyTorch tensors
        x = torch.FloatTensor(node_features_scaled).to(device)
        edge_index = torch.LongTensor(edge_index).to(device)
        y = torch.FloatTensor(labels).to(device)
        train_mask = torch.BoolTensor(train_mask).to(device)
        val_mask = torch.BoolTensor(val_mask).to(device)

        # Tạo model
        self.model = GNNNetwork(
            input_dim=node_features.shape[1],
            hidden_channels=self.config.get('hidden_channels', 64),
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout', 0.3)
        ).to(device)

        # Optimizer và loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.01),
            weight_decay=5e-4
        )

        # Class weights cho imbalanced data
        pos_weight = (1 - labels.mean()) / labels.mean()
        criterion = nn.BCELoss()

        # Training
        epochs = self.config.get('epochs', 100)
        best_val_auc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            optimizer.zero_grad()

            output = self.model(x, edge_index)
            loss = criterion(output[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                output = self.model(x, edge_index)

                train_loss = criterion(output[train_mask], y[train_mask]).item()
                val_loss = criterion(output[val_mask], y[val_mask]).item()

                # AUC
                from sklearn.metrics import roc_auc_score
                train_auc = roc_auc_score(
                    y[train_mask].cpu().numpy(),
                    output[train_mask].cpu().numpy()
                )
                val_auc = roc_auc_score(
                    y[val_mask].cpu().numpy(),
                    output[val_mask].cpu().numpy()
                )

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)

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
                      f"auc={train_auc:.4f}, val_auc={val_auc:.4f}")

        self.is_fitted = True
        if verbose:
            print(f"[GNN] Training hoàn tất! Best val AUC: {best_val_auc:.4f}")

    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Dự đoán fraud cho nodes

        Args:
            node_features: Node features
            edge_index: Edge connections
            threshold: Ngưỡng quyết định

        Returns:
            np.ndarray: Predictions
        """
        proba = self.predict_proba(node_features, edge_index)
        return (proba >= threshold).astype(int)

    def predict_proba(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Tính xác suất fraud

        Args:
            node_features: Node features
            edge_index: Edge connections

        Returns:
            np.ndarray: Fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        node_features_scaled = self.scaler.transform(node_features)

        x = torch.FloatTensor(node_features_scaled).to(device)
        edge_index = torch.LongTensor(edge_index).to(device)

        self.model.eval()
        with torch.no_grad():
            proba = self.model(x, edge_index).cpu().numpy()

        return proba

    def get_node_embeddings(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Lấy node embeddings

        Args:
            node_features: Node features
            edge_index: Edge connections

        Returns:
            np.ndarray: Node embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        node_features_scaled = self.scaler.transform(node_features)

        x = torch.FloatTensor(node_features_scaled).to(device)
        edge_index = torch.LongTensor(edge_index).to(device)

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_node_embeddings(x, edge_index).cpu().numpy()

        return embeddings

    def evaluate(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
        test_mask: np.ndarray = None,
        verbose: bool = True
    ) -> Dict:
        """
        Đánh giá model

        Args:
            node_features: Node features
            edge_index: Edge connections
            labels: True labels
            test_mask: Mask cho test nodes
            verbose: In kết quả

        Returns:
            Dict metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        proba = self.predict_proba(node_features, edge_index)
        pred = (proba >= 0.5).astype(int)

        if test_mask is not None:
            labels = labels[test_mask]
            proba = proba[test_mask]
            pred = pred[test_mask]

        metrics = {
            'accuracy': accuracy_score(labels, pred),
            'precision': precision_score(labels, pred, zero_division=0),
            'recall': recall_score(labels, pred, zero_division=0),
            'f1_score': f1_score(labels, pred, zero_division=0),
            'roc_auc': roc_auc_score(labels, proba),
            'confusion_matrix': confusion_matrix(labels, pred).tolist()
        }

        if verbose:
            print("\n[GNN] Kết quả đánh giá:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def detect_communities(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray = None
    ) -> Dict:
        """
        Phát hiện fraud communities trong graph

        Args:
            node_features: Node features
            edge_index: Edge connections
            labels: True labels (optional)

        Returns:
            Dict chứa thông tin về communities
        """
        embeddings = self.get_node_embeddings(node_features, edge_index)
        proba = self.predict_proba(node_features, edge_index)

        # Simple community detection using clustering on embeddings
        from sklearn.cluster import KMeans

        n_clusters = min(10, len(embeddings) // 100)
        if n_clusters < 2:
            n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Analyze each cluster
        community_stats = []
        for i in range(n_clusters):
            mask = clusters == i
            cluster_proba = proba[mask]

            stats = {
                'cluster_id': i,
                'size': int(mask.sum()),
                'avg_fraud_prob': float(cluster_proba.mean()),
                'high_risk_count': int((cluster_proba >= 0.5).sum())
            }

            if labels is not None:
                cluster_labels = labels[mask]
                stats['actual_fraud_count'] = int(cluster_labels.sum())
                stats['fraud_ratio'] = float(cluster_labels.mean())

            community_stats.append(stats)

        # Sort by fraud probability
        community_stats.sort(key=lambda x: x['avg_fraud_prob'], reverse=True)

        return {
            'num_communities': n_clusters,
            'communities': community_stats,
            'cluster_assignments': clusters.tolist()
        }

    def save(self, path: str = None):
        """Lưu model"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn.pth')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history,
            'is_fitted': self.is_fitted
        }

        torch.save(save_data, path)
        print(f"[GNN] Đã lưu model: {path}")

    def load(self, path: str = None):
        """Load model"""
        if path is None:
            path = os.path.join(config.SAVED_MODELS_DIR, 'gnn.pth')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model: {path}")

        save_data = torch.load(path, map_location=device)

        self.scaler = save_data['scaler']
        self.config = save_data['config']
        self.history = save_data['history']
        self.is_fitted = save_data['is_fitted']

        print(f"[GNN] Đã load model: {path}")

    def explain_prediction(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        node_idx: int
    ) -> Dict:
        """
        Giải thích dự đoán cho một node

        Args:
            node_features: Node features
            edge_index: Edge connections
            node_idx: Index của node cần giải thích

        Returns:
            Dict giải thích
        """
        proba = self.predict_proba(node_features, edge_index)
        embeddings = self.get_node_embeddings(node_features, edge_index)

        # Tìm neighbors của node
        edges = edge_index.T
        neighbor_indices = edges[edges[:, 0] == node_idx][:, 1].tolist()
        neighbor_indices.extend(edges[edges[:, 1] == node_idx][:, 0].tolist())
        neighbor_indices = list(set(neighbor_indices))

        # Thống kê neighbors
        neighbor_proba = proba[neighbor_indices] if neighbor_indices else np.array([])

        return {
            'node_idx': node_idx,
            'fraud_probability': float(proba[node_idx]),
            'prediction': 'fraud' if proba[node_idx] >= 0.5 else 'normal',
            'embedding': embeddings[node_idx].tolist(),
            'num_neighbors': len(neighbor_indices),
            'neighbor_avg_fraud_prob': float(neighbor_proba.mean()) if len(neighbor_proba) > 0 else 0,
            'high_risk_neighbors': int((neighbor_proba >= 0.5).sum()) if len(neighbor_proba) > 0 else 0
        }

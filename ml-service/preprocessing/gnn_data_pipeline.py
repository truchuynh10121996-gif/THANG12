"""
GNN Data Pipeline - Xử lý dữ liệu cho Heterogeneous Graph
==========================================================
Pipeline này xử lý dữ liệu từ thư mục gnn_data/ và build graph dị thể
cho edge-level fraud detection.

Node types: user, recipient, device, ip
Edge types: transfer (có label), uses_device, uses_ip

Author: ML Team - Agribank Vietnam
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime

import torch
from torch_geometric.data import HeteroData

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn mặc định đến thư mục gnn_data
GNN_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gnn_data')


class GNNDataPipeline:
    """
    Pipeline xử lý dữ liệu và xây dựng Heterogeneous Graph cho GNN

    Workflow:
    1. Load tất cả file CSV/JSON từ gnn_data/
    2. Sanity check - kiểm tra tính toàn vẹn dữ liệu
    3. Build PyTorch Geometric HeteroData
    4. Lưu graph và flag file
    """

    def __init__(self, data_dir: str = None, verbose: bool = True):
        """
        Khởi tạo GNN Data Pipeline

        Args:
            data_dir: Đường dẫn đến thư mục chứa dữ liệu GNN
            verbose: In thông tin chi tiết
        """
        self.data_dir = data_dir or GNN_DATA_DIR
        self.verbose = verbose

        # DataFrames sẽ được load
        self.nodes_df = None
        self.edges_transfer_df = None
        self.edges_uses_device_df = None
        self.edges_uses_ip_df = None
        self.edge_labels_df = None
        self.splits_df = None
        self.metadata = None

        # Node mappings (node_id -> index trong mỗi loại)
        self.node_type_mappings = {
            'user': {},
            'recipient': {},
            'device': {},
            'ip': {}
        }

        # Graph data
        self.hetero_data = None

    def log(self, message: str, level: str = "INFO"):
        """In log nếu verbose mode"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def load_all_data(self) -> bool:
        """
        Load tất cả file dữ liệu từ thư mục gnn_data/

        Returns:
            bool: True nếu load thành công
        """
        self.log("=" * 60)
        self.log("BƯỚC 1: LOAD DỮ LIỆU TỪ GNN_DATA/")
        self.log("=" * 60)

        try:
            # 1. Load nodes.csv
            nodes_path = os.path.join(self.data_dir, 'nodes.csv')
            if not os.path.exists(nodes_path):
                raise FileNotFoundError(f"Không tìm thấy file: {nodes_path}")
            self.nodes_df = pd.read_csv(nodes_path)
            self.log(f"  ✓ Đã load nodes.csv: {len(self.nodes_df)} nodes")

            # 2. Load edges_transfer.csv
            edges_transfer_path = os.path.join(self.data_dir, 'edges_transfer.csv')
            if not os.path.exists(edges_transfer_path):
                raise FileNotFoundError(f"Không tìm thấy file: {edges_transfer_path}")
            self.edges_transfer_df = pd.read_csv(edges_transfer_path)
            self.log(f"  ✓ Đã load edges_transfer.csv: {len(self.edges_transfer_df)} edges")

            # 3. Load edges_uses_device.csv
            edges_device_path = os.path.join(self.data_dir, 'edges_uses_device.csv')
            if not os.path.exists(edges_device_path):
                raise FileNotFoundError(f"Không tìm thấy file: {edges_device_path}")
            self.edges_uses_device_df = pd.read_csv(edges_device_path)
            self.log(f"  ✓ Đã load edges_uses_device.csv: {len(self.edges_uses_device_df)} edges")

            # 4. Load edges_uses_ip.csv
            edges_ip_path = os.path.join(self.data_dir, 'edges_uses_ip.csv')
            if not os.path.exists(edges_ip_path):
                raise FileNotFoundError(f"Không tìm thấy file: {edges_ip_path}")
            self.edges_uses_ip_df = pd.read_csv(edges_ip_path)
            self.log(f"  ✓ Đã load edges_uses_ip.csv: {len(self.edges_uses_ip_df)} edges")

            # 5. Load edge_labels.csv
            labels_path = os.path.join(self.data_dir, 'edge_labels.csv')
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Không tìm thấy file: {labels_path}")
            self.edge_labels_df = pd.read_csv(labels_path)
            self.log(f"  ✓ Đã load edge_labels.csv: {len(self.edge_labels_df)} labels")

            # 6. Load splits.csv
            splits_path = os.path.join(self.data_dir, 'splits.csv')
            if not os.path.exists(splits_path):
                raise FileNotFoundError(f"Không tìm thấy file: {splits_path}")
            self.splits_df = pd.read_csv(splits_path)
            self.log(f"  ✓ Đã load splits.csv: {len(self.splits_df)} splits")

            # 7. Load metadata.json
            metadata_path = os.path.join(self.data_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.log(f"  ✓ Đã load metadata.json")
            else:
                self.log(f"  ⚠ Không tìm thấy metadata.json (optional)", "WARNING")
                self.metadata = {}

            self.log("✓ LOAD DỮ LIỆU THÀNH CÔNG!")
            return True

        except Exception as e:
            self.log(f"✗ LỖI KHI LOAD DỮ LIỆU: {str(e)}", "ERROR")
            raise

    def sanity_check(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính toàn vẹn của dữ liệu (SANITY CHECK)

        Kiểm tra:
        1. Mọi src_node_id, dst_node_id trong edges_* PHẢI tồn tại trong nodes.csv
        2. Mọi edge_id trong edge_labels.csv và splits.csv PHẢI match với edges_transfer.csv

        Returns:
            Tuple[bool, List[str]]: (passed, list of errors)
        """
        self.log("=" * 60)
        self.log("BƯỚC 2: SANITY CHECK - KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU")
        self.log("=" * 60)

        errors = []
        all_node_ids = set(self.nodes_df['node_id'].astype(str).values)

        # =============================================
        # CHECK 1: Kiểm tra node_id trong edges_transfer
        # =============================================
        self.log("  Kiểm tra edges_transfer.csv...")
        for idx, row in self.edges_transfer_df.iterrows():
            src_id = str(row['src_node_id'])
            dst_id = str(row['dst_node_id'])

            if src_id not in all_node_ids:
                errors.append(f"edges_transfer.csv dòng {idx+2}: src_node_id '{src_id}' KHÔNG tồn tại trong nodes.csv")

            if dst_id not in all_node_ids:
                errors.append(f"edges_transfer.csv dòng {idx+2}: dst_node_id '{dst_id}' KHÔNG tồn tại trong nodes.csv")

        if not errors:
            self.log("    ✓ Tất cả node_id trong edges_transfer đều hợp lệ")

        # =============================================
        # CHECK 2: Kiểm tra node_id trong edges_uses_device
        # =============================================
        self.log("  Kiểm tra edges_uses_device.csv...")
        for idx, row in self.edges_uses_device_df.iterrows():
            src_id = str(row['src_node_id'])
            device_id = str(row['device_node_id'])

            if src_id not in all_node_ids:
                errors.append(f"edges_uses_device.csv dòng {idx+2}: src_node_id '{src_id}' KHÔNG tồn tại trong nodes.csv")

            if device_id not in all_node_ids:
                errors.append(f"edges_uses_device.csv dòng {idx+2}: device_node_id '{device_id}' KHÔNG tồn tại trong nodes.csv")

        if len([e for e in errors if 'edges_uses_device' in e]) == 0:
            self.log("    ✓ Tất cả node_id trong edges_uses_device đều hợp lệ")

        # =============================================
        # CHECK 3: Kiểm tra node_id trong edges_uses_ip
        # =============================================
        self.log("  Kiểm tra edges_uses_ip.csv...")
        for idx, row in self.edges_uses_ip_df.iterrows():
            src_id = str(row['src_node_id'])
            ip_id = str(row['ip_node_id'])

            if src_id not in all_node_ids:
                errors.append(f"edges_uses_ip.csv dòng {idx+2}: src_node_id '{src_id}' KHÔNG tồn tại trong nodes.csv")

            if ip_id not in all_node_ids:
                errors.append(f"edges_uses_ip.csv dòng {idx+2}: ip_node_id '{ip_id}' KHÔNG tồn tại trong nodes.csv")

        if len([e for e in errors if 'edges_uses_ip' in e]) == 0:
            self.log("    ✓ Tất cả node_id trong edges_uses_ip đều hợp lệ")

        # =============================================
        # CHECK 4: Kiểm tra edge_id trong edge_labels.csv
        # =============================================
        self.log("  Kiểm tra edge_labels.csv...")
        transfer_edge_ids = set(self.edges_transfer_df['edge_id'].astype(str).values)

        for idx, row in self.edge_labels_df.iterrows():
            edge_id = str(row['edge_id'])

            if edge_id not in transfer_edge_ids:
                errors.append(f"edge_labels.csv dòng {idx+2}: edge_id '{edge_id}' KHÔNG tồn tại trong edges_transfer.csv")

        if len([e for e in errors if 'edge_labels' in e]) == 0:
            self.log("    ✓ Tất cả edge_id trong edge_labels đều hợp lệ")

        # =============================================
        # CHECK 5: Kiểm tra edge_id trong splits.csv
        # =============================================
        self.log("  Kiểm tra splits.csv...")
        for idx, row in self.splits_df.iterrows():
            edge_id = str(row['edge_id'])

            if edge_id not in transfer_edge_ids:
                errors.append(f"splits.csv dòng {idx+2}: edge_id '{edge_id}' KHÔNG tồn tại trong edges_transfer.csv")

        if len([e for e in errors if 'splits' in e]) == 0:
            self.log("    ✓ Tất cả edge_id trong splits đều hợp lệ")

        # =============================================
        # CHECK 6: Kiểm tra số lượng khớp nhau
        # =============================================
        self.log("  Kiểm tra số lượng records...")

        num_transfer_edges = len(self.edges_transfer_df)
        num_labels = len(self.edge_labels_df)
        num_splits = len(self.splits_df)

        if num_transfer_edges != num_labels:
            errors.append(f"Số edge_id trong edges_transfer ({num_transfer_edges}) KHÔNG khớp với edge_labels ({num_labels})")
        else:
            self.log(f"    ✓ Số lượng edges_transfer ({num_transfer_edges}) = edge_labels ({num_labels})")

        if num_transfer_edges != num_splits:
            errors.append(f"Số edge_id trong edges_transfer ({num_transfer_edges}) KHÔNG khớp với splits ({num_splits})")
        else:
            self.log(f"    ✓ Số lượng edges_transfer ({num_transfer_edges}) = splits ({num_splits})")

        # =============================================
        # KẾT QUẢ CUỐI CÙNG
        # =============================================
        if errors:
            self.log("=" * 60, "ERROR")
            self.log("✗ SANITY CHECK THẤT BẠI - PHÁT HIỆN LỖI:", "ERROR")
            self.log("=" * 60, "ERROR")
            for i, error in enumerate(errors, 1):
                self.log(f"  {i}. {error}", "ERROR")
            raise Exception(f"SANITY CHECK THẤT BẠI: {len(errors)} lỗi được phát hiện. Xem chi tiết ở trên.")
        else:
            self.log("=" * 60)
            self.log("✓ SANITY CHECK THÀNH CÔNG - DỮ LIỆU TOÀN VẸN!")
            self.log("=" * 60)
            return True, []

    def _create_node_mappings(self):
        """
        Tạo mapping từ node_id -> index cho từng loại node
        """
        self.log("  Tạo node mappings...")

        for node_type in ['user', 'recipient', 'device', 'ip']:
            type_nodes = self.nodes_df[self.nodes_df['node_type'] == node_type]
            for idx, row in type_nodes.iterrows():
                node_id = str(row['node_id'])
                local_idx = len(self.node_type_mappings[node_type])
                self.node_type_mappings[node_type][node_id] = local_idx

            self.log(f"    - {node_type}: {len(self.node_type_mappings[node_type])} nodes")

    def _get_node_features(self, node_type: str) -> torch.Tensor:
        """
        Lấy feature tensor cho một loại node

        Args:
            node_type: Loại node (user, recipient, device, ip)

        Returns:
            torch.Tensor: Feature tensor
        """
        type_nodes = self.nodes_df[self.nodes_df['node_type'] == node_type].copy()

        if node_type == 'user':
            # User có nhiều features
            feature_cols = ['age', 'account_age_days', 'avg_transaction_amount',
                           'credit_score', 'kyc_level', 'is_verified']

            # Xử lý missing values và encoding
            type_nodes['income_level_encoded'] = type_nodes['income_level'].map({
                'low': 0, 'medium': 1, 'high': 2, 'unknown': 0
            }).fillna(0)

            feature_cols.append('income_level_encoded')

            # Chuẩn hóa số liệu
            features = type_nodes[feature_cols].fillna(0).values.astype(np.float32)

            # Normalize một số cột
            if features.shape[0] > 0:
                # age: /100
                features[:, 0] = features[:, 0] / 100.0
                # account_age_days: /1000
                features[:, 1] = features[:, 1] / 1000.0
                # avg_transaction_amount: log scale
                features[:, 2] = np.log1p(features[:, 2]) / 20.0
                # credit_score: /850
                features[:, 3] = features[:, 3] / 850.0
                # kyc_level: /3
                features[:, 4] = features[:, 4] / 3.0
                # income_level_encoded: /2
                features[:, 6] = features[:, 6] / 2.0

        else:
            # recipient, device, ip chỉ có feature đơn giản (có thể mở rộng sau)
            num_nodes = len(type_nodes)
            # Tạo one-hot encoding cho node type
            features = np.zeros((num_nodes, 7), dtype=np.float32)
            features[:, 0] = 1.0  # indicator rằng đây là loại node này

        return torch.FloatTensor(features)

    def _get_edge_features_transfer(self) -> torch.Tensor:
        """
        Lấy feature tensor cho edges transfer

        Returns:
            torch.Tensor: Edge feature tensor
        """
        # Chọn các cột features số
        numeric_cols = ['amount', 'amount_log', 'is_international', 'hour', 'day_of_week',
                       'is_weekend', 'is_night', 'velocity_1h', 'velocity_24h', 'amount_zscore',
                       'device_change', 'ip_change', 'location_change', 'recipient_is_new',
                       'recipient_risk_score', 'sender_avg_amount', 'sender_std_amount',
                       'amount_deviation', 'time_since_last_tx', 'tx_count_1h', 'tx_count_24h',
                       'failed_attempts', 'login_attempts', 'session_duration',
                       'merchant_risk', 'is_recurring', 'is_scheduled',
                       'otp_verified', 'biometric_verified', 'pin_verified', 'risk_score_realtime']

        # Lọc các cột thực sự tồn tại
        available_cols = [col for col in numeric_cols if col in self.edges_transfer_df.columns]

        features = self.edges_transfer_df[available_cols].fillna(0).values.astype(np.float32)

        # Normalize một số cột quan trọng
        if features.shape[0] > 0 and 'amount' in available_cols:
            amount_idx = available_cols.index('amount')
            features[:, amount_idx] = np.log1p(features[:, amount_idx]) / 20.0

            if 'sender_avg_amount' in available_cols:
                idx = available_cols.index('sender_avg_amount')
                features[:, idx] = np.log1p(features[:, idx]) / 20.0

            if 'sender_std_amount' in available_cols:
                idx = available_cols.index('sender_std_amount')
                features[:, idx] = np.log1p(features[:, idx]) / 20.0

            if 'time_since_last_tx' in available_cols:
                idx = available_cols.index('time_since_last_tx')
                features[:, idx] = np.log1p(features[:, idx]) / 15.0

            if 'hour' in available_cols:
                idx = available_cols.index('hour')
                features[:, idx] = features[:, idx] / 24.0

            if 'session_duration' in available_cols:
                idx = available_cols.index('session_duration')
                features[:, idx] = features[:, idx] / 500.0

        return torch.FloatTensor(features)

    def build_hetero_graph(self) -> HeteroData:
        """
        Xây dựng PyTorch Geometric HeteroData từ dữ liệu đã load

        Returns:
            HeteroData: Graph dị thể
        """
        self.log("=" * 60)
        self.log("BƯỚC 3: XÂY DỰNG HETEROGENEOUS GRAPH")
        self.log("=" * 60)

        # Tạo node mappings
        self._create_node_mappings()

        # Khởi tạo HeteroData
        data = HeteroData()

        # =============================================
        # 1. THÊM NODE FEATURES
        # =============================================
        self.log("  Thêm node features...")

        for node_type in ['user', 'recipient', 'device', 'ip']:
            features = self._get_node_features(node_type)
            data[node_type].x = features
            self.log(f"    - {node_type}: shape {features.shape}")

        # =============================================
        # 2. THÊM EDGES VÀ EDGE FEATURES
        # =============================================
        self.log("  Thêm edges và edge features...")

        # --- Edge type: transfer (user -> recipient) ---
        src_indices = []
        dst_indices = []

        for _, row in self.edges_transfer_df.iterrows():
            src_id = str(row['src_node_id'])
            dst_id = str(row['dst_node_id'])

            # Lấy node type của src và dst
            src_type = self.nodes_df[self.nodes_df['node_id'] == src_id]['node_type'].values[0]
            dst_type = self.nodes_df[self.nodes_df['node_id'] == dst_id]['node_type'].values[0]

            if src_type == 'user' and dst_type == 'recipient':
                src_idx = self.node_type_mappings['user'].get(src_id)
                dst_idx = self.node_type_mappings['recipient'].get(dst_id)

                if src_idx is not None and dst_idx is not None:
                    src_indices.append(src_idx)
                    dst_indices.append(dst_idx)

        edge_index_transfer = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data['user', 'transfer', 'recipient'].edge_index = edge_index_transfer

        # Edge features cho transfer
        edge_features = self._get_edge_features_transfer()
        data['user', 'transfer', 'recipient'].edge_attr = edge_features

        self.log(f"    - transfer (user->recipient): {edge_index_transfer.shape[1]} edges, features: {edge_features.shape}")

        # --- Edge type: uses_device (user -> device) ---
        src_indices = []
        dst_indices = []

        for _, row in self.edges_uses_device_df.iterrows():
            src_id = str(row['src_node_id'])
            device_id = str(row['device_node_id'])

            src_idx = self.node_type_mappings['user'].get(src_id)
            dst_idx = self.node_type_mappings['device'].get(device_id)

            if src_idx is not None and dst_idx is not None:
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)

        edge_index_device = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data['user', 'uses_device', 'device'].edge_index = edge_index_device
        self.log(f"    - uses_device (user->device): {edge_index_device.shape[1]} edges")

        # --- Edge type: uses_ip (user -> ip) ---
        src_indices = []
        dst_indices = []

        for _, row in self.edges_uses_ip_df.iterrows():
            src_id = str(row['src_node_id'])
            ip_id = str(row['ip_node_id'])

            src_idx = self.node_type_mappings['user'].get(src_id)
            dst_idx = self.node_type_mappings['ip'].get(ip_id)

            if src_idx is not None and dst_idx is not None:
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)

        edge_index_ip = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data['user', 'uses_ip', 'ip'].edge_index = edge_index_ip
        self.log(f"    - uses_ip (user->ip): {edge_index_ip.shape[1]} edges")

        # =============================================
        # 3. THÊM EDGE LABELS (chỉ cho transfer edges)
        # =============================================
        self.log("  Thêm edge labels cho transfer edges...")

        # Merge labels với transfer edges
        transfer_with_labels = self.edges_transfer_df.merge(
            self.edge_labels_df, on='edge_id', how='left'
        )
        labels = transfer_with_labels['label'].fillna(0).values.astype(np.int64)
        data['user', 'transfer', 'recipient'].y = torch.LongTensor(labels)

        fraud_count = labels.sum()
        normal_count = len(labels) - fraud_count
        self.log(f"    - Labels: {normal_count} normal, {fraud_count} fraud ({fraud_count/len(labels)*100:.1f}%)")

        # =============================================
        # 4. THÊM TRAIN/VAL/TEST MASKS
        # =============================================
        self.log("  Tạo train/val/test masks...")

        # Merge splits với transfer edges
        transfer_with_splits = self.edges_transfer_df.merge(
            self.splits_df, on='edge_id', how='left'
        )

        splits = transfer_with_splits['split'].fillna('train').values
        num_edges = len(splits)

        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        for i, split in enumerate(splits):
            if split == 'train':
                train_mask[i] = True
            elif split == 'val':
                val_mask[i] = True
            elif split == 'test':
                test_mask[i] = True

        data['user', 'transfer', 'recipient'].train_mask = train_mask
        data['user', 'transfer', 'recipient'].val_mask = val_mask
        data['user', 'transfer', 'recipient'].test_mask = test_mask

        self.log(f"    - Train: {train_mask.sum().item()} edges")
        self.log(f"    - Val: {val_mask.sum().item()} edges")
        self.log(f"    - Test: {test_mask.sum().item()} edges")

        self.hetero_data = data

        self.log("=" * 60)
        self.log("✓ XÂY DỰNG HETEROGENEOUS GRAPH THÀNH CÔNG!")
        self.log("=" * 60)

        return data

    def save_graph(self, output_path: str = None) -> str:
        """
        Lưu graph và tạo flag file

        Args:
            output_path: Đường dẫn lưu graph (mặc định: gnn_data/hetero_graph.pt)

        Returns:
            str: Đường dẫn file đã lưu
        """
        if self.hetero_data is None:
            raise ValueError("Graph chưa được build! Gọi build_hetero_graph() trước.")

        if output_path is None:
            output_path = os.path.join(self.data_dir, 'hetero_graph.pt')

        self.log("=" * 60)
        self.log("BƯỚC 4: LƯU GRAPH VÀ TẠO FLAG FILE")
        self.log("=" * 60)

        # Lưu graph
        save_data = {
            'hetero_data': self.hetero_data,
            'node_type_mappings': self.node_type_mappings,
            'metadata': self.metadata,
            'created_at': datetime.now().isoformat()
        }

        torch.save(save_data, output_path)
        self.log(f"  ✓ Đã lưu graph: {output_path}")

        # Tạo flag file
        flag_path = os.path.join(self.data_dir, 'graph_ready.flag')
        with open(flag_path, 'w') as f:
            f.write(f"Graph created at: {datetime.now().isoformat()}\n")
            f.write(f"Graph path: {output_path}\n")
            f.write(f"Num node types: 4 (user, recipient, device, ip)\n")
            f.write(f"Num edge types: 3 (transfer, uses_device, uses_ip)\n")
            f.write(f"Num transfer edges: {self.hetero_data['user', 'transfer', 'recipient'].edge_index.shape[1]}\n")

        self.log(f"  ✓ Đã tạo flag file: {flag_path}")
        self.log("=" * 60)
        self.log("✓ HOÀN TẤT BƯỚC TẠO MẠNG LƯỚI GNN!")
        self.log("=" * 60)

        return output_path

    def run_full_pipeline(self) -> Tuple[HeteroData, str]:
        """
        Chạy toàn bộ pipeline: Load -> Sanity Check -> Build -> Save

        Returns:
            Tuple[HeteroData, str]: (graph data, path to saved graph)
        """
        self.log("*" * 70)
        self.log("     GNN DATA PIPELINE - TẠO MẠNG LƯỚI GNN")
        self.log("     Fraud Detection System - Agribank Vietnam")
        self.log("*" * 70)

        # Bước 1: Load dữ liệu
        self.load_all_data()

        # Bước 2: Sanity check
        self.sanity_check()

        # Bước 3: Build graph
        data = self.build_hetero_graph()

        # Bước 4: Lưu graph
        graph_path = self.save_graph()

        return data, graph_path


def load_gnn_graph(graph_path: str = None) -> Tuple[HeteroData, Dict]:
    """
    Load graph đã được build sẵn

    Args:
        graph_path: Đường dẫn file graph

    Returns:
        Tuple[HeteroData, Dict]: (graph data, metadata)
    """
    if graph_path is None:
        graph_path = os.path.join(GNN_DATA_DIR, 'hetero_graph.pt')

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Không tìm thấy graph file: {graph_path}")

    save_data = torch.load(graph_path)

    return save_data['hetero_data'], save_data.get('metadata', {})


def check_graph_ready(data_dir: str = None) -> bool:
    """
    Kiểm tra xem graph đã được build chưa

    Args:
        data_dir: Đường dẫn thư mục gnn_data

    Returns:
        bool: True nếu graph đã sẵn sàng
    """
    if data_dir is None:
        data_dir = GNN_DATA_DIR

    flag_path = os.path.join(data_dir, 'graph_ready.flag')
    graph_path = os.path.join(data_dir, 'hetero_graph.pt')

    return os.path.exists(flag_path) and os.path.exists(graph_path)


# ====================================================
# MAIN - Chạy pipeline
# ====================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CHẠY GNN DATA PIPELINE")
    print("=" * 70 + "\n")

    pipeline = GNNDataPipeline(verbose=True)
    data, graph_path = pipeline.run_full_pipeline()

    print("\n" + "=" * 70)
    print("KẾT QUẢ:")
    print(f"  - Graph đã được lưu tại: {graph_path}")
    print(f"  - Flag file: {os.path.join(GNN_DATA_DIR, 'graph_ready.flag')}")
    print("=" * 70)

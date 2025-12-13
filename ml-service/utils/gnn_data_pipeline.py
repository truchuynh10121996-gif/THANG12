"""
GNN Data Pipeline - Pipeline xử lý dữ liệu cho GNN Fraud Detection
===================================================================
Pipeline này xử lý bộ dữ liệu GNN từ nhiều file CSV/JSON:
- nodes.csv: Tất cả nodes (user, recipient, device, ip)
- nodes_user.csv, nodes_recipient.csv, nodes_device.csv, nodes_ip.csv: Node features riêng
- edges.csv: Tất cả edges
- edges_transfer.csv: Edges chuyển tiền (user → recipient) - CÓ LABEL
- edges_uses_device.csv: Edges sử dụng thiết bị (user → device)
- edges_uses_ip.csv: Edges sử dụng IP (user → ip)
- edge_labels.csv: Labels cho edges transfer
- splits.csv: Train/val/test split
- metadata.json, graph_preview.json: Metadata

Mục tiêu: Edge-level fraud detection (binary classification) trên edge type "transfer"
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch và PyTorch Geometric
import torch
from torch_geometric.data import HeteroData

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()

# Thư mục lưu graph đã build
GNN_GRAPH_DIR = os.path.join(config.SAVED_MODELS_DIR, 'gnn_graphs')
os.makedirs(GNN_GRAPH_DIR, exist_ok=True)

# Flag file để đánh dấu graph đã ready
GRAPH_READY_FLAG = os.path.join(GNN_GRAPH_DIR, 'graph_ready.flag')
GRAPH_DATA_PATH = os.path.join(GNN_GRAPH_DIR, 'hetero_graph.pt')
GRAPH_METADATA_PATH = os.path.join(GNN_GRAPH_DIR, 'graph_metadata.json')


class GNNDataPipeline:
    """
    Pipeline xử lý dữ liệu cho GNN heterogeneous graph

    Hỗ trợ:
    - Load toàn bộ file CSV/JSON từ thư mục gnn_data
    - Sanity check: kiểm tra tính toàn vẹn dữ liệu
    - Build heterogeneous graph với PyTorch Geometric
    - Edge-level fraud detection trên edge type "transfer"
    """

    def __init__(self, verbose: bool = True):
        """
        Khởi tạo pipeline

        Args:
            verbose: In thông tin chi tiết
        """
        self.verbose = verbose
        self.data = {}  # Lưu trữ dữ liệu đã load
        self.node_mapping = {}  # Mapping node_id -> index cho mỗi node type
        self.reverse_mapping = {}  # Mapping index -> node_id
        self.errors = []  # Lưu các lỗi sanity check
        self.warnings = []  # Lưu các cảnh báo

    def log(self, message: str, level: str = 'info'):
        """In log nếu verbose mode"""
        if self.verbose:
            prefix = {
                'info': '[GNN-Pipeline]',
                'success': '[GNN-Pipeline] ✅',
                'error': '[GNN-Pipeline] ❌',
                'warning': '[GNN-Pipeline] ⚠️'
            }.get(level, '[GNN-Pipeline]')
            print(f"{prefix} {message}")

    def load_data_from_directory(self, data_dir: str) -> Dict[str, Any]:
        """
        Load toàn bộ dữ liệu từ thư mục

        Args:
            data_dir: Đường dẫn đến thư mục chứa dữ liệu GNN

        Returns:
            Dict chứa tất cả dữ liệu đã load
        """
        self.log(f"Bắt đầu load dữ liệu từ: {data_dir}")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

        # Định nghĩa các file cần load
        required_files = {
            'nodes': 'nodes.csv',
            'edges_transfer': 'edges_transfer.csv',
            'edge_labels': 'edge_labels.csv',
            'splits': 'splits.csv',
        }

        optional_files = {
            'nodes_user': 'nodes_user.csv',
            'nodes_recipient': 'nodes_recipient.csv',
            'nodes_device': 'nodes_device.csv',
            'nodes_ip': 'nodes_ip.csv',
            'edges': 'edges.csv',
            'edges_uses_device': 'edges_uses_device.csv',
            'edges_uses_ip': 'edges_uses_ip.csv',
            'metadata': 'metadata.json',
            'graph_preview': 'graph_preview.json',
        }

        # Load required files
        for key, filename in required_files.items():
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Thiếu file bắt buộc: {filename}")

            if filename.endswith('.csv'):
                self.data[key] = pd.read_csv(filepath)
                self.log(f"  Đã load {filename}: {len(self.data[key])} dòng")
            elif filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.data[key] = json.load(f)
                self.log(f"  Đã load {filename}")

        # Load optional files
        for key, filename in optional_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                if filename.endswith('.csv'):
                    self.data[key] = pd.read_csv(filepath)
                    self.log(f"  Đã load {filename}: {len(self.data[key])} dòng")
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.data[key] = json.load(f)
                    self.log(f"  Đã load {filename}")
            else:
                self.log(f"  Bỏ qua {filename} (không tồn tại)", level='warning')

        self.log(f"Load dữ liệu hoàn tất! Tổng cộng {len(self.data)} files", level='success')
        return self.data

    def sanity_check(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính toàn vẹn dữ liệu

        Checks:
        1. Mọi src_node_id, dst_node_id trong edges_* PHẢI tồn tại trong nodes.csv
        2. Mọi edge_id trong edge_labels.csv và splits.csv PHẢI match với edges_transfer.csv
        3. Kiểm tra format (leading zeros, etc.)

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        self.log("="*60)
        self.log("BẮT ĐẦU SANITY CHECK")
        self.log("="*60)

        self.errors = []
        self.warnings = []

        # Lấy tập hợp tất cả node_id từ nodes.csv
        nodes_df = self.data.get('nodes')
        if nodes_df is None:
            self.errors.append("Không có dữ liệu nodes.csv")
            return False, self.errors

        # Xác định cột node_id
        node_id_col = None
        for col in ['node_id', 'id', 'node']:
            if col in nodes_df.columns:
                node_id_col = col
                break

        if node_id_col is None:
            self.errors.append("Không tìm thấy cột node_id trong nodes.csv")
            return False, self.errors

        # Chuyển đổi sang string để đảm bảo so sánh chính xác
        all_node_ids = set(nodes_df[node_id_col].astype(str).unique())
        self.log(f"Tổng số nodes trong nodes.csv: {len(all_node_ids)}")

        # === CHECK 1: Nodes trong edges_transfer ===
        edges_transfer = self.data.get('edges_transfer')
        if edges_transfer is not None:
            self.log("\n[CHECK 1] Kiểm tra edges_transfer.csv...")

            # Xác định cột source và target
            src_col = None
            dst_col = None
            for col in ['src_node_id', 'source', 'src', 'from']:
                if col in edges_transfer.columns:
                    src_col = col
                    break
            for col in ['dst_node_id', 'target', 'dst', 'to']:
                if col in edges_transfer.columns:
                    dst_col = col
                    break

            if src_col is None or dst_col is None:
                self.errors.append("Không tìm thấy cột src/dst trong edges_transfer.csv")
            else:
                # Kiểm tra từng edge
                src_ids = set(edges_transfer[src_col].astype(str).unique())
                dst_ids = set(edges_transfer[dst_col].astype(str).unique())

                missing_src = src_ids - all_node_ids
                missing_dst = dst_ids - all_node_ids

                if missing_src:
                    self.errors.append(
                        f"edges_transfer.csv: {len(missing_src)} src_node_id không tồn tại trong nodes.csv. "
                        f"Ví dụ: {list(missing_src)[:5]}"
                    )
                if missing_dst:
                    self.errors.append(
                        f"edges_transfer.csv: {len(missing_dst)} dst_node_id không tồn tại trong nodes.csv. "
                        f"Ví dụ: {list(missing_dst)[:5]}"
                    )

                if not missing_src and not missing_dst:
                    self.log(f"  ✅ Tất cả {len(edges_transfer)} edges có nodes hợp lệ", level='success')

        # === CHECK 2: Edge IDs consistency ===
        self.log("\n[CHECK 2] Kiểm tra edge_id consistency...")

        # Lấy edge_id từ edges_transfer
        edge_id_col = None
        for col in ['edge_id', 'id', 'edge']:
            if col in edges_transfer.columns:
                edge_id_col = col
                break

        if edge_id_col is None:
            self.errors.append("Không tìm thấy cột edge_id trong edges_transfer.csv")
        else:
            transfer_edge_ids = set(edges_transfer[edge_id_col].astype(str).unique())
            self.log(f"  Số edge_id trong edges_transfer.csv: {len(transfer_edge_ids)}")

            # Kiểm tra edge_labels.csv
            edge_labels = self.data.get('edge_labels')
            if edge_labels is not None:
                label_edge_col = None
                for col in ['edge_id', 'id', 'edge']:
                    if col in edge_labels.columns:
                        label_edge_col = col
                        break

                if label_edge_col:
                    label_edge_ids = set(edge_labels[label_edge_col].astype(str).unique())

                    # Kiểm tra mismatch
                    missing_in_transfer = label_edge_ids - transfer_edge_ids
                    missing_in_labels = transfer_edge_ids - label_edge_ids

                    if missing_in_transfer:
                        self.errors.append(
                            f"edge_labels.csv: {len(missing_in_transfer)} edge_id không tồn tại trong edges_transfer.csv. "
                            f"Ví dụ: {list(missing_in_transfer)[:5]}"
                        )
                    if missing_in_labels:
                        self.warnings.append(
                            f"{len(missing_in_labels)} edges trong edges_transfer.csv không có label (sẽ bị bỏ qua)"
                        )

                    if not missing_in_transfer:
                        self.log(f"  ✅ Tất cả {len(label_edge_ids)} edge_id trong edge_labels.csv hợp lệ", level='success')

            # Kiểm tra splits.csv
            splits = self.data.get('splits')
            if splits is not None:
                split_edge_col = None
                for col in ['edge_id', 'id', 'edge']:
                    if col in splits.columns:
                        split_edge_col = col
                        break

                if split_edge_col:
                    split_edge_ids = set(splits[split_edge_col].astype(str).unique())

                    missing_in_transfer = split_edge_ids - transfer_edge_ids

                    if missing_in_transfer:
                        self.errors.append(
                            f"splits.csv: {len(missing_in_transfer)} edge_id không tồn tại trong edges_transfer.csv. "
                            f"Ví dụ: {list(missing_in_transfer)[:5]}"
                        )
                    else:
                        self.log(f"  ✅ Tất cả {len(split_edge_ids)} edge_id trong splits.csv hợp lệ", level='success')

        # === CHECK 3: Kiểm tra các edge types khác ===
        for edge_key in ['edges_uses_device', 'edges_uses_ip']:
            edge_df = self.data.get(edge_key)
            if edge_df is not None:
                self.log(f"\n[CHECK 3] Kiểm tra {edge_key}.csv...")

                src_col = None
                dst_col = None
                for col in ['src_node_id', 'source', 'src', 'from', 'user_id']:
                    if col in edge_df.columns:
                        src_col = col
                        break
                for col in ['dst_node_id', 'target', 'dst', 'to', 'device_id', 'ip_id']:
                    if col in edge_df.columns:
                        dst_col = col
                        break

                if src_col and dst_col:
                    src_ids = set(edge_df[src_col].astype(str).unique())
                    dst_ids = set(edge_df[dst_col].astype(str).unique())

                    missing_src = src_ids - all_node_ids
                    missing_dst = dst_ids - all_node_ids

                    if missing_src:
                        self.errors.append(
                            f"{edge_key}.csv: {len(missing_src)} src không tồn tại trong nodes. "
                            f"Ví dụ: {list(missing_src)[:5]}"
                        )
                    if missing_dst:
                        self.errors.append(
                            f"{edge_key}.csv: {len(missing_dst)} dst không tồn tại trong nodes. "
                            f"Ví dụ: {list(missing_dst)[:5]}"
                        )

                    if not missing_src and not missing_dst:
                        self.log(f"  ✅ Tất cả {len(edge_df)} edges hợp lệ", level='success')

        # === Tổng kết ===
        self.log("\n" + "="*60)
        self.log("SANITY CHECK HOÀN TẤT")
        self.log("="*60)

        if self.errors:
            self.log(f"❌ Phát hiện {len(self.errors)} lỗi:", level='error')
            for i, err in enumerate(self.errors, 1):
                self.log(f"  {i}. {err}", level='error')
            return False, self.errors

        if self.warnings:
            self.log(f"⚠️ Có {len(self.warnings)} cảnh báo:", level='warning')
            for w in self.warnings:
                self.log(f"  - {w}", level='warning')

        self.log("✅ Dữ liệu hợp lệ, sẵn sàng build graph!", level='success')
        return True, []

    def build_hetero_graph(self) -> HeteroData:
        """
        Build heterogeneous graph từ dữ liệu đã load

        Graph structure:
        - Node types: user, recipient, device, ip
        - Edge types:
          - ('user', 'transfer', 'recipient') [CÓ LABEL]
          - ('user', 'uses', 'device')
          - ('user', 'uses', 'ip')

        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph
        """
        self.log("\n" + "="*60)
        self.log("BẮT ĐẦU BUILD HETEROGENEOUS GRAPH")
        self.log("="*60)

        # Khởi tạo HeteroData
        data = HeteroData()

        # === BƯỚC 1: Xử lý nodes ===
        self.log("\n[BƯỚC 1] Xử lý nodes...")
        self._build_nodes(data)

        # === BƯỚC 2: Xử lý edges ===
        self.log("\n[BƯỚC 2] Xử lý edges...")
        self._build_edges(data)

        # === BƯỚC 3: Thêm labels và splits ===
        self.log("\n[BƯỚC 3] Thêm labels và splits...")
        self._add_labels_and_splits(data)

        # === Tổng kết ===
        self.log("\n" + "="*60)
        self.log("BUILD GRAPH HOÀN TẤT", level='success')
        self.log("="*60)

        self._print_graph_summary(data)

        return data

    def _build_nodes(self, data: HeteroData):
        """Build nodes cho heterogeneous graph"""
        nodes_df = self.data['nodes']

        # Xác định cột node_id và node_type
        node_id_col = 'node_id' if 'node_id' in nodes_df.columns else 'id'
        type_col = 'node_type' if 'node_type' in nodes_df.columns else 'type'

        # Nhóm nodes theo type
        node_types = nodes_df[type_col].unique() if type_col in nodes_df.columns else ['user']

        for node_type in node_types:
            type_nodes = nodes_df[nodes_df[type_col] == node_type] if type_col in nodes_df.columns else nodes_df

            # Tạo mapping node_id -> index
            self.node_mapping[node_type] = {}
            self.reverse_mapping[node_type] = {}

            for idx, row in enumerate(type_nodes[node_id_col]):
                node_id_str = str(row)
                self.node_mapping[node_type][node_id_str] = idx
                self.reverse_mapping[node_type][idx] = node_id_str

            # Lấy features cho node type này
            # Ưu tiên file riêng (nodes_user.csv, nodes_device.csv, ...)
            specific_nodes_key = f'nodes_{node_type}'
            if specific_nodes_key in self.data:
                feature_df = self.data[specific_nodes_key]
            else:
                feature_df = type_nodes

            # Trích xuất features (loại bỏ cột ID và type)
            exclude_cols = [node_id_col, type_col, 'node_id', 'id', 'type', 'node_type']
            feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

            if feature_cols:
                # Chuyển đổi sang numeric
                features = feature_df[feature_cols].copy()
                for col in features.columns:
                    if features[col].dtype == 'object':
                        # Label encode string columns
                        features[col] = pd.factorize(features[col])[0]
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                features = features.fillna(0).values
            else:
                # Nếu không có features, tạo random features (placeholder)
                features = np.random.randn(len(type_nodes), 16).astype(np.float32)

            # Thêm vào HeteroData
            data[node_type].x = torch.FloatTensor(features)
            data[node_type].num_nodes = len(type_nodes)

            self.log(f"  ✅ {node_type}: {len(type_nodes)} nodes, {features.shape[1]} features")

    def _build_edges(self, data: HeteroData):
        """Build edges cho heterogeneous graph"""

        # === Edge type: transfer (user → recipient) ===
        edges_transfer = self.data['edges_transfer']

        # Xác định cột
        src_col = 'src_node_id' if 'src_node_id' in edges_transfer.columns else 'source'
        dst_col = 'dst_node_id' if 'dst_node_id' in edges_transfer.columns else 'target'
        edge_id_col = 'edge_id' if 'edge_id' in edges_transfer.columns else 'id'

        # Xác định node types cho source và destination
        # Dựa trên prefix của node_id hoặc từ nodes.csv
        nodes_df = self.data['nodes']
        type_col = 'node_type' if 'node_type' in nodes_df.columns else 'type'
        node_id_col = 'node_id' if 'node_id' in nodes_df.columns else 'id'

        # Tạo mapping node_id -> node_type
        if type_col in nodes_df.columns:
            node_to_type = dict(zip(nodes_df[node_id_col].astype(str), nodes_df[type_col]))
        else:
            # Giả sử tất cả là 'user' nếu không có type
            node_to_type = {str(nid): 'user' for nid in nodes_df[node_id_col]}

        # Lưu edge mapping cho labels
        self.edge_mapping = {}

        # Xây dựng edge_index cho transfer edges
        src_indices = []
        dst_indices = []
        edge_ids = []

        for idx, row in edges_transfer.iterrows():
            src_id = str(row[src_col])
            dst_id = str(row[dst_col])
            eid = str(row[edge_id_col])

            # Xác định types
            src_type = node_to_type.get(src_id, 'user')
            dst_type = node_to_type.get(dst_id, 'recipient')

            # Đảm bảo node types tồn tại trong mapping
            if src_type not in self.node_mapping:
                continue
            if dst_type not in self.node_mapping:
                # Nếu dst_type không tồn tại, sử dụng src_type
                dst_type = src_type

            if src_id in self.node_mapping.get(src_type, {}) and dst_id in self.node_mapping.get(dst_type, {}):
                src_idx = self.node_mapping[src_type][src_id]
                dst_idx = self.node_mapping[dst_type][dst_id]

                src_indices.append(src_idx)
                dst_indices.append(dst_idx)
                edge_ids.append(eid)

                self.edge_mapping[eid] = len(edge_ids) - 1

        if src_indices:
            # Xác định edge type tuple
            # Ưu tiên ('user', 'transfer', 'recipient') nếu có đủ node types
            if 'user' in self.node_mapping and 'recipient' in self.node_mapping:
                edge_type = ('user', 'transfer', 'recipient')
            else:
                # Fallback: sử dụng node type đầu tiên
                first_type = list(self.node_mapping.keys())[0]
                edge_type = (first_type, 'transfer', first_type)

            data[edge_type].edge_index = torch.LongTensor([src_indices, dst_indices])

            # Thêm edge features nếu có
            feature_cols = [c for c in edges_transfer.columns
                          if c not in [src_col, dst_col, edge_id_col, 'label', 'is_fraud']]
            if feature_cols:
                edge_features = edges_transfer[feature_cols].copy()
                for col in edge_features.columns:
                    if edge_features[col].dtype == 'object':
                        edge_features[col] = pd.factorize(edge_features[col])[0]
                    edge_features[col] = pd.to_numeric(edge_features[col], errors='coerce')
                edge_features = edge_features.fillna(0).values
                data[edge_type].edge_attr = torch.FloatTensor(edge_features)

            self.log(f"  ✅ {edge_type}: {len(src_indices)} edges")

            # Lưu edge_ids để map với labels
            data[edge_type].edge_ids = edge_ids

        # === Edge types khác: uses_device, uses_ip ===
        for edge_key, edge_type_name, dst_type in [
            ('edges_uses_device', 'uses', 'device'),
            ('edges_uses_ip', 'uses', 'ip')
        ]:
            edge_df = self.data.get(edge_key)
            if edge_df is not None and dst_type in self.node_mapping:
                src_col = 'src_node_id' if 'src_node_id' in edge_df.columns else 'user_id'
                dst_col_name = 'dst_node_id' if 'dst_node_id' in edge_df.columns else f'{dst_type}_id'

                if src_col in edge_df.columns and dst_col_name in edge_df.columns:
                    src_indices = []
                    dst_indices = []

                    for _, row in edge_df.iterrows():
                        src_id = str(row[src_col])
                        dst_id = str(row[dst_col_name])

                        if src_id in self.node_mapping.get('user', {}) and dst_id in self.node_mapping.get(dst_type, {}):
                            src_indices.append(self.node_mapping['user'][src_id])
                            dst_indices.append(self.node_mapping[dst_type][dst_id])

                    if src_indices:
                        edge_tuple = ('user', edge_type_name, dst_type)
                        data[edge_tuple].edge_index = torch.LongTensor([src_indices, dst_indices])
                        self.log(f"  ✅ {edge_tuple}: {len(src_indices)} edges")

    def _add_labels_and_splits(self, data: HeteroData):
        """Thêm labels và train/val/test splits"""

        # Lấy edge type chính (transfer)
        edge_type = None
        for et in data.edge_types:
            if 'transfer' in et[1]:
                edge_type = et
                break

        if edge_type is None:
            self.log("Không tìm thấy edge type 'transfer'", level='error')
            return

        edge_ids = data[edge_type].edge_ids
        num_edges = len(edge_ids)

        # === Labels ===
        edge_labels = self.data['edge_labels']
        label_col = 'label' if 'label' in edge_labels.columns else 'is_fraud'
        edge_id_col = 'edge_id' if 'edge_id' in edge_labels.columns else 'id'

        # Tạo mapping edge_id -> label
        label_map = dict(zip(
            edge_labels[edge_id_col].astype(str),
            edge_labels[label_col]
        ))

        # Gán labels theo thứ tự edges
        labels = []
        for eid in edge_ids:
            labels.append(label_map.get(str(eid), 0))  # Default = 0 (không fraud)

        data[edge_type].y = torch.LongTensor(labels)

        fraud_count = sum(labels)
        self.log(f"  ✅ Labels: {fraud_count} fraud / {num_edges - fraud_count} normal ({fraud_count/num_edges*100:.2f}% fraud)")

        # === Splits ===
        splits = self.data['splits']
        split_col = 'split' if 'split' in splits.columns else 'set'
        split_edge_col = 'edge_id' if 'edge_id' in splits.columns else 'id'

        # Tạo mapping edge_id -> split
        split_map = dict(zip(
            splits[split_edge_col].astype(str),
            splits[split_col]
        ))

        # Tạo masks
        train_mask = []
        val_mask = []
        test_mask = []

        for eid in edge_ids:
            split = split_map.get(str(eid), 'train')
            train_mask.append(split == 'train')
            val_mask.append(split in ['val', 'valid', 'validation'])
            test_mask.append(split == 'test')

        data[edge_type].train_mask = torch.BoolTensor(train_mask)
        data[edge_type].val_mask = torch.BoolTensor(val_mask)
        data[edge_type].test_mask = torch.BoolTensor(test_mask)

        self.log(f"  ✅ Splits: {sum(train_mask)} train / {sum(val_mask)} val / {sum(test_mask)} test")

    def _print_graph_summary(self, data: HeteroData):
        """In tổng kết graph"""
        self.log("\n📊 TỔNG KẾT GRAPH:")
        self.log(f"  Node types: {data.node_types}")
        self.log(f"  Edge types: {data.edge_types}")

        for node_type in data.node_types:
            self.log(f"  {node_type}: {data[node_type].num_nodes} nodes, {data[node_type].x.shape[1]} features")

        for edge_type in data.edge_types:
            num_edges = data[edge_type].edge_index.shape[1]
            self.log(f"  {edge_type}: {num_edges} edges")

    def save_graph(self, data: HeteroData, metadata: Dict = None):
        """
        Lưu graph và tạo flag file

        Args:
            data: HeteroData object
            metadata: Metadata bổ sung
        """
        self.log("\n[SAVE] Đang lưu graph...")

        # Lưu graph
        torch.save(data, GRAPH_DATA_PATH)
        self.log(f"  ✅ Đã lưu graph: {GRAPH_DATA_PATH}")

        # Lưu metadata
        graph_metadata = {
            'node_types': list(data.node_types),
            'edge_types': [str(et) for et in data.edge_types],
            'node_mapping': {k: len(v) for k, v in self.node_mapping.items()},
            'created_at': pd.Timestamp.now().isoformat(),
        }
        if metadata:
            graph_metadata.update(metadata)

        with open(GRAPH_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(graph_metadata, f, indent=2, ensure_ascii=False)
        self.log(f"  ✅ Đã lưu metadata: {GRAPH_METADATA_PATH}")

        # Tạo flag file
        with open(GRAPH_READY_FLAG, 'w') as f:
            f.write(f"Graph ready at {pd.Timestamp.now().isoformat()}")
        self.log(f"  ✅ Đã tạo flag: {GRAPH_READY_FLAG}", level='success')

    @staticmethod
    def load_saved_graph() -> Tuple[Optional[HeteroData], Optional[Dict]]:
        """
        Load graph đã lưu

        Returns:
            Tuple[HeteroData, Dict]: Graph và metadata
        """
        if not os.path.exists(GRAPH_READY_FLAG):
            return None, None

        if not os.path.exists(GRAPH_DATA_PATH):
            return None, None

        data = torch.load(GRAPH_DATA_PATH)

        metadata = None
        if os.path.exists(GRAPH_METADATA_PATH):
            with open(GRAPH_METADATA_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        return data, metadata

    @staticmethod
    def is_graph_ready() -> bool:
        """Kiểm tra graph đã sẵn sàng chưa"""
        return os.path.exists(GRAPH_READY_FLAG) and os.path.exists(GRAPH_DATA_PATH)

    @staticmethod
    def clear_graph():
        """Xóa graph đã lưu"""
        for path in [GRAPH_READY_FLAG, GRAPH_DATA_PATH, GRAPH_METADATA_PATH]:
            if os.path.exists(path):
                os.remove(path)


def build_gnn_graph_from_directory(data_dir: str, verbose: bool = True) -> Dict:
    """
    Hàm tiện ích: Load, sanity check, và build graph từ thư mục

    Args:
        data_dir: Đường dẫn thư mục chứa dữ liệu GNN
        verbose: In thông tin chi tiết

    Returns:
        Dict với kết quả build graph
    """
    pipeline = GNNDataPipeline(verbose=verbose)

    # Load data
    pipeline.load_data_from_directory(data_dir)

    # Sanity check
    is_valid, errors = pipeline.sanity_check()

    if not is_valid:
        return {
            'success': False,
            'errors': errors,
            'message': 'Sanity check thất bại'
        }

    # Build graph
    graph = pipeline.build_hetero_graph()

    # Save graph
    pipeline.save_graph(graph, {'source_dir': data_dir})

    return {
        'success': True,
        'message': 'Build graph thành công!',
        'graph_path': GRAPH_DATA_PATH,
        'metadata': {
            'node_types': list(graph.node_types),
            'edge_types': [str(et) for et in graph.edge_types],
            'warnings': pipeline.warnings
        }
    }


if __name__ == '__main__':
    # Test với thư mục mẫu
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.join(config.DATA_DIR, 'gnn_data')

    result = build_gnn_graph_from_directory(data_dir)
    print(f"\nKết quả: {result}")

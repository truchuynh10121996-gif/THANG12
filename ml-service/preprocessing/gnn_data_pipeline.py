"""
GNN Data Pipeline - Pipeline xử lý dữ liệu cho GNN Fraud Detection
====================================================================
Module này xử lý dữ liệu từ các file CSV/JSON để xây dựng
Heterogeneous Graph cho bài toán edge-level fraud detection.

Cấu trúc dữ liệu đầu vào:
- nodes.csv: (node_id, node_type, raw_id, + node features)
- edges_transfer.csv: (edge_id, src_node_id, dst_node_id, edge_type, timestamp, 40 features)
- edges_uses_device.csv: (src_node_id, device_node_id)
- edges_uses_ip.csv: (src_node_id, ip_node_id)
- edge_labels.csv: (edge_id, label)
- splits.csv: (edge_id, split=train/val/test)
- metadata.json: mô tả dataset

Author: Senior ML Engineer
Version: 2.0.0
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

# PyTorch và PyTorch Geometric
import torch
from torch_geometric.data import HeteroData

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class GNNDataPipeline:
    """
    Pipeline xử lý dữ liệu cho GNN Fraud Detection

    Hỗ trợ:
    - Load dữ liệu từ nhiều file CSV/JSON
    - Sanity check toàn diện
    - Xây dựng Heterogeneous Graph với PyTorch Geometric
    - Edge-level fraud detection
    """

    def __init__(self, data_dir: str, verbose: bool = True):
        """
        Khởi tạo GNN Data Pipeline

        Args:
            data_dir: Đường dẫn thư mục chứa dữ liệu GNN
            verbose: In thông tin chi tiết
        """
        self.data_dir = data_dir
        self.verbose = verbose

        # Các file dữ liệu
        self.files = {
            'nodes': 'nodes.csv',
            'nodes_user': 'nodes_user.csv',
            'nodes_recipient': 'nodes_recipient.csv',
            'nodes_device': 'nodes_device.csv',
            'nodes_ip': 'nodes_ip.csv',
            'edges_transfer': 'edges_transfer.csv',
            'edges_uses_device': 'edges_uses_device.csv',
            'edges_uses_ip': 'edges_uses_ip.csv',
            'edge_labels': 'edge_labels.csv',
            'splits': 'splits.csv',
            'metadata': 'metadata.json'
        }

        # DataFrames
        self.nodes_df = None
        self.nodes_user_df = None
        self.nodes_recipient_df = None
        self.nodes_device_df = None
        self.nodes_ip_df = None
        self.edges_transfer_df = None
        self.edges_uses_device_df = None
        self.edges_uses_ip_df = None
        self.edge_labels_df = None
        self.splits_df = None
        self.metadata = None

        # Mappings
        self.node_id_to_idx = {}  # {node_type: {node_id: idx}}
        self.idx_to_node_id = {}  # {node_type: {idx: node_id}}

        # Thống kê
        self.stats = {}

        # Lỗi phát hiện
        self.errors = []

    def log(self, message: str, level: str = "INFO"):
        """In log message"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌"
            }.get(level, "")
            print(f"{prefix} [GNN Pipeline] {message}")

    def load_all_data(self) -> Dict[str, Any]:
        """
        Load toàn bộ dữ liệu từ các file CSV/JSON

        Returns:
            Dict chứa thông tin về dữ liệu đã load
        """
        self.log("Bắt đầu load dữ liệu...", "INFO")

        load_results = {
            'success': True,
            'files_loaded': [],
            'files_missing': [],
            'errors': []
        }

        # Load nodes.csv (file chính chứa tất cả nodes)
        nodes_path = os.path.join(self.data_dir, self.files['nodes'])
        if os.path.exists(nodes_path):
            self.nodes_df = pd.read_csv(nodes_path)
            load_results['files_loaded'].append('nodes.csv')
            self.log(f"  Loaded nodes.csv: {len(self.nodes_df):,} nodes", "SUCCESS")
        else:
            # Thử load các file nodes riêng lẻ
            self._load_separate_node_files(load_results)

        # Load edges_transfer.csv
        edges_transfer_path = os.path.join(self.data_dir, self.files['edges_transfer'])
        if os.path.exists(edges_transfer_path):
            self.edges_transfer_df = pd.read_csv(edges_transfer_path)
            load_results['files_loaded'].append('edges_transfer.csv')
            self.log(f"  Loaded edges_transfer.csv: {len(self.edges_transfer_df):,} edges", "SUCCESS")
        else:
            load_results['files_missing'].append('edges_transfer.csv')
            load_results['errors'].append("edges_transfer.csv không tồn tại - đây là file bắt buộc!")
            load_results['success'] = False

        # Load edges_uses_device.csv (optional)
        edges_device_path = os.path.join(self.data_dir, self.files['edges_uses_device'])
        if os.path.exists(edges_device_path):
            self.edges_uses_device_df = pd.read_csv(edges_device_path)
            load_results['files_loaded'].append('edges_uses_device.csv')
            self.log(f"  Loaded edges_uses_device.csv: {len(self.edges_uses_device_df):,} edges", "SUCCESS")
        else:
            load_results['files_missing'].append('edges_uses_device.csv')
            self.log("  edges_uses_device.csv không tồn tại (optional)", "WARNING")

        # Load edges_uses_ip.csv (optional)
        edges_ip_path = os.path.join(self.data_dir, self.files['edges_uses_ip'])
        if os.path.exists(edges_ip_path):
            self.edges_uses_ip_df = pd.read_csv(edges_ip_path)
            load_results['files_loaded'].append('edges_uses_ip.csv')
            self.log(f"  Loaded edges_uses_ip.csv: {len(self.edges_uses_ip_df):,} edges", "SUCCESS")
        else:
            load_results['files_missing'].append('edges_uses_ip.csv')
            self.log("  edges_uses_ip.csv không tồn tại (optional)", "WARNING")

        # Load edge_labels.csv
        labels_path = os.path.join(self.data_dir, self.files['edge_labels'])
        if os.path.exists(labels_path):
            self.edge_labels_df = pd.read_csv(labels_path)
            load_results['files_loaded'].append('edge_labels.csv')
            self.log(f"  Loaded edge_labels.csv: {len(self.edge_labels_df):,} labels", "SUCCESS")
        else:
            load_results['files_missing'].append('edge_labels.csv')
            load_results['errors'].append("edge_labels.csv không tồn tại - đây là file bắt buộc!")
            load_results['success'] = False

        # Load splits.csv
        splits_path = os.path.join(self.data_dir, self.files['splits'])
        if os.path.exists(splits_path):
            self.splits_df = pd.read_csv(splits_path)
            load_results['files_loaded'].append('splits.csv')
            self.log(f"  Loaded splits.csv: {len(self.splits_df):,} entries", "SUCCESS")
        else:
            load_results['files_missing'].append('splits.csv')
            load_results['errors'].append("splits.csv không tồn tại - đây là file bắt buộc!")
            load_results['success'] = False

        # Load metadata.json
        metadata_path = os.path.join(self.data_dir, self.files['metadata'])
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            load_results['files_loaded'].append('metadata.json')
            self.log("  Loaded metadata.json", "SUCCESS")
        else:
            load_results['files_missing'].append('metadata.json')
            self.log("  metadata.json không tồn tại (optional)", "WARNING")

        return load_results

    def _load_separate_node_files(self, load_results: Dict):
        """Load các file nodes riêng lẻ nếu không có nodes.csv"""
        node_dfs = []

        for node_type in ['user', 'recipient', 'device', 'ip']:
            file_name = f'nodes_{node_type}.csv'
            file_path = os.path.join(self.data_dir, file_name)

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['node_type'] = node_type
                node_dfs.append(df)
                load_results['files_loaded'].append(file_name)
                self.log(f"  Loaded {file_name}: {len(df):,} nodes", "SUCCESS")

                # Lưu riêng
                if node_type == 'user':
                    self.nodes_user_df = df
                elif node_type == 'recipient':
                    self.nodes_recipient_df = df
                elif node_type == 'device':
                    self.nodes_device_df = df
                elif node_type == 'ip':
                    self.nodes_ip_df = df

        if node_dfs:
            # Gộp tất cả nodes vào một DataFrame
            self.nodes_df = pd.concat(node_dfs, ignore_index=True)
            self.log(f"  Tổng hợp: {len(self.nodes_df):,} nodes từ các file riêng lẻ", "SUCCESS")
        else:
            load_results['files_missing'].append('nodes.csv hoặc nodes_*.csv')
            load_results['errors'].append("Không tìm thấy file nodes - đây là file bắt buộc!")
            load_results['success'] = False

    def sanity_check(self) -> Dict[str, Any]:
        """
        Kiểm tra tính toàn vẹn của dữ liệu

        Kiểm tra:
        1. Mọi src_node_id, dst_node_id trong edges_* PHẢI tồn tại trong nodes.csv
        2. Mọi edge_id trong edge_labels.csv và splits.csv PHẢI match với edges_transfer.csv
        3. Không được mất leading zeros, không được lệch format

        Returns:
            Dict chứa kết quả kiểm tra

        Raises:
            Exception: Nếu phát hiện lỗi nghiêm trọng
        """
        self.log("Bắt đầu Sanity Check...", "INFO")

        check_results = {
            'passed': True,
            'checks_performed': 0,
            'checks_passed': 0,
            'checks_failed': 0,
            'errors': [],
            'warnings': []
        }

        # Lấy danh sách tất cả node_id
        all_node_ids = set()
        if self.nodes_df is not None:
            # Đảm bảo node_id là string để so sánh chính xác
            self.nodes_df['node_id'] = self.nodes_df['node_id'].astype(str)
            all_node_ids = set(self.nodes_df['node_id'].unique())
            self.log(f"  Tổng số nodes: {len(all_node_ids):,}", "INFO")

        # ========== CHECK 1: edges_transfer node references ==========
        check_results['checks_performed'] += 1
        if self.edges_transfer_df is not None:
            self.log("  Kiểm tra edges_transfer.csv...", "INFO")

            # Đảm bảo ID là string
            self.edges_transfer_df['edge_id'] = self.edges_transfer_df['edge_id'].astype(str)
            self.edges_transfer_df['src_node_id'] = self.edges_transfer_df['src_node_id'].astype(str)
            self.edges_transfer_df['dst_node_id'] = self.edges_transfer_df['dst_node_id'].astype(str)

            # Kiểm tra src_node_id
            invalid_src = self.edges_transfer_df[~self.edges_transfer_df['src_node_id'].isin(all_node_ids)]
            if len(invalid_src) > 0:
                error_msg = f"edges_transfer.csv: {len(invalid_src)} dòng có src_node_id không tồn tại trong nodes.csv"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False
                check_results['checks_failed'] += 1

                # Chi tiết lỗi (tối đa 5 dòng đầu)
                for idx, row in invalid_src.head(5).iterrows():
                    detail = f"    - Dòng {idx}: edge_id={row['edge_id']}, src_node_id={row['src_node_id']}"
                    check_results['errors'].append(detail)
                    self.log(detail, "ERROR")
            else:
                self.log("    ✓ Tất cả src_node_id hợp lệ", "SUCCESS")

            # Kiểm tra dst_node_id
            invalid_dst = self.edges_transfer_df[~self.edges_transfer_df['dst_node_id'].isin(all_node_ids)]
            if len(invalid_dst) > 0:
                error_msg = f"edges_transfer.csv: {len(invalid_dst)} dòng có dst_node_id không tồn tại trong nodes.csv"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False

                for idx, row in invalid_dst.head(5).iterrows():
                    detail = f"    - Dòng {idx}: edge_id={row['edge_id']}, dst_node_id={row['dst_node_id']}"
                    check_results['errors'].append(detail)
                    self.log(detail, "ERROR")
            else:
                self.log("    ✓ Tất cả dst_node_id hợp lệ", "SUCCESS")
                check_results['checks_passed'] += 1

        # ========== CHECK 2: edges_uses_device node references ==========
        check_results['checks_performed'] += 1
        if self.edges_uses_device_df is not None:
            self.log("  Kiểm tra edges_uses_device.csv...", "INFO")

            self.edges_uses_device_df['src_node_id'] = self.edges_uses_device_df['src_node_id'].astype(str)

            # Tìm tên cột cho device node (có thể là device_node_id hoặc dst_node_id)
            device_col = 'device_node_id' if 'device_node_id' in self.edges_uses_device_df.columns else 'dst_node_id'
            self.edges_uses_device_df[device_col] = self.edges_uses_device_df[device_col].astype(str)

            invalid_src = self.edges_uses_device_df[~self.edges_uses_device_df['src_node_id'].isin(all_node_ids)]
            invalid_device = self.edges_uses_device_df[~self.edges_uses_device_df[device_col].isin(all_node_ids)]

            if len(invalid_src) > 0 or len(invalid_device) > 0:
                error_msg = f"edges_uses_device.csv: {len(invalid_src)} invalid src, {len(invalid_device)} invalid device"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False
                check_results['checks_failed'] += 1
            else:
                self.log("    ✓ Tất cả node references hợp lệ", "SUCCESS")
                check_results['checks_passed'] += 1

        # ========== CHECK 3: edges_uses_ip node references ==========
        check_results['checks_performed'] += 1
        if self.edges_uses_ip_df is not None:
            self.log("  Kiểm tra edges_uses_ip.csv...", "INFO")

            self.edges_uses_ip_df['src_node_id'] = self.edges_uses_ip_df['src_node_id'].astype(str)

            # Tìm tên cột cho IP node
            ip_col = 'ip_node_id' if 'ip_node_id' in self.edges_uses_ip_df.columns else 'dst_node_id'
            self.edges_uses_ip_df[ip_col] = self.edges_uses_ip_df[ip_col].astype(str)

            invalid_src = self.edges_uses_ip_df[~self.edges_uses_ip_df['src_node_id'].isin(all_node_ids)]
            invalid_ip = self.edges_uses_ip_df[~self.edges_uses_ip_df[ip_col].isin(all_node_ids)]

            if len(invalid_src) > 0 or len(invalid_ip) > 0:
                error_msg = f"edges_uses_ip.csv: {len(invalid_src)} invalid src, {len(invalid_ip)} invalid ip"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False
                check_results['checks_failed'] += 1
            else:
                self.log("    ✓ Tất cả node references hợp lệ", "SUCCESS")
                check_results['checks_passed'] += 1

        # ========== CHECK 4: edge_labels.csv edge_id match ==========
        check_results['checks_performed'] += 1
        if self.edge_labels_df is not None and self.edges_transfer_df is not None:
            self.log("  Kiểm tra edge_labels.csv...", "INFO")

            self.edge_labels_df['edge_id'] = self.edge_labels_df['edge_id'].astype(str)
            transfer_edge_ids = set(self.edges_transfer_df['edge_id'].unique())
            label_edge_ids = set(self.edge_labels_df['edge_id'].unique())

            # edge_id trong labels nhưng không có trong transfer
            missing_in_transfer = label_edge_ids - transfer_edge_ids
            if len(missing_in_transfer) > 0:
                error_msg = f"edge_labels.csv: {len(missing_in_transfer)} edge_id không tồn tại trong edges_transfer.csv"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False
                check_results['checks_failed'] += 1

                for edge_id in list(missing_in_transfer)[:5]:
                    detail = f"    - edge_id={edge_id} không tìm thấy"
                    check_results['errors'].append(detail)
                    self.log(detail, "ERROR")
            else:
                self.log("    ✓ Tất cả edge_id trong labels khớp với edges_transfer", "SUCCESS")
                check_results['checks_passed'] += 1

            # Cảnh báo: edge trong transfer nhưng không có label
            missing_in_labels = transfer_edge_ids - label_edge_ids
            if len(missing_in_labels) > 0:
                warning = f"edge_labels.csv: {len(missing_in_labels)} edges trong transfer không có label"
                check_results['warnings'].append(warning)
                self.log(f"    ⚠️ {warning}", "WARNING")

        # ========== CHECK 5: splits.csv edge_id match ==========
        check_results['checks_performed'] += 1
        if self.splits_df is not None and self.edges_transfer_df is not None:
            self.log("  Kiểm tra splits.csv...", "INFO")

            self.splits_df['edge_id'] = self.splits_df['edge_id'].astype(str)
            transfer_edge_ids = set(self.edges_transfer_df['edge_id'].unique())
            split_edge_ids = set(self.splits_df['edge_id'].unique())

            # edge_id trong splits nhưng không có trong transfer
            missing_in_transfer = split_edge_ids - transfer_edge_ids
            if len(missing_in_transfer) > 0:
                error_msg = f"splits.csv: {len(missing_in_transfer)} edge_id không tồn tại trong edges_transfer.csv"
                check_results['errors'].append(error_msg)
                check_results['passed'] = False
                check_results['checks_failed'] += 1

                for edge_id in list(missing_in_transfer)[:5]:
                    detail = f"    - edge_id={edge_id} không tìm thấy"
                    check_results['errors'].append(detail)
                    self.log(detail, "ERROR")
            else:
                self.log("    ✓ Tất cả edge_id trong splits khớp với edges_transfer", "SUCCESS")
                check_results['checks_passed'] += 1

            # Kiểm tra split values
            if 'split' in self.splits_df.columns:
                valid_splits = {'train', 'val', 'test', 'validation'}
                actual_splits = set(self.splits_df['split'].str.lower().unique())
                invalid_splits = actual_splits - valid_splits

                if len(invalid_splits) > 0:
                    warning = f"splits.csv: Giá trị split không hợp lệ: {invalid_splits}"
                    check_results['warnings'].append(warning)
                    self.log(f"    ⚠️ {warning}", "WARNING")
                else:
                    split_counts = self.splits_df['split'].value_counts().to_dict()
                    self.log(f"    ✓ Split distribution: {split_counts}", "SUCCESS")

        # ========== CHECK 6: Kiểm tra format ID (leading zeros) ==========
        check_results['checks_performed'] += 1
        self.log("  Kiểm tra format ID (leading zeros)...", "INFO")

        id_format_issues = []

        # Kiểm tra edge_id format consistency
        if self.edges_transfer_df is not None and self.edge_labels_df is not None:
            # Lấy sample để so sánh format
            transfer_sample = list(self.edges_transfer_df['edge_id'].head(5))
            labels_sample = list(self.edge_labels_df['edge_id'].head(5))

            # Kiểm tra độ dài ID có nhất quán không
            transfer_lengths = [len(str(x)) for x in self.edges_transfer_df['edge_id'].unique()[:100]]
            labels_lengths = [len(str(x)) for x in self.edge_labels_df['edge_id'].unique()[:100]]

            if len(set(transfer_lengths)) > 1:
                id_format_issues.append("edges_transfer.csv: edge_id có độ dài không nhất quán")
            if len(set(labels_lengths)) > 1:
                id_format_issues.append("edge_labels.csv: edge_id có độ dài không nhất quán")

        if id_format_issues:
            for issue in id_format_issues:
                check_results['warnings'].append(issue)
                self.log(f"    ⚠️ {issue}", "WARNING")
        else:
            self.log("    ✓ Format ID nhất quán", "SUCCESS")
            check_results['checks_passed'] += 1

        # ========== TỔNG KẾT ==========
        self.log("=" * 50, "INFO")
        if check_results['passed']:
            self.log(f"SANITY CHECK PASSED: {check_results['checks_passed']}/{check_results['checks_performed']} checks", "SUCCESS")
        else:
            self.log(f"SANITY CHECK FAILED: {check_results['checks_failed']} lỗi phát hiện", "ERROR")

            # Raise exception với chi tiết lỗi
            error_summary = "\n".join(check_results['errors'][:10])
            raise ValueError(f"Sanity Check Failed!\n{error_summary}")

        return check_results

    def build_hetero_graph(self) -> HeteroData:
        """
        Xây dựng Heterogeneous Graph từ dữ liệu đã load

        Graph structure:
        - Node types: user, recipient, device, ip
        - Edge types:
            - transfer: user → recipient (CÓ LABEL)
            - uses_device: user → device
            - uses_ip: user → ip

        Returns:
            HeteroData object (PyTorch Geometric)
        """
        self.log("Bắt đầu xây dựng Heterogeneous Graph...", "INFO")

        data = HeteroData()

        # ========== BƯỚC 1: Tạo node mappings ==========
        self.log("  Bước 1: Tạo node mappings...", "INFO")

        # Phân loại nodes theo type
        node_types = self.nodes_df['node_type'].unique()
        self.log(f"    Node types: {list(node_types)}", "INFO")

        for node_type in node_types:
            type_df = self.nodes_df[self.nodes_df['node_type'] == node_type].reset_index(drop=True)

            self.node_id_to_idx[node_type] = {
                str(row['node_id']): idx for idx, row in type_df.iterrows()
            }
            self.idx_to_node_id[node_type] = {
                idx: str(row['node_id']) for idx, row in type_df.iterrows()
            }

            self.log(f"    {node_type}: {len(type_df)} nodes", "SUCCESS")

        # ========== BƯỚC 2: Tạo node features ==========
        self.log("  Bước 2: Tạo node features...", "INFO")

        for node_type in node_types:
            type_df = self.nodes_df[self.nodes_df['node_type'] == node_type].reset_index(drop=True)

            # Lấy các cột feature (loại bỏ các cột metadata)
            exclude_cols = ['node_id', 'node_type', 'raw_id', 'Unnamed: 0']
            feature_cols = [c for c in type_df.columns if c not in exclude_cols]

            if feature_cols:
                # Chuyển đổi features sang numeric
                features = type_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                node_features = torch.FloatTensor(features.values)
            else:
                # Nếu không có features, tạo embedding learnable
                node_features = torch.ones((len(type_df), 1))

            data[node_type].x = node_features
            data[node_type].num_nodes = len(type_df)

            self.log(f"    {node_type}: features shape = {node_features.shape}", "SUCCESS")

        # ========== BƯỚC 3: Tạo edges transfer (user → recipient) ==========
        self.log("  Bước 3: Tạo edges transfer...", "INFO")

        if self.edges_transfer_df is not None:
            # Xác định node types cho src và dst
            src_type = 'user'
            dst_type = 'recipient'

            # Kiểm tra xem có đủ node types không
            if src_type not in self.node_id_to_idx:
                # Có thể tất cả đều là một loại node
                available_types = list(self.node_id_to_idx.keys())
                src_type = available_types[0]
                dst_type = available_types[0] if len(available_types) == 1 else available_types[1]

            src_indices = []
            dst_indices = []
            valid_edge_indices = []  # Lưu index của các edge hợp lệ

            for idx, row in self.edges_transfer_df.iterrows():
                src_id = str(row['src_node_id'])
                dst_id = str(row['dst_node_id'])

                # Tìm node type chứa src_id và dst_id
                src_idx = None
                dst_idx = None

                for nt in self.node_id_to_idx:
                    if src_id in self.node_id_to_idx[nt]:
                        src_idx = self.node_id_to_idx[nt][src_id]
                        src_type_actual = nt
                    if dst_id in self.node_id_to_idx[nt]:
                        dst_idx = self.node_id_to_idx[nt][dst_id]
                        dst_type_actual = nt

                if src_idx is not None and dst_idx is not None:
                    src_indices.append(src_idx)
                    dst_indices.append(dst_idx)
                    valid_edge_indices.append(idx)

            edge_index = torch.LongTensor([src_indices, dst_indices])
            data[src_type, 'transfer', dst_type].edge_index = edge_index

            # Lấy edge features (loại bỏ các cột metadata)
            exclude_cols = ['edge_id', 'src_node_id', 'dst_node_id', 'edge_type', 'timestamp', 'Unnamed: 0']
            feature_cols = [c for c in self.edges_transfer_df.columns if c not in exclude_cols]

            if feature_cols:
                edge_features = self.edges_transfer_df.loc[valid_edge_indices, feature_cols]
                edge_features = edge_features.apply(pd.to_numeric, errors='coerce').fillna(0)
                data[src_type, 'transfer', dst_type].edge_attr = torch.FloatTensor(edge_features.values)
                self.log(f"    Edge features: {len(feature_cols)} features", "SUCCESS")

            self.log(f"    transfer edges: {len(src_indices):,}", "SUCCESS")

            # Lưu mapping edge_id -> index
            valid_edge_ids = self.edges_transfer_df.loc[valid_edge_indices, 'edge_id'].astype(str).tolist()
            self.edge_id_to_idx = {eid: i for i, eid in enumerate(valid_edge_ids)}

        # ========== BƯỚC 4: Tạo edges uses_device (user → device) ==========
        self.log("  Bước 4: Tạo edges uses_device...", "INFO")

        if self.edges_uses_device_df is not None and 'device' in self.node_id_to_idx:
            device_col = 'device_node_id' if 'device_node_id' in self.edges_uses_device_df.columns else 'dst_node_id'

            src_indices = []
            dst_indices = []

            for _, row in self.edges_uses_device_df.iterrows():
                src_id = str(row['src_node_id'])
                device_id = str(row[device_col])

                # Tìm trong user nodes
                src_idx = None
                for nt in self.node_id_to_idx:
                    if src_id in self.node_id_to_idx[nt]:
                        src_idx = self.node_id_to_idx[nt][src_id]
                        src_type = nt
                        break

                device_idx = self.node_id_to_idx.get('device', {}).get(device_id)

                if src_idx is not None and device_idx is not None:
                    src_indices.append(src_idx)
                    dst_indices.append(device_idx)

            if src_indices:
                edge_index = torch.LongTensor([src_indices, dst_indices])
                data[src_type, 'uses_device', 'device'].edge_index = edge_index
                self.log(f"    uses_device edges: {len(src_indices):,}", "SUCCESS")
        else:
            self.log("    uses_device edges: 0 (không có dữ liệu)", "WARNING")

        # ========== BƯỚC 5: Tạo edges uses_ip (user → ip) ==========
        self.log("  Bước 5: Tạo edges uses_ip...", "INFO")

        if self.edges_uses_ip_df is not None and 'ip' in self.node_id_to_idx:
            ip_col = 'ip_node_id' if 'ip_node_id' in self.edges_uses_ip_df.columns else 'dst_node_id'

            src_indices = []
            dst_indices = []

            for _, row in self.edges_uses_ip_df.iterrows():
                src_id = str(row['src_node_id'])
                ip_id = str(row[ip_col])

                src_idx = None
                for nt in self.node_id_to_idx:
                    if src_id in self.node_id_to_idx[nt]:
                        src_idx = self.node_id_to_idx[nt][src_id]
                        src_type = nt
                        break

                ip_idx = self.node_id_to_idx.get('ip', {}).get(ip_id)

                if src_idx is not None and ip_idx is not None:
                    src_indices.append(src_idx)
                    dst_indices.append(ip_idx)

            if src_indices:
                edge_index = torch.LongTensor([src_indices, dst_indices])
                data[src_type, 'uses_ip', 'ip'].edge_index = edge_index
                self.log(f"    uses_ip edges: {len(src_indices):,}", "SUCCESS")
        else:
            self.log("    uses_ip edges: 0 (không có dữ liệu)", "WARNING")

        # ========== BƯỚC 6: Gán labels cho edges transfer ==========
        self.log("  Bước 6: Gán labels cho edges transfer...", "INFO")

        if self.edge_labels_df is not None:
            # Tạo mảng labels theo thứ tự edge index
            num_edges = len(self.edge_id_to_idx)
            edge_labels = torch.zeros(num_edges, dtype=torch.long)

            labeled_count = 0
            for _, row in self.edge_labels_df.iterrows():
                edge_id = str(row['edge_id'])
                if edge_id in self.edge_id_to_idx:
                    edge_idx = self.edge_id_to_idx[edge_id]
                    edge_labels[edge_idx] = int(row['label'])
                    labeled_count += 1

            # Lưu labels vào edge type
            for edge_type in data.edge_types:
                if edge_type[1] == 'transfer':
                    data[edge_type].y = edge_labels
                    break

            self.log(f"    Labeled edges: {labeled_count:,}/{num_edges:,}", "SUCCESS")

        # ========== BƯỚC 7: Tạo train/val/test masks ==========
        self.log("  Bước 7: Tạo train/val/test masks...", "INFO")

        if self.splits_df is not None:
            num_edges = len(self.edge_id_to_idx)
            train_mask = torch.zeros(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask = torch.zeros(num_edges, dtype=torch.bool)

            for _, row in self.splits_df.iterrows():
                edge_id = str(row['edge_id'])
                if edge_id in self.edge_id_to_idx:
                    edge_idx = self.edge_id_to_idx[edge_id]
                    split = row['split'].lower()

                    if split == 'train':
                        train_mask[edge_idx] = True
                    elif split in ['val', 'validation']:
                        val_mask[edge_idx] = True
                    elif split == 'test':
                        test_mask[edge_idx] = True

            # Lưu masks
            for edge_type in data.edge_types:
                if edge_type[1] == 'transfer':
                    data[edge_type].train_mask = train_mask
                    data[edge_type].val_mask = val_mask
                    data[edge_type].test_mask = test_mask
                    break

            self.log(f"    Train: {train_mask.sum().item():,}, Val: {val_mask.sum().item():,}, Test: {test_mask.sum().item():,}", "SUCCESS")

        # ========== TỔNG KẾT ==========
        self.log("=" * 50, "INFO")
        self.log(f"Graph xây dựng hoàn tất!", "SUCCESS")
        self.log(f"  Node types: {data.node_types}", "INFO")
        self.log(f"  Edge types: {data.edge_types}", "INFO")

        # Thống kê
        for node_type in data.node_types:
            self.log(f"  {node_type}: {data[node_type].num_nodes} nodes", "INFO")

        for edge_type in data.edge_types:
            num_edges = data[edge_type].edge_index.shape[1]
            self.log(f"  {edge_type}: {num_edges} edges", "INFO")

        return data

    def save_graph(self, data: HeteroData, output_path: str = None):
        """
        Lưu graph đã xây dựng

        Args:
            data: HeteroData object
            output_path: Đường dẫn file output
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'hetero_graph.pt')

        # Lưu graph
        torch.save(data, output_path)
        self.log(f"Đã lưu graph: {output_path}", "SUCCESS")

        # Lưu mappings
        mappings_path = os.path.join(os.path.dirname(output_path), 'graph_mappings.json')
        mappings = {
            'node_id_to_idx': self.node_id_to_idx,
            'idx_to_node_id': self.idx_to_node_id,
            'edge_id_to_idx': self.edge_id_to_idx
        }
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2)
        self.log(f"Đã lưu mappings: {mappings_path}", "SUCCESS")

        # Tạo file flag
        flag_path = os.path.join(os.path.dirname(output_path), 'graph_ready.flag')
        with open(flag_path, 'w') as f:
            f.write(f"Graph built at {pd.Timestamp.now()}\n")
            f.write(f"Graph path: {output_path}\n")
        self.log(f"Đã tạo flag file: {flag_path}", "SUCCESS")

        return output_path

    def load_graph(self, graph_path: str = None) -> HeteroData:
        """
        Load graph đã lưu

        Args:
            graph_path: Đường dẫn file graph

        Returns:
            HeteroData object
        """
        if graph_path is None:
            graph_path = os.path.join(self.data_dir, 'hetero_graph.pt')

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Không tìm thấy graph: {graph_path}")

        data = torch.load(graph_path)
        self.log(f"Đã load graph: {graph_path}", "SUCCESS")

        # Load mappings
        mappings_path = os.path.join(os.path.dirname(graph_path), 'graph_mappings.json')
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.node_id_to_idx = mappings.get('node_id_to_idx', {})
                self.idx_to_node_id = mappings.get('idx_to_node_id', {})
                self.edge_id_to_idx = mappings.get('edge_id_to_idx', {})

        return data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về dữ liệu

        Returns:
            Dict chứa thống kê
        """
        stats = {
            'data_dir': self.data_dir,
            'nodes': {},
            'edges': {},
            'labels': {},
            'splits': {}
        }

        # Thống kê nodes
        if self.nodes_df is not None:
            stats['nodes']['total'] = len(self.nodes_df)
            stats['nodes']['by_type'] = self.nodes_df['node_type'].value_counts().to_dict()

        # Thống kê edges
        if self.edges_transfer_df is not None:
            stats['edges']['transfer'] = len(self.edges_transfer_df)
        if self.edges_uses_device_df is not None:
            stats['edges']['uses_device'] = len(self.edges_uses_device_df)
        if self.edges_uses_ip_df is not None:
            stats['edges']['uses_ip'] = len(self.edges_uses_ip_df)

        # Thống kê labels
        if self.edge_labels_df is not None:
            stats['labels']['total'] = len(self.edge_labels_df)
            stats['labels']['distribution'] = self.edge_labels_df['label'].value_counts().to_dict()
            stats['labels']['fraud_ratio'] = float(self.edge_labels_df['label'].mean())

        # Thống kê splits
        if self.splits_df is not None:
            stats['splits']['total'] = len(self.splits_df)
            stats['splits']['distribution'] = self.splits_df['split'].value_counts().to_dict()

        return stats


def run_gnn_data_pipeline(data_dir: str, output_dir: str = None, verbose: bool = True) -> Tuple[HeteroData, Dict]:
    """
    Chạy toàn bộ pipeline xử lý dữ liệu GNN

    Args:
        data_dir: Thư mục chứa dữ liệu GNN
        output_dir: Thư mục lưu output (mặc định = data_dir)
        verbose: In thông tin chi tiết

    Returns:
        Tuple (HeteroData, statistics dict)
    """
    print("\n" + "=" * 60)
    print("🕸️  GNN DATA PIPELINE - Xây dựng mạng lưới GNN")
    print("=" * 60 + "\n")

    if output_dir is None:
        output_dir = data_dir

    # Khởi tạo pipeline
    pipeline = GNNDataPipeline(data_dir, verbose=verbose)

    # Bước 1: Load dữ liệu
    print("\n📁 BƯỚC 1: Load dữ liệu")
    print("-" * 40)
    load_results = pipeline.load_all_data()

    if not load_results['success']:
        raise ValueError(f"Load dữ liệu thất bại: {load_results['errors']}")

    # Bước 2: Sanity check
    print("\n🔍 BƯỚC 2: Sanity Check")
    print("-" * 40)
    check_results = pipeline.sanity_check()

    # Bước 3: Build graph
    print("\n🔨 BƯỚC 3: Build Heterogeneous Graph")
    print("-" * 40)
    data = pipeline.build_hetero_graph()

    # Bước 4: Save graph
    print("\n💾 BƯỚC 4: Lưu Graph")
    print("-" * 40)
    graph_path = os.path.join(output_dir, 'hetero_graph.pt')
    pipeline.save_graph(data, graph_path)

    # Lấy thống kê
    stats = pipeline.get_statistics()

    print("\n" + "=" * 60)
    print("✅ PIPELINE HOÀN TẤT!")
    print("=" * 60)

    return data, stats


if __name__ == '__main__':
    # Test với thư mục mặc định
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gnn_data')

    if os.path.exists(data_dir):
        data, stats = run_gnn_data_pipeline(data_dir)
        print("\nThống kê:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Thư mục không tồn tại: {data_dir}")

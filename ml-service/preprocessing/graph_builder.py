"""
Graph Builder - Xây dựng graph cho GNN
======================================
Module này tạo graph từ dữ liệu giao dịch:
- Nodes: Users và Merchants
- Edges: Giao dịch giữa các nodes
- Edge attributes: Thông tin giao dịch
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()


class GraphBuilder:
    """
    Class xây dựng graph từ dữ liệu giao dịch
    Graph biểu diễn mối quan hệ giữa users và merchants/recipients
    """

    def __init__(self, verbose: bool = True):
        """
        Khởi tạo GraphBuilder

        Args:
            verbose: In thông tin chi tiết
        """
        self.verbose = verbose
        self.node_mapping = {}  # Map node_id -> index
        self.reverse_mapping = {}  # Map index -> node_id
        self.node_types = {}  # Map node_id -> type (user/merchant)

    def log(self, message: str):
        """In log nếu verbose mode"""
        if self.verbose:
            print(f"[GraphBuilder] {message}")

    def build_transaction_graph(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame = None
    ) -> Dict:
        """
        Xây dựng graph từ dữ liệu giao dịch

        Args:
            transactions_df: DataFrame giao dịch
            users_df: DataFrame users (optional, cho node features)

        Returns:
            Dict chứa thông tin graph
        """
        self.log("Bắt đầu xây dựng graph...")

        # Step 1: Tạo nodes
        nodes = self._create_nodes(transactions_df, users_df)

        # Step 2: Tạo edges
        edges = self._create_edges(transactions_df)

        # Step 3: Tạo node features
        node_features = self._create_node_features(transactions_df, users_df)

        # Step 4: Tạo edge features
        edge_features = self._create_edge_features(transactions_df)

        # Step 5: Tạo adjacency list
        adjacency = self._create_adjacency_list(edges)

        graph = {
            'nodes': nodes,
            'edges': edges,
            'node_features': node_features,
            'edge_features': edge_features,
            'adjacency': adjacency,
            'node_mapping': self.node_mapping,
            'node_types': self.node_types,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }

        self.log(f"Hoàn tất! Graph có {graph['num_nodes']:,} nodes và {graph['num_edges']:,} edges")
        return graph

    def _create_nodes(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame = None
    ) -> List[Dict]:
        """
        Tạo danh sách nodes

        Mỗi user là một node
        Mỗi merchant/receiving_bank cũng là một node
        """
        self.log("  Tạo nodes...")

        nodes = []
        node_idx = 0

        # Tạo nodes từ users
        unique_users = transactions_df['user_id'].unique()
        for user_id in unique_users:
            self.node_mapping[user_id] = node_idx
            self.reverse_mapping[node_idx] = user_id
            self.node_types[user_id] = 'user'

            node_data = {
                'id': node_idx,
                'original_id': user_id,
                'type': 'user'
            }

            # Thêm user info nếu có
            if users_df is not None and 'user_id' in users_df.columns:
                user_info = users_df[users_df['user_id'] == user_id]
                if len(user_info) > 0:
                    user_row = user_info.iloc[0]
                    for col in ['age', 'monthly_income', 'credit_score']:
                        if col in user_row:
                            node_data[col] = user_row[col]

            nodes.append(node_data)
            node_idx += 1

        # Tạo nodes từ receiving banks/merchants
        if 'receiving_bank' in transactions_df.columns:
            unique_recipients = transactions_df['receiving_bank'].dropna().unique()
            for recipient in unique_recipients:
                if recipient not in self.node_mapping:
                    self.node_mapping[recipient] = node_idx
                    self.reverse_mapping[node_idx] = recipient
                    self.node_types[recipient] = 'merchant'

                    nodes.append({
                        'id': node_idx,
                        'original_id': recipient,
                        'type': 'merchant'
                    })
                    node_idx += 1

        # Tạo nodes từ merchant categories
        if 'merchant_category' in transactions_df.columns:
            unique_merchants = transactions_df['merchant_category'].dropna().unique()
            for merchant in unique_merchants:
                if merchant not in self.node_mapping and merchant != 'unknown':
                    self.node_mapping[merchant] = node_idx
                    self.reverse_mapping[node_idx] = merchant
                    self.node_types[merchant] = 'merchant'

                    nodes.append({
                        'id': node_idx,
                        'original_id': merchant,
                        'type': 'merchant'
                    })
                    node_idx += 1

        self.log(f"    Tạo được {len(nodes):,} nodes ({len(unique_users):,} users)")
        return nodes

    def _create_edges(self, transactions_df: pd.DataFrame) -> List[Dict]:
        """
        Tạo danh sách edges từ giao dịch

        Edge: user -> recipient (receiving_bank hoặc merchant)
        """
        self.log("  Tạo edges...")

        edges = []

        for idx, row in transactions_df.iterrows():
            user_id = row['user_id']
            if user_id not in self.node_mapping:
                continue

            source_idx = self.node_mapping[user_id]

            # Edge đến receiving bank
            if pd.notna(row.get('receiving_bank')):
                recipient = row['receiving_bank']
                if recipient in self.node_mapping:
                    target_idx = self.node_mapping[recipient]
                    edges.append({
                        'source': source_idx,
                        'target': target_idx,
                        'transaction_id': row.get('transaction_id', idx),
                        'amount': row['amount'],
                        'is_fraud': row.get('is_fraud', 0)
                    })

            # Edge đến merchant category
            if pd.notna(row.get('merchant_category')) and row['merchant_category'] != 'unknown':
                merchant = row['merchant_category']
                if merchant in self.node_mapping:
                    target_idx = self.node_mapping[merchant]
                    edges.append({
                        'source': source_idx,
                        'target': target_idx,
                        'transaction_id': row.get('transaction_id', idx),
                        'amount': row['amount'],
                        'is_fraud': row.get('is_fraud', 0)
                    })

        self.log(f"    Tạo được {len(edges):,} edges")
        return edges

    def _create_node_features(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Tạo feature vector cho mỗi node

        Features cho user nodes:
        - Số giao dịch
        - Tổng số tiền
        - Số tiền trung bình
        - Tỷ lệ fraud (nếu có)
        """
        self.log("  Tạo node features...")

        num_nodes = len(self.node_mapping)
        feature_dim = 8  # Số features cho mỗi node

        features = np.zeros((num_nodes, feature_dim))

        # Thống kê giao dịch theo user
        user_stats = transactions_df.groupby('user_id').agg({
            'amount': ['count', 'sum', 'mean', 'std', 'max'],
            'is_fraud': ['sum', 'mean'] if 'is_fraud' in transactions_df.columns else ['count']
        })

        for user_id, node_idx in self.node_mapping.items():
            if self.node_types.get(user_id) == 'user':
                if user_id in user_stats.index:
                    stats = user_stats.loc[user_id]

                    features[node_idx, 0] = stats[('amount', 'count')]
                    features[node_idx, 1] = np.log1p(stats[('amount', 'sum')])
                    features[node_idx, 2] = np.log1p(stats[('amount', 'mean')])
                    features[node_idx, 3] = np.log1p(stats.get(('amount', 'std'), 0) or 0)
                    features[node_idx, 4] = np.log1p(stats[('amount', 'max')])

                    if 'is_fraud' in transactions_df.columns:
                        features[node_idx, 5] = stats[('is_fraud', 'sum')]
                        features[node_idx, 6] = stats[('is_fraud', 'mean')]

                # Thêm user profile features
                if users_df is not None and 'user_id' in users_df.columns:
                    user_info = users_df[users_df['user_id'] == user_id]
                    if len(user_info) > 0:
                        features[node_idx, 7] = user_info.iloc[0].get('credit_score', 650) / 850

            else:
                # Merchant node - features khác
                features[node_idx, 0] = 1  # Type indicator

        return features

    def _create_edge_features(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Tạo feature vector cho mỗi edge

        Features:
        - Amount (log)
        - Is fraud
        - Hour
        - Day of week
        """
        self.log("  Tạo edge features...")

        # Đếm số edges thực tế
        num_edges = 0
        for _, row in transactions_df.iterrows():
            if pd.notna(row.get('receiving_bank')) or pd.notna(row.get('merchant_category')):
                num_edges += 1

        feature_dim = 4
        features = np.zeros((num_edges, feature_dim))

        edge_idx = 0
        for _, row in transactions_df.iterrows():
            if pd.notna(row.get('receiving_bank')) or pd.notna(row.get('merchant_category')):
                features[edge_idx, 0] = np.log1p(row['amount'])
                features[edge_idx, 1] = row.get('is_fraud', 0)

                if 'timestamp' in row:
                    try:
                        ts = pd.to_datetime(row['timestamp'])
                        features[edge_idx, 2] = ts.hour / 24
                        features[edge_idx, 3] = ts.dayofweek / 7
                    except:
                        pass

                edge_idx += 1

        return features[:edge_idx]

    def _create_adjacency_list(self, edges: List[Dict]) -> Dict[int, List[int]]:
        """
        Tạo adjacency list từ danh sách edges
        """
        self.log("  Tạo adjacency list...")

        adjacency = defaultdict(list)

        for edge in edges:
            source = edge['source']
            target = edge['target']
            adjacency[source].append(target)
            # Undirected graph
            adjacency[target].append(source)

        # Convert to regular dict
        return dict(adjacency)

    def get_edge_index(self, edges: List[Dict]) -> np.ndarray:
        """
        Lấy edge index cho PyTorch Geometric

        Returns:
            np.ndarray shape (2, num_edges)
        """
        sources = [e['source'] for e in edges]
        targets = [e['target'] for e in edges]

        return np.array([sources, targets])

    def save_graph(self, graph: Dict, output_dir: str = None):
        """
        Lưu graph ra files

        Args:
            graph: Dict chứa thông tin graph
            output_dir: Thư mục lưu
        """
        if output_dir is None:
            output_dir = config.DATA_PROCESSED_DIR

        os.makedirs(output_dir, exist_ok=True)

        # Lưu edges
        edges_df = pd.DataFrame(graph['edges'])
        edges_df.to_csv(os.path.join(output_dir, 'graph_edges.csv'), index=False)

        # Lưu node features
        np.save(os.path.join(output_dir, 'node_features.npy'), graph['node_features'])

        # Lưu edge features
        np.save(os.path.join(output_dir, 'edge_features.npy'), graph['edge_features'])

        # Lưu mapping
        import json
        with open(os.path.join(output_dir, 'node_mapping.json'), 'w') as f:
            # Convert keys to string for JSON
            mapping = {str(k): v for k, v in graph['node_mapping'].items()}
            json.dump(mapping, f, indent=2)

        self.log(f"[SAVED] Graph data to {output_dir}")


def build_graph_from_data(
    transactions_path: str = None,
    users_path: str = None,
    output_dir: str = None
) -> Dict:
    """
    Hàm tiện ích xây dựng graph từ dữ liệu

    Args:
        transactions_path: Đường dẫn file giao dịch
        users_path: Đường dẫn file users
        output_dir: Thư mục lưu output

    Returns:
        Dict chứa thông tin graph
    """
    if transactions_path is None:
        transactions_path = os.path.join(config.DATA_PROCESSED_DIR, 'transactions_clean.csv')
    if users_path is None:
        users_path = os.path.join(config.DATA_PROCESSED_DIR, 'users_clean.csv')
    if output_dir is None:
        output_dir = config.DATA_PROCESSED_DIR

    print("\n" + "=" * 50)
    print("XÂY DỰNG GRAPH CHO GNN")
    print("=" * 50)

    # Đọc dữ liệu
    transactions_df = pd.read_csv(transactions_path)

    users_df = None
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)

    # Xây dựng graph
    builder = GraphBuilder(verbose=True)
    graph = builder.build_transaction_graph(transactions_df, users_df)

    # Lưu graph
    builder.save_graph(graph, output_dir)

    return graph


if __name__ == '__main__':
    build_graph_from_data()

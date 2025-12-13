"""
GNN Training Script - 2 Bước Riêng Biệt
========================================
Script này cung cấp 2 chức năng độc lập:

BƯỚC 1: Tạo mạng lưới GNN (build_gnn_graph)
    - Load dữ liệu từ gnn_data/
    - Kiểm tra tính toàn vẹn (Sanity Check)
    - Build Heterogeneous Graph
    - Lưu graph và tạo flag file

BƯỚC 2: Huấn luyện GNN (train_gnn_model)
    - Kiểm tra graph_ready.flag
    - Load graph đã build
    - Train HeteroGNN Model
    - In metrics và lưu model

Author: ML Team - Agribank Vietnam
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Thêm path để import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from preprocessing.gnn_data_pipeline import GNNDataPipeline, load_gnn_graph, check_graph_ready
from models.layer2.hetero_gnn_model import HeteroGNNModel

# Đường dẫn mặc định
GNN_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gnn_data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')


def build_gnn_graph(data_dir: str = None, verbose: bool = True) -> dict:
    """
    BƯỚC 1: TẠO MẠNG LƯỚI GNN

    Thực hiện:
    1. Load dữ liệu từ gnn_data/
    2. Kiểm tra tính toàn vẹn (Sanity Check)
    3. Build Heterogeneous Graph (PyTorch Geometric)
    4. Lưu graph ra file (hetero_graph.pt)
    5. Tạo flag file (graph_ready.flag)

    Args:
        data_dir: Đường dẫn thư mục gnn_data/
        verbose: In thông tin chi tiết

    Returns:
        dict: Kết quả với các thông tin về graph đã build
    """
    if data_dir is None:
        data_dir = GNN_DATA_DIR

    result = {
        'success': False,
        'step': 'build_graph',
        'message': '',
        'graph_path': None,
        'flag_path': None,
        'stats': {},
        'started_at': datetime.now().isoformat()
    }

    try:
        print("\n" + "*" * 70)
        print("     BƯỚC 1: TẠO MẠNG LƯỚI GNN")
        print("     Fraud Detection System - Agribank Vietnam")
        print("*" * 70 + "\n")

        # Khởi tạo pipeline
        pipeline = GNNDataPipeline(data_dir=data_dir, verbose=verbose)

        # Chạy toàn bộ pipeline
        data, graph_path = pipeline.run_full_pipeline()

        # Thống kê
        transfer_key = ('user', 'transfer', 'recipient')
        num_transfer_edges = data[transfer_key].edge_index.shape[1]
        num_fraud = data[transfer_key].y.sum().item()
        num_normal = num_transfer_edges - num_fraud

        stats = {
            'num_users': data['user'].x.shape[0],
            'num_recipients': data['recipient'].x.shape[0],
            'num_devices': data['device'].x.shape[0],
            'num_ips': data['ip'].x.shape[0],
            'num_transfer_edges': num_transfer_edges,
            'num_uses_device_edges': data['user', 'uses_device', 'device'].edge_index.shape[1],
            'num_uses_ip_edges': data['user', 'uses_ip', 'ip'].edge_index.shape[1],
            'num_fraud': num_fraud,
            'num_normal': num_normal,
            'fraud_ratio': num_fraud / num_transfer_edges if num_transfer_edges > 0 else 0,
            'train_edges': data[transfer_key].train_mask.sum().item(),
            'val_edges': data[transfer_key].val_mask.sum().item(),
            'test_edges': data[transfer_key].test_mask.sum().item(),
            'node_feature_dim_user': data['user'].x.shape[1],
            'edge_feature_dim': data[transfer_key].edge_attr.shape[1]
        }

        flag_path = os.path.join(data_dir, 'graph_ready.flag')

        result['success'] = True
        result['message'] = 'Tạo mạng lưới GNN thành công!'
        result['graph_path'] = graph_path
        result['flag_path'] = flag_path
        result['stats'] = stats
        result['completed_at'] = datetime.now().isoformat()

        # In kết quả
        print("\n" + "=" * 70)
        print("KẾT QUẢ BƯỚC 1: TẠO MẠNG LƯỚI GNN")
        print("=" * 70)
        print(f"  ✓ Graph đã được lưu: {graph_path}")
        print(f"  ✓ Flag file: {flag_path}")
        print("\n  THỐNG KÊ GRAPH:")
        print(f"    - Số users:           {stats['num_users']}")
        print(f"    - Số recipients:      {stats['num_recipients']}")
        print(f"    - Số devices:         {stats['num_devices']}")
        print(f"    - Số IPs:             {stats['num_ips']}")
        print(f"    - Số transfer edges:  {stats['num_transfer_edges']}")
        print(f"    - Fraud edges:        {stats['num_fraud']} ({stats['fraud_ratio']*100:.1f}%)")
        print(f"    - Normal edges:       {stats['num_normal']}")
        print(f"    - Train/Val/Test:     {stats['train_edges']}/{stats['val_edges']}/{stats['test_edges']}")
        print("=" * 70)

        return result

    except Exception as e:
        result['success'] = False
        result['message'] = f'Lỗi: {str(e)}'
        result['error'] = str(e)
        result['completed_at'] = datetime.now().isoformat()

        print("\n" + "=" * 70)
        print("✗ BƯỚC 1 THẤT BẠI!")
        print(f"  Lỗi: {str(e)}")
        print("=" * 70)

        return result


def train_gnn_model(
    data_dir: str = None,
    output_dir: str = None,
    config: dict = None,
    verbose: bool = True
) -> dict:
    """
    BƯỚC 2: HUẤN LUYỆN GNN

    Yêu cầu: graph_ready.flag phải tồn tại (Bước 1 đã hoàn thành)

    Thực hiện:
    1. Kiểm tra graph_ready.flag
    2. Load graph đã build
    3. Train HeteroGNN Model
    4. Đánh giá trên test set
    5. In metrics và lưu model

    Args:
        data_dir: Đường dẫn thư mục gnn_data/
        output_dir: Đường dẫn lưu model
        config: Cấu hình model (optional)
        verbose: In thông tin chi tiết

    Returns:
        dict: Kết quả training với metrics
    """
    if data_dir is None:
        data_dir = GNN_DATA_DIR
    if output_dir is None:
        output_dir = MODELS_DIR

    result = {
        'success': False,
        'step': 'train_model',
        'message': '',
        'model_path': None,
        'metrics': {},
        'started_at': datetime.now().isoformat()
    }

    try:
        print("\n" + "*" * 70)
        print("     BƯỚC 2: HUẤN LUYỆN GNN")
        print("     Fraud Detection System - Agribank Vietnam")
        print("*" * 70 + "\n")

        # ========================================
        # KIỂM TRA GRAPH_READY.FLAG
        # ========================================
        print("[1/4] Kiểm tra graph_ready.flag...")

        flag_path = os.path.join(data_dir, 'graph_ready.flag')
        graph_path = os.path.join(data_dir, 'hetero_graph.pt')

        if not os.path.exists(flag_path):
            raise Exception(
                f"Không tìm thấy flag file: {flag_path}\n"
                "  → Vui lòng chạy Bước 1 'Tạo mạng lưới GNN' trước!"
            )

        if not os.path.exists(graph_path):
            raise Exception(
                f"Không tìm thấy graph file: {graph_path}\n"
                "  → Vui lòng chạy Bước 1 'Tạo mạng lưới GNN' trước!"
            )

        print(f"  ✓ Flag file tồn tại: {flag_path}")
        print(f"  ✓ Graph file tồn tại: {graph_path}")

        # ========================================
        # LOAD GRAPH
        # ========================================
        print("\n[2/4] Load graph đã build...")

        data, metadata = load_gnn_graph(graph_path)
        print(f"  ✓ Đã load graph thành công")

        transfer_key = ('user', 'transfer', 'recipient')
        num_edges = data[transfer_key].edge_index.shape[1]
        print(f"  - Số transfer edges: {num_edges}")

        # ========================================
        # TRAIN MODEL
        # ========================================
        print("\n[3/4] Huấn luyện HeteroGNN Model...")

        # Cấu hình mặc định
        default_config = {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15,
            'weight_decay': 5e-4
        }

        if config:
            default_config.update(config)

        model = HeteroGNNModel(model_config=default_config)
        train_result = model.fit(data, verbose=verbose)

        # ========================================
        # ĐÁNH GIÁ TRÊN TEST SET
        # ========================================
        print("\n[4/4] Đánh giá trên test set...")

        metrics = model.evaluate(data, verbose=verbose)

        # ========================================
        # LƯU MODEL
        # ========================================
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'hetero_gnn.pth')
        model.save(model_path)

        # Kết quả cuối cùng
        result['success'] = True
        result['message'] = 'Huấn luyện GNN thành công!'
        result['model_path'] = model_path
        result['metrics'] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        result['training_info'] = {
            'epochs_trained': train_result['epochs_trained'],
            'best_val_auc': train_result['best_val_auc'],
            'final_train_loss': train_result['final_train_loss'],
            'final_val_loss': train_result['final_val_loss']
        }
        result['config'] = default_config
        result['completed_at'] = datetime.now().isoformat()

        # In kết quả
        print("\n" + "=" * 70)
        print("KẾT QUẢ BƯỚC 2: HUẤN LUYỆN GNN")
        print("=" * 70)
        print(f"  ✓ Model đã được lưu: {model_path}")
        print("\n  METRICS TRÊN TEST SET:")
        print(f"    - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    - Precision: {metrics['precision']:.4f}")
        print(f"    - Recall:    {metrics['recall']:.4f}")
        print(f"    - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    - ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\n  TRAINING INFO:")
        print(f"    - Epochs trained:  {train_result['epochs_trained']}")
        print(f"    - Best val AUC:    {train_result['best_val_auc']:.4f}")
        print("=" * 70)

        return result

    except Exception as e:
        result['success'] = False
        result['message'] = f'Lỗi: {str(e)}'
        result['error'] = str(e)
        result['completed_at'] = datetime.now().isoformat()

        print("\n" + "=" * 70)
        print("✗ BƯỚC 2 THẤT BẠI!")
        print(f"  Lỗi: {str(e)}")
        print("=" * 70)

        return result


def get_gnn_status(data_dir: str = None) -> dict:
    """
    Lấy trạng thái hiện tại của GNN pipeline

    Returns:
        dict: Trạng thái với các flags và thông tin
    """
    if data_dir is None:
        data_dir = GNN_DATA_DIR

    status = {
        'data_dir': data_dir,
        'data_exists': os.path.isdir(data_dir),
        'graph_ready': False,
        'model_ready': False,
        'graph_path': None,
        'model_path': None,
        'graph_stats': None,
        'model_metrics': None
    }

    # Kiểm tra graph
    flag_path = os.path.join(data_dir, 'graph_ready.flag')
    graph_path = os.path.join(data_dir, 'hetero_graph.pt')

    if os.path.exists(flag_path) and os.path.exists(graph_path):
        status['graph_ready'] = True
        status['graph_path'] = graph_path

        # Đọc thêm thông tin từ graph
        try:
            data, metadata = load_gnn_graph(graph_path)
            transfer_key = ('user', 'transfer', 'recipient')
            status['graph_stats'] = {
                'num_users': data['user'].x.shape[0],
                'num_transfer_edges': data[transfer_key].edge_index.shape[1],
                'num_fraud': int(data[transfer_key].y.sum().item())
            }
        except:
            pass

    # Kiểm tra model
    model_path = os.path.join(MODELS_DIR, 'hetero_gnn.pth')
    report_path = os.path.join(MODELS_DIR, 'hetero_gnn_report.json')

    if os.path.exists(model_path):
        status['model_ready'] = True
        status['model_path'] = model_path

        # Đọc metrics từ report
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    status['model_metrics'] = report.get('final_metrics', {})
            except:
                pass

    return status


# ====================================================
# MAIN - CLI Interface
# ====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GNN Training Script - 2 Bước Riêng Biệt'
    )

    parser.add_argument(
        '--step',
        choices=['build', 'train', 'status', 'all'],
        required=True,
        help='Bước cần thực hiện: build (Bước 1), train (Bước 2), status (kiểm tra), all (cả 2)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Đường dẫn thư mục gnn_data/'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Đường dẫn lưu model'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Số epochs training'
    )

    parser.add_argument(
        '--hidden_channels',
        type=int,
        default=64,
        help='Số hidden channels'
    )

    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Số GNN layers'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Chế độ im lặng (ít log)'
    )

    args = parser.parse_args()

    # Chuẩn bị config
    config = {
        'epochs': args.epochs,
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'learning_rate': args.learning_rate
    }

    verbose = not args.quiet

    # Thực hiện theo step
    if args.step == 'status':
        status = get_gnn_status(args.data_dir)
        print("\n" + "=" * 50)
        print("TRẠNG THÁI GNN PIPELINE")
        print("=" * 50)
        print(f"  Data directory: {status['data_dir']}")
        print(f"  Data exists:    {status['data_exists']}")
        print(f"  Graph ready:    {status['graph_ready']}")
        print(f"  Model ready:    {status['model_ready']}")

        if status['graph_stats']:
            print(f"\n  Graph stats:")
            print(f"    - Users:          {status['graph_stats']['num_users']}")
            print(f"    - Transfer edges: {status['graph_stats']['num_transfer_edges']}")
            print(f"    - Fraud edges:    {status['graph_stats']['num_fraud']}")

        if status['model_metrics']:
            print(f"\n  Model metrics:")
            for k, v in status['model_metrics'].items():
                if v is not None:
                    print(f"    - {k}: {v:.4f}" if isinstance(v, float) else f"    - {k}: {v}")

        print("=" * 50)

    elif args.step == 'build':
        result = build_gnn_graph(args.data_dir, verbose)
        sys.exit(0 if result['success'] else 1)

    elif args.step == 'train':
        result = train_gnn_model(args.data_dir, args.output_dir, config, verbose)
        sys.exit(0 if result['success'] else 1)

    elif args.step == 'all':
        # Chạy cả 2 bước
        result1 = build_gnn_graph(args.data_dir, verbose)
        if not result1['success']:
            print("\n✗ Bước 1 thất bại, không thể tiếp tục Bước 2")
            sys.exit(1)

        result2 = train_gnn_model(args.data_dir, args.output_dir, config, verbose)
        sys.exit(0 if result2['success'] else 1)

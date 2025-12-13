"""
Streamlit App - Giao diện huấn luyện ML Fraud Detection
========================================================
Ứng dụng web để huấn luyện và quản lý các mô hình ML
cho hệ thống phát hiện giao dịch lừa đảo.

Features:
- Huấn luyện Layer 1 models (Isolation Forest, LightGBM)
- Huấn luyện Layer 2 models (Autoencoder, LSTM, GNN)
- GNN với 2 bước: Tạo mạng lưới + Huấn luyện
- Xem metrics và đánh giá model

Author: Senior ML Engineer
Version: 2.0.0
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Thêm path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config

config = get_config()

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="ML Fraud Detection - Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ========== HELPER FUNCTIONS ==========
def check_gnn_data_exists(data_dir: str) -> Dict[str, bool]:
    """Kiểm tra các file dữ liệu GNN có tồn tại không"""
    required_files = {
        'nodes.csv': False,
        'edges_transfer.csv': False,
        'edge_labels.csv': False,
        'splits.csv': False
    }

    optional_files = {
        'edges_uses_device.csv': False,
        'edges_uses_ip.csv': False,
        'metadata.json': False,
        'nodes_user.csv': False,
        'nodes_recipient.csv': False,
        'nodes_device.csv': False,
        'nodes_ip.csv': False
    }

    for file in required_files:
        required_files[file] = os.path.exists(os.path.join(data_dir, file))

    for file in optional_files:
        optional_files[file] = os.path.exists(os.path.join(data_dir, file))

    # nodes.csv có thể được thay thế bởi các file nodes_*.csv
    if not required_files['nodes.csv']:
        if optional_files['nodes_user.csv'] or optional_files['nodes_recipient.csv']:
            required_files['nodes.csv'] = True

    return {
        'required': required_files,
        'optional': optional_files,
        'all_required_exist': all(required_files.values())
    }


def check_graph_ready(data_dir: str) -> bool:
    """Kiểm tra graph đã được build chưa"""
    flag_path = os.path.join(data_dir, 'graph_ready.flag')
    graph_path = os.path.join(data_dir, 'hetero_graph.pt')
    return os.path.exists(flag_path) and os.path.exists(graph_path)


def check_model_exists(model_name: str) -> bool:
    """Kiểm tra model đã được train chưa"""
    model_paths = {
        'gnn': os.path.join(config.SAVED_MODELS_DIR, 'gnn_hetero.pth'),
        'lstm': os.path.join(config.SAVED_MODELS_DIR, 'lstm.pth'),
        'autoencoder': os.path.join(config.SAVED_MODELS_DIR, 'autoencoder.pth'),
        'lightgbm': os.path.join(config.SAVED_MODELS_DIR, 'lightgbm.pkl'),
        'isolation_forest': os.path.join(config.SAVED_MODELS_DIR, 'isolation_forest.pkl')
    }
    return os.path.exists(model_paths.get(model_name, ''))


def format_number(num: float, decimals: int = 4) -> str:
    """Format số với số chữ số thập phân"""
    return f"{num:.{decimals}f}"


def create_metrics_chart(history: Dict) -> go.Figure:
    """Tạo biểu đồ metrics trong quá trình training"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'ROC-AUC', 'F1-Score', 'Learning Progress')
    )

    epochs = list(range(1, len(history.get('train_loss', [])) + 1))

    # Loss
    if 'train_loss' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='red')), row=1, col=1)

    # AUC
    if 'train_auc' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['train_auc'], name='Train AUC', line=dict(color='blue')), row=1, col=2)
    if 'val_auc' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_auc'], name='Val AUC', line=dict(color='red')), row=1, col=2)

    # F1
    if 'train_f1' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['train_f1'], name='Train F1', line=dict(color='blue')), row=2, col=1)
    if 'val_f1' in history:
        fig.add_trace(go.Scatter(x=epochs, y=history['val_f1'], name='Val F1', line=dict(color='red')), row=2, col=1)

    fig.update_layout(height=500, showlegend=True)
    return fig


def create_confusion_matrix_chart(cm: list, labels: list = ['Normal', 'Fraud']) -> go.Figure:
    """Tạo biểu đồ confusion matrix"""
    cm_array = np.array(cm)

    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm_array,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )

    return fig


# ========== SIDEBAR ==========
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.markdown("# 🤖 ML Training")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Chọn trang:",
        ["🏠 Tổng quan", "🎯 Layer 1", "🧠 Layer 2", "🕸️ GNN Training"],
        index=3  # Mặc định chọn GNN Training
    )

    st.sidebar.markdown("---")

    # Hiển thị trạng thái models
    st.sidebar.markdown("### 📊 Trạng thái Models")

    models_status = {
        'Isolation Forest': check_model_exists('isolation_forest'),
        'LightGBM': check_model_exists('lightgbm'),
        'Autoencoder': check_model_exists('autoencoder'),
        'LSTM': check_model_exists('lstm'),
        'GNN': check_model_exists('gnn')
    }

    for model, exists in models_status.items():
        status = "✅" if exists else "❌"
        st.sidebar.markdown(f"{status} {model}")

    return page


# ========== MAIN PAGES ==========
def render_overview_page():
    """Trang tổng quan"""
    st.markdown('<h1 class="main-header">🤖 ML Fraud Detection Training Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### Chào mừng đến với hệ thống huấn luyện ML Fraud Detection

    Hệ thống này cho phép bạn:
    - 🎯 **Layer 1**: Huấn luyện Isolation Forest và LightGBM
    - 🧠 **Layer 2**: Huấn luyện Autoencoder, LSTM và GNN
    - 📊 **Đánh giá**: Xem metrics và đánh giá model

    #### Kiến trúc hệ thống:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Layer 1 - Fast Detection:**
        - Isolation Forest: Phát hiện anomaly nhanh
        - LightGBM: Classification với gradient boosting

        **Layer 2 - Deep Analysis:**
        - Autoencoder: Reconstruction-based anomaly
        - LSTM: Sequence-based detection
        - GNN: Graph-based fraud detection
        """)

    with col2:
        st.markdown("""
        **Quy trình GNN:**
        1. Upload dữ liệu vào thư mục `gnn_data/`
        2. Bấm "Tạo mạng lưới GNN" để build graph
        3. Bấm "Huấn luyện GNN" để train model
        4. Đánh giá và sử dụng model

        **Lưu ý:**
        - Graph phải được build trước khi train
        - Model được lưu tự động sau khi train
        """)


def render_layer1_page():
    """Trang Layer 1 training"""
    st.markdown('<h2 class="section-header">🎯 Layer 1 - Fast Detection Models</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Isolation Forest")
        st.markdown("Phát hiện anomaly dựa trên isolation của data points.")

        if st.button("🌲 Huấn luyện Isolation Forest", key="train_if"):
            with st.spinner("Đang huấn luyện Isolation Forest..."):
                st.info("Feature đang được phát triển...")

    with col2:
        st.markdown("### LightGBM")
        st.markdown("Gradient boosting cho binary classification.")

        if st.button("🚀 Huấn luyện LightGBM", key="train_lgbm"):
            with st.spinner("Đang huấn luyện LightGBM..."):
                st.info("Feature đang được phát triển...")


def render_layer2_page():
    """Trang Layer 2 training"""
    st.markdown('<h2 class="section-header">🧠 Layer 2 - Deep Learning Models</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Autoencoder", "LSTM", "GNN"])

    with tab1:
        st.markdown("### Autoencoder")
        st.markdown("Reconstruction-based anomaly detection.")
        if st.button("🔄 Huấn luyện Autoencoder", key="train_ae"):
            st.info("Feature đang được phát triển...")

    with tab2:
        st.markdown("### LSTM")
        st.markdown("Sequence-based fraud detection.")
        if st.button("📈 Huấn luyện LSTM", key="train_lstm"):
            st.info("Feature đang được phát triển...")

    with tab3:
        st.markdown("### GNN")
        st.markdown("Graph-based fraud detection. Xem trang **GNN Training** để huấn luyện.")


def render_gnn_training_page():
    """Trang huấn luyện GNN với 2 bước riêng biệt"""
    st.markdown('<h2 class="section-header">🕸️ GNN Fraud Detection - Huấn luyện mô hình</h2>', unsafe_allow_html=True)

    # Cấu hình thư mục dữ liệu
    default_data_dir = os.path.join(config.BASE_DIR, 'gnn_data')

    st.markdown("### 📁 Cấu hình dữ liệu")

    data_dir = st.text_input(
        "Thư mục dữ liệu GNN:",
        value=default_data_dir,
        help="Đường dẫn đến thư mục chứa các file CSV/JSON cho GNN"
    )

    # Kiểm tra dữ liệu
    data_status = check_gnn_data_exists(data_dir)
    graph_ready = check_graph_ready(data_dir)

    # Hiển thị trạng thái dữ liệu
    with st.expander("📋 Trạng thái dữ liệu", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Files bắt buộc:**")
            for file, exists in data_status['required'].items():
                status = "✅" if exists else "❌"
                st.markdown(f"{status} `{file}`")

        with col2:
            st.markdown("**Files tùy chọn:**")
            for file, exists in data_status['optional'].items():
                if exists:
                    st.markdown(f"✅ `{file}`")

        with col3:
            st.markdown("**Trạng thái:**")
            if data_status['all_required_exist']:
                st.success("✅ Dữ liệu đầy đủ")
            else:
                st.error("❌ Thiếu dữ liệu bắt buộc")

            if graph_ready:
                st.success("✅ Graph đã sẵn sàng")
            else:
                st.warning("⚠️ Graph chưa được tạo")

    st.markdown("---")

    # ========== 2 NÚT RIÊNG BIỆT ==========
    st.markdown("### 🎮 Điều khiển huấn luyện")

    col1, col2 = st.columns(2)

    # ========== NÚT 1: TẠO MẠNG LƯỚI GNN ==========
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: white;">🕸️ BƯỚC 1: Tạo mạng lưới</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Load dữ liệu, kiểm tra tính toàn vẹn và xây dựng Heterogeneous Graph
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Disable nếu không có dữ liệu
        build_disabled = not data_status['all_required_exist']

        if st.button(
            "🕸️ Tạo mạng lưới GNN",
            key="build_graph",
            disabled=build_disabled,
            help="Load dữ liệu và xây dựng graph PyTorch Geometric"
        ):
            run_build_graph(data_dir)

        if build_disabled:
            st.warning("⚠️ Cần upload đầy đủ dữ liệu trước khi tạo mạng lưới")

    # ========== NÚT 2: HUẤN LUYỆN GNN ==========
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: white;">🤖 BƯỚC 2: Huấn luyện model</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Train GNN model và đánh giá hiệu suất trên tập test
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Disable nếu graph chưa sẵn sàng
        train_disabled = not graph_ready

        if st.button(
            "🤖 Huấn luyện GNN",
            key="train_gnn",
            disabled=train_disabled,
            help="Train GNN model (yêu cầu graph đã được tạo)"
        ):
            run_train_gnn(data_dir)

        if train_disabled:
            st.warning("⚠️ Cần tạo mạng lưới GNN trước khi huấn luyện")

    st.markdown("---")

    # ========== CẤU HÌNH NÂNG CAO ==========
    with st.expander("⚙️ Cấu hình nâng cao", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            epochs = st.number_input("Số epochs:", min_value=10, max_value=500, value=100)
            learning_rate = st.number_input("Learning rate:", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")

        with col2:
            hidden_dim = st.selectbox("Hidden dimension:", [32, 64, 128, 256], index=1)
            num_layers = st.selectbox("Số GNN layers:", [2, 3, 4], index=1)

        with col3:
            dropout = st.slider("Dropout:", 0.0, 0.5, 0.3)
            patience = st.number_input("Early stopping patience:", min_value=5, max_value=50, value=15)

        # Lưu config vào session state
        st.session_state['gnn_config'] = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'hidden_channels': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'patience': patience
        }

    # ========== HIỂN THỊ KẾT QUẢ ==========
    if 'gnn_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Kết quả huấn luyện")

        results = st.session_state['gnn_results']

        # Metrics cards
        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = results.get('test', results.get('val', {}))

        with col1:
            st.metric("Accuracy", format_number(metrics.get('accuracy', 0)))
        with col2:
            st.metric("Precision", format_number(metrics.get('precision', 0)))
        with col3:
            st.metric("Recall", format_number(metrics.get('recall', 0)))
        with col4:
            st.metric("F1-Score", format_number(metrics.get('f1_score', 0)))
        with col5:
            st.metric("ROC-AUC", format_number(metrics.get('roc_auc', 0)))

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            if 'history' in results:
                st.plotly_chart(create_metrics_chart(results['history']), use_container_width=True)

        with col2:
            if 'confusion_matrix' in metrics:
                st.plotly_chart(create_confusion_matrix_chart(metrics['confusion_matrix']), use_container_width=True)

    # ========== HIỂN THỊ THỐNG KÊ GRAPH ==========
    if 'graph_stats' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Thống kê Graph")

        stats = st.session_state['graph_stats']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Nodes:**")
            if 'nodes' in stats:
                st.write(f"Tổng: {stats['nodes'].get('total', 0):,}")
                if 'by_type' in stats['nodes']:
                    for nt, count in stats['nodes']['by_type'].items():
                        st.write(f"  - {nt}: {count:,}")

        with col2:
            st.markdown("**Edges:**")
            if 'edges' in stats:
                for et, count in stats['edges'].items():
                    st.write(f"  - {et}: {count:,}")

        with col3:
            st.markdown("**Labels:**")
            if 'labels' in stats:
                st.write(f"Tổng: {stats['labels'].get('total', 0):,}")
                st.write(f"Fraud ratio: {stats['labels'].get('fraud_ratio', 0)*100:.2f}%")


def run_build_graph(data_dir: str):
    """Chạy quá trình xây dựng graph"""
    st.markdown("---")
    st.markdown("### 🔄 Đang xây dựng mạng lưới GNN...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    try:
        # Import pipeline
        from preprocessing.gnn_data_pipeline import GNNDataPipeline

        # Bước 1: Load dữ liệu
        status_text.text("📁 Bước 1/4: Load dữ liệu...")
        progress_bar.progress(10)

        pipeline = GNNDataPipeline(data_dir, verbose=False)
        load_results = pipeline.load_all_data()

        with log_container:
            st.success(f"✅ Loaded {len(load_results['files_loaded'])} files")
            for f in load_results['files_loaded']:
                st.write(f"  - {f}")

        if not load_results['success']:
            st.error(f"❌ Lỗi load dữ liệu: {load_results['errors']}")
            return

        progress_bar.progress(30)

        # Bước 2: Sanity check
        status_text.text("🔍 Bước 2/4: Kiểm tra tính toàn vẹn...")

        try:
            check_results = pipeline.sanity_check()
            with log_container:
                st.success(f"✅ Sanity check passed: {check_results['checks_passed']}/{check_results['checks_performed']} checks")
        except ValueError as e:
            with log_container:
                st.error(f"❌ Sanity check failed: {str(e)}")
            return

        progress_bar.progress(50)

        # Bước 3: Build graph
        status_text.text("🔨 Bước 3/4: Xây dựng Heterogeneous Graph...")

        data = pipeline.build_hetero_graph()

        with log_container:
            st.success(f"✅ Graph built: {len(data.node_types)} node types, {len(data.edge_types)} edge types")

        progress_bar.progress(80)

        # Bước 4: Save graph
        status_text.text("💾 Bước 4/4: Lưu graph...")

        graph_path = pipeline.save_graph(data)
        stats = pipeline.get_statistics()

        progress_bar.progress(100)
        status_text.text("✅ Hoàn tất!")

        # Lưu stats vào session state
        st.session_state['graph_stats'] = stats

        with log_container:
            st.success(f"""
            ✅ **XÂY DỰNG MẠNG LƯỚI HOÀN TẤT!**

            - Graph saved: `{graph_path}`
            - Node types: {data.node_types}
            - Edge types: {data.edge_types}

            Bạn có thể tiến hành **Huấn luyện GNN** ngay bây giờ!
            """)

        # Force rerun để update trạng thái
        time.sleep(1)
        st.rerun()

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Lỗi!")
        st.error(f"Lỗi khi xây dựng graph: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def run_train_gnn(data_dir: str):
    """Chạy quá trình huấn luyện GNN"""
    st.markdown("---")
    st.markdown("### 🔄 Đang huấn luyện GNN model...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    try:
        import torch

        # Import modules
        from preprocessing.gnn_data_pipeline import GNNDataPipeline
        from models.layer2.gnn_hetero_model import GNNHeteroTrainer

        # Bước 1: Load graph
        status_text.text("📁 Bước 1/3: Load graph đã build...")
        progress_bar.progress(10)

        pipeline = GNNDataPipeline(data_dir, verbose=False)
        data = pipeline.load_graph()

        with log_container:
            st.success(f"✅ Loaded graph: {len(data.node_types)} node types, {len(data.edge_types)} edge types")

        progress_bar.progress(20)

        # Bước 2: Train model
        status_text.text("🤖 Bước 2/3: Huấn luyện model...")

        # Lấy config từ session state hoặc dùng default
        model_config = st.session_state.get('gnn_config', config.GNN_CONFIG)

        trainer = GNNHeteroTrainer(model_config=model_config, verbose=False)

        # Training với progress updates
        train_results = trainer.fit(data)

        with log_container:
            st.success(f"✅ Training completed: {train_results['epochs_trained']} epochs, Best AUC: {train_results['best_val_auc']:.4f}")

        progress_bar.progress(70)

        # Bước 3: Evaluate
        status_text.text("📊 Bước 3/3: Đánh giá model...")

        metrics = {}
        for split in ['train', 'val', 'test']:
            try:
                split_metrics = trainer.evaluate(data, mask_type=split)
                metrics[split] = split_metrics
            except:
                pass

        # Save model
        model_path = trainer.save()

        progress_bar.progress(100)
        status_text.text("✅ Hoàn tất!")

        # Lưu results vào session state
        st.session_state['gnn_results'] = {
            **metrics,
            'history': train_results['history'],
            'best_threshold': train_results['best_threshold']
        }

        with log_container:
            st.success(f"""
            ✅ **HUẤN LUYỆN HOÀN TẤT!**

            - Model saved: `{model_path}`
            - Epochs trained: {train_results['epochs_trained']}
            - Best validation AUC: {train_results['best_val_auc']:.4f}
            - Optimal threshold: {train_results['best_threshold']:.4f}
            """)

            if 'test' in metrics:
                st.markdown("**Test Metrics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['test']['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['test']['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['test']['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['test']['f1_score']:.4f}")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Lỗi!")
        st.error(f"Lỗi khi huấn luyện: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ========== MAIN ==========
def main():
    """Main function"""
    page = render_sidebar()

    if page == "🏠 Tổng quan":
        render_overview_page()
    elif page == "🎯 Layer 1":
        render_layer1_page()
    elif page == "🧠 Layer 2":
        render_layer2_page()
    elif page == "🕸️ GNN Training":
        render_gnn_training_page()


if __name__ == "__main__":
    main()

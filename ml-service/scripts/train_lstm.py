#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script huấn luyện LSTM Fraud Detection Model
=============================================
Sử dụng: python scripts/train_lstm.py --data_path data/transactions.csv

Script này:
1. Load và xử lý dữ liệu qua lstm_data_pipeline
2. Train LSTM model
3. Đánh giá model và tự động đề xuất threshold
4. In bảng threshold analysis chi tiết
5. Lưu model và config

Author: ML Team - Fraud Detection
Target: Ngân hàng Việt Nam (fraud ratio ~2-4%)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các modules cần thiết
from models.layer2.lstm_sequence import LSTMSequenceModel
from utils.lstm_data_pipeline import prepare_lstm_data, validate_raw_data, FEATURE_NAMES
from utils.threshold_optimizer import (
    compute_roc_curve,
    compute_precision_recall_curve,
    compute_metrics_at_threshold,
    recommend_threshold,
    classify_transactions,
    get_tier_distribution,
    log_threshold_analysis,
    print_summary_table,
    generate_threshold_report,
    FraudThresholdClassifier,
    DEFAULT_THRESHOLDS
)


def load_transaction_data(data_path: str) -> pd.DataFrame:
    """
    Load dữ liệu giao dịch từ file CSV

    Args:
        data_path: Đường dẫn file CSV

    Returns:
        DataFrame với dữ liệu giao dịch
    """
    print("=" * 70)
    print("BƯỚC 1: LOAD DỮ LIỆU")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

    df = pd.read_csv(data_path)

    print(f"  - File: {data_path}")
    print(f"  - Số giao dịch: {len(df):,}")
    print(f"  - Số cột: {len(df.columns)}")

    # Validate
    validation = validate_raw_data(df)

    if not validation['is_valid']:
        print(f"\n  [ERROR] Dữ liệu không hợp lệ:")
        for err in validation['errors']:
            print(f"    - {err}")
        raise ValueError("Dữ liệu không hợp lệ để train LSTM")

    if validation['warnings']:
        print(f"\n  [WARNING]:")
        for warn in validation['warnings']:
            print(f"    - {warn}")

    print(f"\n  - Số users: {validation['info']['users']:,}")
    print(f"  - Fraud ratio: {validation['info']['fraud_ratio']:.2%}")

    return df


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict = None,
    verbose: bool = True
) -> tuple:
    """
    Train LSTM model

    Args:
        X_train: Training sequences (n, seq_len, features)
        y_train: Training labels
        X_test: Test sequences
        y_test: Test labels
        config: Model config (optional)
        verbose: In thông tin training

    Returns:
        Tuple (model, y_pred_prob, metrics)
    """
    print("\n" + "=" * 70)
    print("BƯỚC 3: TRAIN LSTM MODEL")
    print("=" * 70)

    # Khởi tạo model
    model = LSTMSequenceModel(model_config=config)

    # Train
    print("\n[LSTM] Bắt đầu training...")
    model.fit(X_train, y_train, validation_split=0.1, verbose=verbose)

    # Predict trên test set
    print("\n[LSTM] Đánh giá trên test set...")
    y_pred_prob = model.predict_proba(X_test)

    # Basic metrics với threshold mặc định 0.5
    basic_metrics = model.evaluate(X_test, y_test, verbose=True)

    return model, y_pred_prob, basic_metrics


def analyze_and_recommend_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    strategy: str = 'balanced'
) -> dict:
    """
    Phân tích và đề xuất threshold cho production

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud từ model
        strategy: Chiến lược tối ưu

    Returns:
        Dict với threshold recommendations
    """
    print("\n" + "=" * 70)
    print("BƯỚC 4: PHÂN TÍCH VÀ ĐỀ XUẤT THRESHOLD")
    print("=" * 70)

    # 1. Tính ROC và PR curves
    print("\n[THRESHOLD] Tính ROC Curve và PR Curve...")
    fpr, tpr, thresholds_roc, auc = compute_roc_curve(y_true, y_pred_prob)
    prec, rec, thresholds_pr, ap = compute_precision_recall_curve(y_true, y_pred_prob)

    print(f"  - ROC-AUC: {auc:.4f}")
    print(f"  - Average Precision: {ap:.4f}")

    # 2. Log threshold analysis
    print("\n[THRESHOLD] Phân tích chi tiết tại các ngưỡng...")
    df_analysis = log_threshold_analysis(y_true, y_pred_prob)

    # 3. Recommend threshold
    print(f"\n[THRESHOLD] Đề xuất threshold (strategy: {strategy})...")
    recommendation = recommend_threshold(y_true, y_pred_prob, strategy=strategy)

    recommended_th = recommendation['recommended_threshold']
    metrics = recommendation['metrics']

    print(f"\n  *** THRESHOLD ĐỀ XUẤT: {recommended_th:.4f} ***")
    print(f"  - Recall: {metrics.recall:.2%}")
    print(f"  - Precision: {metrics.precision:.2%}")
    print(f"  - F1-Score: {metrics.f1:.4f}")
    print(f"  - FPR: {metrics.fpr:.2%}")

    if recommendation['warnings']:
        print("\n  [WARNINGS]:")
        for warn in recommendation['warnings']:
            print(f"    - {warn}")

    # 4. In bảng summary
    print_summary_table(y_true, y_pred_prob)

    # 5. Tier distribution
    print("\n[THRESHOLD] Phân bổ giao dịch theo tier:")
    tier_dist = get_tier_distribution(y_pred_prob, y_true)
    print(tier_dist.to_string(index=False))

    return {
        'recommended_threshold': recommended_th,
        'roc_auc': auc,
        'average_precision': ap,
        'metrics_at_recommended': metrics.to_dict(),
        'alternatives': recommendation['alternatives'],
        'analysis_table': df_analysis.to_dict('records'),
        'tier_distribution': tier_dist.to_dict('records')
    }


def save_model_and_config(
    model: LSTMSequenceModel,
    threshold_result: dict,
    output_dir: str,
    training_info: dict
):
    """
    Lưu model, threshold config và báo cáo

    Args:
        model: Trained LSTM model
        threshold_result: Kết quả phân tích threshold
        output_dir: Thư mục lưu
        training_info: Thông tin training
    """
    print("\n" + "=" * 70)
    print("BƯỚC 5: LƯU MODEL VÀ CONFIG")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save LSTM model
    model_path = os.path.join(output_dir, 'lstm_fraud_detection.pth')
    model.save(model_path)

    # 2. Save threshold config
    threshold_config = {
        'recommended_threshold': threshold_result['recommended_threshold'],
        'thresholds': DEFAULT_THRESHOLDS.copy(),
        'roc_auc': threshold_result['roc_auc'],
        'average_precision': threshold_result['average_precision'],
        'metrics_at_recommended': threshold_result['metrics_at_recommended'],
        'created_at': datetime.now().isoformat(),
        'training_info': training_info
    }

    config_path = os.path.join(output_dir, 'threshold_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_config, f, indent=2, ensure_ascii=False)
    print(f"  - Threshold config: {config_path}")

    # 3. Save FraudThresholdClassifier config
    classifier = FraudThresholdClassifier(
        default_threshold=threshold_result['recommended_threshold']
    )
    classifier_config = classifier.to_config()

    classifier_path = os.path.join(output_dir, 'classifier_config.json')
    with open(classifier_path, 'w', encoding='utf-8') as f:
        json.dump(classifier_config, f, indent=2, ensure_ascii=False)
    print(f"  - Classifier config: {classifier_path}")

    # 4. Save analysis table
    analysis_path = os.path.join(output_dir, 'threshold_analysis.csv')
    pd.DataFrame(threshold_result['analysis_table']).to_csv(analysis_path, index=False)
    print(f"  - Analysis table: {analysis_path}")

    # 5. Save tier distribution
    tier_path = os.path.join(output_dir, 'tier_distribution.csv')
    pd.DataFrame(threshold_result['tier_distribution']).to_csv(tier_path, index=False)
    print(f"  - Tier distribution: {tier_path}")

    # 6. Save feature names
    features_path = os.path.join(output_dir, 'lstm_feature_names.json')
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(FEATURE_NAMES, f, indent=2, ensure_ascii=False)
    print(f"  - Feature names: {features_path}")

    print(f"\n  Tất cả files đã lưu trong: {output_dir}")


def print_final_summary(threshold_result: dict, training_info: dict):
    """In tổng kết cuối cùng"""
    print("\n" + "=" * 70)
    print("TỔNG KẾT TRAINING LSTM FRAUD DETECTION")
    print("=" * 70)

    print(f"""
    MODEL PERFORMANCE:
    ------------------
    - ROC-AUC: {threshold_result['roc_auc']:.4f}
    - Average Precision: {threshold_result['average_precision']:.4f}

    THRESHOLD ĐỀ XUẤT CHO PRODUCTION:
    ---------------------------------
    - Threshold: {threshold_result['recommended_threshold']:.4f}
    - Recall: {threshold_result['metrics_at_recommended']['recall']:.2%}
    - Precision: {threshold_result['metrics_at_recommended']['precision']:.2%}
    - FPR: {threshold_result['metrics_at_recommended']['fpr']:.2%}

    3 MỨC THRESHOLD:
    ----------------
    - LOW_RISK:    {DEFAULT_THRESHOLDS['LOW_RISK']:.2f} (monitoring)
    - MEDIUM_RISK: {DEFAULT_THRESHOLDS['MEDIUM_RISK']:.2f} (review, 2h SLA)
    - HIGH_RISK:   {DEFAULT_THRESHOLDS['HIGH_RISK']:.2f} (urgent, 15min SLA)

    TRAINING INFO:
    --------------
    - Train samples: {training_info['train_samples']:,}
    - Test samples: {training_info['test_samples']:,}
    - Train fraud ratio: {training_info['train_fraud_ratio']:.2%}
    - Test fraud ratio: {training_info['test_fraud_ratio']:.2%}
    """)

    print("=" * 70)
    print("HƯỚNG DẪN SỬ DỤNG TRONG PRODUCTION")
    print("=" * 70)
    print("""
    # Load model và classifier
    from utils.threshold_optimizer import FraudThresholdClassifier
    import json

    # Load config
    with open('saved_models/classifier_config.json') as f:
        config = json.load(f)

    classifier = FraudThresholdClassifier.from_config(config)

    # Predict và classify
    prob = model.predict_proba(sequence)
    result = classifier.classify_single(prob)
    # result = {'risk_tier': 'MEDIUM_RISK', 'is_flagged': True, 'action': '...'}
    """)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM Fraud Detection với Threshold Optimization'
    )
    parser.add_argument(
        '--data_path', type=str,
        default='data/synthetic/transactions.csv',
        help='Đường dẫn file CSV giao dịch'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='saved_models',
        help='Thư mục lưu model và config'
    )
    parser.add_argument(
        '--seq_length', type=int, default=7,
        help='Độ dài sequence (default: 7)'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        help='Tỷ lệ test set (default: 0.2)'
    )
    parser.add_argument(
        '--strategy', type=str, default='balanced',
        choices=['recall_focused', 'balanced', 'precision_focused', 'fpr_controlled'],
        help='Chiến lược chọn threshold'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Số epochs training (default: 50)'
    )

    args = parser.parse_args()

    # Xác định đường dẫn
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, args.data_path)
    output_dir = os.path.join(script_dir, args.output_dir)

    print("\n" + "=" * 70)
    print("LSTM FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    print(f"  - Data path: {data_path}")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Sequence length: {args.seq_length}")
    print(f"  - Test ratio: {args.test_ratio}")
    print(f"  - Strategy: {args.strategy}")
    print(f"  - Epochs: {args.epochs}")

    # =====================================================
    # BƯỚC 1: Load dữ liệu
    # =====================================================
    df_raw = load_transaction_data(data_path)

    # =====================================================
    # BƯỚC 2: Chuẩn bị data cho LSTM
    # =====================================================
    print("\n" + "=" * 70)
    print("BƯỚC 2: XỬ LÝ DỮ LIỆU QUA LSTM PIPELINE")
    print("=" * 70)

    result = prepare_lstm_data(
        df_raw,
        seq_length=args.seq_length,
        test_ratio=args.test_ratio,
        include_padding=True,
        verbose=True
    )

    X_train = result['X_train']
    y_train = result['y_train']
    X_test = result['X_test']
    y_test = result['y_test']

    training_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_fraud_ratio': float(y_train.mean()),
        'test_fraud_ratio': float(y_test.mean()),
        'seq_length': args.seq_length,
        'num_features': X_train.shape[2] if len(X_train) > 0 else 40
    }

    # =====================================================
    # BƯỚC 3: Train LSTM
    # =====================================================
    lstm_config = {
        'sequence_length': args.seq_length,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': args.epochs,
        'batch_size': 128
    }

    model, y_pred_prob, basic_metrics = train_lstm_model(
        X_train, y_train, X_test, y_test,
        config=lstm_config,
        verbose=True
    )

    # =====================================================
    # BƯỚC 4: Phân tích và đề xuất threshold
    # =====================================================
    threshold_result = analyze_and_recommend_threshold(
        y_test, y_pred_prob, strategy=args.strategy
    )

    # =====================================================
    # BƯỚC 5: Lưu model và config
    # =====================================================
    save_model_and_config(model, threshold_result, output_dir, training_info)

    # =====================================================
    # TỔNG KẾT
    # =====================================================
    print_final_summary(threshold_result, training_info)

    print("\n TRAINING HOÀN TẤT!")


if __name__ == '__main__':
    main()

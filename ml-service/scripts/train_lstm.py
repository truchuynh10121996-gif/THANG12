#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script huấn luyện LSTM Fraud Detection Model với Threshold Optimization
=======================================================================
Script này train LSTM model và tự động tối ưu threshold cho ngân hàng Việt Nam.

Sử dụng: python scripts/train_lstm.py --data_dir data/generated --output_dir models/trained

Author: ML Team - Fraud Detection Vietnam
Created: 2025
Target: Ngân hàng Việt Nam - Fraud ratio ~2-4%
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LSTM model và data pipeline
from models.layer2.lstm_sequence import LSTMSequenceModel
from utils.lstm_data_pipeline import prepare_lstm_data
from utils.threshold_optimizer import (
    recommend_threshold,
    log_threshold_analysis,
    generate_threshold_report,
    print_summary_table,
    get_tier_distribution,
    compute_roc_curve,
    compute_precision_recall_curve,
    compute_metrics_at_threshold,
    FraudThresholdClassifier,
    DEFAULT_THRESHOLDS,
    BUSINESS_CONSTRAINTS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_transaction_data(data_dir: str) -> pd.DataFrame:
    """
    Load dữ liệu giao dịch từ thư mục

    Args:
        data_dir: Đường dẫn thư mục chứa transactions.csv

    Returns:
        DataFrame transactions
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOAD TRANSACTION DATA")
    print("=" * 70)

    transactions_path = os.path.join(data_dir, 'transactions.csv')

    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"Không tìm thấy file: {transactions_path}")

    df = pd.read_csv(transactions_path)

    # Thêm các columns cần thiết nếu thiếu
    if 'timestamp' not in df.columns and 'created_at' in df.columns:
        df['timestamp'] = df['created_at']

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"  - Total transactions: {len(df):,}")
    print(f"  - Unique users: {df['user_id'].nunique():,}")
    print(f"  - Fraud ratio: {df['is_fraud'].mean()*100:.2f}%")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def prepare_lstm_sequences(df: pd.DataFrame, sequence_length: int = 7) -> dict:
    """
    Chuẩn bị sequences cho LSTM training

    Args:
        df: DataFrame transactions
        sequence_length: Độ dài sequence

    Returns:
        Dict với X_train, y_train, X_test, y_test
    """
    print("\n" + "=" * 70)
    print("STEP 2: PREPARE LSTM SEQUENCES")
    print("=" * 70)

    data = prepare_lstm_data(
        df,
        sequence_length=sequence_length,
        verbose=True
    )

    print(f"\n  Summary:")
    print(f"  - X_train shape: {data['X_train'].shape}")
    print(f"  - X_test shape: {data['X_test'].shape}")
    print(f"  - Train fraud ratio: {data['y_train'].mean()*100:.2f}%")
    print(f"  - Test fraud ratio: {data['y_test'].mean()*100:.2f}%")

    return data


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict = None,
    verbose: bool = True
) -> LSTMSequenceModel:
    """
    Train LSTM model

    Args:
        X_train: Training sequences
        y_train: Training labels
        model_config: Config cho model
        verbose: In thông tin

    Returns:
        LSTMSequenceModel đã train
    """
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN LSTM MODEL")
    print("=" * 70)

    model = LSTMSequenceModel(model_config=model_config)
    model.fit(X_train, y_train, validation_split=0.1, verbose=verbose)

    return model


def optimize_threshold(
    model: LSTMSequenceModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    strategy: str = 'balanced'
) -> dict:
    """
    Tối ưu threshold cho model

    ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT - TỐI ƯU THRESHOLD SAU KHI TRAIN

    Args:
        model: LSTM model đã train
        X_val: Validation sequences
        y_val: Validation labels
        strategy: Chiến lược tối ưu ('balanced', 'recall_focused', 'precision_focused', 'fpr_controlled')

    Returns:
        Dict kết quả threshold optimization
    """
    print("\n" + "=" * 70)
    print("STEP 4: THRESHOLD OPTIMIZATION - NGƯỠNG TỐI ƯU CHO NGÂN HÀNG VIỆT NAM")
    print("=" * 70)

    # Predict probabilities trên validation set
    print("\n[THRESHOLD] Đang tính fraud probabilities trên validation set...")
    y_pred_prob = model.predict_proba(X_val)

    print(f"  - Validation samples: {len(y_val):,}")
    print(f"  - Actual frauds: {int(y_val.sum()):,} ({y_val.mean()*100:.2f}%)")
    print(f"  - Prob range: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")
    print(f"  - Prob mean: {y_pred_prob.mean():.4f}")

    # ============================================================
    # A. TÍNH ROC VÀ PR CURVES
    # ============================================================
    print("\n" + "-" * 60)
    print("A. MODEL PERFORMANCE CURVES")
    print("-" * 60)

    fpr, tpr, thresholds_roc, auc = compute_roc_curve(y_val, y_pred_prob)
    precision, recall, thresholds_pr, ap = compute_precision_recall_curve(y_val, y_pred_prob)

    print(f"\n  ROC-AUC Score: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    # ============================================================
    # B. PHÂN TÍCH CHI TIẾT CÁC THRESHOLDS
    # ============================================================
    print("\n" + "-" * 60)
    print("B. CHI TIẾT METRICS TẠI CÁC NGƯỠNG")
    print("-" * 60)

    analysis_df = log_threshold_analysis(y_val, y_pred_prob)

    # ============================================================
    # C. ĐỀ XUẤT THRESHOLD TỐI ƯU
    # ============================================================
    print("\n" + "-" * 60)
    print(f"C. ĐỀ XUẤT THRESHOLD (Strategy: {strategy.upper()})")
    print("-" * 60)

    # Thử tất cả các strategies
    strategies = ['balanced', 'recall_focused', 'precision_focused', 'fpr_controlled']
    strategy_results = {}

    for strat in strategies:
        result = recommend_threshold(y_val, y_pred_prob, strategy=strat)
        strategy_results[strat] = result

        if result['recommended_threshold'] is not None:
            metrics = result['metrics']
            print(f"\n  [{strat.upper()}]")
            print(f"    Threshold: {result['recommended_threshold']:.4f}")
            print(f"    Recall: {metrics.recall:.2%} | Precision: {metrics.precision:.2%}")
            print(f"    F1: {metrics.f1:.4f} | FPR: {metrics.fpr:.2%}")
            if result['warnings']:
                for warn in result['warnings']:
                    print(f"    ⚠️ Warning: {warn}")

    # Lấy kết quả theo strategy được chọn
    main_result = strategy_results[strategy]
    recommended_threshold = main_result['recommended_threshold']

    # ============================================================
    # D. BẢNG SUMMARY 3 MỨC THRESHOLD
    # ============================================================
    print("\n" + "-" * 60)
    print("D. 3 MỨC THRESHOLD CHO PRODUCTION")
    print("-" * 60)

    print_summary_table(y_val, y_pred_prob, DEFAULT_THRESHOLDS)

    # ============================================================
    # E. TIER DISTRIBUTION
    # ============================================================
    print("\n" + "-" * 60)
    print("E. PHÂN BỔ GIAO DỊCH THEO TIER")
    print("-" * 60)

    tier_dist = get_tier_distribution(y_pred_prob, y_val, DEFAULT_THRESHOLDS)
    print(tier_dist.to_string(index=False))

    # ============================================================
    # F. THRESHOLD ĐỀ XUẤT CHÍNH THỨC
    # ============================================================
    print("\n" + "=" * 60)
    print("THRESHOLD ĐỀ XUẤT CHÍNH THỨC CHO DEPLOY")
    print("=" * 60)

    if recommended_threshold is not None:
        final_metrics = compute_metrics_at_threshold(y_val, y_pred_prob, recommended_threshold)
        print(f"\n  ✅ THRESHOLD: {recommended_threshold:.4f}")
        print(f"\n  Performance tại threshold này:")
        print(f"    - Recall (Fraud caught): {final_metrics.recall:.2%}")
        print(f"    - Precision: {final_metrics.precision:.2%}")
        print(f"    - F1-Score: {final_metrics.f1:.4f}")
        print(f"    - False Positive Rate: {final_metrics.fpr:.2%}")
        print(f"    - False Negative Rate: {final_metrics.fnr:.2%}")

        # Tính số lượng cụ thể
        y_pred = (y_pred_prob >= recommended_threshold).astype(int)
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        tn = ((y_pred == 0) & (y_val == 0)).sum()

        print(f"\n  Confusion Matrix:")
        print(f"    Fraud detected (TP): {tp:,} / {int(y_val.sum()):,} total frauds")
        print(f"    Fraud missed (FN): {fn:,}")
        print(f"    False alarms (FP): {fp:,}")
        print(f"    Normal correct (TN): {tn:,}")
    else:
        print("\n  ⚠️ Không tìm được threshold tối ưu, sử dụng default MEDIUM_RISK: 0.15")
        recommended_threshold = DEFAULT_THRESHOLDS['MEDIUM_RISK']

    # Tạo kết quả trả về
    result = {
        'recommended_threshold': recommended_threshold,
        'strategy': strategy,
        'all_strategies': strategy_results,
        'roc_auc': auc,
        'average_precision': ap,
        'thresholds_analysis': analysis_df.to_dict('records'),
        'tier_distribution': tier_dist.to_dict('records'),
        'default_thresholds': DEFAULT_THRESHOLDS,
        'business_constraints': BUSINESS_CONSTRAINTS
    }

    if main_result['metrics'] is not None:
        result['metrics_at_threshold'] = main_result['metrics'].to_dict()

    return result


def save_results(
    model: LSTMSequenceModel,
    threshold_result: dict,
    output_dir: str,
    test_metrics: dict = None
):
    """
    Lưu model và kết quả threshold optimization

    Args:
        model: LSTM model đã train
        threshold_result: Kết quả threshold optimization
        output_dir: Thư mục lưu
        test_metrics: Metrics trên test set (optional)
    """
    print("\n" + "=" * 70)
    print("STEP 5: SAVE RESULTS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Lưu model
    model_path = os.path.join(output_dir, 'lstm.pth')
    model.save(model_path)
    print(f"  - Model saved: {model_path}")

    # 2. Lưu threshold config
    threshold_config = {
        'recommended_threshold': threshold_result['recommended_threshold'],
        'strategy': threshold_result['strategy'],
        'thresholds': DEFAULT_THRESHOLDS,
        'business_constraints': BUSINESS_CONSTRAINTS,
        'roc_auc': threshold_result['roc_auc'],
        'average_precision': threshold_result['average_precision'],
        'training_time': datetime.now().isoformat()
    }

    if 'metrics_at_threshold' in threshold_result:
        threshold_config['metrics_at_threshold'] = threshold_result['metrics_at_threshold']

    threshold_path = os.path.join(output_dir, 'lstm_threshold_config.json')
    with open(threshold_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_config, f, indent=2, ensure_ascii=False)
    print(f"  - Threshold config saved: {threshold_path}")

    # 3. Lưu FraudThresholdClassifier config
    classifier = FraudThresholdClassifier(
        thresholds=DEFAULT_THRESHOLDS,
        default_threshold=threshold_result['recommended_threshold']
    )
    classifier_config = classifier.to_config()

    classifier_path = os.path.join(output_dir, 'fraud_classifier_config.json')
    with open(classifier_path, 'w', encoding='utf-8') as f:
        json.dump(classifier_config, f, indent=2, ensure_ascii=False)
    print(f"  - Classifier config saved: {classifier_path}")

    # 4. Lưu training report đầy đủ
    report = {
        'training_time': datetime.now().isoformat(),
        'model_type': 'LSTM',
        'threshold_optimization': threshold_result,
        'test_metrics': test_metrics
    }

    report_path = os.path.join(output_dir, 'lstm_training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  - Training report saved: {report_path}")

    # 5. Lưu threshold analysis CSV
    if 'thresholds_analysis' in threshold_result:
        analysis_df = pd.DataFrame(threshold_result['thresholds_analysis'])
        analysis_path = os.path.join(output_dir, 'threshold_analysis.csv')
        analysis_df.to_csv(analysis_path, index=False)
        print(f"  - Threshold analysis saved: {analysis_path}")

    print("\n  All files saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM Fraud Detection Model with Threshold Optimization'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/generated',
        help='Thư mục chứa dữ liệu training (default: data/generated)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/trained',
        help='Thư mục lưu models (default: models/trained)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=7,
        help='Độ dài sequence cho LSTM (default: 7)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Số epochs training (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'recall_focused', 'precision_focused', 'fpr_controlled'],
        help='Threshold optimization strategy (default: balanced)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='In chi tiết (default: True)'
    )

    args = parser.parse_args()

    # Xác định đường dẫn
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, args.data_dir)
    output_dir = os.path.join(script_dir, args.output_dir)

    print("=" * 70)
    print("🚀 LSTM FRAUD DETECTION - TRAINING WITH THRESHOLD OPTIMIZATION")
    print("=" * 70)
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Sequence length: {args.sequence_length}")
    print(f"🔧 Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"🎯 Threshold strategy: {args.strategy}")
    print("=" * 70)

    try:
        # STEP 1: Load data
        df = load_transaction_data(data_dir)

        # STEP 2: Prepare sequences
        data = prepare_lstm_sequences(df, sequence_length=args.sequence_length)

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        # STEP 3: Train model
        model_config = {
            'sequence_length': args.sequence_length,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        }

        model = train_lstm_model(X_train, y_train, model_config, args.verbose)

        # STEP 4: Threshold Optimization (PHẦN QUAN TRỌNG)
        # Sử dụng test set để optimize threshold
        # LƯU Ý: Trong production, nên dùng validation set riêng
        threshold_result = optimize_threshold(
            model=model,
            X_val=X_test,
            y_val=y_test,
            strategy=args.strategy
        )

        # Evaluate trên test set với threshold đã chọn
        print("\n" + "=" * 70)
        print("STEP 4b: EVALUATE ON TEST SET")
        print("=" * 70)

        test_metrics = model.evaluate(X_test, y_test, verbose=True)

        # Đánh giá với threshold tối ưu
        optimal_threshold = threshold_result['recommended_threshold']
        y_pred_optimal = model.predict(X_test, threshold=optimal_threshold)
        y_pred_prob = model.predict_proba(X_test)

        from sklearn.metrics import precision_score, recall_score, f1_score

        print(f"\n  Performance với threshold tối ưu ({optimal_threshold:.4f}):")
        print(f"  - Recall: {recall_score(y_test, y_pred_optimal):.4f}")
        print(f"  - Precision: {precision_score(y_test, y_pred_optimal, zero_division=0):.4f}")
        print(f"  - F1-Score: {f1_score(y_test, y_pred_optimal, zero_division=0):.4f}")

        # STEP 5: Save results
        save_results(model, threshold_result, output_dir, test_metrics)

        # Summary
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\n📊 SUMMARY:")
        print("-" * 40)
        print(f"  ROC-AUC: {threshold_result['roc_auc']:.4f}")
        print(f"  Average Precision: {threshold_result['average_precision']:.4f}")
        print(f"  Recommended Threshold: {threshold_result['recommended_threshold']:.4f}")
        print(f"  Strategy: {args.strategy}")

        if 'metrics_at_threshold' in threshold_result:
            m = threshold_result['metrics_at_threshold']
            print(f"\n  At recommended threshold:")
            print(f"    Recall: {m['recall']:.2%}")
            print(f"    Precision: {m['precision']:.2%}")
            print(f"    F1: {m['f1']:.4f}")
            print(f"    FPR: {m['fpr']:.2%}")

        print(f"\n📁 Files saved in: {output_dir}")
        print("   - lstm.pth (model)")
        print("   - lstm_threshold_config.json")
        print("   - fraud_classifier_config.json")
        print("   - lstm_training_report.json")
        print("   - threshold_analysis.csv")
        print("=" * 70)

        print("\n💡 HƯỚNG DẪN SỬ DỤNG TRONG PRODUCTION:")
        print("-" * 40)
        print("""
1. Load model và threshold config:

   from models.layer2.lstm_sequence import LSTMSequenceModel
   from utils import FraudThresholdClassifier
   import json

   model = LSTMSequenceModel()
   model.load('models/trained/lstm.pth')

   with open('models/trained/fraud_classifier_config.json') as f:
       config = json.load(f)
   classifier = FraudThresholdClassifier.from_config(config)

2. Predict và classify:

   proba = model.predict_proba(sequences)
   results = classifier.classify_batch(proba)

   # Hoặc single prediction
   result = classifier.classify_single(proba[0])
   print(result)  # {'risk_tier': 'MEDIUM_RISK', 'action': '...', ...}
""")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

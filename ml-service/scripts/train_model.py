#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script huấn luyện model ML Fraud Detection
Sử dụng: python scripts/train_model.py --data_dir data/generated --output_dir models/trained
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.layer1.isolation_forest import IsolationForestModel
from models.layer1.lightgbm_model import LightGBMModel


def load_data(data_dir):
    """
    Load dữ liệu từ thư mục

    Args:
        data_dir: Đường dẫn thư mục chứa users.csv và transactions.csv

    Returns:
        users_df, transactions_df
    """
    print("📂 Đang load dữ liệu...")

    users_path = os.path.join(data_dir, 'users.csv')
    transactions_path = os.path.join(data_dir, 'transactions.csv')

    if not os.path.exists(users_path):
        raise FileNotFoundError(f"Không tìm thấy file: {users_path}")
    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"Không tìm thấy file: {transactions_path}")

    users_df = pd.read_csv(users_path)
    transactions_df = pd.read_csv(transactions_path)

    print(f"   ✅ Users: {len(users_df):,} records")
    print(f"   ✅ Transactions: {len(transactions_df):,} records")
    print(f"   ✅ Fraud rate: {transactions_df['is_fraud'].mean()*100:.2f}%")

    return users_df, transactions_df


def prepare_features(transactions_df, users_df):
    """
    Chuẩn bị features cho training

    Args:
        transactions_df: DataFrame giao dịch
        users_df: DataFrame người dùng

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: Tên các features
    """
    print("🔧 Đang chuẩn bị features...")

    # Merge với user data
    df = transactions_df.merge(users_df[['user_id', 'age', 'income_level',
                                          'account_age_days', 'kyc_level',
                                          'avg_monthly_transactions',
                                          'avg_transaction_amount',
                                          'risk_score_historical']],
                               on='user_id', how='left')

    # Các features số
    numeric_features = [
        'amount',
        'session_duration_sec',
        'login_attempts',
        'time_since_last_transaction_min',
        'hour_of_day',
        'day_of_week',
        'velocity_1h',
        'velocity_24h',
        'amount_deviation_ratio',
        'age',
        'account_age_days',
        'kyc_level',
        'avg_monthly_transactions',
        'avg_transaction_amount',
        'risk_score_historical'
    ]

    # Các features binary
    binary_features = [
        'is_international',
        'is_new_recipient',
        'is_new_device',
        'is_new_location',
        'is_weekend'
    ]

    # Các features categorical cần encode
    categorical_features = [
        'transaction_type',
        'channel',
        'recipient_type',
        'device_type',
        'income_level'
    ]

    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('unknown'))
        label_encoders[col] = le

    # Tạo feature matrix
    feature_cols = (numeric_features + binary_features +
                    [f'{col}_encoded' for col in categorical_features])

    X = df[feature_cols].fillna(0).values
    y = df['is_fraud'].values

    print(f"   ✅ Feature matrix: {X.shape}")
    print(f"   ✅ Features: {len(feature_cols)}")

    return X, y, feature_cols, label_encoders


def train_isolation_forest(X_train, y_train, X_test, y_test, output_dir):
    """
    Huấn luyện Isolation Forest model

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        output_dir: Thư mục lưu model

    Returns:
        model, metrics
    """
    print("\n" + "=" * 50)
    print("🌲 TRAINING ISOLATION FOREST")
    print("=" * 50)

    model = IsolationForestModel()

    # Fit model (Isolation Forest là unsupervised, chỉ cần X)
    print("   📊 Fitting model...")
    model.fit(X_train)

    # Predict
    print("   🔍 Predicting...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Chuyển đổi: Isolation Forest trả về 1 cho normal, -1 cho anomaly
    # Ta cần chuyển -1 -> 1 (fraud), 1 -> 0 (normal)
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    # Tính metrics
    metrics = calculate_metrics(y_test, y_pred_binary, y_prob)
    print_metrics("Isolation Forest", metrics)

    # Save model
    model_path = os.path.join(output_dir, 'isolation_forest.pkl')
    model.save(model_path)
    print(f"   💾 Model saved: {model_path}")

    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names, output_dir):
    """
    Huấn luyện LightGBM model

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Tên các features
        output_dir: Thư mục lưu model

    Returns:
        model, metrics
    """
    print("\n" + "=" * 50)
    print("🚀 TRAINING LIGHTGBM")
    print("=" * 50)

    model = LightGBMModel()

    # Fit model
    print("   📊 Fitting model...")
    model.fit(X_train, y_train, feature_names=feature_names)

    # Predict
    print("   🔍 Predicting...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Tính metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_metrics("LightGBM", metrics)

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print("\n   📊 Top 10 Important Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, score in sorted_importance:
            print(f"      - {name}: {score:.4f}")

    # Save model
    model_path = os.path.join(output_dir, 'lightgbm.pkl')
    model.save(model_path)
    print(f"   💾 Model saved: {model_path}")

    return model, metrics


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Tính toán các metrics đánh giá model

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary chứa các metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positive'] = int(tp)
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)

    return metrics


def print_metrics(model_name, metrics):
    """In metrics đẹp ra console"""
    print(f"\n   📈 {model_name} Performance:")
    print(f"      - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      - Precision: {metrics['precision']:.4f}")
    print(f"      - Recall:    {metrics['recall']:.4f}")
    print(f"      - F1-Score:  {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"      - AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"      - Confusion Matrix:")
    print(f"        TP: {metrics['true_positive']:,}  FP: {metrics['false_positive']:,}")
    print(f"        FN: {metrics['false_negative']:,}  TN: {metrics['true_negative']:,}")


def save_training_report(output_dir, metrics_dict, training_info):
    """
    Lưu báo cáo training

    Args:
        output_dir: Thư mục lưu báo cáo
        metrics_dict: Dictionary chứa metrics của các models
        training_info: Thông tin training
    """
    report = {
        'training_time': datetime.now().isoformat(),
        'training_info': training_info,
        'models': metrics_dict
    }

    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Training report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Huấn luyện ML Fraud Detection models')
    parser.add_argument('--data_dir', type=str, default='data/generated',
                        help='Thư mục chứa dữ liệu training (default: data/generated)')
    parser.add_argument('--output_dir', type=str, default='models/trained',
                        help='Thư mục lưu models (default: models/trained)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Tỷ lệ test set (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    # Xác định đường dẫn
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, args.data_dir)
    output_dir = os.path.join(script_dir, args.output_dir)

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 ML FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Test size: {args.test_size * 100}%")
    print("=" * 60)

    # Load data
    users_df, transactions_df = load_data(data_dir)

    # Prepare features
    X, y, feature_names, label_encoders = prepare_features(transactions_df, users_df)

    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"   ✅ Training set: {len(X_train):,} samples")
    print(f"   ✅ Test set: {len(X_test):,} samples")

    # Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler và label encoders
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   💾 Scaler saved: {scaler_path}")

    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"   💾 Label encoders saved: {encoders_path}")

    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"   💾 Feature names saved: {features_path}")

    # Training
    metrics_dict = {}

    # 1. Train Isolation Forest
    iso_model, iso_metrics = train_isolation_forest(
        X_train_scaled, y_train, X_test_scaled, y_test, output_dir
    )
    metrics_dict['isolation_forest'] = iso_metrics

    # 2. Train LightGBM
    lgb_model, lgb_metrics = train_lightgbm(
        X_train, y_train, X_test, y_test, feature_names, output_dir
    )
    metrics_dict['lightgbm'] = lgb_metrics

    # Save training report
    training_info = {
        'total_samples': len(X),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_names),
        'fraud_rate': float(y.mean()),
        'test_size': args.test_size,
        'random_state': args.random_state
    }
    save_training_report(output_dir, metrics_dict, training_info)

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print("\n📊 Model Performance Summary:")
    print("-" * 40)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'AUC-ROC':<10}")
    print("-" * 40)
    for model_name, metrics in metrics_dict.items():
        auc = metrics.get('auc_roc', 0)
        print(f"{model_name:<20} {metrics['accuracy']:.4f}     {metrics['f1']:.4f}     {auc:.4f}")
    print("-" * 40)

    print(f"\n📁 Models saved in: {output_dir}")
    print("   - isolation_forest.pkl")
    print("   - lightgbm.pkl")
    print("   - scaler.pkl")
    print("   - label_encoders.pkl")
    print("   - feature_names.json")
    print("   - training_report.json")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script predict giao dịch nhanh từ command line
Sử dụng: python scripts/predict.py --amount 5000000 --user_id USR001
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_models(models_dir):
    """Load các model đã train"""
    models = {}

    # Load Isolation Forest
    iso_path = os.path.join(models_dir, 'isolation_forest.pkl')
    if os.path.exists(iso_path):
        with open(iso_path, 'rb') as f:
            models['isolation_forest'] = pickle.load(f)
        print("✅ Loaded Isolation Forest")

    # Load LightGBM
    lgb_path = os.path.join(models_dir, 'lightgbm.pkl')
    if os.path.exists(lgb_path):
        with open(lgb_path, 'rb') as f:
            models['lightgbm'] = pickle.load(f)
        print("✅ Loaded LightGBM")

    # Load Scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
        print("✅ Loaded Scaler")

    # Load Feature names
    features_path = os.path.join(models_dir, 'feature_names.json')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            models['feature_names'] = json.load(f)
        print("✅ Loaded Feature names")

    # Load Label encoders
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            models['label_encoders'] = pickle.load(f)
        print("✅ Loaded Label encoders")

    return models


def prepare_transaction_features(transaction, models):
    """Chuẩn bị features từ transaction data"""

    # Default values cho các features
    defaults = {
        'amount': 1000000,
        'session_duration_sec': 180,
        'login_attempts': 1,
        'time_since_last_transaction_min': 60,
        'hour_of_day': 10,
        'day_of_week': 1,
        'velocity_1h': 1,
        'velocity_24h': 3,
        'amount_deviation_ratio': 0.5,
        'age': 30,
        'account_age_days': 365,
        'kyc_level': 2,
        'avg_monthly_transactions': 20,
        'avg_transaction_amount': 3000000,
        'risk_score_historical': 0.1,
        'is_international': 0,
        'is_new_recipient': 0,
        'is_new_device': 0,
        'is_new_location': 0,
        'is_weekend': 0,
        'transaction_type': 'transfer',
        'channel': 'mobile',
        'recipient_type': 'individual',
        'device_type': 'android',
        'income_level': 'medium'
    }

    # Merge với input
    for key, value in transaction.items():
        if key in defaults:
            defaults[key] = value

    # Tính toán derived features
    if 'amount' in transaction and 'avg_transaction_amount' in defaults:
        defaults['amount_deviation_ratio'] = transaction['amount'] / defaults['avg_transaction_amount']

    # Encode categorical features
    label_encoders = models.get('label_encoders', {})
    categorical_features = ['transaction_type', 'channel', 'recipient_type', 'device_type', 'income_level']

    for col in categorical_features:
        if col in label_encoders:
            try:
                defaults[f'{col}_encoded'] = label_encoders[col].transform([defaults[col]])[0]
            except:
                defaults[f'{col}_encoded'] = 0
        else:
            defaults[f'{col}_encoded'] = 0

    # Tạo feature vector theo thứ tự
    feature_names = models.get('feature_names', [])
    if not feature_names:
        # Default feature order
        feature_names = [
            'amount', 'session_duration_sec', 'login_attempts',
            'time_since_last_transaction_min', 'hour_of_day', 'day_of_week',
            'velocity_1h', 'velocity_24h', 'amount_deviation_ratio',
            'age', 'account_age_days', 'kyc_level',
            'avg_monthly_transactions', 'avg_transaction_amount',
            'risk_score_historical', 'is_international', 'is_new_recipient',
            'is_new_device', 'is_new_location', 'is_weekend',
            'transaction_type_encoded', 'channel_encoded',
            'recipient_type_encoded', 'device_type_encoded', 'income_level_encoded'
        ]

    features = []
    for name in feature_names:
        if name in defaults:
            features.append(defaults[name])
        else:
            features.append(0)

    return np.array(features).reshape(1, -1)


def predict(transaction, models):
    """Predict fraud cho một giao dịch"""

    # Prepare features
    X = prepare_transaction_features(transaction, models)

    results = {
        'transaction': transaction,
        'model_scores': {},
        'risk_factors': []
    }

    # Scale for Isolation Forest
    scaler = models.get('scaler')
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Predict với Isolation Forest
    if 'isolation_forest' in models:
        iso_model = models['isolation_forest']
        try:
            # Isolation Forest trả về -1 cho anomaly, 1 cho normal
            iso_pred = iso_model.model.predict(X_scaled)[0]
            iso_score = iso_model.model.decision_function(X_scaled)[0]
            # Chuyển đổi score thành probability (0-1)
            iso_prob = 1 / (1 + np.exp(iso_score))  # Sigmoid
            results['model_scores']['isolation_forest'] = float(iso_prob)
        except Exception as e:
            print(f"⚠️ Isolation Forest error: {e}")
            results['model_scores']['isolation_forest'] = 0.5

    # Predict với LightGBM
    if 'lightgbm' in models:
        lgb_model = models['lightgbm']
        try:
            lgb_prob = lgb_model.model.predict_proba(X)[0][1]
            results['model_scores']['lightgbm'] = float(lgb_prob)
        except Exception as e:
            print(f"⚠️ LightGBM error: {e}")
            results['model_scores']['lightgbm'] = 0.5

    # Tính combined score
    scores = list(results['model_scores'].values())
    if scores:
        # Weighted average (LightGBM thường chính xác hơn)
        weights = {'isolation_forest': 0.3, 'lightgbm': 0.7}
        weighted_sum = sum(results['model_scores'].get(m, 0) * w
                          for m, w in weights.items())
        weight_total = sum(w for m, w in weights.items() if m in results['model_scores'])
        results['risk_score'] = weighted_sum / weight_total if weight_total > 0 else 0.5
    else:
        results['risk_score'] = 0.5

    # Xác định risk level
    score = results['risk_score']
    if score < 0.3:
        results['risk_level'] = 'LOW'
        results['recommendation'] = 'APPROVE'
    elif score < 0.6:
        results['risk_level'] = 'MEDIUM'
        results['recommendation'] = 'REVIEW'
    elif score < 0.8:
        results['risk_level'] = 'HIGH'
        results['recommendation'] = 'MANUAL_REVIEW'
    else:
        results['risk_level'] = 'CRITICAL'
        results['recommendation'] = 'BLOCK'

    results['is_fraud'] = score >= 0.5

    # Phân tích risk factors
    if transaction.get('amount', 0) > 10000000:
        results['risk_factors'].append('Số tiền lớn (> 10M VND)')
    if transaction.get('is_new_recipient'):
        results['risk_factors'].append('Người nhận mới')
    if transaction.get('is_new_device'):
        results['risk_factors'].append('Thiết bị mới')
    if transaction.get('is_international'):
        results['risk_factors'].append('Giao dịch quốc tế')
    if transaction.get('hour_of_day', 10) in [1, 2, 3, 4, 5]:
        results['risk_factors'].append('Thời gian bất thường (1-5 giờ sáng)')
    if transaction.get('login_attempts', 1) > 3:
        results['risk_factors'].append('Nhiều lần thử đăng nhập')
    if transaction.get('velocity_1h', 1) > 5:
        results['risk_factors'].append('Tốc độ giao dịch cao')

    return results


def print_result(result):
    """In kết quả đẹp ra console"""
    print("\n" + "=" * 60)
    print("🔍 KẾT QUẢ PHÂN TÍCH GIAO DỊCH")
    print("=" * 60)

    # Risk Score
    score = result['risk_score']
    level = result['risk_level']

    if level == 'LOW':
        emoji = '✅'
        color = '\033[92m'  # Green
    elif level == 'MEDIUM':
        emoji = '⚠️'
        color = '\033[93m'  # Yellow
    elif level == 'HIGH':
        emoji = '🚨'
        color = '\033[91m'  # Red
    else:
        emoji = '🛑'
        color = '\033[91m'  # Red

    reset = '\033[0m'

    print(f"\n{emoji} Risk Score: {color}{score:.2%}{reset}")
    print(f"   Risk Level: {color}{level}{reset}")
    print(f"   Is Fraud: {'Yes' if result['is_fraud'] else 'No'}")
    print(f"   Recommendation: {result['recommendation']}")

    # Model scores
    print("\n📊 Model Scores:")
    for model, score in result['model_scores'].items():
        print(f"   - {model}: {score:.4f}")

    # Risk factors
    if result['risk_factors']:
        print("\n⚠️ Risk Factors:")
        for factor in result['risk_factors']:
            print(f"   - {factor}")
    else:
        print("\n✅ Không phát hiện yếu tố rủi ro đáng ngờ")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Predict giao dịch')

    # Required
    parser.add_argument('--amount', type=int, required=True,
                        help='Số tiền giao dịch (VND)')

    # Optional
    parser.add_argument('--user_id', type=str, default='USR000001',
                        help='ID người dùng')
    parser.add_argument('--transaction_type', type=str, default='transfer',
                        choices=['transfer', 'payment', 'withdrawal'],
                        help='Loại giao dịch')
    parser.add_argument('--channel', type=str, default='mobile',
                        choices=['mobile', 'web', 'atm'],
                        help='Kênh giao dịch')
    parser.add_argument('--hour', type=int, default=10,
                        help='Giờ giao dịch (0-23)')
    parser.add_argument('--is_new_recipient', action='store_true',
                        help='Người nhận mới')
    parser.add_argument('--is_new_device', action='store_true',
                        help='Thiết bị mới')
    parser.add_argument('--is_international', action='store_true',
                        help='Giao dịch quốc tế')
    parser.add_argument('--login_attempts', type=int, default=1,
                        help='Số lần thử đăng nhập')
    parser.add_argument('--velocity_1h', type=int, default=1,
                        help='Số giao dịch trong 1h')
    parser.add_argument('--models_dir', type=str, default='models/trained',
                        help='Thư mục chứa models')

    args = parser.parse_args()

    # Xác định đường dẫn models
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(script_dir, args.models_dir)

    print("=" * 60)
    print("🚀 ML FRAUD DETECTION - PREDICT")
    print("=" * 60)

    # Load models
    print("\n📂 Loading models...")
    if not os.path.exists(models_dir):
        print(f"❌ Không tìm thấy thư mục models: {models_dir}")
        print("   Hãy chạy train_model.py trước!")
        sys.exit(1)

    models = load_models(models_dir)

    if not models:
        print("❌ Không load được model nào!")
        sys.exit(1)

    # Tạo transaction từ args
    transaction = {
        'user_id': args.user_id,
        'amount': args.amount,
        'transaction_type': args.transaction_type,
        'channel': args.channel,
        'hour_of_day': args.hour,
        'is_new_recipient': 1 if args.is_new_recipient else 0,
        'is_new_device': 1 if args.is_new_device else 0,
        'is_international': 1 if args.is_international else 0,
        'login_attempts': args.login_attempts,
        'velocity_1h': args.velocity_1h
    }

    print(f"\n📝 Transaction Input:")
    for key, value in transaction.items():
        print(f"   - {key}: {value}")

    # Predict
    result = predict(transaction, models)

    # Print result
    print_result(result)

    # Return exit code based on risk
    if result['risk_level'] in ['HIGH', 'CRITICAL']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

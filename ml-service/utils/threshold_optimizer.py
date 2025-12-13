"""
Threshold Optimizer - Tối ưu hóa ngưỡng quyết định cho LSTM Fraud Detection
============================================================================
Module này cung cấp các hàm để:
1. Tính và visualize ROC Curve, Precision-Recall Curve
2. Tự động đề xuất threshold dựa trên constraints nghiệp vụ
3. Phân tầng giao dịch theo mức độ rủi ro (LOW/MEDIUM/HIGH)
4. Log chi tiết metrics tại mỗi threshold

Author: ML Team - Fraud Detection
Created: 2025
Target: Ngân hàng Việt Nam - Fraud ratio ~2-4%

QUAN TRỌNG:
- Module này KHÔNG nhìn vào test label để chọn threshold đẹp
- Threshold được đề xuất dựa trên validation set hoặc business rules
- Mục đích: Cảnh báo + Review (KHÔNG auto-block)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Sklearn imports
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Cấu hình mặc định cho ngân hàng Việt Nam
# =============================================================================

# Default thresholds cho 3 mức rủi ro
DEFAULT_THRESHOLDS = {
    'LOW_RISK': 0.08,      # Sensitive, dùng cho monitoring
    'MEDIUM_RISK': 0.15,   # Balanced, recommend cho production
    'HIGH_RISK': 0.30      # Conservative, high confidence fraud
}

# Constraints nghiệp vụ
BUSINESS_CONSTRAINTS = {
    'min_recall': 0.70,           # Recall tối thiểu 70%
    'min_precision': 0.05,        # Precision tối thiểu 5% (với fraud ratio 2-4%)
    'max_fpr': 0.35,              # False Positive Rate tối đa 35%
    'expected_fraud_ratio': 0.03  # Fraud ratio kỳ vọng 3%
}


# =============================================================================
# DATA CLASSES - Cấu trúc dữ liệu
# =============================================================================

@dataclass
class ThresholdMetrics:
    """
    Metrics tại một threshold cụ thể

    Attributes:
        threshold: Giá trị threshold
        precision: Tỷ lệ đúng trong số flagged
        recall: Tỷ lệ fraud bắt được
        f1: F1-score (harmonic mean của precision và recall)
        fpr: False Positive Rate
        fnr: False Negative Rate (= 1 - recall)
        support: Số lượng positive samples
    """
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    support: int

    def to_dict(self) -> Dict:
        """Chuyển đổi sang dictionary"""
        return {
            'threshold': round(self.threshold, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'fpr': round(self.fpr, 4),
            'fnr': round(self.fnr, 4),
            'support': self.support
        }


@dataclass
class RiskTier:
    """
    Định nghĩa một tier rủi ro

    Attributes:
        name: Tên tier (LOW/MEDIUM/HIGH)
        threshold_min: Ngưỡng dưới (inclusive)
        threshold_max: Ngưỡng trên (exclusive)
        action: Hành động đề xuất
        sla_minutes: SLA review (phút)
    """
    name: str
    threshold_min: float
    threshold_max: float
    action: str
    sla_minutes: int


# =============================================================================
# CORE FUNCTIONS - Tính toán curves
# =============================================================================

def compute_roc_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Tính ROC Curve và AUC

    ROC Curve thể hiện trade-off giữa True Positive Rate (Recall) và
    False Positive Rate tại các threshold khác nhau.

    Args:
        y_true: Labels thực tế (0 = normal, 1 = fraud)
        y_pred_prob: Xác suất fraud từ model (0 đến 1)

    Returns:
        Tuple chứa:
        - fpr: Array False Positive Rate tại các thresholds
        - tpr: Array True Positive Rate (Recall) tại các thresholds
        - thresholds: Array các threshold tương ứng
        - auc: ROC-AUC score

    Example:
        >>> fpr, tpr, thresholds, auc = compute_roc_curve(y_true, y_pred_prob)
        >>> print(f"ROC-AUC: {auc:.4f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    if len(y_true) != len(y_pred_prob):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred_prob={len(y_pred_prob)}")

    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true phải chỉ chứa 0 và 1")

    # Tính ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    logger.info(f"[ROC] Computed ROC curve: {len(thresholds)} thresholds, AUC={auc:.4f}")

    return fpr, tpr, thresholds, auc


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Tính Precision-Recall Curve và Average Precision

    PR Curve quan trọng hơn ROC Curve khi data mất cân bằng mạnh (fraud ~2-4%)
    vì nó không bị ảnh hưởng bởi số lượng lớn True Negatives.

    Args:
        y_true: Labels thực tế (0 = normal, 1 = fraud)
        y_pred_prob: Xác suất fraud từ model (0 đến 1)

    Returns:
        Tuple chứa:
        - precision: Array precision tại các thresholds
        - recall: Array recall tại các thresholds
        - thresholds: Array các threshold tương ứng
        - ap: Average Precision score

    Note:
        - precision và recall arrays có length = len(thresholds) + 1
        - Phần tử cuối là precision=1, recall=0 (convention của sklearn)

    Example:
        >>> prec, rec, thresholds, ap = compute_precision_recall_curve(y_true, y_pred_prob)
        >>> print(f"Average Precision: {ap:.4f}")
    """
    # Validate inputs
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    if len(y_true) != len(y_pred_prob):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred_prob={len(y_pred_prob)}")

    # Tính PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    logger.info(f"[PR] Computed PR curve: {len(thresholds)} thresholds, AP={ap:.4f}")

    return precision, recall, thresholds, ap


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float
) -> ThresholdMetrics:
    """
    Tính toàn bộ metrics tại một threshold cụ thể

    Đây là hàm core để evaluate một threshold trước khi deploy.

    Args:
        y_true: Labels thực tế (0 = normal, 1 = fraud)
        y_pred_prob: Xác suất fraud từ model
        threshold: Ngưỡng quyết định (0 đến 1)

    Returns:
        ThresholdMetrics object chứa tất cả metrics

    Example:
        >>> metrics = compute_metrics_at_threshold(y_true, y_pred_prob, 0.15)
        >>> print(f"Recall: {metrics.recall:.2%}, Precision: {metrics.precision:.2%}")
    """
    # Validate
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold phải trong khoảng [0, 1], got {threshold}")

    # Predictions tại threshold
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Tính metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    support = int(tp + fn)  # Số positive samples

    return ThresholdMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        support=support
    )


# =============================================================================
# THRESHOLD RECOMMENDATION - Đề xuất threshold tự động
# =============================================================================

def find_threshold_by_recall(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    target_recall: float = 0.70,
    min_precision: float = 0.05
) -> Tuple[float, ThresholdMetrics]:
    """
    Tìm threshold đạt recall target với precision constraint

    Chiến lược: Tìm threshold THẤP NHẤT (sensitive nhất) mà vẫn đạt:
    - Recall >= target_recall
    - Precision >= min_precision

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud từ model
        target_recall: Recall mục tiêu (default 70%)
        min_precision: Precision tối thiểu (default 5%)

    Returns:
        Tuple (threshold, ThresholdMetrics)
        Nếu không tìm được threshold phù hợp, trả về (None, None)

    Note:
        - Ưu tiên recall trước, precision sau
        - Với data imbalanced, precision 5-10% là realistic
    """
    logger.info(f"[FIND] Finding threshold for recall >= {target_recall:.0%}, precision >= {min_precision:.0%}")

    # Tính PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

    # precision, recall có length = len(thresholds) + 1
    # Bỏ phần tử cuối để match length
    precision = precision[:-1]
    recall = recall[:-1]

    # Tìm các thresholds thỏa mãn constraints
    valid_mask = (recall >= target_recall) & (precision >= min_precision)

    if not np.any(valid_mask):
        logger.warning(f"[FIND] Không tìm được threshold thỏa mãn constraints!")
        logger.warning(f"  - Max recall với precision >= {min_precision:.0%}: {recall[precision >= min_precision].max():.2%}"
                      if np.any(precision >= min_precision) else "  - Không có threshold nào đạt precision constraint")
        return None, None

    # Lấy threshold cao nhất trong các valid (recall cao nhất có thể)
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    best_threshold = thresholds[best_idx]

    # Tính metrics tại threshold này
    metrics = compute_metrics_at_threshold(y_true, y_pred_prob, best_threshold)

    logger.info(f"[FIND] Found threshold: {best_threshold:.4f}")
    logger.info(f"  - Recall: {metrics.recall:.2%}")
    logger.info(f"  - Precision: {metrics.precision:.2%}")
    logger.info(f"  - FPR: {metrics.fpr:.2%}")

    return best_threshold, metrics


def find_threshold_by_fpr(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    max_fpr: float = 0.30,
    min_recall: float = 0.50
) -> Tuple[float, ThresholdMetrics]:
    """
    Tìm threshold với FPR constraint (kiểm soát false alarms)

    Chiến lược: Tìm threshold mà:
    - FPR <= max_fpr (không quá nhiều false alarms)
    - Recall >= min_recall (vẫn bắt được phần lớn fraud)

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud từ model
        max_fpr: FPR tối đa chấp nhận được (default 30%)
        min_recall: Recall tối thiểu (default 50%)

    Returns:
        Tuple (threshold, ThresholdMetrics)

    Use case:
        - Khi team review có capacity giới hạn
        - Cần control số lượng alerts/ngày
    """
    logger.info(f"[FIND] Finding threshold for FPR <= {max_fpr:.0%}, recall >= {min_recall:.0%}")

    # Tính ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # tpr = recall, fpr = false positive rate
    recall = tpr

    # Tìm các thresholds thỏa mãn constraints
    valid_mask = (fpr <= max_fpr) & (recall >= min_recall)

    if not np.any(valid_mask):
        logger.warning(f"[FIND] Không tìm được threshold thỏa mãn constraints!")
        return None, None

    # Lấy threshold có recall cao nhất trong các valid
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    best_threshold = thresholds[best_idx]

    # Tính metrics
    metrics = compute_metrics_at_threshold(y_true, y_pred_prob, best_threshold)

    logger.info(f"[FIND] Found threshold: {best_threshold:.4f}")
    logger.info(f"  - Recall: {metrics.recall:.2%}")
    logger.info(f"  - FPR: {metrics.fpr:.2%}")

    return best_threshold, metrics


def recommend_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    strategy: str = 'balanced',
    constraints: Dict = None
) -> Dict:
    """
    Đề xuất threshold dựa trên strategy và constraints nghiệp vụ

    Đây là hàm chính để recommend threshold cho production.

    Args:
        y_true: Labels thực tế (validation set, KHÔNG PHẢI test set)
        y_pred_prob: Xác suất fraud từ model
        strategy: Chiến lược tối ưu
            - 'recall_focused': Ưu tiên recall (bắt fraud), chấp nhận FPR cao
            - 'balanced': Cân bằng recall và precision
            - 'precision_focused': Ưu tiên precision (ít false alarm)
            - 'fpr_controlled': Kiểm soát FPR ở mức cố định
        constraints: Dict override constraints mặc định

    Returns:
        Dict chứa:
        {
            'recommended_threshold': float,
            'strategy': str,
            'metrics': ThresholdMetrics,
            'alternatives': List[Dict],  # Các threshold khác để tham khảo
            'warnings': List[str]
        }

    Example:
        >>> result = recommend_threshold(y_val, probs, strategy='balanced')
        >>> print(f"Recommend: {result['recommended_threshold']:.4f}")
    """
    logger.info(f"[RECOMMEND] Strategy: {strategy}")

    # Merge constraints với defaults
    cons = BUSINESS_CONSTRAINTS.copy()
    if constraints:
        cons.update(constraints)

    result = {
        'recommended_threshold': None,
        'strategy': strategy,
        'metrics': None,
        'alternatives': [],
        'warnings': []
    }

    # Tìm threshold theo strategy
    if strategy == 'recall_focused':
        # Ưu tiên recall >= 80%, precision >= 3%
        threshold, metrics = find_threshold_by_recall(
            y_true, y_pred_prob,
            target_recall=0.80,
            min_precision=0.03
        )

    elif strategy == 'balanced':
        # Recall >= 70%, precision >= 5%
        threshold, metrics = find_threshold_by_recall(
            y_true, y_pred_prob,
            target_recall=cons['min_recall'],
            min_precision=cons['min_precision']
        )

    elif strategy == 'precision_focused':
        # Recall >= 50%, precision >= 10%
        threshold, metrics = find_threshold_by_recall(
            y_true, y_pred_prob,
            target_recall=0.50,
            min_precision=0.10
        )

    elif strategy == 'fpr_controlled':
        # FPR <= 30%, recall >= 60%
        threshold, metrics = find_threshold_by_fpr(
            y_true, y_pred_prob,
            max_fpr=cons['max_fpr'],
            min_recall=0.60
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Xử lý trường hợp không tìm được threshold phù hợp
    if threshold is None:
        result['warnings'].append("Không tìm được threshold tối ưu theo constraints. Sử dụng default.")
        threshold = DEFAULT_THRESHOLDS['MEDIUM_RISK']
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, threshold)

    result['recommended_threshold'] = threshold
    result['metrics'] = metrics

    # Tính alternatives (3 mức cố định)
    for name, th in DEFAULT_THRESHOLDS.items():
        alt_metrics = compute_metrics_at_threshold(y_true, y_pred_prob, th)
        result['alternatives'].append({
            'name': name,
            'threshold': th,
            'metrics': alt_metrics.to_dict()
        })

    # Warnings
    if metrics.recall < cons['min_recall']:
        result['warnings'].append(f"Recall {metrics.recall:.1%} < target {cons['min_recall']:.0%}")

    if metrics.fpr > cons['max_fpr']:
        result['warnings'].append(f"FPR {metrics.fpr:.1%} > max {cons['max_fpr']:.0%}")

    return result


# =============================================================================
# RISK TIERING - Phân tầng giao dịch
# =============================================================================

def define_risk_tiers(
    thresholds: Dict[str, float] = None
) -> List[RiskTier]:
    """
    Định nghĩa các tier rủi ro với thresholds và actions tương ứng

    Args:
        thresholds: Dict custom thresholds, default sử dụng DEFAULT_THRESHOLDS

    Returns:
        List[RiskTier] định nghĩa các tier

    Note:
        - Tier được sắp xếp từ LOW đến HIGH
        - Mỗi giao dịch thuộc đúng 1 tier
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    tiers = [
        RiskTier(
            name='NORMAL',
            threshold_min=0.0,
            threshold_max=thresholds['LOW_RISK'],
            action='Không cần action, log để audit',
            sla_minutes=0  # No SLA
        ),
        RiskTier(
            name='LOW_RISK',
            threshold_min=thresholds['LOW_RISK'],
            threshold_max=thresholds['MEDIUM_RISK'],
            action='Monitor, review nếu có capacity',
            sla_minutes=480  # 8 giờ
        ),
        RiskTier(
            name='MEDIUM_RISK',
            threshold_min=thresholds['MEDIUM_RISK'],
            threshold_max=thresholds['HIGH_RISK'],
            action='Queue review, delay tx nếu amount > 10M',
            sla_minutes=120  # 2 giờ
        ),
        RiskTier(
            name='HIGH_RISK',
            threshold_min=thresholds['HIGH_RISK'],
            threshold_max=1.01,  # > 1.0 để cover = 1.0
            action='Review ngay, gọi điện xác nhận nếu amount > 50M',
            sla_minutes=15  # 15 phút
        ),
    ]

    return tiers


def classify_transactions(
    fraud_probabilities: np.ndarray,
    thresholds: Dict[str, float] = None,
    return_details: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
    """
    Phân loại giao dịch vào các tier rủi ro

    Args:
        fraud_probabilities: Array xác suất fraud từ model
        thresholds: Dict custom thresholds
        return_details: Nếu True, trả về thêm DataFrame chi tiết

    Returns:
        - Array risk_tier names (NORMAL/LOW_RISK/MEDIUM_RISK/HIGH_RISK)
        - Nếu return_details=True: (risk_tiers, DataFrame với chi tiết)

    Example:
        >>> probs = model.predict_proba(X_test)
        >>> tiers = classify_transactions(probs)
        >>> print(f"HIGH_RISK: {(tiers == 'HIGH_RISK').sum()}")
    """
    probs = np.asarray(fraud_probabilities).ravel()
    n = len(probs)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Khởi tạo array risk tiers
    risk_tiers = np.array(['NORMAL'] * n, dtype=object)

    # Phân loại theo thứ tự từ thấp đến cao
    risk_tiers[probs >= thresholds['LOW_RISK']] = 'LOW_RISK'
    risk_tiers[probs >= thresholds['MEDIUM_RISK']] = 'MEDIUM_RISK'
    risk_tiers[probs >= thresholds['HIGH_RISK']] = 'HIGH_RISK'

    if return_details:
        tiers_def = define_risk_tiers(thresholds)

        # Tạo DataFrame chi tiết
        details = pd.DataFrame({
            'fraud_probability': probs,
            'risk_tier': risk_tiers,
        })

        # Add action và SLA
        tier_map = {t.name: (t.action, t.sla_minutes) for t in tiers_def}
        details['recommended_action'] = details['risk_tier'].map(lambda x: tier_map.get(x, ('', 0))[0])
        details['sla_minutes'] = details['risk_tier'].map(lambda x: tier_map.get(x, ('', 0))[1])

        return risk_tiers, details

    return risk_tiers


def get_tier_distribution(
    fraud_probabilities: np.ndarray,
    y_true: np.ndarray = None,
    thresholds: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Thống kê phân bổ giao dịch theo tier và fraud rate

    Args:
        fraud_probabilities: Array xác suất fraud
        y_true: Labels thực tế (optional, để tính actual fraud rate)
        thresholds: Dict custom thresholds

    Returns:
        DataFrame với columns:
        - tier: Tên tier
        - count: Số lượng giao dịch
        - percentage: % trong tổng số
        - fraud_count: Số fraud thực (nếu có y_true)
        - fraud_rate: % fraud trong tier
        - threshold_range: Khoảng threshold
    """
    probs = np.asarray(fraud_probabilities).ravel()
    risk_tiers = classify_transactions(probs, thresholds)

    tier_order = ['NORMAL', 'LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Tính distribution
    rows = []
    for tier in tier_order:
        mask = risk_tiers == tier
        count = mask.sum()

        row = {
            'tier': tier,
            'count': count,
            'percentage': count / len(probs) * 100 if len(probs) > 0 else 0,
        }

        # Threshold range
        if tier == 'NORMAL':
            row['threshold_range'] = f"[0, {thresholds['LOW_RISK']})"
        elif tier == 'LOW_RISK':
            row['threshold_range'] = f"[{thresholds['LOW_RISK']}, {thresholds['MEDIUM_RISK']})"
        elif tier == 'MEDIUM_RISK':
            row['threshold_range'] = f"[{thresholds['MEDIUM_RISK']}, {thresholds['HIGH_RISK']})"
        else:
            row['threshold_range'] = f"[{thresholds['HIGH_RISK']}, 1.0]"

        # Fraud stats nếu có labels
        if y_true is not None:
            y_true_arr = np.asarray(y_true).ravel()
            fraud_in_tier = y_true_arr[mask].sum() if count > 0 else 0
            row['fraud_count'] = int(fraud_in_tier)
            row['fraud_rate'] = (fraud_in_tier / count * 100) if count > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# LOGGING & REPORTING - Log và báo cáo chi tiết
# =============================================================================

def log_threshold_analysis(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds_to_analyze: List[float] = None
) -> pd.DataFrame:
    """
    Log chi tiết metrics tại nhiều thresholds để so sánh

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud
        thresholds_to_analyze: List các threshold cần phân tích
            Default: [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    Returns:
        DataFrame với metrics tại mỗi threshold

    Example:
        >>> df = log_threshold_analysis(y_true, y_pred_prob)
        >>> print(df.to_string())
    """
    if thresholds_to_analyze is None:
        thresholds_to_analyze = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    logger.info("=" * 70)
    logger.info("THRESHOLD ANALYSIS - Chi tiết metrics tại các ngưỡng")
    logger.info("=" * 70)

    rows = []
    for th in thresholds_to_analyze:
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, th)
        rows.append(metrics.to_dict())

        logger.info(f"Threshold {th:.2f}: "
                   f"Recall={metrics.recall:.2%}, "
                   f"Precision={metrics.precision:.2%}, "
                   f"F1={metrics.f1:.4f}, "
                   f"FPR={metrics.fpr:.2%}")

    df = pd.DataFrame(rows)

    # Tìm threshold tốt nhất theo F1
    best_f1_idx = df['f1'].idxmax()
    logger.info("-" * 70)
    logger.info(f"Best F1: threshold={df.loc[best_f1_idx, 'threshold']:.2f}, F1={df.loc[best_f1_idx, 'f1']:.4f}")

    return df


def generate_threshold_report(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    recommended_threshold: float = None,
    model_name: str = 'LSTM'
) -> str:
    """
    Sinh báo cáo text đầy đủ về threshold analysis

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud
        recommended_threshold: Threshold được recommend (optional)
        model_name: Tên model

    Returns:
        String báo cáo formatted
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    # Tính các metrics cơ bản
    _, _, _, auc = compute_roc_curve(y_true, y_pred_prob)
    _, _, _, ap = compute_precision_recall_curve(y_true, y_pred_prob)

    fraud_ratio = y_true.mean()
    total_samples = len(y_true)
    total_fraud = int(y_true.sum())

    report_lines = [
        "=" * 70,
        f"BÁO CÁO THRESHOLD ANALYSIS - {model_name} Fraud Detection",
        "=" * 70,
        "",
        "1. THỐNG KÊ DATA",
        "-" * 40,
        f"   Tổng số samples: {total_samples:,}",
        f"   Tổng số fraud: {total_fraud:,}",
        f"   Fraud ratio: {fraud_ratio:.2%}",
        "",
        "2. MODEL PERFORMANCE",
        "-" * 40,
        f"   ROC-AUC: {auc:.4f}",
        f"   Average Precision: {ap:.4f}",
        "",
        "3. PHÂN TÍCH THRESHOLD",
        "-" * 40,
    ]

    # Phân tích các threshold
    df = log_threshold_analysis(y_true, y_pred_prob)
    report_lines.append(df.to_string(index=False))

    report_lines.extend([
        "",
        "4. TIER DISTRIBUTION",
        "-" * 40,
    ])

    tier_dist = get_tier_distribution(y_pred_prob, y_true)
    report_lines.append(tier_dist.to_string(index=False))

    if recommended_threshold is not None:
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, recommended_threshold)
        report_lines.extend([
            "",
            "5. THRESHOLD ĐỀ XUẤT",
            "-" * 40,
            f"   Threshold: {recommended_threshold:.4f}",
            f"   Recall: {metrics.recall:.2%}",
            f"   Precision: {metrics.precision:.2%}",
            f"   F1-Score: {metrics.f1:.4f}",
            f"   FPR: {metrics.fpr:.2%}",
        ])

    report_lines.extend([
        "",
        "=" * 70,
        "LƯU Ý QUAN TRỌNG",
        "=" * 70,
        "- Threshold này được tính từ validation/test set",
        "- KHÔNG nên điều chỉnh threshold dựa trên test set để 'đẹp số'",
        "- Monitor precision/recall trên production và điều chỉnh nếu cần",
        "- Fraud patterns thay đổi theo thời gian, cần re-evaluate định kỳ",
        "",
    ])

    return "\n".join(report_lines)


def print_summary_table(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: Dict[str, float] = None
) -> None:
    """
    In bảng summary rõ ràng ra console

    Args:
        y_true: Labels thực tế
        y_pred_prob: Xác suất fraud
        thresholds: Dict 3 mức threshold
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    print("\n" + "=" * 80)
    print("SUMMARY TABLE - THRESHOLD ANALYSIS FOR VIETNAM BANKING FRAUD DETECTION")
    print("=" * 80)

    # Header
    print(f"\n{'Risk Level':<15} {'Threshold':>10} {'Recall':>10} {'Precision':>12} {'F1':>10} {'FPR':>10}")
    print("-" * 80)

    # Rows
    for name, th in thresholds.items():
        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, th)
        print(f"{name:<15} {th:>10.4f} {metrics.recall:>9.2%} {metrics.precision:>11.2%} "
              f"{metrics.f1:>10.4f} {metrics.fpr:>9.2%}")

    print("-" * 80)

    # Tier distribution
    print("\nTIER DISTRIBUTION:")
    tier_dist = get_tier_distribution(y_pred_prob, y_true, thresholds)

    print(f"\n{'Tier':<15} {'Count':>10} {'%':>10} {'Frauds':>10} {'Fraud Rate':>12}")
    print("-" * 60)

    for _, row in tier_dist.iterrows():
        fraud_count = row.get('fraud_count', 'N/A')
        fraud_rate = row.get('fraud_rate', 'N/A')
        if isinstance(fraud_rate, float):
            fraud_rate = f"{fraud_rate:.2f}%"
        print(f"{row['tier']:<15} {row['count']:>10,} {row['percentage']:>9.2f}% "
              f"{fraud_count:>10} {fraud_rate:>12}")

    print("\n" + "=" * 80)


# =============================================================================
# PRODUCTION INTEGRATION - Hàm cho production
# =============================================================================

class FraudThresholdClassifier:
    """
    Class wrapper để sử dụng trong production

    Có thể dùng cho:
    - Offline batch scoring
    - Realtime API (FastAPI/Flask)
    - Scheduled evaluation

    Example:
        >>> classifier = FraudThresholdClassifier()
        >>> classifier.set_thresholds(LOW_RISK=0.08, MEDIUM_RISK=0.15, HIGH_RISK=0.30)
        >>>
        >>> # Single prediction
        >>> result = classifier.classify_single(0.25)
        >>> print(result)  # {'risk_tier': 'MEDIUM_RISK', 'action': '...', 'sla_minutes': 120}
        >>>
        >>> # Batch prediction
        >>> results = classifier.classify_batch(prob_array)
    """

    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        default_threshold: float = 0.15
    ):
        """
        Khởi tạo classifier

        Args:
            thresholds: Dict 3 mức threshold
            default_threshold: Threshold mặc định cho binary decision
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.default_threshold = default_threshold
        self.risk_tiers = define_risk_tiers(self.thresholds)

        logger.info(f"[CLASSIFIER] Initialized with thresholds: {self.thresholds}")

    def set_thresholds(
        self,
        LOW_RISK: float = None,
        MEDIUM_RISK: float = None,
        HIGH_RISK: float = None
    ) -> None:
        """Cập nhật thresholds"""
        if LOW_RISK is not None:
            self.thresholds['LOW_RISK'] = LOW_RISK
        if MEDIUM_RISK is not None:
            self.thresholds['MEDIUM_RISK'] = MEDIUM_RISK
        if HIGH_RISK is not None:
            self.thresholds['HIGH_RISK'] = HIGH_RISK

        self.risk_tiers = define_risk_tiers(self.thresholds)
        logger.info(f"[CLASSIFIER] Updated thresholds: {self.thresholds}")

    def classify_single(self, fraud_probability: float) -> Dict:
        """
        Phân loại 1 giao dịch

        Args:
            fraud_probability: Xác suất fraud (0 đến 1)

        Returns:
            Dict chứa:
            - fraud_probability: float
            - risk_tier: str
            - is_flagged: bool (True nếu >= default_threshold)
            - recommended_action: str
            - sla_minutes: int
        """
        # Determine tier
        tier_name = 'NORMAL'
        if fraud_probability >= self.thresholds['HIGH_RISK']:
            tier_name = 'HIGH_RISK'
        elif fraud_probability >= self.thresholds['MEDIUM_RISK']:
            tier_name = 'MEDIUM_RISK'
        elif fraud_probability >= self.thresholds['LOW_RISK']:
            tier_name = 'LOW_RISK'

        # Get tier details
        tier = next((t for t in self.risk_tiers if t.name == tier_name), None)

        return {
            'fraud_probability': round(fraud_probability, 4),
            'risk_tier': tier_name,
            'is_flagged': fraud_probability >= self.default_threshold,
            'recommended_action': tier.action if tier else '',
            'sla_minutes': tier.sla_minutes if tier else 0
        }

    def classify_batch(
        self,
        fraud_probabilities: np.ndarray,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Phân loại batch giao dịch

        Args:
            fraud_probabilities: Array xác suất fraud
            return_dataframe: True = return DataFrame, False = return list of dicts

        Returns:
            DataFrame hoặc List[Dict] với kết quả phân loại
        """
        probs = np.asarray(fraud_probabilities).ravel()
        results = [self.classify_single(p) for p in probs]

        if return_dataframe:
            return pd.DataFrame(results)
        return results

    def get_flagged_indices(
        self,
        fraud_probabilities: np.ndarray,
        threshold: float = None
    ) -> np.ndarray:
        """
        Lấy indices của các giao dịch cần flag

        Args:
            fraud_probabilities: Array xác suất fraud
            threshold: Threshold để flag (default = self.default_threshold)

        Returns:
            Array indices của flagged transactions
        """
        if threshold is None:
            threshold = self.default_threshold

        probs = np.asarray(fraud_probabilities).ravel()
        return np.where(probs >= threshold)[0]

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float = None
    ) -> Dict:
        """
        Evaluate performance tại threshold

        Args:
            y_true: Labels thực tế
            y_pred_prob: Xác suất fraud
            threshold: Threshold để evaluate (default = self.default_threshold)

        Returns:
            Dict metrics
        """
        if threshold is None:
            threshold = self.default_threshold

        metrics = compute_metrics_at_threshold(y_true, y_pred_prob, threshold)
        return metrics.to_dict()

    def to_config(self) -> Dict:
        """Export config để lưu/load"""
        return {
            'thresholds': self.thresholds.copy(),
            'default_threshold': self.default_threshold
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'FraudThresholdClassifier':
        """Load từ config"""
        return cls(
            thresholds=config.get('thresholds'),
            default_threshold=config.get('default_threshold', 0.15)
        )


# =============================================================================
# MAIN - Demo usage
# =============================================================================

if __name__ == '__main__':
    """
    Demo usage của module threshold_optimizer
    """
    print("=" * 70)
    print("THRESHOLD OPTIMIZER - Demo")
    print("=" * 70)

    # Tạo synthetic data để demo
    np.random.seed(42)
    n_samples = 10000
    fraud_ratio = 0.03  # 3% fraud

    # Labels
    y_true = np.random.binomial(1, fraud_ratio, n_samples)

    # Simulated predictions (model output)
    # Fraud cases có probability cao hơn
    y_pred_prob = np.zeros(n_samples)
    y_pred_prob[y_true == 0] = np.random.beta(2, 10, (y_true == 0).sum())  # Normal: skew thấp
    y_pred_prob[y_true == 1] = np.random.beta(5, 3, (y_true == 1).sum())   # Fraud: skew cao
    y_pred_prob = np.clip(y_pred_prob, 0, 1)

    print(f"\nSynthetic Data:")
    print(f"  - Total samples: {n_samples:,}")
    print(f"  - Fraud ratio: {y_true.mean():.2%}")
    print(f"  - Prob range: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")

    # 1. Compute curves
    print("\n" + "-" * 70)
    fpr, tpr, thresholds_roc, auc = compute_roc_curve(y_true, y_pred_prob)
    prec, rec, thresholds_pr, ap = compute_precision_recall_curve(y_true, y_pred_prob)

    print(f"\nModel Performance:")
    print(f"  - ROC-AUC: {auc:.4f}")
    print(f"  - Average Precision: {ap:.4f}")

    # 2. Recommend threshold
    print("\n" + "-" * 70)
    result = recommend_threshold(y_true, y_pred_prob, strategy='balanced')
    print(f"\nRecommended Threshold: {result['recommended_threshold']:.4f}")
    if result['metrics']:
        print(f"  - Recall: {result['metrics'].recall:.2%}")
        print(f"  - Precision: {result['metrics'].precision:.2%}")
        print(f"  - FPR: {result['metrics'].fpr:.2%}")

    # 3. Print summary table
    print_summary_table(y_true, y_pred_prob)

    # 4. Demo FraudThresholdClassifier
    print("\n" + "-" * 70)
    print("\nFraudThresholdClassifier Demo:")

    classifier = FraudThresholdClassifier()

    # Single prediction
    test_prob = 0.25
    result = classifier.classify_single(test_prob)
    print(f"\nSingle prediction (prob={test_prob}):")
    print(f"  {result}")

    # Batch prediction
    batch_probs = np.array([0.05, 0.12, 0.22, 0.45, 0.8])
    batch_results = classifier.classify_batch(batch_probs)
    print(f"\nBatch predictions:")
    print(batch_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)

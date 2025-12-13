/**
 * API Service - Kết nối với ML Service Backend
 * =============================================
 * Tất cả các API calls đến ML service
 *
 * Các API mới bổ sung:
 * - GET /api/users - Danh sách users
 * - GET /api/users/:id - Chi tiết user + profile
 * - GET /api/users/:id/transactions - Lịch sử giao dịch của user
 * - POST /api/users - Tạo user mới (cho demo)
 * - GET /api/transactions/recent - Giao dịch gần đây của hệ thống
 */

import axios from 'axios';

// Base URL cho ML service
const API_BASE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'http://localhost:5001/api';

// Tạo axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Timeout dài hơn cho các API training (30 phút)
const TRAINING_TIMEOUT = 1800000;

// Request interceptor - Log các request
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Xử lý response
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('[API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============ USER APIs (MỚI) ============

/**
 * Lấy danh sách tất cả users
 * @param {number} limit - Số lượng user tối đa
 * @returns {Promise} - Danh sách users
 */
export const getUsers = async (limit = 100) => {
  return api.get('/users', { params: { limit } });
};

/**
 * Lấy chi tiết user bao gồm profile và behavioral features
 * @param {string} userId - ID của user
 * @returns {Promise} - Thông tin chi tiết user
 */
export const getUserDetail = async (userId) => {
  return api.get(`/users/${userId}`);
};

/**
 * Lấy lịch sử giao dịch của user
 * @param {string} userId - ID của user
 * @param {number} limit - Số giao dịch tối đa
 * @returns {Promise} - Danh sách giao dịch
 */
export const getUserTransactions = async (userId, limit = 10) => {
  return api.get(`/users/${userId}/transactions`, { params: { limit } });
};

/**
 * Tạo user mới (cho demo)
 * @param {object} userData - Dữ liệu user
 * @returns {Promise} - User đã tạo
 */
export const createUser = async (userData) => {
  return api.post('/users', userData);
};

// ============ TRANSACTION APIs (MỚI) ============

/**
 * Lấy giao dịch gần đây của hệ thống
 * @param {number} limit - Số giao dịch tối đa
 * @returns {Promise} - Danh sách giao dịch gần đây
 */
export const getRecentTransactions = async (limit = 50) => {
  return api.get('/transactions/recent', { params: { limit } });
};

// ============ PREDICTION APIs ============

/**
 * Dự đoán fraud cho một giao dịch
 * Đã cập nhật để trả về behavioral_features nếu có user_id
 * @param {object} transaction - Dữ liệu giao dịch
 * @returns {Promise} - Kết quả prediction và behavioral features
 */
export const predictSingle = async (transaction) => {
  return api.post('/predict', transaction);
};

/**
 * Dự đoán fraud cho nhiều giao dịch
 * @param {array} transactions - Danh sách giao dịch
 * @returns {Promise} - Danh sách kết quả prediction
 */
export const predictBatch = async (transactions) => {
  return api.post('/predict/batch', { transactions });
};

// ============ MODEL APIs ============

/**
 * Lấy trạng thái các models
 * @returns {Promise} - Trạng thái models
 */
export const getModelStatus = async () => {
  return api.get('/models/status');
};

/**
 * Train Layer 1
 * @param {string} dataPath - Đường dẫn đến file data (optional)
 * @returns {Promise} - Kết quả training
 */
export const trainLayer1 = async (dataPath = null) => {
  return api.post('/train/layer1', { data_path: dataPath });
};

/**
 * Train Layer 2
 * @param {object} params - Tham số training
 * @returns {Promise} - Kết quả training
 */
export const trainLayer2 = async (params = {}) => {
  return api.post('/train/layer2', params);
};

/**
 * Train tất cả models
 * @param {object} params - Tham số training
 * @returns {Promise} - Kết quả training
 */
export const trainAll = async (params = {}) => {
  return api.post('/train/all', params);
};

// ============ INDIVIDUAL MODEL TRAINING APIs ============

/**
 * Train Isolation Forest model
 * @param {FormData} formData - FormData chứa file CSV (không cần label)
 * @returns {Promise} - Kết quả training
 */
export const trainIsolationForest = async (formData) => {
  return api.post('/train/isolation_forest', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * Train LightGBM model
 * @param {FormData} formData - FormData chứa file CSV (cần có label is_fraud)
 * @returns {Promise} - Kết quả training
 */
export const trainLightGBM = async (formData) => {
  return api.post('/train/lightgbm', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * Train Autoencoder model
 * @param {FormData} formData - FormData chứa file CSV (không cần label)
 * @returns {Promise} - Kết quả training
 */
export const trainAutoencoder = async (formData) => {
  return api.post('/train/autoencoder', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * Train LSTM model
 * @param {FormData} formData - FormData chứa file CSV (cần có label is_fraud + user_id)
 * @returns {Promise} - Kết quả training
 */
export const trainLSTM = async (formData) => {
  return api.post('/train/lstm', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * Train GNN model (LEGACY - dùng cho file CSV đơn giản)
 * @param {FormData} formData - FormData chứa file CSV (cần thông tin nodes và edges)
 * @returns {Promise} - Kết quả training
 */
export const trainGNN = async (formData) => {
  return api.post('/train/gnn', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

// ============ GNN HETEROGENEOUS APIs (MỚI - 2 BƯỚC) ============

/**
 * 🕸️ BƯỚC 1: Tạo mạng lưới GNN
 * Upload các file CSV/JSON hoặc file ZIP chứa dữ liệu GNN
 *
 * Files cần có:
 * - nodes.csv: Tất cả nodes (user, recipient, device, ip)
 * - edges_transfer.csv: Edges chuyển tiền (user → recipient)
 * - edge_labels.csv: Labels cho edges (fraud/normal)
 * - splits.csv: Train/val/test split
 *
 * @param {FormData} formData - FormData chứa các file CSV/JSON hoặc ZIP
 * @returns {Promise} - Kết quả build graph (graph_stats, warnings)
 */
export const buildGNNGraph = async (formData) => {
  return api.post('/train/gnn/build-graph', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * 🎯 BƯỚC 2: Huấn luyện GNN
 * CHỈ chạy khi đã build graph (Bước 1)
 *
 * @returns {Promise} - Kết quả training (metrics, training_info)
 */
export const trainGNNHetero = async () => {
  return api.post('/train/gnn/train', {}, {
    timeout: TRAINING_TIMEOUT,
  });
};

/**
 * Kiểm tra trạng thái GNN graph
 * @returns {Promise} - { graph_ready: boolean, metadata: object }
 */
export const getGNNStatus = async () => {
  return api.get('/train/gnn/status');
};

/**
 * Xóa graph GNN đã build (để build lại với dữ liệu mới)
 * @returns {Promise} - Kết quả xóa
 */
export const clearGNNGraph = async () => {
  return api.post('/train/gnn/clear');
};

/**
 * Train tất cả models với file upload
 * @param {FormData} formData - FormData chứa file CSV
 * @returns {Promise} - Kết quả training
 */
export const trainAllWithData = async (formData) => {
  return api.post('/train/all/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: TRAINING_TIMEOUT,
  });
};

// ============ METRICS APIs ============

/**
 * Lấy metrics đánh giá
 * @returns {Promise} - Metrics của các models
 */
export const getMetrics = async () => {
  return api.get('/metrics');
};

/**
 * Lấy lịch sử metrics
 * @returns {Promise} - Lịch sử metrics
 */
export const getMetricsHistory = async () => {
  return api.get('/metrics/history');
};

// ============ DASHBOARD APIs ============

/**
 * Lấy thống kê dashboard
 * @returns {Promise} - Thống kê dashboard
 */
export const getDashboardStats = async () => {
  return api.get('/dashboard/stats');
};

/**
 * Lấy dữ liệu graph
 * @returns {Promise} - Dữ liệu graph và communities
 */
export const getGraphData = async () => {
  return api.get('/graph/community');
};

// ============ USER PROFILE APIs ============

/**
 * Lấy user profile
 * @param {string} userId - ID của user
 * @returns {Promise} - Profile của user
 */
export const getUserProfile = async (userId) => {
  return api.get(`/user/${userId}/profile`);
};

/**
 * Lấy sequence giao dịch của user
 * @param {string} userId - ID của user
 * @returns {Promise} - Sequence giao dịch
 */
export const getUserSequence = async (userId) => {
  return api.get(`/user/${userId}/sequence`);
};

// ============ EXPLAIN APIs ============

/**
 * Giải thích prediction
 * @param {object} transaction - Dữ liệu giao dịch
 * @returns {Promise} - Giải thích chi tiết
 */
export const explainPrediction = async (transaction) => {
  return api.post('/explain', { transaction });
};

// ============ DATA APIs ============

/**
 * Tạo dữ liệu giả lập
 * @param {number} numUsers - Số users
 * @param {number} numTransactions - Số giao dịch
 * @returns {Promise} - Kết quả tạo dữ liệu
 */
export const generateData = async (numUsers = 1000, numTransactions = 10000) => {
  return api.post('/data/generate', {
    num_users: numUsers,
    num_transactions: numTransactions
  });
};

// ============ HEALTH CHECK ============

/**
 * Kiểm tra trạng thái service
 * @returns {Promise} - Trạng thái service và database
 */
export const healthCheck = async () => {
  return api.get('/health');
};

// ============ UTILITY FUNCTIONS ============

/**
 * Format số tiền VND
 * @param {number} amount - Số tiền
 * @returns {string} - Số tiền đã format
 */
export const formatCurrency = (amount) => {
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency: 'VND'
  }).format(amount);
};

/**
 * Format ngày giờ
 * @param {string} dateString - Chuỗi ngày giờ
 * @returns {string} - Ngày giờ đã format
 */
export const formatDateTime = (dateString) => {
  if (!dateString) return 'N/A';
  const date = new Date(dateString);
  return date.toLocaleString('vi-VN');
};

/**
 * Lấy màu theo risk level
 * @param {string} riskLevel - Mức độ rủi ro
 * @returns {string} - Mã màu
 */
export const getRiskColor = (riskLevel) => {
  const colors = {
    low: '#4caf50',
    medium: '#ff9800',
    high: '#D32F2F',
    critical: '#7B1FA2'
  };
  return colors[riskLevel] || '#666';
};

/**
 * Lấy nhãn tiếng Việt cho risk level
 * @param {string} riskLevel - Mức độ rủi ro
 * @returns {string} - Nhãn tiếng Việt
 */
export const getRiskLabel = (riskLevel) => {
  const labels = {
    low: 'Thấp',
    medium: 'Trung bình',
    high: 'Cao',
    critical: 'Nghiêm trọng'
  };
  return labels[riskLevel] || riskLevel;
};

/**
 * Lấy nhãn tiếng Việt cho transaction type
 * @param {string} type - Loại giao dịch
 * @returns {string} - Nhãn tiếng Việt
 */
export const getTransactionTypeLabel = (type) => {
  const labels = {
    transfer: 'Chuyển khoản',
    payment: 'Thanh toán',
    withdrawal: 'Rút tiền',
    deposit: 'Nạp tiền'
  };
  return labels[type] || type;
};

/**
 * Lấy nhãn tiếng Việt cho income level
 * @param {string} level - Mức thu nhập
 * @returns {string} - Nhãn tiếng Việt
 */
export const getIncomeLevelLabel = (level) => {
  const labels = {
    low: 'Thấp',
    medium: 'Trung bình',
    high: 'Cao',
    very_high: 'Rất cao'
  };
  return labels[level] || level;
};

export default api;

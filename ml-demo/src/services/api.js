/**
 * API Service - Kết nối với ML Service Backend
 * =============================================
 * Tất cả các API calls đến ML service
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

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('[API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============ PREDICTION APIs ============

/**
 * Dự đoán fraud cho một giao dịch
 */
export const predictSingle = async (transaction) => {
  return api.post('/predict', transaction);
};

/**
 * Dự đoán fraud cho nhiều giao dịch
 */
export const predictBatch = async (transactions) => {
  return api.post('/predict/batch', { transactions });
};

// ============ MODEL APIs ============

/**
 * Lấy trạng thái các models
 */
export const getModelStatus = async () => {
  return api.get('/models/status');
};

/**
 * Train Layer 1
 */
export const trainLayer1 = async (dataPath = null) => {
  return api.post('/train/layer1', { data_path: dataPath });
};

/**
 * Train Layer 2
 */
export const trainLayer2 = async (params = {}) => {
  return api.post('/train/layer2', params);
};

/**
 * Train tất cả models
 */
export const trainAll = async (params = {}) => {
  return api.post('/train/all', params);
};

// ============ METRICS APIs ============

/**
 * Lấy metrics đánh giá
 */
export const getMetrics = async () => {
  return api.get('/metrics');
};

/**
 * Lấy lịch sử metrics
 */
export const getMetricsHistory = async () => {
  return api.get('/metrics/history');
};

// ============ DASHBOARD APIs ============

/**
 * Lấy thống kê dashboard
 */
export const getDashboardStats = async () => {
  return api.get('/dashboard/stats');
};

/**
 * Lấy dữ liệu graph
 */
export const getGraphData = async () => {
  return api.get('/graph/community');
};

// ============ USER APIs ============

/**
 * Lấy user profile
 */
export const getUserProfile = async (userId) => {
  return api.get(`/user/${userId}/profile`);
};

/**
 * Lấy sequence giao dịch của user
 */
export const getUserSequence = async (userId) => {
  return api.get(`/user/${userId}/sequence`);
};

// ============ EXPLAIN APIs ============

/**
 * Giải thích prediction
 */
export const explainPrediction = async (transaction) => {
  return api.post('/explain', { transaction });
};

// ============ DATA APIs ============

/**
 * Tạo dữ liệu giả lập
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
 */
export const healthCheck = async () => {
  return api.get('/health');
};

export default api;

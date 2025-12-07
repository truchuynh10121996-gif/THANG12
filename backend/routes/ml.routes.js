/**
 * ML Routes - Proxy endpoints cho ML Service
 * ===========================================
 * Các endpoints này proxy requests đến ML Service
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

// URL của ML Service
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001/api';

// Tạo axios instance với timeout
const mlApi = axios.create({
  baseURL: ML_SERVICE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// ============ PREDICTION ENDPOINTS ============

/**
 * POST /api/ml/predict
 * Dự đoán fraud cho một giao dịch
 */
router.post('/predict', async (req, res) => {
  try {
    const response = await mlApi.post('/predict', req.body);
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Predict error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

/**
 * POST /api/ml/predict/batch
 * Dự đoán fraud cho nhiều giao dịch
 */
router.post('/predict/batch', async (req, res) => {
  try {
    const response = await mlApi.post('/predict/batch', req.body);
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Batch predict error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

// ============ MODEL ENDPOINTS ============

/**
 * GET /api/ml/status
 * Lấy trạng thái các models
 */
router.get('/status', async (req, res) => {
  try {
    const response = await mlApi.get('/models/status');
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Status error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

/**
 * POST /api/ml/train
 * Train tất cả models
 */
router.post('/train', async (req, res) => {
  try {
    const response = await mlApi.post('/train/all', req.body);
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Train error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

// ============ METRICS ENDPOINTS ============

/**
 * GET /api/ml/metrics
 * Lấy metrics đánh giá
 */
router.get('/metrics', async (req, res) => {
  try {
    const response = await mlApi.get('/metrics');
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Metrics error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

// ============ DASHBOARD ENDPOINTS ============

/**
 * GET /api/ml/dashboard
 * Lấy thống kê dashboard
 */
router.get('/dashboard', async (req, res) => {
  try {
    const response = await mlApi.get('/dashboard/stats');
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Dashboard error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

// ============ EXPLAIN ENDPOINT ============

/**
 * POST /api/ml/explain
 * Giải thích prediction
 */
router.post('/explain', async (req, res) => {
  try {
    const response = await mlApi.post('/explain', req.body);
    res.json(response.data);
  } catch (error) {
    console.error('[ML Proxy] Explain error:', error.message);
    res.status(error.response?.status || 500).json({
      success: false,
      error: 'ML Service không khả dụng',
      message: error.message
    });
  }
});

// ============ HEALTH CHECK ============

/**
 * GET /api/ml/health
 * Kiểm tra ML Service
 */
router.get('/health', async (req, res) => {
  try {
    const response = await mlApi.get('/health');
    res.json(response.data);
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      service: 'ML Fraud Detection',
      error: 'Không thể kết nối đến ML Service'
    });
  }
});

module.exports = router;

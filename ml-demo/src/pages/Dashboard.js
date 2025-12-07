/**
 * Dashboard Page - Trang tổng quan
 * ==================================
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  TrendingUp as TrendingIcon,
} from '@mui/icons-material';

// Components
import StatCard from '../components/Cards/StatCard';
import FraudTrendChart from '../components/Charts/FraudTrendChart';
import RiskDistribution from '../components/Charts/RiskDistribution';
import ModelPerformance from '../components/Charts/ModelPerformance';

// API
import { getDashboardStats, getModelStatus } from '../services/api';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsRes, modelRes] = await Promise.all([
          getDashboardStats(),
          getModelStatus(),
        ]);

        setStats(statsRes.stats);
        setModelStatus(modelRes.status);
        setError(null);
      } catch (err) {
        setError('Không thể kết nối đến ML Service. Đang hiển thị dữ liệu mẫu.');
        // Dữ liệu mẫu khi không kết nối được
        setStats({
          total_transactions: 500000,
          total_fraud: 25000,
          fraud_rate: 0.05,
          total_predictions: 10000,
          fraud_detected: 500,
          high_risk_alerts: 50,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '50vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Title */}
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Tổng quan Hệ thống
      </Typography>

      {/* Error alert */}
      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Stats cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Tổng giao dịch"
            value={stats?.total_transactions?.toLocaleString() || '0'}
            subtitle="Trong hệ thống"
            icon={<SecurityIcon />}
            color="primary"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Fraud phát hiện"
            value={stats?.fraud_detected?.toLocaleString() || '0'}
            subtitle={`Tỷ lệ: ${((stats?.fraud_rate || 0) * 100).toFixed(1)}%`}
            icon={<WarningIcon />}
            color="error"
            trend="up"
            trendValue="+12% so với tuần trước"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Cảnh báo cao"
            value={stats?.high_risk_alerts?.toLocaleString() || '0'}
            subtitle="Cần xem xét"
            icon={<TrendingIcon />}
            color="warning"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Độ chính xác"
            value="93%"
            subtitle="ROC-AUC của Ensemble"
            icon={<CheckIcon />}
            color="success"
            showProgress
            progressValue={93}
          />
        </Grid>
      </Grid>

      {/* Charts row 1 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <FraudTrendChart />
        </Grid>
        <Grid item xs={12} md={4}>
          <RiskDistribution />
        </Grid>
      </Grid>

      {/* Charts row 2 */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <ModelPerformance />
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;

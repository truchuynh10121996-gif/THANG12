/**
 * RealtimeMonitor Page - Giám sát real-time
 * ==========================================
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Alert,
} from '@mui/material';
import AlertCard from '../components/Cards/AlertCard';
import StatCard from '../components/Cards/StatCard';
import {
  Visibility as MonitorIcon,
  Warning as WarningIcon,
  Block as BlockIcon,
} from '@mui/icons-material';

// Dữ liệu mẫu cho real-time alerts
const sampleAlerts = [
  {
    id: 1,
    title: 'Giao dịch số tiền lớn bất thường',
    description: 'Chuyển khoản 500 triệu lúc 3h sáng',
    riskLevel: 'critical',
    timestamp: '2 phút trước',
    transactionId: 'TXN_001',
    amount: 500000000,
    userId: 'USR_123',
  },
  {
    id: 2,
    title: 'Nhiều giao dịch liên tiếp',
    description: '5 giao dịch trong 10 phút',
    riskLevel: 'high',
    timestamp: '5 phút trước',
    transactionId: 'TXN_002',
    amount: 50000000,
    userId: 'USR_456',
  },
  {
    id: 3,
    title: 'Thiết bị mới đăng nhập',
    description: 'Giao dịch từ thiết bị chưa đăng ký',
    riskLevel: 'medium',
    timestamp: '8 phút trước',
    transactionId: 'TXN_003',
    amount: 10000000,
    userId: 'USR_789',
  },
  {
    id: 4,
    title: 'Giao dịch quốc tế',
    description: 'Chuyển tiền sang tài khoản nước ngoài',
    riskLevel: 'high',
    timestamp: '12 phút trước',
    transactionId: 'TXN_004',
    amount: 100000000,
    userId: 'USR_101',
  },
];

function RealtimeMonitor() {
  const [alerts, setAlerts] = useState(sampleAlerts);
  const [expandedAlert, setExpandedAlert] = useState(null);
  const [stats, setStats] = useState({
    monitoring: 1234,
    flagged: 23,
    blocked: 5,
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        monitoring: prev.monitoring + Math.floor(Math.random() * 5),
        flagged: prev.flagged + (Math.random() > 0.7 ? 1 : 0),
        blocked: prev.blocked + (Math.random() > 0.9 ? 1 : 0),
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Giám sát Real-time
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Đang mô phỏng real-time monitoring. Trong production, sẽ kết nối WebSocket với ML Service.
      </Alert>

      {/* Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <StatCard
            title="Đang theo dõi"
            value={stats.monitoring.toLocaleString()}
            subtitle="Giao dịch trong 24h"
            icon={<MonitorIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard
            title="Cảnh báo"
            value={stats.flagged}
            subtitle="Cần xem xét"
            icon={<WarningIcon />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard
            title="Đã chặn"
            value={stats.blocked}
            subtitle="Hôm nay"
            icon={<BlockIcon />}
            color="error"
          />
        </Grid>
      </Grid>

      {/* Alerts */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Cảnh báo gần đây
          </Typography>

          {alerts.map((alert) => (
            <AlertCard
              key={alert.id}
              title={alert.title}
              description={alert.description}
              riskLevel={alert.riskLevel}
              timestamp={alert.timestamp}
              transactionId={alert.transactionId}
              amount={alert.amount}
              userId={alert.userId}
              expanded={expandedAlert === alert.id}
              onToggle={() =>
                setExpandedAlert(expandedAlert === alert.id ? null : alert.id)
              }
            />
          ))}
        </CardContent>
      </Card>
    </Box>
  );
}

export default RealtimeMonitor;

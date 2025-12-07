/**
 * Reports Page - Báo cáo
 * =======================
 */

import React from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Alert,
} from '@mui/material';
import ModelPerformance from '../components/Charts/ModelPerformance';
import FraudTrendChart from '../components/Charts/FraudTrendChart';
import RiskDistribution from '../components/Charts/RiskDistribution';

function Reports() {
  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Báo cáo & Phân tích
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Báo cáo chi tiết về hiệu suất hệ thống và xu hướng fraud.
        Trong production, dữ liệu sẽ được cập nhật real-time từ ML Service.
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <ModelPerformance title="So sánh hiệu suất các Models" />
        </Grid>

        <Grid item xs={12} md={8}>
          <FraudTrendChart title="Xu hướng Fraud 7 ngày gần nhất" />
        </Grid>

        <Grid item xs={12} md={4}>
          <RiskDistribution title="Phân bố mức độ rủi ro" />
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Tóm tắt
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Model tốt nhất
                  </Typography>
                  <Typography variant="h6">Ensemble (F1: 81%)</Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Tỷ lệ phát hiện fraud
                  </Typography>
                  <Typography variant="h6">93% (ROC-AUC)</Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    False Positive Rate
                  </Typography>
                  <Typography variant="h6">15%</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Reports;

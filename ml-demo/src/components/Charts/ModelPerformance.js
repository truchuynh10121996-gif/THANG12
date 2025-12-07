/**
 * ModelPerformance - Biểu đồ hiệu suất các models
 * ================================================
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { chartColors } from '../../styles/theme';

// Dữ liệu mẫu
const sampleData = [
  {
    name: 'Isolation Forest',
    precision: 0.75,
    recall: 0.80,
    f1: 0.77,
    auc: 0.88,
  },
  {
    name: 'LightGBM',
    precision: 0.85,
    recall: 0.78,
    f1: 0.81,
    auc: 0.92,
  },
  {
    name: 'Autoencoder',
    precision: 0.72,
    recall: 0.82,
    f1: 0.77,
    auc: 0.85,
  },
  {
    name: 'LSTM',
    precision: 0.80,
    recall: 0.75,
    f1: 0.77,
    auc: 0.87,
  },
  {
    name: 'Ensemble',
    precision: 0.83,
    recall: 0.80,
    f1: 0.81,
    auc: 0.93,
  },
];

function ModelPerformance({ data = sampleData, title = 'Hiệu suất các Models' }) {
  // Chuyển đổi dữ liệu thành phần trăm
  const chartData = data.map((item) => ({
    ...item,
    precision: Math.round(item.precision * 100),
    recall: Math.round(item.recall * 100),
    f1: Math.round(item.f1 * 100),
    auc: Math.round(item.auc * 100),
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
          {title}
        </Typography>

        <Box sx={{ width: '100%', height: 350 }}>
          <ResponsiveContainer>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis type="number" domain={[0, 100]} stroke="#666" fontSize={12} />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#666"
                fontSize={12}
                width={100}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e0e0e0',
                  borderRadius: 8,
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                }}
                formatter={(value) => `${value}%`}
              />
              <Legend />

              <Bar
                dataKey="precision"
                name="Precision"
                fill={chartColors.primary}
                radius={[0, 4, 4, 0]}
              />
              <Bar
                dataKey="recall"
                name="Recall"
                fill={chartColors.tertiary}
                radius={[0, 4, 4, 0]}
              />
              <Bar
                dataKey="f1"
                name="F1-Score"
                fill={chartColors.quaternary}
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
}

export default ModelPerformance;

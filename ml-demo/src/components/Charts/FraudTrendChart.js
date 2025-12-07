/**
 * FraudTrendChart - Biểu đồ xu hướng fraud theo thời gian
 * ========================================================
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import {
  LineChart,
  Line,
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
  { date: '01/01', total: 1200, fraud: 45, detected: 42 },
  { date: '02/01', total: 1350, fraud: 52, detected: 48 },
  { date: '03/01', total: 1100, fraud: 38, detected: 36 },
  { date: '04/01', total: 1420, fraud: 58, detected: 55 },
  { date: '05/01', total: 1380, fraud: 48, detected: 46 },
  { date: '06/01', total: 1550, fraud: 62, detected: 60 },
  { date: '07/01', total: 1480, fraud: 55, detected: 53 },
];

function FraudTrendChart({ data = sampleData, title = 'Xu hướng Fraud theo thời gian' }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
          {title}
        </Typography>

        <Box sx={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="date" stroke="#666" fontSize={12} />
              <YAxis stroke="#666" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e0e0e0',
                  borderRadius: 8,
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                }}
                labelStyle={{ fontWeight: 500 }}
              />
              <Legend />

              <Line
                type="monotone"
                dataKey="total"
                name="Tổng giao dịch"
                stroke={chartColors.primary}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="fraud"
                name="Fraud thực tế"
                stroke={chartColors.secondary}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="detected"
                name="Đã phát hiện"
                stroke={chartColors.tertiary}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
}

export default FraudTrendChart;

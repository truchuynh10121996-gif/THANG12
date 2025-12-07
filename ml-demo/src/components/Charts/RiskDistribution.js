/**
 * RiskDistribution - Biểu đồ phân bố mức độ rủi ro
 * =================================================
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import { riskColors } from '../../styles/theme';

// Dữ liệu mẫu
const sampleData = [
  { name: 'Rủi ro thấp', value: 75, color: riskColors.low },
  { name: 'Rủi ro trung bình', value: 15, color: riskColors.medium },
  { name: 'Rủi ro cao', value: 7, color: riskColors.high },
  { name: 'Nghiêm trọng', value: 3, color: riskColors.critical },
];

function RiskDistribution({ data = sampleData, title = 'Phân bố mức độ rủi ro' }) {
  const total = data.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <Box
          sx={{
            backgroundColor: '#fff',
            border: '1px solid #e0e0e0',
            borderRadius: 1,
            p: 1.5,
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {item.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {item.value}% ({Math.round((item.value / 100) * total).toLocaleString()} giao dịch)
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
          {title}
        </Typography>

        <Box sx={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value, entry) => (
                  <span style={{ color: '#333', fontSize: '0.85rem' }}>{value}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </Box>

        {/* Legend chi tiết */}
        <Box sx={{ mt: 2 }}>
          {data.map((item) => (
            <Box
              key={item.name}
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 0.5,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: item.color,
                  }}
                />
                <Typography variant="body2">{item.name}</Typography>
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {item.value}%
              </Typography>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
}

export default RiskDistribution;

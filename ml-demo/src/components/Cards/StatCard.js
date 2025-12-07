/**
 * StatCard Component - Card hiển thị thống kê
 * ============================================
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
} from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';

function StatCard({
  title,
  value,
  subtitle,
  icon,
  color = 'primary',
  trend,
  trendValue,
  showProgress,
  progressValue,
}) {
  // Màu cho từng loại
  const colorMap = {
    primary: '#1976d2',
    secondary: '#dc004e',
    success: '#4caf50',
    warning: '#ff9800',
    error: '#f44336',
    info: '#2196f3',
  };

  const bgColorMap = {
    primary: 'rgba(25, 118, 210, 0.1)',
    secondary: 'rgba(220, 0, 78, 0.1)',
    success: 'rgba(76, 175, 80, 0.1)',
    warning: 'rgba(255, 152, 0, 0.1)',
    error: 'rgba(244, 67, 54, 0.1)',
    info: 'rgba(33, 150, 243, 0.1)',
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Box>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 0.5, fontWeight: 500 }}
            >
              {title}
            </Typography>
            <Typography
              variant="h4"
              sx={{ fontWeight: 600, color: colorMap[color] }}
            >
              {value}
            </Typography>
          </Box>

          {icon && (
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: bgColorMap[color],
                color: colorMap[color],
              }}
            >
              {icon}
            </Box>
          )}
        </Box>

        {/* Trend indicator */}
        {trend && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
            {trend === 'up' ? (
              <TrendingUp sx={{ fontSize: 18, color: '#4caf50' }} />
            ) : (
              <TrendingDown sx={{ fontSize: 18, color: '#f44336' }} />
            )}
            <Typography
              variant="body2"
              sx={{ color: trend === 'up' ? '#4caf50' : '#f44336' }}
            >
              {trendValue}
            </Typography>
          </Box>
        )}

        {/* Subtitle */}
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}

        {/* Progress bar */}
        {showProgress && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress
              variant="determinate"
              value={progressValue}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: bgColorMap[color],
                '& .MuiLinearProgress-bar': {
                  backgroundColor: colorMap[color],
                  borderRadius: 3,
                },
              }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default StatCard;

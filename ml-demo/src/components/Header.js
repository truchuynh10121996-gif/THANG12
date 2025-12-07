/**
 * Header Component - Thanh header trên cùng
 * ==========================================
 * Đồng bộ style với web-admin và web-app
 *
 * Màu sắc:
 * - Gradient header: linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)
 * - Box shadow: 0 4px 12px rgba(0,0,0,0.1)
 */

import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Box,
  Chip,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  CheckCircle as HealthyIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';
import { healthCheck } from '../services/api';
import { gradients } from '../styles/theme';

function Header({ drawerWidth, onDrawerToggle }) {
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [lastChecked, setLastChecked] = useState(null);

  // Kiểm tra trạng thái service
  const checkServiceHealth = async () => {
    try {
      setServiceStatus('checking');
      const response = await healthCheck();
      if (response.status === 'healthy') {
        setServiceStatus('healthy');
      } else {
        setServiceStatus('error');
      }
      setLastChecked(new Date());
    } catch (error) {
      setServiceStatus('error');
      setLastChecked(new Date());
    }
  };

  useEffect(() => {
    checkServiceHealth();
    // Kiểm tra mỗi 30 giây
    const interval = setInterval(checkServiceHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <AppBar
      position="fixed"
      sx={{
        width: { sm: `calc(100% - ${drawerWidth}px)` },
        ml: { sm: `${drawerWidth}px` },
        // Gradient header theo yêu cầu
        background: gradients.header,
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
      }}
    >
      <Toolbar>
        {/* Menu button for mobile */}
        <IconButton
          color="inherit"
          aria-label="Mở menu"
          edge="start"
          onClick={onDrawerToggle}
          sx={{ mr: 2, display: { sm: 'none' } }}
        >
          <MenuIcon />
        </IconButton>

        {/* Icon và Title */}
        <SecurityIcon sx={{ mr: 1.5, fontSize: 28 }} />
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{
            flexGrow: 1,
            fontWeight: 700,
            letterSpacing: '0.5px'
          }}
        >
          AGRIBANK DIGITAL GUARD - ML Fraud Detection
        </Typography>

        {/* Service status và các nút chức năng */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Chip hiển thị trạng thái service */}
          <Tooltip title={`Kiểm tra lần cuối: ${lastChecked?.toLocaleTimeString() || 'Chưa kiểm tra'}`}>
            <Chip
              icon={serviceStatus === 'healthy' ? <HealthyIcon /> : <ErrorIcon />}
              label={
                serviceStatus === 'healthy' ? 'ML Service: Hoạt động' :
                serviceStatus === 'checking' ? 'Đang kiểm tra...' : 'ML Service: Ngắt kết nối'
              }
              sx={{
                // Style chip theo trạng thái
                backgroundColor: serviceStatus === 'healthy' ? 'rgba(76, 175, 80, 0.3)' :
                                 serviceStatus === 'checking' ? 'rgba(255, 255, 255, 0.2)' : 'rgba(244, 67, 54, 0.3)',
                color: '#fff',
                fontWeight: 600,
                border: '1px solid rgba(255,255,255,0.3)',
                '& .MuiChip-icon': {
                  color: '#fff'
                }
              }}
              size="small"
            />
          </Tooltip>

          {/* Nút refresh trạng thái */}
          <Tooltip title="Kiểm tra lại kết nối">
            <IconButton
              onClick={checkServiceHealth}
              sx={{ color: '#fff' }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          {/* Nút thông báo */}
          <Tooltip title="Thông báo">
            <IconButton sx={{ color: '#fff' }}>
              <NotificationsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;

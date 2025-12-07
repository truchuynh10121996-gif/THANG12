/**
 * Header Component - Thanh header trên cùng
 * ==========================================
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
} from '@mui/icons-material';
import { healthCheck } from '../services/api';

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
        backgroundColor: '#fff',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      }}
    >
      <Toolbar>
        {/* Menu button for mobile */}
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={onDrawerToggle}
          sx={{ mr: 2, display: { sm: 'none' }, color: 'text.primary' }}
        >
          <MenuIcon />
        </IconButton>

        {/* Title */}
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{ flexGrow: 1, color: 'text.primary', fontWeight: 500 }}
        >
          ML Fraud Detection Dashboard
        </Typography>

        {/* Service status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title={`Kiểm tra lần cuối: ${lastChecked?.toLocaleTimeString() || 'N/A'}`}>
            <Chip
              icon={serviceStatus === 'healthy' ? <HealthyIcon /> : <ErrorIcon />}
              label={serviceStatus === 'healthy' ? 'ML Service: Online' :
                     serviceStatus === 'checking' ? 'Đang kiểm tra...' : 'ML Service: Offline'}
              color={serviceStatus === 'healthy' ? 'success' :
                     serviceStatus === 'checking' ? 'default' : 'error'}
              size="small"
              variant="outlined"
            />
          </Tooltip>

          <Tooltip title="Kiểm tra lại">
            <IconButton
              onClick={checkServiceHealth}
              sx={{ color: 'text.secondary' }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Thông báo">
            <IconButton sx={{ color: 'text.secondary' }}>
              <NotificationsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;

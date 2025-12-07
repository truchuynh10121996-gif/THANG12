/**
 * Sidebar Component - Menu điều hướng
 * =====================================
 */

import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  MonitorHeart as MonitorIcon,
  Science as TestIcon,
  BatchPrediction as BatchIcon,
  ModelTraining as TrainingIcon,
  Storage as DataIcon,
  Assessment as ReportsIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';

// Menu items
const menuItems = [
  {
    text: 'Tổng quan',
    icon: <DashboardIcon />,
    path: '/dashboard',
    description: 'Dashboard metrics'
  },
  {
    text: 'Giám sát Real-time',
    icon: <MonitorIcon />,
    path: '/realtime',
    description: 'Theo dõi trực tiếp'
  },
  {
    text: 'Test Giao dịch',
    icon: <TestIcon />,
    path: '/test',
    description: 'Test giao dịch đơn lẻ'
  },
  {
    text: 'Phân tích Batch',
    icon: <BatchIcon />,
    path: '/batch',
    description: 'Phân tích nhiều giao dịch'
  },
  {
    text: 'Training Models',
    icon: <TrainingIcon />,
    path: '/training',
    description: 'Train và đánh giá models'
  },
  {
    text: 'Dữ liệu',
    icon: <DataIcon />,
    path: '/data',
    description: 'Khám phá dữ liệu'
  },
  {
    text: 'Báo cáo',
    icon: <ReportsIcon />,
    path: '/reports',
    description: 'Báo cáo chi tiết'
  },
];

function Sidebar({ drawerWidth, mobileOpen, onDrawerToggle }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const location = useLocation();
  const navigate = useNavigate();

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          borderBottom: '1px solid rgba(255,255,255,0.1)'
        }}
      >
        <SecurityIcon sx={{ fontSize: 32, color: '#4caf50' }} />
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700, color: '#fff' }}>
            ML Fraud Detection
          </Typography>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            Agribank Digital Guard
          </Typography>
        </Box>
      </Box>

      {/* Menu items */}
      <List sx={{ flexGrow: 1, py: 2 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path ||
                          (item.path === '/dashboard' && location.pathname === '/');

          return (
            <ListItem key={item.text} disablePadding sx={{ px: 1, mb: 0.5 }}>
              <ListItemButton
                onClick={() => {
                  navigate(item.path);
                  if (isMobile) onDrawerToggle();
                }}
                sx={{
                  borderRadius: 2,
                  backgroundColor: isActive ? 'rgba(25, 118, 210, 0.2)' : 'transparent',
                  '&:hover': {
                    backgroundColor: isActive
                      ? 'rgba(25, 118, 210, 0.3)'
                      : 'rgba(255,255,255,0.08)',
                  },
                  color: isActive ? '#64b5f6' : 'rgba(255,255,255,0.7)',
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isActive ? '#64b5f6' : 'rgba(255,255,255,0.5)',
                    minWidth: 40
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  primaryTypographyProps={{
                    fontSize: '0.9rem',
                    fontWeight: isActive ? 600 : 400,
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {/* Footer */}
      <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.4)' }}>
          Version 1.0.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            backgroundColor: '#1a1a2e',
          },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', sm: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            backgroundColor: '#1a1a2e',
            borderRight: 'none',
          },
        }}
        open
      >
        {drawerContent}
      </Drawer>
    </Box>
  );
}

export default Sidebar;

/**
 * Sidebar Component - Menu điều hướng
 * =====================================
 * Đồng bộ style với web-admin và web-app
 *
 * Màu sắc:
 * - Gradient sidebar: linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)
 * - Màu chủ đạo: #FF8DAD (hồng), #FF6B99 (hồng đậm)
 *
 * Nhãn tiếng Việt:
 * - Dashboard → Bảng điều khiển
 * - Transaction Test → Kiểm tra giao dịch
 * - Batch Analysis → Phân tích hàng loạt
 * - Real-time Monitor → Giám sát thời gian thực
 * - Model Training → Huấn luyện mô hình
 * - Data Explorer → Khám phá dữ liệu
 * - Reports → Báo cáo
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
import { gradients } from '../styles/theme';

// Menu items - Nhãn tiếng Việt theo yêu cầu
const menuItems = [
  {
    text: 'Bảng điều khiển',      // Dashboard → Bảng điều khiển
    icon: <DashboardIcon />,
    path: '/dashboard',
    description: 'Tổng quan hệ thống'
  },
  {
    text: 'Giám sát thời gian thực',  // Real-time Monitor → Giám sát thời gian thực
    icon: <MonitorIcon />,
    path: '/realtime',
    description: 'Theo dõi giao dịch trực tiếp'
  },
  {
    text: 'Kiểm tra giao dịch',       // Transaction Test → Kiểm tra giao dịch
    icon: <TestIcon />,
    path: '/test',
    description: 'Test giao dịch đơn lẻ'
  },
  {
    text: 'Phân tích hàng loạt',      // Batch Analysis → Phân tích hàng loạt
    icon: <BatchIcon />,
    path: '/batch',
    description: 'Phân tích nhiều giao dịch'
  },
  {
    text: 'Huấn luyện mô hình',       // Model Training → Huấn luyện mô hình
    icon: <TrainingIcon />,
    path: '/training',
    description: 'Train và đánh giá models'
  },
  {
    text: 'Khám phá dữ liệu',         // Data Explorer → Khám phá dữ liệu
    icon: <DataIcon />,
    path: '/data',
    description: 'Xem và phân tích dữ liệu'
  },
  {
    text: 'Báo cáo',                  // Reports → Báo cáo
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

  // Nội dung sidebar
  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo và tiêu đề */}
      <Box
        sx={{
          p: 3,
          textAlign: 'center',
          background: gradients.sidebar,
          borderBottom: '1px solid rgba(255, 107, 153, 0.2)'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5 }}>
          <SecurityIcon sx={{ fontSize: 36, color: '#FF6B99' }} />
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#FF6B99' }}>
              ML Fraud Detection
            </Typography>
            <Typography variant="caption" sx={{ color: '#FF8DAD', fontWeight: 500 }}>
              Agribank Digital Guard
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Menu items */}
      <List sx={{ flexGrow: 1, py: 2, px: 1 }}>
        {menuItems.map((item) => {
          // Kiểm tra trang hiện tại
          const isActive = location.pathname === item.path ||
                          (item.path === '/dashboard' && location.pathname === '/');

          return (
            <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => {
                  navigate(item.path);
                  if (isMobile) onDrawerToggle();
                }}
                sx={{
                  borderRadius: 2,
                  mx: 1,
                  // Style khi được chọn - sử dụng gradient hồng
                  ...(isActive && {
                    background: gradients.cardPink,
                    color: '#fff',
                    '& .MuiListItemIcon-root': {
                      color: '#fff'
                    },
                    '&:hover': {
                      background: gradients.cardPink,
                    }
                  }),
                  // Style khi không được chọn
                  ...(!isActive && {
                    color: '#555',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 141, 173, 0.15)',
                    },
                    '& .MuiListItemIcon-root': {
                      color: '#FF8DAD'
                    }
                  }),
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 40
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  secondary={!isActive ? item.description : null}
                  primaryTypographyProps={{
                    fontSize: '0.9rem',
                    fontWeight: isActive ? 700 : 600,
                  }}
                  secondaryTypographyProps={{
                    fontSize: '0.7rem',
                    sx: { color: 'rgba(0,0,0,0.5)' }
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {/* Footer */}
      <Divider sx={{ borderColor: 'rgba(255, 107, 153, 0.2)' }} />
      <Box sx={{ p: 2, textAlign: 'center', background: 'rgba(255,255,255,0.5)' }}>
        <Typography variant="caption" sx={{ color: '#666', display: 'block' }}>
          Version 2.0.0
        </Typography>
        <Typography variant="caption" sx={{ color: '#FF6B99' }}>
          © 2024 Agribank
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
    >
      {/* Mobile drawer - Hiển thị trên mobile */}
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
            background: gradients.sidebar,
          },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Desktop drawer - Hiển thị trên desktop */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', sm: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            background: gradients.sidebar,
            borderRight: '2px solid rgba(255, 107, 153, 0.1)',
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

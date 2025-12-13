import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Container
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  QuestionAnswer as QAIcon,
  Chat as ChatIcon,
  Security as SecurityIcon,
  Psychology as TrainIcon
} from '@mui/icons-material';

const drawerWidth = 280;

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Quản lý Q&A', icon: <QAIcon />, path: '/qa-management' },
  { text: 'Huấn luyện mô hình', icon: <TrainIcon />, path: '/model-training' },
  { text: 'Xem trước Chatbot', icon: <ChatIcon />, path: '/chatbot-preview' }
];

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Box sx={{ display: 'flex' }}>
      {/* AppBar */}
      <AppBar
        position="fixed"
        sx={{
          width: `calc(100% - ${drawerWidth}px)`,
          ml: `${drawerWidth}px`,
          background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}
      >
        <Toolbar>
          <SecurityIcon sx={{ mr: 2 }} />
          <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
            AGRIBANK DIGITAL GUARD - Admin Dashboard
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            background: 'rgba(255, 255, 255, 0.95)',
            borderRight: '2px solid rgba(46, 125, 50, 0.1)'
          }
        }}
        variant="permanent"
        anchor="left"
      >
        <Box
          sx={{
            p: 3,
            textAlign: 'center',
            background: 'linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)'
          }}
        >
          <Typography variant="h5" sx={{ fontWeight: 700, color: '#FF8DAD' }}>
            🛡️ Digital Guard
          </Typography>
          <Typography variant="body2" sx={{ color: '#FF6B99', mt: 0.5 }}>
            Admin Panel
          </Typography>
        </Box>

        <Divider />

        <List sx={{ p: 2 }}>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
              <ListItemButton
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
                    color: '#fff',
                    '& .MuiListItemIcon-root': {
                      color: '#fff'
                    },
                    '&:hover': {
                      background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)'
                    }
                  }
                }}
              >
                <ListItemIcon sx={{ color: '#FF8DAD' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  primaryTypographyProps={{ fontWeight: 600 }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        <Box sx={{ flexGrow: 1 }} />

        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" sx={{ color: '#666' }}>
            Version 1.0.0
          </Typography>
          <Typography variant="caption" display="block" sx={{ color: '#666' }}>
            © 2024 Agribank
          </Typography>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: `calc(100% - ${drawerWidth}px)`,
          minHeight: '100vh'
        }}
      >
        <Toolbar />
        <Container maxWidth="xl" sx={{ mt: 2 }}>
          {children}
        </Container>
      </Box>
    </Box>
  );
}

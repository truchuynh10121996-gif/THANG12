/**
 * Theme Configuration - Cấu hình màu sắc đồng bộ với dự án
 * =========================================================
 * Sử dụng màu sắc giống web-app và web-admin
 */

import { createTheme } from '@mui/material/styles';

// Màu sắc chính của dự án
const colors = {
  primary: {
    main: '#1976d2',      // Xanh dương - màu chủ đạo
    light: '#42a5f5',
    dark: '#1565c0',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#dc004e',      // Hồng đậm
    light: '#ff5c8d',
    dark: '#9a0036',
    contrastText: '#ffffff',
  },
  success: {
    main: '#4caf50',      // Xanh lá - an toàn
    light: '#80e27e',
    dark: '#087f23',
    contrastText: '#ffffff',
  },
  warning: {
    main: '#ff9800',      // Cam - cảnh báo
    light: '#ffc947',
    dark: '#c66900',
    contrastText: '#000000',
  },
  error: {
    main: '#f44336',      // Đỏ - nguy hiểm/fraud
    light: '#ff7961',
    dark: '#ba000d',
    contrastText: '#ffffff',
  },
  info: {
    main: '#2196f3',
    light: '#6ec6ff',
    dark: '#0069c0',
    contrastText: '#ffffff',
  },
  background: {
    default: '#f5f5f5',   // Nền chung
    paper: '#ffffff',     // Nền card
  },
  text: {
    primary: '#333333',
    secondary: '#666666',
  },
};

// Risk level colors
export const riskColors = {
  low: '#4caf50',         // Xanh lá
  medium: '#ff9800',      // Cam
  high: '#f44336',        // Đỏ
  critical: '#9c27b0',    // Tím
};

// Chart colors
export const chartColors = {
  primary: '#1976d2',
  secondary: '#dc004e',
  tertiary: '#4caf50',
  quaternary: '#ff9800',
  quinary: '#9c27b0',
  series: [
    '#1976d2', '#dc004e', '#4caf50', '#ff9800',
    '#9c27b0', '#00bcd4', '#795548', '#607d8b'
  ],
};

// Create theme
const theme = createTheme({
  palette: colors,
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#1a1a2e',
          color: '#ffffff',
        },
      },
    },
  },
});

export default theme;

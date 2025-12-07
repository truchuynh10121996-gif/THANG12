/**
 * Theme Configuration - Cấu hình màu sắc đồng bộ với dự án
 * =========================================================
 * Sử dụng màu sắc giống web-app và web-admin
 *
 * Màu chủ đạo: #FF8DAD (hồng), #FF6B99 (hồng đậm)
 * Gradient header: linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)
 * Gradient sidebar: linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)
 * Nền trang: linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)
 */

import { createTheme } from '@mui/material/styles';

// Màu sắc chính của dự án - Đồng bộ với web-admin và web-app
const colors = {
  primary: {
    main: '#FF8DAD',       // Hồng - màu chủ đạo
    light: '#FFAFC5',
    dark: '#FF6B99',       // Hồng đậm
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#FF6B99',       // Hồng đậm
    light: '#FF8DAD',
    dark: '#E5507F',
    contrastText: '#ffffff',
  },
  success: {
    main: '#4caf50',       // Xanh lá - an toàn
    light: '#80e27e',
    dark: '#087f23',
    contrastText: '#ffffff',
  },
  warning: {
    main: '#ff9800',       // Cam - cảnh báo
    light: '#ffc947',
    dark: '#c66900',
    contrastText: '#000000',
  },
  error: {
    main: '#D32F2F',       // Đỏ - nguy hiểm/fraud
    light: '#EF5350',
    dark: '#B71C1C',
    contrastText: '#ffffff',
  },
  info: {
    main: '#1976D2',       // Xanh dương
    light: '#42a5f5',
    dark: '#0D47A1',
    contrastText: '#ffffff',
  },
  background: {
    default: '#FFF5F8',    // Nền hồng nhạt
    paper: '#ffffff',      // Nền card
  },
  text: {
    primary: '#333333',
    secondary: '#666666',
  },
};

// Risk level colors - Màu sắc theo mức độ rủi ro
export const riskColors = {
  low: '#4caf50',          // Xanh lá - An toàn
  medium: '#ff9800',       // Cam - Cảnh báo
  high: '#D32F2F',         // Đỏ - Nguy hiểm
  critical: '#7B1FA2',     // Tím - Nghiêm trọng
};

// Chart colors - Màu sắc cho biểu đồ
export const chartColors = {
  primary: '#FF8DAD',
  secondary: '#FF6B99',
  tertiary: '#4caf50',
  quaternary: '#ff9800',
  quinary: '#7B1FA2',
  series: [
    '#FF8DAD', '#FF6B99', '#4caf50', '#ff9800',
    '#7B1FA2', '#1976D2', '#00bcd4', '#607d8b'
  ],
};

// Gradient colors - Các gradient chính
export const gradients = {
  // Gradient header
  header: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
  // Gradient sidebar
  sidebar: 'linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)',
  // Gradient nền trang
  background: 'linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)',
  // Card gradients
  cardPink: 'linear-gradient(135deg, #FF8DAD 0%, #FF6B99 100%)',
  cardRed: 'linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%)',
  cardBlue: 'linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)',
  cardPurple: 'linear-gradient(135deg, #7B1FA2 0%, #4A148C 100%)',
  cardGreen: 'linear-gradient(135deg, #388E3C 0%, #1B5E20 100%)',
  cardOrange: 'linear-gradient(135deg, #F57C00 0%, #E65100 100%)',
};

// Create theme - Tạo theme MUI
const theme = createTheme({
  palette: colors,
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 3, // Border radius 2-3 cho cards/papers
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)', // Box shadow chuẩn
          borderRadius: 12,
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-5px)', // Hiệu ứng hover
            boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
        contained: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          width: 280, // Drawer width chuẩn
          background: 'linear-gradient(135deg, #FBD6E3 0%, #A9EDE9 100%)',
          borderRight: 'none',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '&:hover fieldset': {
              borderColor: '#FF8DAD',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#FF6B99',
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#FF8DAD',
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#FF6B99',
          },
        },
      },
    },
  },
});

export default theme;

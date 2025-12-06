import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Pages
import Dashboard from './pages/Dashboard';
import QAManagement from './pages/QAManagement';
import ChatbotPreview from './pages/ChatbotPreview';
import Layout from './components/Layout';

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#FF8DAD',
      light: '#FFB3C6',
      dark: '#FF6B99'
    },
    secondary: {
      main: '#FFC9DD',
      light: '#FFE6F0',
      dark: '#FFB3C6'
    },
    background: {
      default: 'transparent',
      paper: 'rgba(255, 255, 255, 0.9)'
    }
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h4: {
      fontWeight: 700
    },
    h5: {
      fontWeight: 600
    }
  },
  shape: {
    borderRadius: 12
  }
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/qa-management" element={<QAManagement />} />
            <Route path="/chatbot-preview" element={<ChatbotPreview />} />
          </Routes>
        </Layout>
      </Router>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#333'
          },
          success: {
            iconTheme: {
              primary: '#FF8DAD',
              secondary: '#fff'
            }
          },
          error: {
            iconTheme: {
              primary: '#d32f2f',
              secondary: '#fff'
            }
          }
        }}
      />
    </ThemeProvider>
  );
}

export default App;

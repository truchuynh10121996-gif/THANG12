/**
 * ML Fraud Detection Demo - Main App
 * ====================================
 * Dashboard demo cho hệ thống phát hiện lừa đảo bằng ML
 *
 * Đồng bộ style với web-admin và web-app:
 * - Nền trang: linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)
 * - Drawer width: 280px
 */

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box, Toolbar } from '@mui/material';

// Layout components
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Pages
import Dashboard from './pages/Dashboard';
import RealtimeMonitor from './pages/RealtimeMonitor';
import TransactionTest from './pages/TransactionTest';
import BatchAnalysis from './pages/BatchAnalysis';
import ModelTraining from './pages/ModelTraining';
import DataExplorer from './pages/DataExplorer';
import Reports from './pages/Reports';

// Theme gradients
import { gradients } from './styles/theme';

// Chiều rộng sidebar - 280px theo chuẩn
const drawerWidth = 280;

function App() {
  // State cho mobile drawer
  const [mobileOpen, setMobileOpen] = useState(false);

  // Toggle mobile drawer
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  return (
    <Router>
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        {/* Header - Gradient header */}
        <Header
          drawerWidth={drawerWidth}
          onDrawerToggle={handleDrawerToggle}
        />

        {/* Sidebar - Gradient sidebar */}
        <Sidebar
          drawerWidth={drawerWidth}
          mobileOpen={mobileOpen}
          onDrawerToggle={handleDrawerToggle}
        />

        {/* Main content - Nền gradient */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            // Nền trang gradient theo yêu cầu
            background: gradients.background,
            minHeight: '100vh',
          }}
        >
          {/* Spacer cho header */}
          <Toolbar />

          {/* Các routes của ứng dụng */}
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/realtime" element={<RealtimeMonitor />} />
            <Route path="/test" element={<TransactionTest />} />
            <Route path="/batch" element={<BatchAnalysis />} />
            <Route path="/training" element={<ModelTraining />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}

export default App;

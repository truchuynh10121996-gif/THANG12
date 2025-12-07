/**
 * DataExplorer Page - Khám phá dữ liệu
 * =====================================
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  TextField,
} from '@mui/material';
import { Dataset as DataIcon, Autorenew as GenerateIcon } from '@mui/icons-material';
import StatCard from '../components/Cards/StatCard';
import { generateData } from '../services/api';

function DataExplorer() {
  const [numUsers, setNumUsers] = useState(1000);
  const [numTransactions, setNumTransactions] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await generateData(numUsers, numTransactions);
      setResult(response.stats);
    } catch (err) {
      setError('Không thể tạo dữ liệu. ML Service có thể chưa khởi động.');
      // Kết quả mẫu
      setResult({
        num_users: numUsers,
        num_transactions: numTransactions,
        num_fraud: Math.floor(numTransactions * 0.05),
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Dữ liệu & Thống kê
      </Typography>

      <Grid container spacing={3}>
        {/* Stats */}
        <Grid item xs={12} md={4}>
          <StatCard
            title="Users"
            value={result?.num_users?.toLocaleString() || '50,000'}
            subtitle="Trong hệ thống"
            icon={<DataIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard
            title="Giao dịch"
            value={result?.num_transactions?.toLocaleString() || '500,000'}
            subtitle="Tổng số"
            icon={<DataIcon />}
            color="info"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <StatCard
            title="Fraud"
            value={result?.num_fraud?.toLocaleString() || '25,000'}
            subtitle="5% tổng giao dịch"
            icon={<DataIcon />}
            color="error"
          />
        </Grid>

        {/* Generate Data */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Tạo dữ liệu giả lập
              </Typography>

              <Alert severity="info" sx={{ mb: 2 }}>
                Tạo dữ liệu giả lập cho việc demo và test hệ thống.
                Dữ liệu bao gồm users, giao dịch bình thường và giao dịch lừa đảo với các patterns thực tế.
              </Alert>

              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Số lượng Users"
                    type="number"
                    value={numUsers}
                    onChange={(e) => setNumUsers(parseInt(e.target.value) || 1000)}
                    inputProps={{ min: 100, max: 100000 }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Số lượng Giao dịch"
                    type="number"
                    value={numTransactions}
                    onChange={(e) => setNumTransactions(parseInt(e.target.value) || 10000)}
                    inputProps={{ min: 1000, max: 1000000 }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={loading ? <CircularProgress size={20} /> : <GenerateIcon />}
                    onClick={handleGenerate}
                    disabled={loading}
                    sx={{ height: '56px' }}
                  >
                    {loading ? 'Đang tạo...' : 'Tạo dữ liệu'}
                  </Button>
                </Grid>
              </Grid>

              {error && <Alert severity="warning">{error}</Alert>}

              {result && (
                <Alert severity="success">
                  Đã tạo thành công {result.num_users?.toLocaleString()} users và{' '}
                  {result.num_transactions?.toLocaleString()} giao dịch
                  (bao gồm {result.num_fraud?.toLocaleString()} fraud).
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default DataExplorer;

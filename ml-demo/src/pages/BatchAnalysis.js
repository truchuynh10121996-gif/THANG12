/**
 * BatchAnalysis Page - Phân tích nhiều giao dịch
 * ===============================================
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { Upload as UploadIcon, Send as SendIcon } from '@mui/icons-material';
import { riskColors } from '../styles/theme';
import { predictBatch } from '../services/api';

// Dữ liệu mẫu
const sampleTransactions = [
  { transaction_id: 'TXN001', user_id: 'USR001', amount: 5000000, channel: 'mobile_app' },
  { transaction_id: 'TXN002', user_id: 'USR002', amount: 500000000, channel: 'web_banking' },
  { transaction_id: 'TXN003', user_id: 'USR003', amount: 1000000, channel: 'mobile_app' },
  { transaction_id: 'TXN004', user_id: 'USR004', amount: 50000000, channel: 'atm' },
  { transaction_id: 'TXN005', user_id: 'USR005', amount: 200000, channel: 'mobile_app' },
];

function BatchAnalysis() {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(sampleTransactions, null, 2));
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);

      const transactions = JSON.parse(jsonInput);
      const response = await predictBatch(transactions);

      setResults(response.predictions);
      setSummary(response.summary);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError('JSON không hợp lệ');
      } else {
        setError('Không thể kết nối đến ML Service');
        // Kết quả mẫu
        setResults(sampleTransactions.map((t, idx) => ({
          ...t,
          fraud_probability: Math.random() * 0.5 + (idx === 1 ? 0.5 : 0),
          risk_level: idx === 1 ? 'critical' : idx === 3 ? 'medium' : 'low',
          prediction: idx === 1 ? 'fraud' : 'normal',
        })));
        setSummary({ total: 5, fraud_count: 1, fraud_ratio: 0.2 });
      }
    } finally {
      setLoading(false);
    }
  };

  const getRiskChip = (level) => (
    <Chip
      label={level}
      size="small"
      sx={{
        backgroundColor: riskColors[level] || riskColors.low,
        color: '#fff',
        fontWeight: 500,
      }}
    />
  );

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Phân tích Batch
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Nhập dữ liệu giao dịch (JSON)
          </Typography>

          <TextField
            fullWidth
            multiline
            rows={10}
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
            placeholder="Nhập JSON array của các giao dịch..."
            sx={{
              fontFamily: 'monospace',
              '& .MuiInputBase-input': { fontFamily: 'monospace' },
            }}
          />

          <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? 'Đang phân tích...' : 'Phân tích'}
            </Button>
            <Button
              variant="outlined"
              startIcon={<UploadIcon />}
              onClick={() => setJsonInput(JSON.stringify(sampleTransactions, null, 2))}
            >
              Load dữ liệu mẫu
            </Button>
          </Box>
        </CardContent>
      </Card>

      {error && <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert>}

      {summary && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Đã phân tích {summary.total} giao dịch. Phát hiện {summary.fraud_count} giao dịch đáng ngờ
          ({(summary.fraud_ratio * 100).toFixed(1)}%)
        </Alert>
      )}

      {results && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Kết quả phân tích
            </Typography>

            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Transaction ID</TableCell>
                    <TableCell>User ID</TableCell>
                    <TableCell align="right">Số tiền</TableCell>
                    <TableCell align="right">Xác suất Fraud</TableCell>
                    <TableCell>Mức rủi ro</TableCell>
                    <TableCell>Dự đoán</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {results.map((r) => (
                    <TableRow key={r.transaction_id}>
                      <TableCell>{r.transaction_id}</TableCell>
                      <TableCell>{r.user_id}</TableCell>
                      <TableCell align="right">{r.amount?.toLocaleString()}</TableCell>
                      <TableCell align="right">
                        {(r.fraud_probability * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell>{getRiskChip(r.risk_level)}</TableCell>
                      <TableCell>
                        <Chip
                          label={r.prediction === 'fraud' ? 'Fraud' : 'Normal'}
                          size="small"
                          color={r.prediction === 'fraud' ? 'error' : 'success'}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default BatchAnalysis;

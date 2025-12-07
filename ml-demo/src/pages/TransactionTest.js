/**
 * TransactionTest Page - Test giao dịch đơn lẻ
 * ==============================================
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Send as SendIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  TipsAndUpdates as TipIcon,
} from '@mui/icons-material';
import { riskColors } from '../styles/theme';
import { predictSingle, explainPrediction } from '../services/api';

function TransactionTest() {
  const [formData, setFormData] = useState({
    transaction_id: 'TXN_TEST_001',
    user_id: 'USR_001',
    amount: 5000000,
    transaction_type: 'transfer',
    channel: 'mobile_app',
    device_type: 'android',
    hour: 14,
    is_international: false,
  });

  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError(null);

      const [predRes, explainRes] = await Promise.all([
        predictSingle(formData),
        explainPrediction(formData),
      ]);

      setResult(predRes.prediction);
      setExplanation(explainRes.explanation);
    } catch (err) {
      setError('Không thể kết nối đến ML Service');
      // Kết quả mẫu
      setResult({
        fraud_probability: 0.75,
        prediction: 'fraud',
        risk_level: 'high',
        should_block: true,
        confidence: 0.50,
      });
      setExplanation({
        summary: 'Giao dịch có 2 yếu tố rủi ro: Số tiền lớn, Chuyển khoản',
        risk_factors: [
          { factor: 'Số tiền lớn', importance: 'high' },
          { factor: 'Loại giao dịch chuyển khoản', importance: 'medium' },
        ],
        recommendations: ['Xác nhận OTP', 'Ghi nhận cảnh báo'],
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical':
        return <ErrorIcon sx={{ color: riskColors.critical }} />;
      case 'high':
        return <WarningIcon sx={{ color: riskColors.high }} />;
      case 'medium':
        return <InfoIcon sx={{ color: riskColors.medium }} />;
      default:
        return <CheckIcon sx={{ color: riskColors.low }} />;
    }
  };

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Test Giao dịch Đơn lẻ
      </Typography>

      <Grid container spacing={3}>
        {/* Form input */}
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Thông tin giao dịch
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Transaction ID"
                    name="transaction_id"
                    value={formData.transaction_id}
                    onChange={handleChange}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="User ID"
                    name="user_id"
                    value={formData.user_id}
                    onChange={handleChange}
                    size="small"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Số tiền (VND)"
                    name="amount"
                    type="number"
                    value={formData.amount}
                    onChange={handleChange}
                    size="small"
                  />
                </Grid>

                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Loại giao dịch</InputLabel>
                    <Select
                      name="transaction_type"
                      value={formData.transaction_type}
                      onChange={handleChange}
                      label="Loại giao dịch"
                    >
                      <MenuItem value="transfer">Chuyển khoản</MenuItem>
                      <MenuItem value="payment">Thanh toán</MenuItem>
                      <MenuItem value="withdrawal">Rút tiền</MenuItem>
                      <MenuItem value="deposit">Nạp tiền</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Kênh</InputLabel>
                    <Select
                      name="channel"
                      value={formData.channel}
                      onChange={handleChange}
                      label="Kênh"
                    >
                      <MenuItem value="mobile_app">Mobile App</MenuItem>
                      <MenuItem value="web_banking">Web Banking</MenuItem>
                      <MenuItem value="atm">ATM</MenuItem>
                      <MenuItem value="branch">Chi nhánh</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Giờ (0-23)"
                    name="hour"
                    type="number"
                    value={formData.hour}
                    onChange={handleChange}
                    size="small"
                    inputProps={{ min: 0, max: 23 }}
                  />
                </Grid>

                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Quốc tế</InputLabel>
                    <Select
                      name="is_international"
                      value={formData.is_international}
                      onChange={handleChange}
                      label="Quốc tế"
                    >
                      <MenuItem value={false}>Không</MenuItem>
                      <MenuItem value={true}>Có</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <Button
                fullWidth
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                onClick={handleSubmit}
                disabled={loading}
                sx={{ mt: 3 }}
              >
                {loading ? 'Đang phân tích...' : 'Phân tích giao dịch'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={7}>
          {error && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {result && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  {getRiskIcon(result.risk_level)}
                  <Typography variant="h6">Kết quả phân tích</Typography>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Xác suất Fraud
                    </Typography>
                    <Typography variant="h4" sx={{ color: riskColors[result.risk_level] }}>
                      {(result.fraud_probability * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Mức độ rủi ro
                    </Typography>
                    <Chip
                      label={result.risk_level?.toUpperCase()}
                      sx={{
                        backgroundColor: riskColors[result.risk_level],
                        color: '#fff',
                        fontWeight: 600,
                        mt: 1,
                      }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Dự đoán
                    </Typography>
                    <Typography variant="h6">
                      {result.prediction === 'fraud' ? 'Nghi ngờ lừa đảo' : 'Bình thường'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Hành động
                    </Typography>
                    <Typography variant="h6" color={result.should_block ? 'error' : 'success'}>
                      {result.should_block ? 'Nên chặn' : 'Cho phép'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {explanation && (
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Giải thích chi tiết
                </Typography>

                <Alert severity="info" sx={{ mb: 2 }}>
                  {explanation.summary}
                </Alert>

                {explanation.risk_factors?.length > 0 && (
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Yếu tố rủi ro:
                    </Typography>
                    <List dense>
                      {explanation.risk_factors.map((factor, idx) => (
                        <ListItem key={idx}>
                          <ListItemIcon>
                            <WarningIcon
                              sx={{
                                color:
                                  factor.importance === 'high'
                                    ? riskColors.high
                                    : riskColors.medium,
                              }}
                            />
                          </ListItemIcon>
                          <ListItemText primary={factor.factor} />
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Khuyến nghị:
                </Typography>
                <List dense>
                  {explanation.recommendations?.map((rec, idx) => (
                    <ListItem key={idx}>
                      <ListItemIcon>
                        <TipIcon sx={{ color: 'primary.main' }} />
                      </ListItemIcon>
                      <ListItemText primary={rec} />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

export default TransactionTest;

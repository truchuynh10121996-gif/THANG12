/**
 * TransactionTest Page - Kiểm tra giao dịch với dữ liệu khách hàng thực
 * =====================================================================
 * Trang này cho phép:
 * 1. Chọn khách hàng từ 3 file Excel (USR_000001, USR_000002, USR_000003)
 * 2. Xem thông tin profile và lịch sử giao dịch của khách hàng
 * 3. Nhập thông tin giao dịch mới
 * 4. Phân tích giao dịch với 5 ML models (Isolation Forest, LightGBM, Autoencoder, LSTM, GNN)
 * 5. Xem kết quả phân tích và giải thích chi tiết
 *
 * Flow hoạt động:
 * - Người dùng chọn user_id từ dropdown (3 users cố định)
 * - Frontend gọi API lấy profile + lịch sử giao dịch từ file Excel
 * - Hiển thị thông tin user và 15 giao dịch gần nhất
 * - Người dùng nhập thông tin giao dịch mới
 * - Khi nhấn "Phân tích giao dịch", gửi đến pipeline 5 models
 * - Hiển thị kết quả với màu sắc tương ứng mức độ rủi ro
 */

import React, { useState, useEffect } from 'react';
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
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Skeleton,
  Tooltip,
  IconButton,
  LinearProgress,
  Collapse,
} from '@mui/material';
import {
  Send as SendIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  TipsAndUpdates as TipIcon,
  Person as PersonIcon,
  History as HistoryIcon,
  AccountBalance as AccountIcon,
  Speed as SpeedIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Security as SecurityIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { riskColors, gradients } from '../styles/theme';
import {
  getDemoCustomers,
  getDemoCustomerDetail,
  getDemoCustomerTransactions,
  analyzeDemoTransaction,
  formatCurrency,
  getRiskColor,
} from '../services/api';

function TransactionTest() {
  // ===== STATE MANAGEMENT =====

  // Danh sách customers và customer được chọn
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState('');
  const [customerProfile, setCustomerProfile] = useState(null);
  const [customerTransactions, setCustomerTransactions] = useState([]);
  const [behavioralFeatures, setBehavioralFeatures] = useState(null);

  // Loading states
  const [loadingCustomers, setLoadingCustomers] = useState(true);
  const [loadingCustomerData, setLoadingCustomerData] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // Form data cho giao dịch mới
  const [formData, setFormData] = useState({
    amount: 5000000,
    transaction_type: 'transfer',
    channel: 'mobile_app',
    hour: new Date().getHours(),
    recipient_id: '',
    is_international: false,
  });

  // Kết quả phân tích
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  // Expand model details
  const [expandModelDetails, setExpandModelDetails] = useState(false);

  // ===== LOAD CUSTOMERS ON MOUNT =====

  useEffect(() => {
    loadCustomers();
  }, []);

  // ===== API CALLS =====

  /**
   * Tải danh sách 3 khách hàng demo
   */
  const loadCustomers = async () => {
    try {
      setLoadingCustomers(true);
      const response = await getDemoCustomers();
      setCustomers(response.customers || []);
    } catch (err) {
      console.error('Lỗi tải danh sách khách hàng:', err);
      setError('Không thể tải danh sách khách hàng. Vui lòng kiểm tra kết nối ML Service.');
    } finally {
      setLoadingCustomers(false);
    }
  };

  /**
   * Khi chọn customer, load profile và lịch sử giao dịch
   */
  const handleCustomerSelect = async (event) => {
    const userId = event.target.value;
    setSelectedCustomer(userId);
    setAnalysisResult(null);
    setError(null);

    if (!userId) {
      setCustomerProfile(null);
      setCustomerTransactions([]);
      setBehavioralFeatures(null);
      return;
    }

    try {
      setLoadingCustomerData(true);

      // Gọi đồng thời cả 2 API
      const [detailRes, txRes] = await Promise.all([
        getDemoCustomerDetail(userId),
        getDemoCustomerTransactions(userId, 15),
      ]);

      setCustomerProfile(detailRes.profile);
      setBehavioralFeatures(detailRes.behavioral_features);
      setCustomerTransactions(txRes.transactions || []);

    } catch (err) {
      console.error('Lỗi tải thông tin khách hàng:', err);
      setError('Không thể tải thông tin khách hàng');
    } finally {
      setLoadingCustomerData(false);
    }
  };

  /**
   * Xử lý thay đổi form
   */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  /**
   * Submit phân tích giao dịch
   */
  const handleAnalyze = async () => {
    if (!selectedCustomer) {
      setError('Vui lòng chọn khách hàng trước');
      return;
    }

    try {
      setLoadingAnalysis(true);
      setError(null);

      // Chuẩn bị dữ liệu giao dịch
      const transactionData = {
        ...formData,
        user_id: selectedCustomer,
        transaction_id: `TX_${selectedCustomer}_${Date.now()}`,
      };

      // Gọi API phân tích
      const response = await analyzeDemoTransaction(transactionData);
      setAnalysisResult(response.result);

      // Cập nhật behavioral features từ response
      if (response.result?.behavioral_features) {
        setBehavioralFeatures(response.result.behavioral_features);
      }

    } catch (err) {
      console.error('Lỗi phân tích giao dịch:', err);
      setError('Không thể kết nối đến ML Service. Vui lòng đảm bảo service đang chạy.');
    } finally {
      setLoadingAnalysis(false);
    }
  };

  /**
   * Format số tiền VND
   */
  const formatMoney = (amount) => {
    if (!amount && amount !== 0) return 'N/A';
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND'
    }).format(amount);
  };

  /**
   * Lấy icon theo risk level
   */
  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical':
        return <ErrorIcon sx={{ color: riskColors.critical, fontSize: 32 }} />;
      case 'high':
        return <WarningIcon sx={{ color: riskColors.high, fontSize: 32 }} />;
      case 'medium':
        return <InfoIcon sx={{ color: riskColors.medium, fontSize: 32 }} />;
      default:
        return <CheckIcon sx={{ color: riskColors.low, fontSize: 32 }} />;
    }
  };

  /**
   * Lấy màu theo risk level
   */
  const getRiskBgColor = (level) => {
    switch (level) {
      case 'critical':
        return 'linear-gradient(135deg, #7B1FA2 0%, #9C27B0 100%)';
      case 'high':
        return 'linear-gradient(135deg, #D32F2F 0%, #F44336 100%)';
      case 'medium':
        return 'linear-gradient(135deg, #ED6C02 0%, #FF9800 100%)';
      default:
        return 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)';
    }
  };

  /**
   * Lấy label tiếng Việt cho risk level
   */
  const getRiskLabel = (level) => {
    const labels = {
      low: 'An toàn',
      medium: 'Cần xác minh',
      high: 'Nghi ngờ cao',
      critical: 'Rủi ro nghiêm trọng'
    };
    return labels[level] || level;
  };

  // ===== RENDER COMPONENTS =====

  /**
   * Render phần chọn khách hàng
   */
  const renderCustomerSelector = () => (
    <Card sx={{ mb: 3, background: gradients.cardPink, color: '#fff' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <PersonIcon />
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Chọn khách hàng
          </Typography>
        </Box>

        <FormControl fullWidth>
          <Select
            value={selectedCustomer}
            onChange={handleCustomerSelect}
            displayEmpty
            sx={{
              backgroundColor: 'rgba(255,255,255,0.95)',
              borderRadius: 2,
              '& .MuiOutlinedInput-notchedOutline': { border: 'none' },
            }}
          >
            <MenuItem value="" disabled>
              -- Chọn một khách hàng --
            </MenuItem>
            {loadingCustomers ? (
              <MenuItem disabled>
                <CircularProgress size={20} sx={{ mr: 2 }} />
                Đang tải...
              </MenuItem>
            ) : (
              customers.map((customer) => (
                <MenuItem key={customer.user_id} value={customer.user_id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PersonIcon sx={{ color: '#FF6B99' }} />
                    <Box>
                      <Typography sx={{ fontWeight: 600 }}>
                        {customer.ho_ten}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {customer.user_id} • {customer.nghe_nghiep} • {customer.total_transactions} giao dịch
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))
            )}
          </Select>
        </FormControl>

        {customers.length > 0 && (
          <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.9 }}>
            Dữ liệu từ 3 file Excel: USR_000001, USR_000002, USR_000003
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  /**
   * Render thông tin profile khách hàng
   */
  const renderCustomerProfile = () => {
    if (!selectedCustomer) return null;

    if (loadingCustomerData) {
      return (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Skeleton variant="text" width="60%" />
            <Skeleton variant="rectangular" height={150} sx={{ mt: 2 }} />
          </CardContent>
        </Card>
      );
    }

    if (!customerProfile) return null;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <AccountIcon sx={{ color: '#FF6B99' }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#FF6B99' }}>
              Thông tin khách hàng
            </Typography>
          </Box>

          <Grid container spacing={2}>
            {/* Thông tin cơ bản */}
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Họ tên</Typography>
              <Typography sx={{ fontWeight: 600 }}>{customerProfile.ho_ten}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Tuổi</Typography>
              <Typography sx={{ fontWeight: 600 }}>{customerProfile.tuoi} tuổi</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Nghề nghiệp</Typography>
              <Typography sx={{ fontWeight: 600 }}>{customerProfile.nghe_nghiep}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Mức thu nhập</Typography>
              <Typography sx={{ fontWeight: 600 }}>
                {formatMoney(customerProfile.thu_nhap_hang_thang)}/tháng
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
            </Grid>

            {/* Thông tin tài khoản */}
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Số ngày sử dụng</Typography>
              <Typography sx={{ fontWeight: 600 }}>{customerProfile.so_ngay_mo_tai_khoan} ngày</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Ngân hàng</Typography>
              <Typography sx={{ fontWeight: 600 }}>{customerProfile.ngan_hang}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">GD trung bình</Typography>
              <Typography sx={{ fontWeight: 600 }}>
                {formatMoney(customerProfile.avg_transaction_amount)}
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Thiết bị</Typography>
              <Typography sx={{ fontWeight: 600 }}>
                {customerProfile.loai_thiet_bi}
              </Typography>
            </Grid>
          </Grid>

          {/* Behavioral features */}
          {behavioralFeatures && (
            <>
              <Divider sx={{ my: 2 }} />
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <SpeedIcon sx={{ color: '#FF6B99', fontSize: 20 }} />
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Chỉ số hành vi thời gian thực
                </Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">GD trong 1 giờ</Typography>
                  <Typography sx={{ fontWeight: 600 }}>{behavioralFeatures.velocity_1h}</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">GD trong 24 giờ</Typography>
                  <Typography sx={{ fontWeight: 600 }}>{behavioralFeatures.velocity_24h}</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">Thời gian từ GD cuối</Typography>
                  <Typography sx={{ fontWeight: 600 }}>
                    {behavioralFeatures.time_since_last_transaction}h
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">Độ lệch số tiền</Typography>
                  <Typography sx={{ fontWeight: 600 }}>
                    x{behavioralFeatures.amount_deviation_ratio}
                  </Typography>
                </Grid>
              </Grid>
            </>
          )}
        </CardContent>
      </Card>
    );
  };

  /**
   * Render lịch sử giao dịch
   */
  const renderTransactionHistory = () => {
    if (!selectedCustomer) return null;

    if (loadingCustomerData) {
      return (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Skeleton variant="text" width="60%" />
            <Skeleton variant="rectangular" height={200} sx={{ mt: 2 }} />
          </CardContent>
        </Card>
      );
    }

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <HistoryIcon sx={{ color: '#FF6B99' }} />
              <Typography variant="h6" sx={{ fontWeight: 700, color: '#FF6B99' }}>
                Lịch sử giao dịch gần đây (15 GD)
              </Typography>
            </Box>
            <Tooltip title="Tải lại">
              <IconButton size="small" onClick={() => handleCustomerSelect({ target: { value: selectedCustomer } })}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {customerTransactions.length === 0 ? (
            <Alert severity="info">Chưa có lịch sử giao dịch</Alert>
          ) : (
            <TableContainer component={Paper} sx={{ maxHeight: 350 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>Ngày</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Giờ</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Loại GD</TableCell>
                    <TableCell sx={{ fontWeight: 600 }} align="right">Số tiền</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Người nhận</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Kênh</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Trạng thái</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {customerTransactions.map((tx, idx) => (
                    <TableRow key={tx.transaction_id || idx} hover>
                      <TableCell>{tx.ngay_giao_dich}</TableCell>
                      <TableCell>{tx.gio_giao_dich}</TableCell>
                      <TableCell>
                        <Chip
                          label={tx.loai_giao_dich}
                          size="small"
                          sx={{ fontSize: '0.7rem' }}
                        />
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 500, color: '#D32F2F' }}>
                        {formatMoney(tx.so_tien)}
                      </TableCell>
                      <TableCell sx={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {tx.ten_nguoi_nhan || 'N/A'}
                      </TableCell>
                      <TableCell>{tx.channel}</TableCell>
                      <TableCell>
                        <Chip
                          label={tx.trang_thai}
                          size="small"
                          color={tx.trang_thai === 'Thanh cong' ? 'success' : 'default'}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    );
  };

  /**
   * Render form nhập giao dịch
   */
  const renderTransactionForm = () => (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: '#FF6B99' }}>
          Thông tin giao dịch mới
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Số tiền (VND)"
              name="amount"
              type="number"
              value={formData.amount}
              onChange={handleChange}
              size="small"
              helperText={customerProfile ? `Trung bình của user: ${formatMoney(customerProfile.avg_transaction_amount)}` : ''}
              InputProps={{
                inputProps: { min: 0 }
              }}
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
              <InputLabel>Kênh giao dịch</InputLabel>
              <Select
                name="channel"
                value={formData.channel}
                onChange={handleChange}
                label="Kênh giao dịch"
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
              label="ID Người nhận"
              name="recipient_id"
              value={formData.recipient_id}
              onChange={handleChange}
              size="small"
              placeholder="VD: RCP_001"
              helperText="Để trống nếu không có"
            />
          </Grid>

          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Giờ giao dịch (0-23)"
              name="hour"
              type="number"
              value={formData.hour}
              onChange={handleChange}
              size="small"
              inputProps={{ min: 0, max: 23 }}
            />
          </Grid>

          <Grid item xs={12}>
            <FormControl fullWidth size="small">
              <InputLabel>Giao dịch quốc tế</InputLabel>
              <Select
                name="is_international"
                value={formData.is_international}
                onChange={handleChange}
                label="Giao dịch quốc tế"
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
          startIcon={loadingAnalysis ? <CircularProgress size={20} color="inherit" /> : <AnalyticsIcon />}
          onClick={handleAnalyze}
          disabled={loadingAnalysis || !selectedCustomer}
          sx={{
            mt: 3,
            py: 1.5,
            background: gradients.cardPink,
            fontSize: '1rem',
            fontWeight: 600,
            '&:hover': {
              background: gradients.cardPink,
              opacity: 0.9,
            },
          }}
        >
          {loadingAnalysis ? 'Đang phân tích với 5 models...' : 'Phân tích giao dịch'}
        </Button>

        {loadingAnalysis && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress color="secondary" />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
              Đang chạy: Isolation Forest → LightGBM → Autoencoder → LSTM → GNN
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  /**
   * Render kết quả phân tích
   */
  const renderAnalysisResults = () => (
    <>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {analysisResult && (
        <>
          {/* Main result card */}
          <Card
            sx={{
              mb: 2,
              background: getRiskBgColor(analysisResult.prediction?.risk_level),
              color: '#fff'
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                {getRiskIcon(analysisResult.prediction?.risk_level)}
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    {analysisResult.prediction?.risk_level === 'low'
                      ? '✅ AN TOÀN - Chuyển tiền thành công!'
                      : analysisResult.prediction?.risk_level === 'medium'
                        ? '⚡ CẦN XÁC MINH THÊM'
                        : analysisResult.prediction?.risk_level === 'high'
                          ? '⚠️ NGHI NGỜ CAO - Cần kiểm tra'
                          : '🛑 CHẶN GIAO DỊCH - Rủi ro nghiêm trọng'}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Mức độ rủi ro: {getRiskLabel(analysisResult.prediction?.risk_level)}
                  </Typography>
                </Box>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>Xác suất gian lận</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 700 }}>
                    {((analysisResult.prediction?.fraud_probability || 0) * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>Độ tin cậy</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 700 }}>
                    {((analysisResult.prediction?.confidence || 0) * 100).toFixed(0)}%
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>Dự đoán</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    {analysisResult.prediction?.prediction === 'fraud' ? 'Gian lận' : 'Bình thường'}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>Hành động</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>
                    {analysisResult.prediction?.should_block ? 'CHẶN' : 'CHO PHÉP'}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Model scores */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer'
                }}
                onClick={() => setExpandModelDetails(!expandModelDetails)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SecurityIcon sx={{ color: '#FF6B99' }} />
                  <Typography variant="h6" sx={{ fontWeight: 700, color: '#FF6B99' }}>
                    Kết quả từ 5 Models
                  </Typography>
                </Box>
                {expandModelDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </Box>

              {/* Quick view of model scores */}
              <Grid container spacing={1} sx={{ mt: 2 }}>
                {analysisResult.model_scores && Object.entries(analysisResult.model_scores).map(([model, score]) => (
                  <Grid item xs={12} sm={6} md={2.4} key={model}>
                    <Box sx={{ textAlign: 'center', p: 1, borderRadius: 1, bgcolor: 'grey.100' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase' }}>
                        {model.replace('_', ' ')}
                      </Typography>
                      <Typography
                        variant="h6"
                        sx={{
                          fontWeight: 700,
                          color: score > 0.7 ? riskColors.high : score > 0.5 ? riskColors.medium : riskColors.low
                        }}
                      >
                        {(score * 100).toFixed(0)}%
                      </Typography>
                      <Chip
                        label={analysisResult.models_status?.[model] ? 'Loaded' : 'N/A'}
                        size="small"
                        color={analysisResult.models_status?.[model] ? 'success' : 'default'}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>

              {/* Expanded details */}
              <Collapse in={expandModelDetails}>
                <Divider sx={{ my: 2 }} />
                <Grid container spacing={2}>
                  {analysisResult.model_details && Object.entries(analysisResult.model_details).map(([model, details]) => (
                    <Grid item xs={12} md={6} key={model}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, textTransform: 'uppercase' }}>
                          {model.replace('_', ' ')}
                        </Typography>
                        {details.loaded ? (
                          <>
                            <Typography variant="body2" color="text.secondary">
                              {details.description}
                            </Typography>
                            {details.anomaly_score !== undefined && (
                              <Typography variant="body2">
                                Anomaly Score: {details.anomaly_score.toFixed(4)}
                              </Typography>
                            )}
                            {details.reconstruction_error !== undefined && (
                              <Typography variant="body2">
                                Reconstruction Error: {details.reconstruction_error.toFixed(4)}
                              </Typography>
                            )}
                          </>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            Model chưa được load
                          </Typography>
                        )}
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Collapse>
            </CardContent>
          </Card>

          {/* Explanation */}
          {analysisResult.explanation && (
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: '#FF6B99' }}>
                  Giải thích chi tiết
                </Typography>

                <Alert
                  severity={
                    analysisResult.prediction?.risk_level === 'critical' ? 'error' :
                    analysisResult.prediction?.risk_level === 'high' ? 'warning' :
                    analysisResult.prediction?.risk_level === 'medium' ? 'info' : 'success'
                  }
                  sx={{ mb: 2 }}
                >
                  {analysisResult.explanation.summary}
                </Alert>

                {/* Risk factors */}
                {analysisResult.explanation.risk_factors?.length > 0 && (
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                      Yếu tố rủi ro phát hiện:
                    </Typography>
                    <List dense>
                      {analysisResult.explanation.risk_factors.map((factor, idx) => (
                        <ListItem key={idx}>
                          <ListItemIcon>
                            <WarningIcon
                              sx={{
                                color: factor.importance === 'high' ? riskColors.high : riskColors.medium,
                              }}
                            />
                          </ListItemIcon>
                          <ListItemText primary={factor.factor} />
                          <Chip
                            label={factor.importance === 'high' ? 'Quan trọng' : 'Trung bình'}
                            size="small"
                            color={factor.importance === 'high' ? 'error' : 'warning'}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}

                {/* Positive factors */}
                {analysisResult.explanation.positive_factors?.length > 0 && (
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600, mt: 2 }}>
                      Yếu tố tích cực:
                    </Typography>
                    <List dense>
                      {analysisResult.explanation.positive_factors.map((factor, idx) => (
                        <ListItem key={idx}>
                          <ListItemIcon>
                            <CheckIcon sx={{ color: riskColors.low }} />
                          </ListItemIcon>
                          <ListItemText primary={factor} />
                        </ListItem>
                      ))}
                    </List>
                  </>
                )}

                <Divider sx={{ my: 2 }} />

                {/* Recommendations */}
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Khuyến nghị hành động:
                </Typography>
                <List dense>
                  {analysisResult.explanation.recommendations?.map((rec, idx) => (
                    <ListItem key={idx}>
                      <ListItemIcon>
                        <TipIcon sx={{ color: '#FF8DAD' }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={rec}
                        primaryTypographyProps={{
                          fontWeight: idx === 0 ? 600 : 400
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </>
  );

  // ===== MAIN RENDER =====

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 1, fontWeight: 700, color: '#FF6B99' }}>
        Kiểm tra giao dịch
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Phân tích giao dịch với dữ liệu khách hàng thực từ file Excel và 5 ML models
      </Typography>

      <Grid container spacing={3}>
        {/* Cột trái - Thông tin khách hàng */}
        <Grid item xs={12} md={7}>
          {renderCustomerSelector()}
          {renderCustomerProfile()}
          {renderTransactionHistory()}
        </Grid>

        {/* Cột phải - Form và kết quả */}
        <Grid item xs={12} md={5}>
          {renderTransactionForm()}
          {renderAnalysisResults()}
        </Grid>
      </Grid>
    </Box>
  );
}

export default TransactionTest;

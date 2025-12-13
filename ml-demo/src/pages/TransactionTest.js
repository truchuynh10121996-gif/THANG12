/**
 * TransactionTest Page - Kiểm tra giao dịch đơn lẻ
 * ==================================================
 * Trang này cho phép:
 * 1. Chọn 1 trong 3 khách hàng demo (USR_0000001, USR_0000002, USR_0000003)
 * 2. Xem thông tin profile và lịch sử giao dịch của khách hàng
 * 3. Nhập thông tin giao dịch mới
 * 4. Phân tích giao dịch với ML models
 * 5. Xem kết quả và giải thích chi tiết
 *
 * Flow hoạt động:
 * - Người dùng chọn user_id từ dropdown (3 khách hàng demo)
 * - Dữ liệu khách hàng được load từ file local (customerData.js)
 * - Hiển thị thông tin user và lịch sử cho người dùng xem
 * - Người dùng nhập thông tin giao dịch mới
 * - Khi nhấn "Phân tích", gửi request đến ML Service
 * - ML Service tính toán features và trả về kết quả
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
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  IconButton,
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
} from '@mui/icons-material';
import { riskColors, gradients } from '../styles/theme';
import {
  predictSingle,
  explainPrediction,
  formatCurrency,
  formatDateTime,
  getTransactionTypeLabel,
  getIncomeLevelLabel,
  getRiskLabel,
  getRiskColor,
} from '../services/api';

// Import dữ liệu 3 khách hàng demo
import {
  DEMO_CUSTOMERS,
  getDemoUserById,
  getDemoTransactions,
  getDemoUserProfile,
  getDemoBehavioralFeatures,
} from '../data/customerData';

function TransactionTest() {
  // ===== STATE MANAGEMENT =====

  // User được chọn (từ 3 khách hàng demo)
  const [selectedUserId, setSelectedUserId] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [userTransactions, setUserTransactions] = useState([]);
  const [behavioralFeatures, setBehavioralFeatures] = useState(null);

  // Loading states
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  // Form data cho giao dịch mới
  const [formData, setFormData] = useState({
    transaction_id: `TXN_${Date.now()}`,
    amount: 5000000,
    transaction_type: 'transfer',
    channel: 'mobile_app',
    device_type: 'android',
    hour: new Date().getHours(),
    is_international: false,
    recipient_id: '',
  });

  // Kết quả phân tích
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState(null);

  // ===== HANDLER CHỌN KHÁCH HÀNG DEMO =====

  /**
   * Khi chọn user từ dropdown 3 khách hàng demo
   */
  const handleUserSelect = (event) => {
    const userId = event.target.value;
    setSelectedUserId(userId);
    setResult(null);
    setExplanation(null);
    setError(null);

    if (!userId) {
      setSelectedUser(null);
      setUserProfile(null);
      setUserTransactions([]);
      setBehavioralFeatures(null);
      return;
    }

    // Load dữ liệu từ file local (3 khách hàng demo)
    const user = getDemoUserById(userId);
    const profile = getDemoUserProfile(userId);
    const transactions = getDemoTransactions(userId);
    const features = getDemoBehavioralFeatures(userId);

    setSelectedUser(user);
    setUserProfile(profile);
    setUserTransactions(transactions);
    setBehavioralFeatures(features);

    // Cập nhật form với user_id
    setFormData(prev => ({
      ...prev,
      user_id: userId,
      transaction_id: `TXN_${userId}_${Date.now()}`,
    }));
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
  const handleSubmit = async () => {
    if (!selectedUser) {
      setError('Vui lòng chọn khách hàng trước');
      return;
    }

    try {
      setLoadingAnalysis(true);
      setError(null);

      // Chuẩn bị dữ liệu giao dịch
      const transactionData = {
        ...formData,
        user_id: selectedUser.user_id,
      };

      // Gọi API prediction và explanation
      const [predRes, explainRes] = await Promise.all([
        predictSingle(transactionData),
        explainPrediction(transactionData),
      ]);

      setResult(predRes.prediction);

      // Cập nhật behavioral features từ response
      if (predRes.behavioral_features) {
        setBehavioralFeatures(predRes.behavioral_features);
      }

      setExplanation(explainRes.explanation);

    } catch (err) {
      console.error('Lỗi phân tích giao dịch:', err);
      setError('Không thể kết nối đến ML Service. Đang hiển thị kết quả mẫu...');

      // Kết quả mẫu khi không kết nối được
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
          { factor: 'Số tiền lớn so với trung bình', importance: 'high' },
          { factor: 'Loại giao dịch chuyển khoản', importance: 'medium' },
        ],
        recommendations: ['Xác nhận OTP', 'Ghi nhận cảnh báo'],
      });
    } finally {
      setLoadingAnalysis(false);
    }
  };

  /**
   * Lấy icon theo risk level
   */
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

  // ===== RENDER COMPONENTS =====

  /**
   * Render phần chọn khách hàng (3 khách hàng demo)
   */
  const renderUserSelector = () => (
    <Card sx={{ mb: 3, background: gradients.cardPink, color: '#fff' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <PersonIcon />
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Chọn khách hàng demo
          </Typography>
        </Box>

        <Alert severity="info" sx={{ mb: 2, backgroundColor: 'rgba(255,255,255,0.9)' }}>
          Chọn 1 trong 3 khách hàng demo để kiểm tra giao dịch
        </Alert>

        <FormControl fullWidth>
          <Select
            value={selectedUserId}
            onChange={handleUserSelect}
            displayEmpty
            sx={{
              backgroundColor: 'rgba(255,255,255,0.95)',
              borderRadius: 2,
              '& .MuiOutlinedInput-notchedOutline': { border: 'none' },
            }}
          >
            <MenuItem value="" disabled>
              -- Chọn khách hàng --
            </MenuItem>
            {DEMO_CUSTOMERS.map((customer) => (
              <MenuItem key={customer.user_id} value={customer.user_id}>
                <Box>
                  <Typography sx={{ fontWeight: 600 }}>
                    {customer.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {customer.user_id} • {customer.occupation} • {customer.city}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </CardContent>
    </Card>
  );

  /**
   * Render thông tin profile khách hàng
   */
  const renderUserProfile = () => {
    if (!selectedUser || !userProfile) return null;

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
              <Typography sx={{ fontWeight: 600 }}>{userProfile.name}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Tuổi</Typography>
              <Typography sx={{ fontWeight: 600 }}>{userProfile.age} tuổi</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Nghề nghiệp</Typography>
              <Typography sx={{ fontWeight: 600 }}>{userProfile.occupation}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Mức thu nhập</Typography>
              <Typography sx={{ fontWeight: 600 }}>
                {getIncomeLevelLabel(userProfile.income_level)}
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
            </Grid>

            {/* Thông tin tài khoản */}
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Số ngày sử dụng</Typography>
              <Typography sx={{ fontWeight: 600 }}>{userProfile.account_age_days} ngày</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Cấp độ KYC</Typography>
              <Chip
                label={`KYC ${userProfile.kyc_level}`}
                size="small"
                color={userProfile.kyc_level >= 3 ? 'success' : userProfile.kyc_level === 2 ? 'warning' : 'default'}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">GD trung bình</Typography>
              <Typography sx={{ fontWeight: 600 }}>
                {formatCurrency(userProfile.avg_transaction_amount)}
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption" color="text.secondary">Điểm rủi ro lịch sử</Typography>
              <Chip
                label={`${(userProfile.historical_risk_score * 100).toFixed(0)}%`}
                size="small"
                sx={{
                  backgroundColor: getRiskColor(
                    userProfile.historical_risk_score > 0.6 ? 'high' :
                    userProfile.historical_risk_score > 0.3 ? 'medium' : 'low'
                  ),
                  color: '#fff',
                }}
              />
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
    if (!selectedUser) return null;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <HistoryIcon sx={{ color: '#FF6B99' }} />
              <Typography variant="h6" sx={{ fontWeight: 700, color: '#FF6B99' }}>
                Lịch sử giao dịch gần đây
              </Typography>
            </Box>
            <Tooltip title="Tải lại">
              <IconButton size="small" onClick={() => {
                if (selectedUserId) {
                  const transactions = getDemoTransactions(selectedUserId);
                  setUserTransactions(transactions);
                }
              }}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {userTransactions.length === 0 ? (
            <Alert severity="info">Chưa có lịch sử giao dịch</Alert>
          ) : (
            <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>Thời gian</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Số tiền</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Loại GD</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Người nhận</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Trạng thái</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {userTransactions.map((tx, idx) => (
                    <TableRow key={tx.transaction_id || idx} hover>
                      <TableCell>{formatDateTime(tx.timestamp)}</TableCell>
                      <TableCell sx={{ fontWeight: 500 }}>
                        {formatCurrency(tx.amount)}
                      </TableCell>
                      <TableCell>{getTransactionTypeLabel(tx.transaction_type)}</TableCell>
                      <TableCell>{tx.recipient_id || 'N/A'}</TableCell>
                      <TableCell>
                        <Chip
                          label={tx.status === 'completed' ? 'Hoàn thành' :
                                 tx.status === 'pending' ? 'Đang xử lý' : 'Thất bại'}
                          size="small"
                          color={tx.status === 'completed' ? 'success' :
                                 tx.status === 'pending' ? 'warning' : 'error'}
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
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Transaction ID"
              name="transaction_id"
              value={formData.transaction_id}
              onChange={handleChange}
              size="small"
              disabled
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="User ID"
              value={selectedUser?.user_id || ''}
              size="small"
              disabled
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
              helperText={`Trung bình của user: ${formatCurrency(userProfile?.avg_transaction_amount || 0)}`}
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

          <Grid item xs={6}>
            <FormControl fullWidth size="small">
              <InputLabel>Loại thiết bị</InputLabel>
              <Select
                name="device_type"
                value={formData.device_type}
                onChange={handleChange}
                label="Loại thiết bị"
              >
                <MenuItem value="android">Android</MenuItem>
                <MenuItem value="ios">iOS</MenuItem>
                <MenuItem value="web">Web Browser</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={6}>
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
          startIcon={loadingAnalysis ? <CircularProgress size={20} /> : <SendIcon />}
          onClick={handleSubmit}
          disabled={loadingAnalysis || !selectedUser}
          sx={{
            mt: 3,
            background: gradients.cardPink,
            '&:hover': {
              background: gradients.cardPink,
              opacity: 0.9,
            },
          }}
        >
          {loadingAnalysis ? 'Đang phân tích...' : 'Phân tích giao dịch'}
        </Button>
      </CardContent>
    </Card>
  );

  /**
   * Render kết quả phân tích
   */
  const renderResults = () => (
    <>
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
              <Typography variant="h6" sx={{ fontWeight: 700 }}>Kết quả phân tích</Typography>
            </Box>

            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Xác suất Fraud
                </Typography>
                <Typography variant="h4" sx={{ color: riskColors[result.risk_level], fontWeight: 700 }}>
                  {(result.fraud_probability * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Mức độ rủi ro
                </Typography>
                <Chip
                  label={getRiskLabel(result.risk_level)}
                  sx={{
                    backgroundColor: riskColors[result.risk_level],
                    color: '#fff',
                    fontWeight: 700,
                    mt: 1,
                    fontSize: '1rem',
                    padding: '4px 8px',
                  }}
                />
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Dự đoán
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {result.prediction === 'fraud' ? 'Nghi ngờ lừa đảo' : 'Bình thường'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Hành động đề xuất
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 600,
                    color: result.should_block ? '#D32F2F' : '#4caf50'
                  }}
                >
                  {result.should_block ? 'Nên chặn giao dịch' : 'Cho phép giao dịch'}
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {explanation && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: '#FF6B99' }}>
              Giải thích chi tiết
            </Typography>

            <Alert
              severity={result?.risk_level === 'high' || result?.risk_level === 'critical' ? 'error' :
                       result?.risk_level === 'medium' ? 'warning' : 'info'}
              sx={{ mb: 2 }}
            >
              {explanation.summary}
            </Alert>

            {explanation.risk_factors?.length > 0 && (
              <>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Yếu tố rủi ro:
                </Typography>
                <List dense>
                  {explanation.risk_factors.map((factor, idx) => (
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

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
              Khuyến nghị:
            </Typography>
            <List dense>
              {explanation.recommendations?.map((rec, idx) => (
                <ListItem key={idx}>
                  <ListItemIcon>
                    <TipIcon sx={{ color: '#FF8DAD' }} />
                  </ListItemIcon>
                  <ListItemText primary={rec} />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}
    </>
  );

  // ===== MAIN RENDER =====

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 700, color: '#FF6B99' }}>
        Kiểm tra giao dịch
      </Typography>

      <Grid container spacing={3}>
        {/* Cột trái - Thông tin khách hàng */}
        <Grid item xs={12} md={7}>
          {renderUserSelector()}
          {renderUserProfile()}
          {renderTransactionHistory()}
        </Grid>

        {/* Cột phải - Form và kết quả */}
        <Grid item xs={12} md={5}>
          {renderTransactionForm()}
          {renderResults()}
        </Grid>
      </Grid>
    </Box>
  );
}

export default TransactionTest;

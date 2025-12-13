/**
 * ModelTraining Page - Training và đánh giá models với upload file
 * ================================================================
 * Hỗ trợ upload file CSV để train riêng từng model hoặc tất cả models
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Chip,
  LinearProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  CloudUpload as UploadIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Delete as DeleteIcon,
  Description as FileIcon,
  Help as HelpIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import {
  getModelStatus,
  trainLayer1,
  trainLayer2,
  trainAll,
  getMetrics,
  trainIsolationForest,
  trainLightGBM,
  trainAutoencoder,
  trainLSTM,
  trainGNN,
  trainAllWithData,
  // GNN Heterogeneous APIs (MỚI)
  buildGNNGraph,
  trainGNNHetero,
  getGNNStatus,
  clearGNNGraph,
} from '../services/api';

// Model configurations với hướng dẫn chi tiết
const MODEL_CONFIGS = {
  isolation_forest: {
    name: 'Isolation Forest',
    layer: 'Layer 1',
    type: 'Unsupervised',
    color: '#4caf50',
    icon: '🌲',
    description: 'Phát hiện bất thường dựa trên cách ly điểm dữ liệu trong không gian nhiều chiều',
    trainFunction: trainIsolationForest,
    dataRequirements: {
      format: 'CSV (Comma Separated Values)',
      hasLabel: false,
      minRows: 100,
      recommendedRows: '1,000 - 50,000',
      requiredColumns: [
        { name: 'amount', type: 'number', description: 'Số tiền giao dịch (VND)' },
        { name: 'hour_of_day', type: 'number (0-23)', description: 'Giờ trong ngày' },
        { name: 'day_of_week', type: 'number (0-6)', description: 'Ngày trong tuần' },
        { name: 'is_weekend', type: 'number (0/1)', description: 'Có phải cuối tuần không' },
        { name: 'velocity_1h', type: 'number', description: 'Số giao dịch trong 1 giờ' },
        { name: 'velocity_24h', type: 'number', description: 'Số giao dịch trong 24 giờ' },
      ],
      optionalColumns: [
        { name: 'amount_log', type: 'number', description: 'Log của số tiền giao dịch' },
        { name: 'amount_norm', type: 'number', description: 'Số tiền chuẩn hóa' },
        { name: 'time_gap_prev_min', type: 'number', description: 'Khoảng cách thời gian với giao dịch trước (phút)' },
        { name: 'freq_norm', type: 'number', description: 'Tần suất giao dịch chuẩn hóa' },
        { name: 'is_new_recipient', type: 'number (0/1)', description: 'Người nhận mới' },
        { name: 'recipient_count_30d', type: 'number', description: 'Số người nhận trong 30 ngày' },
        { name: 'is_new_device', type: 'number (0/1)', description: 'Thiết bị mới' },
        { name: 'device_count_30d', type: 'number', description: 'Số thiết bị trong 30 ngày' },
        { name: 'location_diff_km', type: 'number', description: 'Khoảng cách vị trí (km)' },
        { name: 'channel', type: 'string/number', description: 'Kênh giao dịch' },
        { name: 'account_age_days', type: 'number', description: 'Tuổi tài khoản (ngày)' },
        { name: 'amount_percentile_system', type: 'number', description: 'Phần trăm số tiền trong hệ thống' },
        { name: 'global_anomaly_score_prev', type: 'number', description: 'Điểm bất thường trước đó' },
        { name: 'amount_vs_avg_user_1m', type: 'number', description: '🆕 Tỷ lệ số tiền so với TB user 1 tháng' },
        { name: 'is_first_large_tx', type: 'number (0/1)', description: '🆕 Giao dịch lớn đầu tiên' },
        { name: 'recipient_is_suspicious', type: 'number (0/1)', description: '🆕 Người nhận nghi ngờ' },
      ],
      notes: [
        'KHÔNG cần cột nhãn is_fraud - đây là mô hình unsupervised',
        'Tất cả các cột phải là dữ liệu số (channel sẽ tự động chuyển đổi)',
        'Hỗ trợ đầy đủ 24 features (22 features training sau khi loại tx_id, user_id)',
        'Model sẽ tự học pattern bình thường và phát hiện bất thường',
      ],
    },
    sampleFile: 'isolation_forest_sample.csv',
  },
  lightgbm: {
    name: 'LightGBM',
    layer: 'Layer 1',
    type: 'Supervised',
    color: '#2196f3',
    icon: '🚀',
    description: 'Gradient Boosting phân loại giao dịch gian lận với độ chính xác cao',
    trainFunction: trainLightGBM,
    dataRequirements: {
      format: 'CSV (Comma Separated Values)',
      hasLabel: true,
      minRows: 500,
      recommendedRows: '5,000 - 100,000',
      requiredColumns: [
        { name: 'is_fraud', type: 'number (0/1)', description: '⚠️ BẮT BUỘC: Nhãn gian lận (0=bình thường, 1=gian lận)' },
        { name: 'amount', type: 'number', description: 'Số tiền giao dịch (VND)' },
        { name: 'hour_of_day', type: 'number (0-23)', description: 'Giờ trong ngày' },
        { name: 'velocity_1h', type: 'number', description: 'Số giao dịch trong 1 giờ' },
        { name: 'velocity_24h', type: 'number', description: 'Số giao dịch trong 24 giờ' },
      ],
      optionalColumns: [
        { name: 'is_international', type: 'number (0/1)', description: 'Giao dịch quốc tế' },
        { name: 'is_new_recipient', type: 'number (0/1)', description: 'Người nhận mới' },
        { name: 'is_new_device', type: 'number (0/1)', description: 'Thiết bị mới' },
        { name: 'is_new_location', type: 'number (0/1)', description: 'Vị trí mới' },
        { name: 'amount_deviation_ratio', type: 'number', description: 'Tỷ lệ lệch chuẩn số tiền' },
        { name: 'time_since_last_transaction_min', type: 'number', description: 'Thời gian từ giao dịch trước (phút)' },
        { name: 'login_attempts', type: 'number', description: 'Số lần đăng nhập' },
      ],
      notes: [
        'BẮT BUỘC có cột is_fraud với giá trị 0 hoặc 1',
        'Nên có tỷ lệ fraud/normal khoảng 1-10% để model học tốt',
        'Càng nhiều features càng tốt cho model này',
        'Model sẽ tự động xử lý class imbalance',
      ],
    },
    sampleFile: 'lightgbm_sample.csv',
  },
  autoencoder: {
    name: 'Autoencoder',
    layer: 'Layer 2',
    type: 'Unsupervised',
    color: '#9c27b0',
    icon: '🧠',
    description: 'Neural Network học biểu diễn hành vi người dùng và phát hiện bất thường',
    trainFunction: trainAutoencoder,
    dataRequirements: {
      format: 'CSV (Comma Separated Values)',
      hasLabel: false,
      minRows: 500,
      recommendedRows: '5,000 - 100,000',
      requiredColumns: [
        { name: 'user_id', type: 'string', description: 'ID người dùng' },
        { name: 'amount', type: 'number', description: 'Số tiền giao dịch' },
        { name: 'hour_of_day', type: 'number (0-23)', description: 'Giờ trong ngày' },
        { name: 'day_of_week', type: 'number (0-6)', description: 'Ngày trong tuần' },
      ],
      optionalColumns: [
        { name: 'velocity_1h', type: 'number', description: 'Số giao dịch trong 1 giờ' },
        { name: 'velocity_24h', type: 'number', description: 'Số giao dịch trong 24 giờ' },
        { name: 'is_international', type: 'number (0/1)', description: 'Giao dịch quốc tế' },
        { name: 'session_duration_sec', type: 'number', description: 'Thời gian phiên' },
        { name: 'avg_transaction_amount', type: 'number', description: 'Số tiền giao dịch trung bình của user' },
        { name: 'login_frequency', type: 'string', description: 'Tần suất đăng nhập' },
      ],
      notes: [
        'KHÔNG cần cột nhãn - model học pattern hành vi bình thường',
        'Mỗi user nên có ít nhất 5-10 giao dịch để model học tốt',
        'Model sẽ tạo embedding 8 chiều cho mỗi user',
        'Reconstruction error cao = hành vi bất thường',
      ],
    },
    sampleFile: 'autoencoder_sample.csv',
  },
  lstm: {
    name: 'LSTM',
    layer: 'Layer 2',
    type: 'Supervised',
    color: '#ff9800',
    icon: '📊',
    description: 'Recurrent Neural Network phân tích chuỗi giao dịch theo thời gian',
    trainFunction: trainLSTM,
    dataRequirements: {
      format: 'CSV (Comma Separated Values)',
      hasLabel: true,
      minRows: 1000,
      recommendedRows: '10,000 - 200,000',
      requiredColumns: [
        { name: 'user_id', type: 'string', description: '⚠️ BẮT BUỘC: ID người dùng (để nhóm sequence)' },
        { name: 'timestamp', type: 'datetime', description: '⚠️ BẮT BUỘC: Thời gian giao dịch (YYYY-MM-DD HH:MM:SS)' },
        { name: 'is_fraud', type: 'number (0/1)', description: '⚠️ BẮT BUỘC: Nhãn gian lận' },
        { name: 'amount', type: 'number', description: 'Số tiền giao dịch' },
      ],
      optionalColumns: [
        { name: 'transaction_type', type: 'string', description: 'Loại giao dịch (transfer, payment, etc.)' },
        { name: 'channel', type: 'string', description: 'Kênh giao dịch (mobile, web)' },
        { name: 'recipient_type', type: 'string', description: 'Loại người nhận' },
        { name: 'is_new_recipient', type: 'number (0/1)', description: 'Người nhận mới' },
        { name: 'hour_of_day', type: 'number', description: 'Giờ trong ngày' },
        { name: 'velocity_1h', type: 'number', description: 'Số giao dịch trong 1 giờ' },
      ],
      notes: [
        'Mỗi user cần có ít nhất 10 giao dịch (sequence_length = 10)',
        'Dữ liệu PHẢI được sắp xếp theo timestamp',
        'Model học pattern theo thời gian của từng user',
        'Phát hiện fraud trong ngữ cảnh lịch sử giao dịch',
      ],
    },
    sampleFile: 'lstm_sample.csv',
  },
  gnn: {
    name: 'GNN (Graph Neural Network)',
    layer: 'Layer 2',
    type: 'Supervised',
    color: '#f44336',
    icon: '🕸️',
    description: 'Graph Neural Network phân tích mạng lưới quan hệ giao dịch',
    trainFunction: trainGNN,
    dataRequirements: {
      format: 'CSV (Comma Separated Values)',
      hasLabel: true,
      minRows: 1000,
      recommendedRows: '10,000 - 500,000',
      requiredColumns: [
        { name: 'user_id', type: 'string', description: '⚠️ BẮT BUỘC: ID người gửi (node nguồn)' },
        { name: 'recipient_id', type: 'string', description: '⚠️ BẮT BUỘC: ID người nhận (node đích)' },
        { name: 'is_fraud', type: 'number (0/1)', description: '⚠️ BẮT BUỘC: Nhãn gian lận' },
        { name: 'amount', type: 'number', description: 'Số tiền giao dịch' },
      ],
      optionalColumns: [
        { name: 'recipient_type', type: 'string', description: 'Loại người nhận (individual, merchant, atm)' },
        { name: 'transaction_type', type: 'string', description: 'Loại giao dịch' },
        { name: 'is_international', type: 'number (0/1)', description: 'Giao dịch quốc tế' },
        { name: 'timestamp', type: 'datetime', description: 'Thời gian giao dịch' },
        { name: 'merchant_category', type: 'string', description: 'Danh mục merchant' },
      ],
      notes: [
        'Model xây dựng graph từ quan hệ user → recipient',
        'Phát hiện fraud dựa trên cấu trúc mạng lưới',
        'Hiệu quả với fraud rings, money laundering',
        'Cần đa dạng về các mối quan hệ giao dịch',
      ],
    },
    sampleFile: 'gnn_sample.csv',
  },
};

// GNN Heterogeneous Config (MỚI - 2 BƯỚC)
const GNN_HETERO_CONFIG = {
  name: 'GNN Heterogeneous (Edge-Level Fraud Detection)',
  layer: 'Layer 2',
  type: 'Supervised',
  color: '#f44336',
  icon: '🕸️',
  description: 'Graph Neural Network với heterogeneous graph - Phát hiện fraud trên edge (giao dịch)',
  dataRequirements: {
    format: 'Nhiều file CSV/JSON hoặc ZIP',
    hasLabel: true,
    minRows: 1000,
    recommendedRows: '10,000 - 500,000',
    requiredFiles: [
      { name: 'nodes.csv', description: 'Tất cả nodes với node_id, node_type (user/recipient/device/ip)' },
      { name: 'edges_transfer.csv', description: 'Edges chuyển tiền: edge_id, src_node_id, dst_node_id, features' },
      { name: 'edge_labels.csv', description: 'Labels cho edges: edge_id, label (0/1)' },
      { name: 'splits.csv', description: 'Train/val/test split: edge_id, split (train/val/test)' },
    ],
    optionalFiles: [
      { name: 'nodes_user.csv', description: 'Features riêng cho user nodes' },
      { name: 'nodes_recipient.csv', description: 'Features riêng cho recipient nodes' },
      { name: 'nodes_device.csv', description: 'Features riêng cho device nodes' },
      { name: 'nodes_ip.csv', description: 'Features riêng cho IP nodes' },
      { name: 'edges_uses_device.csv', description: 'Edges user → device' },
      { name: 'edges_uses_ip.csv', description: 'Edges user → IP' },
      { name: 'metadata.json', description: 'Metadata graph (optional)' },
    ],
    notes: [
      'QUAN TRỌNG: Pipeline 2 bước - Build graph trước, Train sau',
      'Hỗ trợ heterogeneous graph (nhiều loại nodes và edges)',
      'Edge-level classification: Phát hiện fraud trên từng giao dịch',
      'Có thể upload ZIP chứa toàn bộ thư mục gnn_data',
      'Sanity check tự động kiểm tra tính toàn vẹn dữ liệu',
    ],
  },
};

// Tab Panel component
function TabPanel({ children, value, index, ...other }) {
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

// Model Card Component với trạng thái trained/untrained rõ ràng
function ModelCard({ modelKey, config, status, onTrain, training, currentModel, trainedModels }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const isTraining = training && currentModel === modelKey;

  // Kiểm tra model đã trained thành công từ session hiện tại
  const isTrainedInSession = trainedModels?.includes(modelKey);
  // Kiểm tra từ status API
  const isFittedFromAPI = status?.layer1?.fitted || status?.layer2?.models?.[modelKey];
  // Model được coi là đã train nếu có trong session hoặc từ API
  const isTrained = isTrainedInSession || isFittedFromAPI;

  const handleFileSelect = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.name.endsWith('.csv')) {
      setSelectedFile(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleTrain = () => {
    if (selectedFile) {
      onTrain(modelKey, selectedFile);
    }
  };

  return (
    <Card
      sx={{
        height: '100%',
        border: `3px solid ${isTrained ? '#4caf50' : (dragOver ? config.color : '#e0e0e0')}`,
        bgcolor: isTrained ? 'rgba(76, 175, 80, 0.05)' : 'background.paper',
        transition: 'all 0.3s ease',
        '&:hover': { boxShadow: 6 },
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Badge hiển thị trạng thái train */}
      {isTrained && (
        <Box
          sx={{
            position: 'absolute',
            top: -10,
            right: 16,
            bgcolor: '#4caf50',
            color: 'white',
            px: 2,
            py: 0.5,
            borderRadius: 2,
            fontSize: '0.75rem',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            boxShadow: 2,
          }}
        >
          <CheckCircleIcon sx={{ fontSize: 16 }} />
          ĐÃ HUẤN LUYỆN
        </Box>
      )}
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" sx={{ mr: 1 }}>{config.icon}</Typography>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: isTrained ? 'success.main' : 'text.primary' }}>
              {config.name}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
              <Chip
                label={config.layer}
                size="small"
                sx={{ bgcolor: config.color, color: 'white', fontSize: '0.7rem' }}
              />
              <Chip
                label={config.type}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            </Box>
          </Box>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {config.description}
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Data Requirements Accordion */}
        <Accordion sx={{ mb: 2, boxShadow: 'none', '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <InfoIcon sx={{ mr: 1, fontSize: 20, color: 'primary.main' }} />
              <Typography variant="subtitle2" color="primary">
                Yêu cầu dữ liệu
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ px: 0 }}>
            <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Định dạng:</Typography>
                  <Typography variant="body2" fontWeight={600}>{config.dataRequirements.format}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Cần nhãn (label):</Typography>
                  <Typography variant="body2" fontWeight={600}>
                    {config.dataRequirements.hasLabel ? '✅ Có (is_fraud)' : '❌ Không'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Số dòng tối thiểu:</Typography>
                  <Typography variant="body2" fontWeight={600}>{config.dataRequirements.minRows} dòng</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">Khuyến nghị:</Typography>
                  <Typography variant="body2" fontWeight={600}>{config.dataRequirements.recommendedRows} dòng</Typography>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ mb: 1, color: 'error.main' }}>
                📋 Cột BẮT BUỘC:
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.100' }}>
                      <TableCell sx={{ fontWeight: 600 }}>Tên cột</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Kiểu dữ liệu</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>Mô tả</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {config.dataRequirements.requiredColumns.map((col) => (
                      <TableRow key={col.name}>
                        <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600, color: 'primary.main' }}>
                          {col.name}
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.75rem' }}>{col.type}</TableCell>
                        <TableCell sx={{ fontSize: '0.75rem' }}>{col.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Typography variant="subtitle2" sx={{ mb: 1, color: 'info.main' }}>
                📋 Cột TÙY CHỌN (tăng độ chính xác):
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableBody>
                    {config.dataRequirements.optionalColumns.map((col) => (
                      <TableRow key={col.name}>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {col.name}
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.7rem' }}>{col.type}</TableCell>
                        <TableCell sx={{ fontSize: '0.7rem' }}>{col.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Alert severity="info" sx={{ mb: 1 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>📝 Lưu ý quan trọng:</Typography>
                <List dense sx={{ py: 0 }}>
                  {config.dataRequirements.notes.map((note, idx) => (
                    <ListItem key={idx} sx={{ py: 0, px: 0 }}>
                      <ListItemText
                        primary={`• ${note}`}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Upload Area */}
        <Box
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          sx={{
            border: '2px dashed',
            borderColor: dragOver ? config.color : 'grey.300',
            borderRadius: 2,
            p: 2,
            textAlign: 'center',
            bgcolor: dragOver ? `${config.color}10` : 'grey.50',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            mb: 2,
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            hidden
            onChange={handleFileSelect}
          />
          {selectedFile ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
              <FileIcon color="primary" />
              <Typography variant="body2" fontWeight={600}>{selectedFile.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </Typography>
              <IconButton
                size="small"
                onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Box>
          ) : (
            <>
              <UploadIcon sx={{ fontSize: 40, color: 'grey.400', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                Kéo thả file CSV vào đây hoặc click để chọn
              </Typography>
            </>
          )}
        </Box>

        {/* Actions - Chỉ có nút Train */}
        <Button
          variant="contained"
          fullWidth
          startIcon={isTraining ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
          onClick={handleTrain}
          disabled={!selectedFile || training}
          sx={{
            bgcolor: isTrained ? '#2e7d32' : config.color,
            '&:hover': { bgcolor: isTrained ? '#1b5e20' : undefined },
            fontWeight: 600,
          }}
        >
          {isTraining ? 'Đang huấn luyện...' : (isTrained ? 'Train lại' : 'Huấn luyện Model')}
        </Button>

        {isTraining && <LinearProgress sx={{ mt: 2 }} color="success" />}
      </CardContent>
    </Card>
  );
}

// Generate sample data for download
function generateSampleData(modelKey) {
  switch (modelKey) {
    case 'isolation_forest':
      return `amount,hour_of_day,day_of_week,is_weekend,velocity_1h,velocity_24h,is_international,is_new_recipient,is_new_device,amount_deviation_ratio,session_duration_sec,login_attempts
2500000,9,6,1,1,5,0,0,0,0.45,245,1
150000,10,6,1,2,6,0,0,0,0.03,120,1
850000,11,6,1,1,3,0,0,0,0.30,380,1
45000000,2,6,1,5,8,1,1,1,3.00,45,3
320000,14,6,1,1,4,0,0,0,0.40,200,1
1200000,8,6,1,1,1,0,0,0,0.27,90,1
8500000,3,6,1,3,9,1,1,1,1.55,30,5
3500000,12,6,1,1,4,0,0,0,0.41,310,1
1800000,15,6,1,2,6,0,0,0,0.56,450,1
4200000,9,6,1,1,3,0,0,0,0.62,280,1`;

    case 'lightgbm':
      return `amount,hour_of_day,day_of_week,is_weekend,velocity_1h,velocity_24h,is_international,is_new_recipient,is_new_device,is_new_location,amount_deviation_ratio,time_since_last_transaction_min,login_attempts,is_fraud
2500000,9,6,1,1,5,0,0,0,0,0.45,1440,1,0
150000,10,6,1,2,6,0,0,0,0,0.03,75,1,0
850000,11,6,1,1,3,0,0,0,0,0.30,2880,1,0
45000000,2,6,1,5,8,1,1,1,1,3.00,15,3,1
320000,14,6,1,1,4,0,0,0,0,0.40,360,1,0
1200000,8,6,1,1,1,0,0,0,0,0.27,10080,1,0
8500000,3,6,1,3,9,1,1,1,1,1.55,285,5,1
3500000,12,6,1,1,4,0,0,0,0,0.41,720,1,0
1800000,15,6,1,2,6,0,0,0,0,0.56,180,1,0
25000000,4,6,1,1,4,1,1,1,1,8.93,1013,4,1`;

    case 'autoencoder':
      return `user_id,amount,hour_of_day,day_of_week,velocity_1h,velocity_24h,is_international,session_duration_sec,avg_transaction_amount,login_frequency
USR001,2500000,9,6,1,5,0,245,5500000,daily
USR001,150000,10,6,2,6,0,120,5500000,daily
USR002,850000,11,6,1,3,0,380,2800000,weekly
USR003,45000000,2,6,5,8,1,45,15000000,daily
USR004,320000,14,6,1,4,0,200,800000,daily
USR005,1200000,8,6,1,1,0,90,4500000,monthly
USR001,8500000,3,6,3,9,1,30,5500000,daily
USR006,3500000,12,6,1,4,0,310,8500000,daily
USR007,1800000,15,6,2,6,0,450,3200000,weekly
USR008,4200000,9,6,1,3,0,280,6800000,daily`;

    case 'lstm':
      return `user_id,timestamp,amount,transaction_type,channel,recipient_type,is_new_recipient,hour_of_day,velocity_1h,is_fraud
USR001,2024-12-07 09:15:23,2500000,transfer,mobile,individual,0,9,1,0
USR001,2024-12-07 10:30:45,150000,payment,mobile,merchant,0,10,2,0
USR001,2024-12-07 11:45:18,450000,payment,mobile,merchant,0,11,3,0
USR001,2024-12-07 03:22:11,8500000,transfer,mobile,individual,1,3,3,1
USR002,2024-12-07 11:22:18,850000,transfer,web,individual,0,11,1,0
USR002,2024-12-07 04:15:00,25000000,transfer,web,individual,1,4,1,1
USR003,2024-12-07 02:45:00,45000000,transfer,mobile,individual,1,2,5,1
USR003,2024-12-07 02:48:00,42000000,transfer,mobile,individual,1,2,6,1
USR003,2024-12-07 02:51:00,38000000,transfer,mobile,individual,1,2,7,1
USR004,2024-12-07 14:20:33,320000,payment,mobile,merchant,0,14,1,0`;

    case 'gnn':
      return `user_id,recipient_id,amount,recipient_type,transaction_type,is_international,timestamp,merchant_category,is_fraud
USR001,USR002,2500000,individual,transfer,0,2024-12-07 09:15:23,peer_transfer,0
USR001,MER001,150000,merchant,payment,0,2024-12-07 10:30:45,food_delivery,0
USR002,USR005,850000,individual,transfer,0,2024-12-07 11:22:18,peer_transfer,0
USR003,USR999,45000000,individual,transfer,1,2024-12-07 02:45:00,peer_transfer,1
USR004,MER002,320000,merchant,payment,0,2024-12-07 14:20:33,shopping,0
USR005,ATM001,1200000,atm,withdrawal,0,2024-12-07 08:10:55,atm_withdrawal,0
USR001,USR888,8500000,individual,transfer,1,2024-12-07 03:22:11,peer_transfer,1
USR006,USR003,3500000,individual,transfer,0,2024-12-07 12:45:22,peer_transfer,0
USR007,MER003,1800000,merchant,payment,0,2024-12-07 15:33:45,electronics,0
USR002,USR777,25000000,individual,transfer,1,2024-12-07 04:15:00,peer_transfer,1`;

    default:
      return '';
  }
}

// GNN Heterogeneous Card Component với 2 nút riêng biệt
function GNNHeteroCard({ onBuildGraph, onTrain, graphStatus, buildingGraph, trainingGNN }) {
  const fileInputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [graphStats, setGraphStats] = useState(null);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      setSelectedFiles(files);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const files = Array.from(event.dataTransfer.files || []);
    const validFiles = files.filter(f =>
      f.name.endsWith('.csv') || f.name.endsWith('.json') || f.name.endsWith('.zip')
    );
    if (validFiles.length > 0) {
      setSelectedFiles(validFiles);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleBuildGraph = async () => {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    selectedFiles.forEach((file, idx) => {
      formData.append(`file${idx}`, file);
    });

    const result = await onBuildGraph(formData);
    if (result?.graph_stats) {
      setGraphStats(result.graph_stats);
    }
  };

  const handleTrain = () => {
    onTrain();
  };

  const handleClearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isGraphReady = graphStatus?.graph_ready || false;

  return (
    <Card
      sx={{
        border: '3px solid #f44336',
        bgcolor: isGraphReady ? 'rgba(76, 175, 80, 0.05)' : 'background.paper',
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Badge trạng thái */}
      {isGraphReady && (
        <Box
          sx={{
            position: 'absolute',
            top: -10,
            right: 16,
            bgcolor: '#4caf50',
            color: 'white',
            px: 2,
            py: 0.5,
            borderRadius: 2,
            fontSize: '0.75rem',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            boxShadow: 2,
          }}
        >
          <CheckCircleIcon sx={{ fontSize: 16 }} />
          GRAPH ĐÃ SẴN SÀNG
        </Box>
      )}

      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" sx={{ mr: 1 }}>🕸️</Typography>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f44336' }}>
              {GNN_HETERO_CONFIG.name}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
              <Chip
                label={GNN_HETERO_CONFIG.layer}
                size="small"
                sx={{ bgcolor: '#f44336', color: 'white', fontSize: '0.7rem' }}
              />
              <Chip
                label={GNN_HETERO_CONFIG.type}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
              <Chip
                label="2 BƯỚC"
                size="small"
                sx={{ bgcolor: '#ff9800', color: 'white', fontSize: '0.7rem', fontWeight: 600 }}
              />
            </Box>
          </Box>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {GNN_HETERO_CONFIG.description}
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Data Requirements Accordion */}
        <Accordion sx={{ mb: 2, boxShadow: 'none', '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <InfoIcon sx={{ mr: 1, fontSize: 20, color: 'primary.main' }} />
              <Typography variant="subtitle2" color="primary">
                Yêu cầu dữ liệu (Nhiều files)
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ px: 0 }}>
            <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: 'error.main' }}>
                📋 Files BẮT BUỘC:
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableBody>
                    {GNN_HETERO_CONFIG.dataRequirements.requiredFiles.map((file) => (
                      <TableRow key={file.name}>
                        <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600, color: 'error.main' }}>
                          {file.name}
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.75rem' }}>{file.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Typography variant="subtitle2" sx={{ mb: 1, color: 'info.main' }}>
                📋 Files TÙY CHỌN:
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableBody>
                    {GNN_HETERO_CONFIG.dataRequirements.optionalFiles.map((file) => (
                      <TableRow key={file.name}>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {file.name}
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.7rem' }}>{file.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Alert severity="warning" sx={{ mb: 1 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>⚠️ Lưu ý quan trọng:</Typography>
                <List dense sx={{ py: 0 }}>
                  {GNN_HETERO_CONFIG.dataRequirements.notes.map((note, idx) => (
                    <ListItem key={idx} sx={{ py: 0, px: 0 }}>
                      <ListItemText
                        primary={`• ${note}`}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Alert>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Upload Area */}
        <Box
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          sx={{
            border: '2px dashed',
            borderColor: dragOver ? '#f44336' : 'grey.300',
            borderRadius: 2,
            p: 2,
            textAlign: 'center',
            bgcolor: dragOver ? 'rgba(244, 67, 54, 0.1)' : 'grey.50',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            mb: 2,
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.zip"
            multiple
            hidden
            onChange={handleFileSelect}
          />
          {selectedFiles.length > 0 ? (
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 1 }}>
                <FileIcon color="primary" />
                <Typography variant="body2" fontWeight={600}>
                  {selectedFiles.length} file(s) đã chọn
                </Typography>
                <IconButton
                  size="small"
                  onClick={(e) => { e.stopPropagation(); handleClearFiles(); }}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                {selectedFiles.map((file, idx) => (
                  <Chip
                    key={idx}
                    label={file.name}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.7rem' }}
                  />
                ))}
              </Box>
            </Box>
          ) : (
            <>
              <UploadIcon sx={{ fontSize: 40, color: 'grey.400', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                Kéo thả các file CSV/JSON hoặc file ZIP vào đây
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Hoặc click để chọn nhiều files
              </Typography>
            </>
          )}
        </Box>

        {/* Graph Stats nếu đã build */}
        {(graphStats || graphStatus?.metadata) && (
          <Alert severity="success" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">📊 Thông tin Graph:</Typography>
            <Typography variant="body2">
              Node types: {(graphStats?.node_types || graphStatus?.metadata?.node_types || []).join(', ')}
            </Typography>
            <Typography variant="body2">
              Edge types: {(graphStats?.edge_types || graphStatus?.metadata?.edge_types || []).join(', ')}
            </Typography>
          </Alert>
        )}

        {/* 2 NÚT RIÊNG BIỆT */}
        <Grid container spacing={2}>
          {/* NÚT 1: Tạo mạng lưới */}
          <Grid item xs={12} md={6}>
            <Button
              variant="contained"
              fullWidth
              size="large"
              startIcon={buildingGraph ? <CircularProgress size={20} color="inherit" /> : null}
              onClick={handleBuildGraph}
              disabled={selectedFiles.length === 0 || buildingGraph || trainingGNN}
              sx={{
                bgcolor: '#9c27b0',
                '&:hover': { bgcolor: '#7b1fa2' },
                fontWeight: 600,
                py: 1.5,
              }}
            >
              {buildingGraph ? 'Đang xây dựng...' : '🕸️ Tạo mạng lưới GNN'}
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}>
              Bước 1: Load data, sanity check, build graph
            </Typography>
          </Grid>

          {/* NÚT 2: Huấn luyện */}
          <Grid item xs={12} md={6}>
            <Button
              variant="contained"
              fullWidth
              size="large"
              startIcon={trainingGNN ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
              onClick={handleTrain}
              disabled={!isGraphReady || buildingGraph || trainingGNN}
              sx={{
                bgcolor: isGraphReady ? '#4caf50' : '#9e9e9e',
                '&:hover': { bgcolor: isGraphReady ? '#388e3c' : '#757575' },
                fontWeight: 600,
                py: 1.5,
              }}
            >
              {trainingGNN ? 'Đang huấn luyện...' : '🎯 Huấn luyện GNN'}
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}>
              Bước 2: Train model với graph đã build
            </Typography>
          </Grid>
        </Grid>

        {/* Warning nếu chưa build graph */}
        {!isGraphReady && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              ⚠️ Cần tạo mạng lưới GNN trước khi huấn luyện. Upload files và bấm "Tạo mạng lưới GNN".
            </Typography>
          </Alert>
        )}

        {(buildingGraph || trainingGNN) && <LinearProgress sx={{ mt: 2 }} color="success" />}
      </CardContent>
    </Card>
  );
}

function ModelTraining() {
  const [modelStatus, setModelStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainingModel, setTrainingModel] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [trainingResult, setTrainingResult] = useState(null); // Kết quả training chi tiết
  const [trainedModels, setTrainedModels] = useState([]); // Danh sách models đã train thành công

  // GNN Heterogeneous states (MỚI)
  const [gnnGraphStatus, setGnnGraphStatus] = useState(null); // Trạng thái graph GNN
  const [buildingGraph, setBuildingGraph] = useState(false);  // Đang build graph
  const [trainingGNN, setTrainingGNN] = useState(false);       // Đang train GNN
  const [gnnResult, setGnnResult] = useState(null);            // Kết quả GNN

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statusRes, metricsRes, gnnStatusRes] = await Promise.all([
        getModelStatus(),
        getMetrics(),
        getGNNStatus().catch(() => ({ graph_ready: false })), // GNN status
      ]);

      setModelStatus(statusRes.status);
      setMetrics(metricsRes.metrics);
      setGnnGraphStatus(gnnStatusRes); // GNN status
      setError(null);
    } catch (err) {
      setError('Không thể kết nối đến ML Service - hiển thị dữ liệu mẫu');
      // Dữ liệu mẫu
      setModelStatus({
        layer1: { fitted: true, models: ['isolation_forest', 'lightgbm'] },
        layer2: {
          fitted: true,
          models: { autoencoder: true, lstm: true, gnn: false },
        },
        weights: { layer1: 0.4, layer2: 0.6 },
      });
      setMetrics({
        isolation_forest: { precision: 0.75, recall: 0.80, f1_score: 0.77, roc_auc: 0.88 },
        lightgbm: { precision: 0.85, recall: 0.78, f1_score: 0.81, roc_auc: 0.92 },
        autoencoder: { precision: 0.72, recall: 0.75, f1_score: 0.73, roc_auc: 0.85 },
        lstm: { precision: 0.80, recall: 0.82, f1_score: 0.81, roc_auc: 0.90 },
        gnn: { precision: 0.78, recall: 0.76, f1_score: 0.77, roc_auc: 0.87 },
        ensemble: { precision: 0.83, recall: 0.80, f1_score: 0.81, roc_auc: 0.93 },
      });
      setGnnGraphStatus({ graph_ready: false });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // === GNN HETERO HANDLERS (MỚI) ===

  // Handler: Build GNN Graph (Bước 1)
  const handleBuildGNNGraph = async (formData) => {
    try {
      setBuildingGraph(true);
      setError(null);
      setSuccess(null);
      setGnnResult(null);

      console.log('[GNN] Bắt đầu build graph...');
      const result = await buildGNNGraph(formData);
      console.log('[GNN] Kết quả build graph:', result);

      if (result.success) {
        setSuccess(`✅ TẠO MẠNG LƯỚI GNN THÀNH CÔNG!\n${result.message}\nBấm "Huấn luyện GNN" để train model.`);
        setGnnGraphStatus({ graph_ready: true, metadata: result.graph_stats });
        setGnnResult(result);
        return result;
      } else {
        const errorMsg = result.sanity_errors
          ? `Lỗi dữ liệu:\n${result.sanity_errors.join('\n')}`
          : result.error;
        setError(`❌ Tạo mạng lưới thất bại: ${errorMsg}`);
        return null;
      }
    } catch (err) {
      console.error('[GNN] Lỗi build graph:', err);
      const errorMsg = err.response?.data?.error || err.message;
      const details = err.response?.data?.sanity_errors || err.response?.data?.details;
      setError(`❌ Tạo mạng lưới thất bại: ${errorMsg}${details ? '\n' + (Array.isArray(details) ? details.join('\n') : details) : ''}`);
      return null;
    } finally {
      setBuildingGraph(false);
    }
  };

  // Handler: Train GNN (Bước 2)
  const handleTrainGNN = async () => {
    try {
      setTrainingGNN(true);
      setError(null);
      setSuccess(null);
      setGnnResult(null);

      console.log('[GNN] Bắt đầu training...');
      const result = await trainGNNHetero();
      console.log('[GNN] Kết quả training:', result);

      if (result.success) {
        // Thêm vào danh sách models đã train
        setTrainedModels(prev => {
          if (!prev.includes('gnn_hetero')) {
            return [...prev, 'gnn_hetero'];
          }
          return prev;
        });

        // Format success message
        const metrics = result.metrics?.test || {};
        let successMsg = `✅ HUẤN LUYỆN GNN THÀNH CÔNG!\n`;
        successMsg += `📊 Metrics trên Test set:\n`;
        if (metrics.accuracy) successMsg += `  • Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%\n`;
        if (metrics.precision) successMsg += `  • Precision: ${(metrics.precision * 100).toFixed(2)}%\n`;
        if (metrics.recall) successMsg += `  • Recall: ${(metrics.recall * 100).toFixed(2)}%\n`;
        if (metrics.f1_score) successMsg += `  • F1-Score: ${(metrics.f1_score * 100).toFixed(2)}%\n`;
        if (metrics.roc_auc) successMsg += `  • ROC-AUC: ${(metrics.roc_auc * 100).toFixed(2)}%`;

        setSuccess(successMsg);
        setGnnResult(result);
        setTrainingResult({
          modelName: 'GNN Heterogeneous',
          modelKey: 'gnn_hetero',
          success: true,
          metrics: metrics,
          training_info: result.training_info,
          graph_info: result.graph_info,
        });

        await fetchData();
      } else {
        setError(`❌ Huấn luyện GNN thất bại: ${result.error}`);
      }
    } catch (err) {
      console.error('[GNN] Lỗi training:', err);
      const errorMsg = err.response?.data?.error || err.message;
      const hint = err.response?.data?.hint;
      setError(`❌ Huấn luyện GNN thất bại: ${errorMsg}${hint ? '\n💡 ' + hint : ''}`);
    } finally {
      setTrainingGNN(false);
    }
  };

  const handleTrainModel = async (modelKey, file) => {
    try {
      setTraining(true);
      setTrainingModel(modelKey);
      setSuccess(null);
      setError(null);
      setTrainingResult(null);

      const formData = new FormData();
      formData.append('file', file);

      const config = MODEL_CONFIGS[modelKey];
      if (config && config.trainFunction) {
        const result = await config.trainFunction(formData);
        console.log('[Training Result]', result);

        // Lưu kết quả training chi tiết
        setTrainingResult({
          modelName: config.name,
          modelKey: modelKey,
          ...result
        });

        // Tạo message thông báo chi tiết
        if (result.success) {
          // Thêm model vào danh sách đã train thành công
          setTrainedModels(prev => {
            if (!prev.includes(modelKey)) {
              return [...prev, modelKey];
            }
            return prev;
          });

          let successMsg = `✅ HUẤN LUYỆN ${config.name.toUpperCase()} THÀNH CÔNG!\n`;
          if (result.training_info) {
            successMsg += `📊 Dữ liệu: ${result.training_info.samples_count?.toLocaleString() || 'N/A'} mẫu, ${result.training_info.features_count || 'N/A'} features`;
          }
          if (result.metrics) {
            if (result.metrics.accuracy !== undefined) {
              successMsg += `\n🎯 Accuracy: ${(result.metrics.accuracy * 100).toFixed(2)}%`;
            }
            if (result.metrics.roc_auc !== undefined) {
              successMsg += ` | ROC-AUC: ${(result.metrics.roc_auc * 100).toFixed(2)}%`;
            }
            if (result.metrics.anomaly_ratio !== undefined) {
              successMsg += `\n🔍 Phát hiện ${result.metrics.detected_anomalies} bất thường (${(result.metrics.anomaly_ratio * 100).toFixed(2)}%)`;
            }
          }
          setSuccess(successMsg);
        } else {
          setError(`Huấn luyện ${config.name} thất bại: ${result.error || 'Unknown error'}`);
        }
      }

      await fetchData();
    } catch (err) {
      console.error('[Training Error]', err);
      const errorMsg = err.response?.data?.error || err.message || 'Lỗi không xác định';
      const errorDetails = err.response?.data?.details || '';
      setError(`Training ${modelKey} thất bại: ${errorMsg}`);
      setTrainingResult({
        modelKey: modelKey,
        success: false,
        error: errorMsg,
        details: errorDetails
      });
    } finally {
      setTraining(false);
      setTrainingModel(null);
    }
  };

  const handleTrainLayer = async (layer) => {
    try {
      setTraining(true);
      setTrainingModel(layer);
      setSuccess(null);
      setError(null);

      if (layer === 'layer1') {
        await trainLayer1();
      } else if (layer === 'layer2') {
        await trainLayer2();
      } else {
        await trainAll();
      }

      setSuccess(`Training ${layer} hoàn tất!`);
      await fetchData();
    } catch (err) {
      setError(`Training ${layer} thất bại: ${err.message}`);
    } finally {
      setTraining(false);
      setTrainingModel(null);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            🎯 Training Models
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Upload dữ liệu và huấn luyện từng mô hình riêng biệt
          </Typography>
        </Box>
        <Button startIcon={<RefreshIcon />} onClick={fetchData} variant="outlined">
          Refresh
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2, whiteSpace: 'pre-line' }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2, whiteSpace: 'pre-line' }}>{success}</Alert>}

      {/* Hiển thị kết quả training chi tiết */}
      {trainingResult && trainingResult.success && (
        <Card sx={{ mb: 3, border: '2px solid #4caf50' }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, color: 'success.main', display: 'flex', alignItems: 'center', gap: 1 }}>
              <CheckCircleIcon color="success" />
              Kết quả Training: {trainingResult.modelName || trainingResult.modelKey}
            </Typography>

            <Grid container spacing={2}>
              {/* Thông tin dữ liệu */}
              {trainingResult.training_info && (
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="primary" sx={{ mb: 1 }}>
                      📊 Thông tin dữ liệu đã train
                    </Typography>
                    <Typography variant="body2">
                      • Số mẫu: <strong>{trainingResult.training_info.samples_count?.toLocaleString()}</strong>
                    </Typography>
                    <Typography variant="body2">
                      • Số features: <strong>{trainingResult.training_info.features_count}</strong>
                    </Typography>
                    {trainingResult.training_info.feature_names && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        • Features: <code style={{ fontSize: '0.75rem' }}>
                          {trainingResult.training_info.feature_names.slice(0, 5).join(', ')}
                          {trainingResult.training_info.feature_names.length > 5 && ` và ${trainingResult.training_info.feature_names.length - 5} cột khác...`}
                        </code>
                      </Typography>
                    )}
                    {trainingResult.training_info.config && (
                      <>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="caption" color="text.secondary">
                          Config: n_estimators={trainingResult.training_info.config.n_estimators},
                          contamination={trainingResult.training_info.config.contamination}
                        </Typography>
                      </>
                    )}
                  </Paper>
                </Grid>
              )}

              {/* Metrics */}
              {trainingResult.metrics && (
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, bgcolor: 'success.50' }}>
                    <Typography variant="subtitle2" color="success.dark" sx={{ mb: 1 }}>
                      🎯 Độ chính xác Model
                    </Typography>
                    {trainingResult.metrics.mode === 'unsupervised' ? (
                      <>
                        <Typography variant="body2">
                          • Tổng số mẫu: <strong>{trainingResult.metrics.total_samples?.toLocaleString()}</strong>
                        </Typography>
                        <Typography variant="body2">
                          • Phát hiện bất thường: <strong style={{ color: '#d32f2f' }}>
                            {trainingResult.metrics.detected_anomalies} ({(trainingResult.metrics.anomaly_ratio * 100).toFixed(2)}%)
                          </strong>
                        </Typography>
                        <Typography variant="body2">
                          • Avg Fraud Probability: <strong>{(trainingResult.metrics.avg_fraud_probability * 100).toFixed(2)}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          • Anomaly Score: <strong>
                            {trainingResult.metrics.min_anomaly_score?.toFixed(4)} → {trainingResult.metrics.max_anomaly_score?.toFixed(4)}
                          </strong>
                        </Typography>
                      </>
                    ) : (
                      <>
                        {trainingResult.metrics.accuracy !== undefined && (
                          <Typography variant="body2">
                            • Accuracy: <strong>{(trainingResult.metrics.accuracy * 100).toFixed(2)}%</strong>
                          </Typography>
                        )}
                        {trainingResult.metrics.precision !== undefined && (
                          <Typography variant="body2">
                            • Precision: <strong>{(trainingResult.metrics.precision * 100).toFixed(2)}%</strong>
                          </Typography>
                        )}
                        {trainingResult.metrics.recall !== undefined && (
                          <Typography variant="body2">
                            • Recall: <strong>{(trainingResult.metrics.recall * 100).toFixed(2)}%</strong>
                          </Typography>
                        )}
                        {trainingResult.metrics.f1_score !== undefined && (
                          <Typography variant="body2">
                            • F1-Score: <strong>{(trainingResult.metrics.f1_score * 100).toFixed(2)}%</strong>
                          </Typography>
                        )}
                        {trainingResult.metrics.roc_auc !== undefined && (
                          <Typography variant="body2">
                            • ROC-AUC: <strong style={{ color: trainingResult.metrics.roc_auc >= 0.9 ? '#4caf50' : trainingResult.metrics.roc_auc >= 0.8 ? '#ff9800' : '#d32f2f' }}>
                              {(trainingResult.metrics.roc_auc * 100).toFixed(2)}%
                            </strong>
                          </Typography>
                        )}
                      </>
                    )}
                  </Paper>
                </Grid>
              )}

              {/* Data Summary */}
              {trainingResult.training_info?.data_summary && (
                <Grid item xs={12}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle2">📋 Chi tiết xử lý dữ liệu</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Cột trong file:</strong> {trainingResult.training_info.data_summary.columns_in_file?.join(', ')}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Cột dùng để train:</strong> {trainingResult.training_info.data_summary.columns_used_for_training?.join(', ')}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Cột bị loại bỏ:</strong> {trainingResult.training_info.data_summary.columns_excluded?.join(', ') || 'Không có'}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Giá trị missing đã fill:</strong> {trainingResult.training_info.data_summary.missing_values_filled || 0}
                      </Typography>
                    </AccordionDetails>
                  </Accordion>
                </Grid>
              )}
            </Grid>

            <Button
              size="small"
              onClick={() => setTrainingResult(null)}
              sx={{ mt: 2 }}
            >
              Đóng
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(e, v) => setTabValue(v)}
          variant="fullWidth"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="🎯 Train riêng từng Model" />
          <Tab label="⚡ Train theo Layer" />
          <Tab label="📊 Metrics & Performance" />
          <Tab label="📖 Hướng dẫn chi tiết" />
        </Tabs>
      </Paper>

      {/* Tab 0: Train Individual Models */}
      <TabPanel value={tabValue} index={0}>
        {/* GNN Heterogeneous Section (MỚI - 2 BƯỚC) */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            🕸️ GNN Heterogeneous (Pipeline 2 Bước)
            <Chip label="MỚI" size="small" color="error" sx={{ fontWeight: 600 }} />
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Pipeline chuyên nghiệp cho edge-level fraud detection với heterogeneous graph.
            Upload thư mục gnn_data (các file CSV/JSON hoặc ZIP) và thực hiện 2 bước riêng biệt.
          </Typography>
          <GNNHeteroCard
            onBuildGraph={handleBuildGNNGraph}
            onTrain={handleTrainGNN}
            graphStatus={gnnGraphStatus}
            buildingGraph={buildingGraph}
            trainingGNN={trainingGNN}
          />
        </Box>

        <Divider sx={{ my: 3 }}>
          <Chip label="Các models khác" size="small" />
        </Divider>

        <Grid container spacing={3}>
          {Object.entries(MODEL_CONFIGS).filter(([key]) => key !== 'gnn').map(([key, config]) => (
            <Grid item xs={12} md={6} key={key}>
              <ModelCard
                modelKey={key}
                config={config}
                status={modelStatus}
                onTrain={handleTrainModel}
                training={training}
                currentModel={trainingModel}
                trainedModels={trainedModels}
              />
            </Grid>
          ))}
        </Grid>

        {/* Legacy GNN (cho file CSV đơn giản) */}
        <Divider sx={{ my: 3 }}>
          <Chip label="GNN Legacy (file CSV đơn giản)" size="small" variant="outlined" />
        </Divider>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <ModelCard
              modelKey="gnn"
              config={MODEL_CONFIGS.gnn}
              status={modelStatus}
              onTrain={handleTrainModel}
              training={training}
              currentModel={trainingModel}
              trainedModels={trainedModels}
            />
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 1: Train by Layer */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                  🔷 Layer 1: Global Fraud Detection
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Bao gồm: Isolation Forest + LightGBM
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  <Chip
                    icon={modelStatus?.layer1?.fitted ? <CheckIcon /> : <CloseIcon />}
                    label="Isolation Forest"
                    color={modelStatus?.layer1?.fitted ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer1?.fitted ? <CheckIcon /> : <CloseIcon />}
                    label="LightGBM"
                    color={modelStatus?.layer1?.fitted ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={
                    training && trainingModel === 'layer1' ? (
                      <CircularProgress size={20} color="inherit" />
                    ) : (
                      <PlayIcon />
                    )
                  }
                  onClick={() => handleTrainLayer('layer1')}
                  disabled={training}
                >
                  {training && trainingModel === 'layer1' ? 'Đang training...' : 'Train Layer 1'}
                </Button>
                {training && trainingModel === 'layer1' && <LinearProgress sx={{ mt: 2 }} />}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                  🔶 Layer 2: User Profile (Advanced)
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Bao gồm: Autoencoder + LSTM + GNN
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  <Chip
                    icon={modelStatus?.layer2?.models?.autoencoder ? <CheckIcon /> : <CloseIcon />}
                    label="Autoencoder"
                    color={modelStatus?.layer2?.models?.autoencoder ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer2?.models?.lstm ? <CheckIcon /> : <CloseIcon />}
                    label="LSTM"
                    color={modelStatus?.layer2?.models?.lstm ? 'success' : 'default'}
                    size="small"
                  />
                  <Chip
                    icon={modelStatus?.layer2?.models?.gnn ? <CheckIcon /> : <CloseIcon />}
                    label="GNN"
                    color={modelStatus?.layer2?.models?.gnn ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
                <Button
                  variant="contained"
                  color="secondary"
                  fullWidth
                  startIcon={
                    training && trainingModel === 'layer2' ? (
                      <CircularProgress size={20} color="inherit" />
                    ) : (
                      <PlayIcon />
                    )
                  }
                  onClick={() => handleTrainLayer('layer2')}
                  disabled={training}
                >
                  {training && trainingModel === 'layer2' ? 'Đang training...' : 'Train Layer 2'}
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Button
                  variant="outlined"
                  size="large"
                  fullWidth
                  startIcon={
                    training && trainingModel === 'all' ? (
                      <CircularProgress size={20} />
                    ) : (
                      <PlayIcon />
                    )
                  }
                  onClick={() => handleTrainLayer('all')}
                  disabled={training}
                >
                  {training && trainingModel === 'all' ? 'Đang training tất cả...' : '🚀 Train Tất cả Models'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 2: Metrics */}
      <TabPanel value={tabValue} index={2}>
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>📊 Metrics đánh giá các Models</Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: 'grey.100' }}>
                    <TableCell sx={{ fontWeight: 600 }}>Model</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Precision</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>Recall</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>F1-Score</TableCell>
                    <TableCell align="center" sx={{ fontWeight: 600 }}>ROC-AUC</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {metrics && Object.entries(metrics).map(([name, m]) => (
                    <TableRow key={name} sx={{ '&:hover': { bgcolor: 'grey.50' } }}>
                      <TableCell component="th" scope="row">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography>
                            {MODEL_CONFIGS[name]?.icon || '📈'}
                          </Typography>
                          <Typography fontWeight={name === 'ensemble' ? 700 : 400}>
                            {name.replace(/_/g, ' ').toUpperCase()}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={`${(m.precision * 100).toFixed(1)}%`} size="small" color="primary" variant="outlined" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={`${(m.recall * 100).toFixed(1)}%`} size="small" color="secondary" variant="outlined" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip label={`${(m.f1_score * 100).toFixed(1)}%`} size="small" color="info" variant="outlined" />
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={`${(m.roc_auc * 100).toFixed(1)}%`}
                          size="small"
                          color={m.roc_auc >= 0.9 ? 'success' : m.roc_auc >= 0.8 ? 'warning' : 'error'}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </TabPanel>

      {/* Tab 3: Detailed Guide */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          {/* Overview */}
          <Grid item xs={12}>
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="subtitle1" fontWeight={600}>📖 Hướng dẫn sử dụng file mẫu có sẵn trong dự án</Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Dự án đã có sẵn 2 file dữ liệu mẫu tại thư mục: <code>ml-service/data/samples/</code>
              </Typography>
            </Alert>
          </Grid>

          {/* Sample Files Guide */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>📁 File: transactions.csv</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  40 giao dịch mẫu, bao gồm cả fraud và normal
                </Typography>
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="body2" fontWeight={600}>Dùng để train các model:</Typography>
                  <Typography variant="body2">✅ Isolation Forest</Typography>
                  <Typography variant="body2">✅ LightGBM</Typography>
                  <Typography variant="body2">✅ LSTM (có user_id + timestamp)</Typography>
                  <Typography variant="body2">✅ GNN (có user_id + recipient_id)</Typography>
                </Alert>
                <Typography variant="caption" color="text.secondary">
                  <strong>Có các cột:</strong> transaction_id, user_id, timestamp, amount, transaction_type, channel, recipient_id, recipient_type, device_id, device_type, ip_address, location_city, location_country, merchant_category, is_international, session_duration_sec, login_attempts, time_since_last_transaction_min, is_new_recipient, is_new_device, is_new_location, hour_of_day, day_of_week, is_weekend, velocity_1h, velocity_24h, amount_deviation_ratio, <strong style={{color: 'red'}}>is_fraud</strong>, fraud_type
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>📁 File: users.csv</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  20 users mẫu với thông tin profile
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2" fontWeight={600}>Dùng để train model:</Typography>
                  <Typography variant="body2">✅ Autoencoder (user behavior profile)</Typography>
                </Alert>
                <Typography variant="caption" color="text.secondary">
                  <strong>Có các cột:</strong> user_id, age, gender, occupation, income_level, account_age_days, city, region, phone_verified, email_verified, kyc_level, avg_monthly_transactions, avg_transaction_amount, preferred_channel, device_count, login_frequency, last_login_days_ago, risk_score_historical, is_premium, created_at
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Model-specific guides */}
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ mb: 2 }}>🎯 Hướng dẫn chi tiết cho từng Model</Typography>
          </Grid>

          {Object.entries(MODEL_CONFIGS).map(([key, config]) => (
            <Grid item xs={12} key={key}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography variant="h5">{config.icon}</Typography>
                    <Box>
                      <Typography variant="h6">{config.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {config.layer} | {config.type} | {config.dataRequirements.hasLabel ? 'Cần nhãn is_fraud' : 'Không cần nhãn'}
                      </Typography>
                    </Box>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                        <Typography variant="subtitle2" color="primary" sx={{ mb: 1 }}>📋 Yêu cầu cơ bản</Typography>
                        <Typography variant="body2">• Định dạng: {config.dataRequirements.format}</Typography>
                        <Typography variant="body2">• Cần nhãn: {config.dataRequirements.hasLabel ? 'Có (is_fraud)' : 'Không'}</Typography>
                        <Typography variant="body2">• Tối thiểu: {config.dataRequirements.minRows} dòng</Typography>
                        <Typography variant="body2">• Khuyến nghị: {config.dataRequirements.recommendedRows}</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, bgcolor: 'error.50' }}>
                        <Typography variant="subtitle2" color="error" sx={{ mb: 1 }}>⚠️ Cột BẮT BUỘC</Typography>
                        {config.dataRequirements.requiredColumns.map((col) => (
                          <Typography key={col.name} variant="body2">
                            • <code>{col.name}</code>: {col.description}
                          </Typography>
                        ))}
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, bgcolor: 'success.50' }}>
                        <Typography variant="subtitle2" color="success.dark" sx={{ mb: 1 }}>💡 File mẫu sử dụng</Typography>
                        {key === 'autoencoder' ? (
                          <Typography variant="body2">
                            Dùng file: <strong>users.csv</strong> hoặc kết hợp với <strong>transactions.csv</strong>
                          </Typography>
                        ) : (
                          <Typography variant="body2">
                            Dùng file: <strong>transactions.csv</strong>
                          </Typography>
                        )}
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Hoặc click nút <DownloadIcon fontSize="small" /> trên card model để tải file mẫu riêng.
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12}>
                      <Alert severity="info">
                        <Typography variant="body2">
                          <strong>Lưu ý:</strong> {config.dataRequirements.notes.join(' | ')}
                        </Typography>
                      </Alert>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          ))}
        </Grid>
      </TabPanel>
    </Box>
  );
}

export default ModelTraining;

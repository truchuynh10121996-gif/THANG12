import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormControlLabel,
  Switch,
  Stack
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import api from '../services/api';

export default function QAManagement() {
  const [qas, setQas] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingQA, setEditingQA] = useState(null);
  const [formData, setFormData] = useState({
    question: '',
    answer: '',
    language: 'vi',
    category: 'general',
    keywords: '',
    isFraudScenario: false
  });

  useEffect(() => {
    loadQAs();
  }, []);

  const loadQAs = async () => {
    try {
      const response = await api.get('/qa');
      setQas(response.data.data.qas || []);
    } catch (error) {
      console.error('Failed to load QAs:', error);
      toast.error('Không thể tải dữ liệu Q&A');
    }
  };

  const handleOpenDialog = (qa = null) => {
    if (qa) {
      setEditingQA(qa);
      setFormData({
        question: qa.question,
        answer: qa.answer,
        language: qa.language,
        category: qa.category,
        keywords: qa.keywords.join(', '),
        isFraudScenario: qa.isFraudScenario
      });
    } else {
      setEditingQA(null);
      setFormData({
        question: '',
        answer: '',
        language: 'vi',
        category: 'general',
        keywords: '',
        isFraudScenario: false
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingQA(null);
  };

  const handleSubmit = async () => {
    try {
      const data = {
        ...formData,
        keywords: formData.keywords.split(',').map(k => k.trim()).filter(k => k)
      };

      if (editingQA) {
        await api.put(`/qa/${editingQA._id}`, data);
        toast.success('Cập nhật Q&A thành công!');
      } else {
        await api.post('/qa', data);
        toast.success('Thêm Q&A thành công!');
      }

      loadQAs();
      handleCloseDialog();
    } catch (error) {
      console.error('Failed to save QA:', error);
      toast.error('Không thể lưu Q&A');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc muốn xóa Q&A này?')) return;

    try {
      await api.delete(`/qa/${id}`);
      toast.success('Xóa Q&A thành công!');
      loadQAs();
    } catch (error) {
      console.error('Failed to delete QA:', error);
      toast.error('Không thể xóa Q&A');
    }
  };

  const handleTrain = async () => {
    try {
      await api.post('/qa/train');
      toast.success('Huấn luyện chatbot thành công!');
    } catch (error) {
      console.error('Failed to train:', error);
      toast.error('Không thể huấn luyện chatbot');
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, color: '#FF8DAD' }}>
          📝 Quản lý Q&A
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => handleOpenDialog()}
            sx={{
              background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
              boxShadow: '0 4px 12px rgba(46, 125, 50, 0.3)'
            }}
          >
            Thêm Q&A mới
          </Button>
          <Button
            variant="outlined"
            onClick={handleTrain}
            sx={{ borderColor: '#FF8DAD', color: '#FF8DAD' }}
          >
            🤖 Huấn luyện Chatbot
          </Button>
        </Stack>
      </Box>

      <TableContainer component={Paper} sx={{ borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
        <Table>
          <TableHead sx={{ bgcolor: '#FF8DAD' }}>
            <TableRow>
              <TableCell sx={{ color: '#fff', fontWeight: 700 }}>Câu hỏi</TableCell>
              <TableCell sx={{ color: '#fff', fontWeight: 700 }}>Ngôn ngữ</TableCell>
              <TableCell sx={{ color: '#fff', fontWeight: 700 }}>Danh mục</TableCell>
              <TableCell sx={{ color: '#fff', fontWeight: 700 }}>Lừa đảo</TableCell>
              <TableCell sx={{ color: '#fff', fontWeight: 700 }}>Thao tác</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {qas.map((qa) => (
              <TableRow key={qa._id} hover>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {qa.question.substring(0, 80)}...
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={qa.language.toUpperCase()}
                    size="small"
                    color="primary"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={qa.category}
                    size="small"
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  {qa.isFraudScenario ? (
                    <Warning sx={{ color: '#D32F2F' }} />
                  ) : (
                    <CheckCircle sx={{ color: '#FF8DAD' }} />
                  )}
                </TableCell>
                <TableCell>
                  <IconButton
                    size="small"
                    onClick={() => handleOpenDialog(qa)}
                    sx={{ color: '#1976D2' }}
                  >
                    <Edit />
                  </IconButton>
                  <IconButton
                    size="small"
                    onClick={() => handleDelete(qa._id)}
                    sx={{ color: '#D32F2F' }}
                  >
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle sx={{ bgcolor: '#FF8DAD', color: '#fff', fontWeight: 700 }}>
          {editingQA ? 'Chỉnh sửa Q&A' : 'Thêm Q&A mới'}
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <TextField
            fullWidth
            label="Câu hỏi"
            value={formData.question}
            onChange={(e) => setFormData({ ...formData, question: e.target.value })}
            margin="normal"
            multiline
            rows={2}
          />
          <TextField
            fullWidth
            label="Câu trả lời"
            value={formData.answer}
            onChange={(e) => setFormData({ ...formData, answer: e.target.value })}
            margin="normal"
            multiline
            rows={6}
          />
          <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Ngôn ngữ</InputLabel>
              <Select
                value={formData.language}
                label="Ngôn ngữ"
                onChange={(e) => setFormData({ ...formData, language: e.target.value })}
              >
                <MenuItem value="vi">Tiếng Việt</MenuItem>
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="km">Khmer</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Danh mục</InputLabel>
              <Select
                value={formData.category}
                label="Danh mục"
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              >
                <MenuItem value="general">General</MenuItem>
                <MenuItem value="fraud-alert">Fraud Alert</MenuItem>
                <MenuItem value="otp">OTP</MenuItem>
                <MenuItem value="transfer">Transfer</MenuItem>
                <MenuItem value="phishing">Phishing</MenuItem>
                <MenuItem value="security">Security</MenuItem>
              </Select>
            </FormControl>
          </Stack>
          <TextField
            fullWidth
            label="Từ khóa (phân cách bằng dấu phẩy)"
            value={formData.keywords}
            onChange={(e) => setFormData({ ...formData, keywords: e.target.value })}
            margin="normal"
            placeholder="lừa đảo, otp, chuyển tiền..."
          />
          <FormControlLabel
            control={
              <Switch
                checked={formData.isFraudScenario}
                onChange={(e) => setFormData({ ...formData, isFraudScenario: e.target.checked })}
              />
            }
            label="Đây là kịch bản lừa đảo"
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={handleCloseDialog}>Hủy</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            sx={{ background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)' }}
          >
            {editingQA ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

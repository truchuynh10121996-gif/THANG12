import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent
} from '@mui/material';
import {
  Security,
  QuestionAnswer,
  Chat,
  Warning
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import api from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState({
    totalQA: 0,
    fraudScenarios: 0,
    conversations: 0,
    languages: 3
  });

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await api.get('/qa');
      const qas = response.data.data.qas || [];

      setStats({
        totalQA: qas.length,
        fraudScenarios: qas.filter(qa => qa.isFraudScenario).length,
        conversations: 0,
        languages: 3
      });
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const statCards = [
    {
      title: 'Tổng Q&A',
      value: stats.totalQA,
      icon: <QuestionAnswer sx={{ fontSize: 40 }} />,
      color: '#FF8DAD',
      bg: 'linear-gradient(135deg, #FF8DAD 0%, #FF6B99 100%)'
    },
    {
      title: 'Kịch bản lừa đảo',
      value: stats.fraudScenarios,
      icon: <Warning sx={{ fontSize: 40 }} />,
      color: '#D32F2F',
      bg: 'linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%)'
    },
    {
      title: 'Hội thoại',
      value: stats.conversations,
      icon: <Chat sx={{ fontSize: 40 }} />,
      color: '#1976D2',
      bg: 'linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)'
    },
    {
      title: 'Ngôn ngữ hỗ trợ',
      value: stats.languages,
      icon: <Security sx={{ fontSize: 40 }} />,
      color: '#7B1FA2',
      bg: 'linear-gradient(135deg, #7B1FA2 0%, #4A148C 100%)'
    }
  ];

  const chartData = [
    { name: 'Tiếng Việt', qas: Math.floor(stats.totalQA * 0.6) },
    { name: 'English', qas: Math.floor(stats.totalQA * 0.3) },
    { name: 'Khmer', qas: Math.floor(stats.totalQA * 0.1) }
  ];

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 700, color: '#FF8DAD' }}>
        📊 Dashboard
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                background: card.bg,
                color: '#fff',
                borderRadius: 3,
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                transition: 'transform 0.3s',
                '&:hover': {
                  transform: 'translateY(-5px)'
                }
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box>
                    <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                      {card.title}
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 700 }}>
                      {card.value}
                    </Typography>
                  </Box>
                  <Box sx={{ opacity: 0.8 }}>
                    {card.icon}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: '#FF8DAD' }}>
              Phân bố Q&A theo ngôn ngữ
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="qas" fill="#FF8DAD" name="Số lượng Q&A" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: '#FF8DAD' }}>
              Hướng dẫn sử dụng
            </Typography>
            <Box>
              <Typography variant="body2" sx={{ mb: 2 }}>
                <strong>✨ Chức năng chính:</strong>
              </Typography>
              <ul style={{ paddingLeft: 20 }}>
                <li style={{ marginBottom: 10 }}>
                  <strong>Quản lý Q&A:</strong> Thêm, sửa, xóa các kịch bản hỏi đáp
                </li>
                <li style={{ marginBottom: 10 }}>
                  <strong>Xem trước Chatbot:</strong> Test chatbot trực tiếp
                </li>
                <li style={{ marginBottom: 10 }}>
                  <strong>Huấn luyện:</strong> Cập nhật dữ liệu mới cho chatbot
                </li>
              </ul>

              <Typography variant="body2" sx={{ mt: 3, p: 2, bgcolor: '#FBD6E3', borderRadius: 2 }}>
                💡 <strong>Mẹo:</strong> Thường xuyên cập nhật các kịch bản lừa đảo mới để chatbot
                luôn có thông tin cập nhật nhất.
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

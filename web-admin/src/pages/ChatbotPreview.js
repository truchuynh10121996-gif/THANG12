import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Stack,
  Chip
} from '@mui/material';
import { Send } from '@mui/icons-material';
import api from '../services/api';
import toast from 'react-hot-toast';

export default function ChatbotPreview() {
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: 'Xin chào! Tôi là Agribank Digital Guard. Hãy mô tả tình huống bạn gặp phải.',
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await api.post('/chatbot/message', {
        message: inputText,
        conversationId,
        language: 'vi'
      });

      if (!conversationId) {
        setConversationId(response.data.data.conversationId);
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: response.data.data.response,
        sender: 'bot',
        timestamp: new Date(),
        isFraudAlert: response.data.data.isFraudAlert
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Send message error:', error);
      toast.error('Không thể gửi tin nhắn');

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Xin lỗi, tôi gặp sự cố. Vui lòng thử lại.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 700, color: '#FF8DAD' }}>
        💬 Xem trước Chatbot
      </Typography>

      <Paper
        sx={{
          height: '70vh',
          borderRadius: 3,
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}
      >
        {/* Chat Header */}
        <Box
          sx={{
            p: 2,
            background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
            color: '#fff'
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            🛡️ Digital Guard Chatbot
          </Typography>
          <Typography variant="caption">
            Test trực tiếp chatbot
          </Typography>
        </Box>

        {/* Messages */}
        <Box
          sx={{
            flex: 1,
            p: 2,
            overflowY: 'auto',
            bgcolor: '#f5f5f5'
          }}
        >
          {messages.map((message) => (
            <Box
              key={message.id}
              sx={{
                display: 'flex',
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                mb: 2
              }}
            >
              <Box
                sx={{
                  maxWidth: '70%',
                  p: 2,
                  borderRadius: 2,
                  background: message.sender === 'user'
                    ? 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)'
                    : '#fff',
                  color: message.sender === 'user' ? '#fff' : '#333',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  borderBottomLeftRadius: message.sender === 'bot' ? 0 : 2,
                  borderBottomRightRadius: message.sender === 'user' ? 0 : 2
                }}
              >
                {message.isFraudAlert && (
                  <Chip
                    label="⚠️ CẢNH BÁO LỪA ĐẢO"
                    size="small"
                    sx={{ bgcolor: '#D32F2F', color: '#fff', mb: 1 }}
                  />
                )}
                <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                  {message.text}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    mt: 0.5,
                    opacity: 0.7
                  }}
                >
                  {message.timestamp.toLocaleTimeString('vi-VN', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </Typography>
              </Box>
            </Box>
          ))}

          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: '#fff',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
              >
                <Typography variant="body2" color="textSecondary">
                  Đang suy nghĩ...
                </Typography>
              </Box>
            </Box>
          )}
        </Box>

        {/* Input */}
        <Box sx={{ p: 2, bgcolor: '#fff', borderTop: '1px solid #e0e0e0' }}>
          <Stack direction="row" spacing={2}>
            <TextField
              fullWidth
              placeholder="Nhập tin nhắn..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isLoading}
            />
            <Button
              variant="contained"
              endIcon={<Send />}
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
              sx={{
                background: 'linear-gradient(90deg, #FF8DAD 0%, #FF6B99 100%)',
                minWidth: 120
              }}
            >
              Gửi
            </Button>
          </Stack>
        </Box>
      </Paper>
    </Box>
  );
}

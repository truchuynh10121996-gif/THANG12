import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  TextField,
  IconButton,
  Typography,
  AppBar,
  Toolbar,
  Avatar,
  Chip,
  CircularProgress,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import HomeIcon from '@mui/icons-material/Home';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import WarningIcon from '@mui/icons-material/Warning';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import toast, { Toaster } from 'react-hot-toast';
import { sendMessage, synthesizeSpeech } from '../services/api';

const translations = {
  vi: {
    title: 'Agribank Digital Guard',
    subtitle: 'Trợ lý AI chống lừa đảo',
    placeholder: 'Nhập câu hỏi của bạn...',
    fraudAlert: 'Cảnh báo lừa đảo',
    errorConnection: 'Gặp sự cố kết nối. Vui lòng thử lại.',
    welcome: 'Xin chào! Tôi là trợ lý AI của Agribank. Tôi có thể giúp gì cho bạn?'
  },
  en: {
    title: 'Agribank Digital Guard',
    subtitle: 'AI Anti-Fraud Assistant',
    placeholder: 'Type your question...',
    fraudAlert: 'Fraud Alert',
    errorConnection: 'Connection error. Please try again.',
    welcome: 'Hello! I am Agribank AI assistant. How can I help you?'
  },
  km: {
    title: 'Agribank Digital Guard',
    subtitle: 'ជំនួយការ AI ប្រឆាំងការក្លែងបន្លំ',
    placeholder: 'វាយសំណួររបស់អ្នក...',
    fraudAlert: 'ការជូនដំណឹងក្លែងបន្លំ',
    errorConnection: 'បញ្ហាការតភ្ជាប់។ សូមព្យាយាមម្តងទៀត។',
    welcome: 'សួស្តី! ខ្ញុំជាជំនួយការ AI របស់ Agribank ។ តើខ្ញុំអាចជួយអ្នកបានដូចម្តេច?'
  }
};

function ChatPage() {
  const navigate = useNavigate();
  const [language, setLanguage] = useState('vi');
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);
  const [playingAudio, setPlayingAudio] = useState(null);

  const t = translations[language] || translations.vi;

  useEffect(() => {
    const savedLanguage = localStorage.getItem('selectedLanguage');
    if (savedLanguage) {
      setLanguage(savedLanguage);
    } else {
      navigate('/');
      return;
    }

    // Add welcome message
    setMessages([
      {
        id: 'welcome',
        text: t.welcome,
        isBot: true,
        timestamp: new Date()
      }
    ]);
  }, [navigate]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      text: inputText,
      isBot: false,
      timestamp: new Date()
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      const response = await sendMessage({
        message: inputText,
        conversationId,
        language
      });

      if (!conversationId) {
        setConversationId(response.conversationId);
      }

      const botMessage = {
        id: Date.now() + 1,
        text: response.response,
        isBot: true,
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert
      };

      setMessages((prev) => [...prev, botMessage]);

      if (response.isFraudAlert) {
        toast.error(t.fraudAlert, {
          icon: '⚠️',
          duration: 4000
        });
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error(t.errorConnection);

      const errorMessage = {
        id: Date.now() + 1,
        text: t.errorConnection,
        isBot: true,
        timestamp: new Date(),
        isError: true
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handlePlayAudio = async (text, messageId) => {
    try {
      if (playingAudio) {
        playingAudio.pause();
        setPlayingAudio(null);
      }

      const response = await synthesizeSpeech({
        text,
        language,
        gender: 'FEMALE'
      });

      if (response.audioContent) {
        const audio = new Audio(`data:audio/mp3;base64,${response.audioContent}`);
        audio.play();
        setPlayingAudio(audio);

        audio.onended = () => {
          setPlayingAudio(null);
        };
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      toast.error('Không thể phát âm thanh');
    }
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Toaster position="top-center" />

      {/* AppBar */}
      <AppBar
        position="static"
        sx={{
          background: 'linear-gradient(135deg, #FF8DAD 0%, #FF6B99 100%)',
          boxShadow: '0 2px 10px rgba(255, 141, 173, 0.3)'
        }}
      >
        <Toolbar>
          <Avatar
            sx={{
              bgcolor: 'white',
              color: '#FF8DAD',
              mr: 2
            }}
          >
            <SmartToyIcon />
          </Avatar>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {t.title}
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.9 }}>
              {t.subtitle}
            </Typography>
          </Box>
          <Tooltip title="Về trang chủ">
            <IconButton
              color="inherit"
              onClick={() => navigate('/home')}
            >
              <HomeIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {/* Chat Messages */}
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          background: 'linear-gradient(135deg, #FFE6F0 0%, #FFC9DD 100%)',
          padding: 2
        }}
      >
        <Container maxWidth="md">
          {messages.map((message) => (
            <Box
              key={message.id}
              sx={{
                display: 'flex',
                justifyContent: message.isBot ? 'flex-start' : 'flex-end',
                mb: 2
              }}
            >
              <Box
                sx={{
                  maxWidth: '70%',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1,
                  flexDirection: message.isBot ? 'row' : 'row-reverse'
                }}
              >
                <Avatar
                  sx={{
                    bgcolor: message.isBot ? '#FF8DAD' : '#FF6B99',
                    width: 36,
                    height: 36
                  }}
                >
                  {message.isBot ? <SmartToyIcon /> : <PersonIcon />}
                </Avatar>
                <Box>
                  <Paper
                    elevation={2}
                    sx={{
                      padding: 2,
                      borderRadius: message.isBot ? '16px 16px 16px 4px' : '16px 16px 4px 16px',
                      bgcolor: message.isBot ? 'white' : '#FF8DAD',
                      color: message.isBot ? '#333' : 'white',
                      border: message.isFraudAlert ? '2px solid #ff4444' : 'none'
                    }}
                  >
                    {message.isFraudAlert && (
                      <Chip
                        icon={<WarningIcon />}
                        label={t.fraudAlert}
                        color="error"
                        size="small"
                        sx={{ mb: 1 }}
                      />
                    )}
                    <Typography
                      variant="body1"
                      sx={{
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word'
                      }}
                    >
                      {message.text}
                    </Typography>
                    {message.isBot && !message.isError && (
                      <IconButton
                        size="small"
                        onClick={() => handlePlayAudio(message.text, message.id)}
                        sx={{
                          mt: 1,
                          color: '#FF8DAD'
                        }}
                      >
                        <VolumeUpIcon fontSize="small" />
                      </IconButton>
                    )}
                  </Paper>
                  <Typography
                    variant="caption"
                    sx={{
                      display: 'block',
                      mt: 0.5,
                      px: 1,
                      color: '#888'
                    }}
                  >
                    {message.timestamp.toLocaleTimeString('vi-VN', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </Typography>
                </Box>
              </Box>
            </Box>
          ))}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Avatar sx={{ bgcolor: '#FF8DAD', width: 36, height: 36 }}>
                  <SmartToyIcon />
                </Avatar>
                <Paper
                  elevation={2}
                  sx={{
                    padding: 2,
                    borderRadius: '16px 16px 16px 4px',
                    bgcolor: 'white'
                  }}
                >
                  <CircularProgress size={20} sx={{ color: '#FF8DAD' }} />
                </Paper>
              </Box>
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Container>
      </Box>

      {/* Input Area */}
      <Paper
        elevation={8}
        sx={{
          padding: 2,
          borderRadius: 0,
          background: 'white'
        }}
      >
        <Container maxWidth="md">
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <TextField
              fullWidth
              multiline
              maxRows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={t.placeholder}
              disabled={loading}
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 3,
                  '&.Mui-focused fieldset': {
                    borderColor: '#FF8DAD',
                    borderWidth: 2
                  }
                }
              }}
            />
            <IconButton
              color="primary"
              onClick={handleSendMessage}
              disabled={!inputText.trim() || loading}
              sx={{
                bgcolor: '#FF8DAD',
                color: 'white',
                width: 56,
                height: 56,
                '&:hover': {
                  bgcolor: '#FF6B99'
                },
                '&.Mui-disabled': {
                  bgcolor: '#FFD6E6',
                  color: 'white'
                }
              }}
            >
              <SendIcon />
            </IconButton>
          </Box>
        </Container>
      </Paper>
    </Box>
  );
}

export default ChatPage;

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
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  LinearProgress
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import HomeIcon from '@mui/icons-material/Home';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import WarningIcon from '@mui/icons-material/Warning';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import ImageIcon from '@mui/icons-material/Image';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import CloseIcon from '@mui/icons-material/Close';
import toast, { Toaster } from 'react-hot-toast';
import { sendMessage, synthesizeSpeech, analyzeImage } from '../services/api';

const translations = {
  vi: {
    title: 'Agribank Digital Guard',
    subtitle: 'Trợ lý AI chống lừa đảo',
    placeholder: 'Nhập câu hỏi của bạn...',
    fraudAlert: 'Cảnh báo lừa đảo',
    errorConnection: 'Gặp sự cố kết nối. Vui lòng thử lại.',
    welcome: 'Xin chào! Tôi là trợ lý AI của Agribank. Tôi có thể giúp gì cho bạn?',
    uploadImage: 'Tải ảnh lên',
    uploadImageTooltip: 'Chụp/tải ảnh tin nhắn để kiểm tra lừa đảo',
    analyzeImage: 'Phân tích ảnh',
    analyzing: 'Đang phân tích ảnh...',
    extractedText: 'Văn bản trích xuất',
    previewImage: 'Xem trước ảnh',
    cancel: 'Hủy',
    send: 'Gửi phân tích',
    imageError: 'Không thể đọc ảnh. Vui lòng thử lại.',
    ocrProcessing: 'Đang trích xuất văn bản từ ảnh...',
    imageUploaded: '[Ảnh chụp màn hình]'
  },
  en: {
    title: 'Agribank Digital Guard',
    subtitle: 'AI Anti-Fraud Assistant',
    placeholder: 'Type your question...',
    fraudAlert: 'Fraud Alert',
    errorConnection: 'Connection error. Please try again.',
    welcome: 'Hello! I am Agribank AI assistant. How can I help you?',
    uploadImage: 'Upload Image',
    uploadImageTooltip: 'Upload screenshot to check for fraud',
    analyzeImage: 'Analyze Image',
    analyzing: 'Analyzing image...',
    extractedText: 'Extracted Text',
    previewImage: 'Image Preview',
    cancel: 'Cancel',
    send: 'Send for Analysis',
    imageError: 'Cannot read image. Please try again.',
    ocrProcessing: 'Extracting text from image...',
    imageUploaded: '[Screenshot]'
  },
  km: {
    title: 'Agribank Digital Guard',
    subtitle: 'ជំនួយការ AI ប្រឆាំងការក្លែងបន្លំ',
    placeholder: 'វាយសំណួររបស់អ្នក...',
    fraudAlert: 'ការជូនដំណឹងក្លែងបន្លំ',
    errorConnection: 'បញ្ហាការតភ្ជាប់។ សូមព្យាយាមម្តងទៀត។',
    welcome: 'សួស្តី! ខ្ញុំជាជំនួយការ AI របស់ Agribank ។ តើខ្ញុំអាចជួយអ្នកបានដូចម្តេច?',
    uploadImage: 'ផ្ទុកឡើងរូបភាព',
    uploadImageTooltip: 'ផ្ទុកឡើងរូបថតអេក្រង់ដើម្បីពិនិត្យការក្លែងបន្លំ',
    analyzeImage: 'វិភាគរូបភាព',
    analyzing: 'កំពុងវិភាគរូបភាព...',
    extractedText: 'អត្ថបទដែលបានស្រង់ចេញ',
    previewImage: 'មើលរូបភាពជាមុន',
    cancel: 'បោះបង់',
    send: 'ផ្ញើសម្រាប់ការវិភាគ',
    imageError: 'មិនអាចអានរូបភាពបានទេ។ សូមព្យាយាមម្តងទៀត។',
    ocrProcessing: 'កំពុងស្រង់អត្ថបទពីរូបភាព...',
    imageUploaded: '[រូបថតអេក្រង់]'
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

  // Image upload states
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showImageDialog, setShowImageDialog] = useState(false);
  const [analyzingImage, setAnalyzingImage] = useState(false);
  const fileInputRef = useRef(null);

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

  // Image upload handlers
  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      // Kiểm tra loại file
      if (!file.type.startsWith('image/')) {
        toast.error(t.imageError);
        return;
      }

      // Kiểm tra kích thước (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File quá lớn. Tối đa 10MB');
        return;
      }

      setSelectedImage(file);

      // Tạo preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
        setShowImageDialog(true);
      };
      reader.readAsDataURL(file);
    }

    // Reset input để có thể chọn cùng file lại
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleImageUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleCloseImageDialog = () => {
    setShowImageDialog(false);
    setSelectedImage(null);
    setImagePreview(null);
  };

  const handleAnalyzeImage = async () => {
    if (!imagePreview) return;

    setAnalyzingImage(true);
    setShowImageDialog(false);

    // Thêm message người dùng với ảnh
    const userMessage = {
      id: Date.now(),
      text: t.imageUploaded,
      isBot: false,
      timestamp: new Date(),
      hasImage: true,
      imageUrl: imagePreview
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      toast.loading(t.ocrProcessing, { id: 'ocr-loading' });

      const response = await analyzeImage({
        imageBase64: imagePreview,
        conversationId,
        language
      });

      toast.dismiss('ocr-loading');

      if (!conversationId && response.conversationId) {
        setConversationId(response.conversationId);
      }

      const botMessage = {
        id: Date.now() + 1,
        text: response.response,
        isBot: true,
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        ocrResult: response.ocrResult
      };

      setMessages((prev) => [...prev, botMessage]);

      if (response.isFraudAlert) {
        toast.error(t.fraudAlert, {
          icon: '⚠️',
          duration: 4000
        });
      }

    } catch (error) {
      toast.dismiss('ocr-loading');
      console.error('Error analyzing image:', error);
      toast.error(t.errorConnection);

      const errorMessage = {
        id: Date.now() + 1,
        text: t.imageError,
        isBot: true,
        timestamp: new Date(),
        isError: true
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setAnalyzingImage(false);
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  // Map ngôn ngữ sang mã BCP-47 cho Web Speech API
  const getLanguageCode = (lang) => {
    const languageMap = {
      vi: 'vi-VN',
      en: 'en-US',
      km: 'km-KH'
    };
    return languageMap[lang] || 'vi-VN';
  };

  // Hàm lấy voice phù hợp với ngôn ngữ
  const getVoiceForLanguage = (langCode) => {
    const voices = window.speechSynthesis.getVoices();

    // Tìm voice phù hợp với ngôn ngữ (ví dụ: vi-VN hoặc vi)
    let voice = voices.find(v => v.lang === langCode);

    // Nếu không tìm thấy, thử tìm với mã ngôn ngữ ngắn (vi, en, km)
    if (!voice) {
      const shortLang = langCode.split('-')[0];
      voice = voices.find(v => v.lang.startsWith(shortLang));
    }

    return voice;
  };

  // Sử dụng Web Speech API (native browser TTS) làm fallback
  const speakWithNativeTTS = (text) => {
    return new Promise((resolve, reject) => {
      if (!window.speechSynthesis) {
        reject(new Error('Web Speech API not supported'));
        return;
      }

      // Dừng bất kỳ speech nào đang phát
      window.speechSynthesis.cancel();

      const langCode = getLanguageCode(language);
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = langCode;
      utterance.rate = 0.9;
      utterance.pitch = 1.0;

      // Hàm thực thi speech sau khi có voices
      const speak = () => {
        // Tìm và đặt voice phù hợp với ngôn ngữ
        const voice = getVoiceForLanguage(langCode);
        if (voice) {
          utterance.voice = voice;
        }

        utterance.onend = () => {
          setPlayingAudio(null);
          resolve();
        };

        utterance.onerror = (error) => {
          setPlayingAudio(null);
          reject(error);
        };

        // Lưu utterance để có thể dừng sau
        setPlayingAudio({ type: 'speech', utterance });
        window.speechSynthesis.speak(utterance);
      };

      // Voices có thể chưa được load, cần đợi
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        speak();
      } else {
        // Đợi voices được load
        window.speechSynthesis.onvoiceschanged = () => {
          speak();
        };
      }
    });
  };

  const handlePlayAudio = async (text, messageId) => {
    try {
      // Nếu đang phát, dừng lại
      if (playingAudio) {
        if (playingAudio.type === 'speech') {
          window.speechSynthesis.cancel();
        } else {
          playingAudio.pause();
        }
        setPlayingAudio(null);
        return;
      }

      // Thử lấy audio từ API backend trước
      try {
        const response = await synthesizeSpeech({
          text,
          language,
          gender: 'FEMALE'
        });

        if (response && response.audioContent) {
          const base64Str = response.audioContent;

          // Kiểm tra header của MP3 (ID3 tag hoặc MPEG sync word)
          // MP3 files bắt đầu với "ID3" hoặc 0xFF 0xFB/0xFA/0xF3/0xF2
          const isValidMP3 = base64Str.startsWith('SUQz') || // "ID3" in base64
                            base64Str.startsWith('//') ||    // 0xFF in base64
                            base64Str.startsWith('/+');      // 0xFF variants

          if (isValidMP3) {
            const audio = new Audio(`data:audio/mp3;base64,${base64Str}`);
            audio.play();
            setPlayingAudio(audio);

            audio.onended = () => {
              setPlayingAudio(null);
            };

            audio.onerror = () => {
              // Nếu audio fail, thử fallback
              console.log('Audio playback failed, using native TTS');
              speakWithNativeTTS(text);
            };

            return;
          }
        }

        // Nếu không hợp lệ, throw error để fallback
        throw new Error('Invalid audio response from backend');

      } catch (backendError) {
        console.log('Backend TTS failed, using native TTS fallback:', backendError.message);
      }

      // Fallback: Sử dụng Web Speech API
      await speakWithNativeTTS(text);

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
                    {/* Hiển thị ảnh nếu có */}
                    {message.hasImage && message.imageUrl && (
                      <Box sx={{ mb: 1 }}>
                        <img
                          src={message.imageUrl}
                          alt="Screenshot"
                          style={{
                            maxWidth: '100%',
                            maxHeight: 200,
                            borderRadius: 8,
                            cursor: 'pointer'
                          }}
                          onClick={() => window.open(message.imageUrl, '_blank')}
                        />
                      </Box>
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
                    {/* Hiển thị văn bản trích xuất nếu có */}
                    {message.ocrResult && message.ocrResult.extractedText && (
                      <Box
                        sx={{
                          mt: 1,
                          p: 1,
                          bgcolor: 'rgba(0,0,0,0.05)',
                          borderRadius: 1,
                          fontSize: '0.85em'
                        }}
                      >
                        <Typography variant="caption" sx={{ fontWeight: 600 }}>
                          {t.extractedText}:
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            maxHeight: 100,
                            overflow: 'auto',
                            mt: 0.5
                          }}
                        >
                          {message.ocrResult.extractedText.substring(0, 300)}
                          {message.ocrResult.extractedText.length > 300 ? '...' : ''}
                        </Typography>
                      </Box>
                    )}
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
          {(loading || analyzingImage) && (
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
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CircularProgress size={20} sx={{ color: '#FF8DAD' }} />
                    {analyzingImage && (
                      <Typography variant="body2" sx={{ color: '#666' }}>
                        {t.analyzing}
                      </Typography>
                    )}
                  </Box>
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
            {/* Hidden file input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*"
              style={{ display: 'none' }}
            />

            {/* Upload image button */}
            <Tooltip title={t.uploadImageTooltip}>
              <IconButton
                onClick={handleImageUploadClick}
                disabled={loading || analyzingImage}
                sx={{
                  bgcolor: '#FFE6F0',
                  color: '#FF8DAD',
                  width: 48,
                  height: 48,
                  '&:hover': {
                    bgcolor: '#FFD6E6'
                  },
                  '&.Mui-disabled': {
                    bgcolor: '#f5f5f5',
                    color: '#ccc'
                  }
                }}
              >
                <CameraAltIcon />
              </IconButton>
            </Tooltip>

            <TextField
              fullWidth
              multiline
              maxRows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={t.placeholder}
              disabled={loading || analyzingImage}
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
              disabled={!inputText.trim() || loading || analyzingImage}
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

      {/* Image Preview Dialog */}
      <Dialog
        open={showImageDialog}
        onClose={handleCloseImageDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ImageIcon sx={{ color: '#FF8DAD' }} />
            {t.previewImage}
          </Box>
          <IconButton onClick={handleCloseImageDialog} size="small">
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {imagePreview && (
            <Box sx={{ textAlign: 'center' }}>
              <img
                src={imagePreview}
                alt="Preview"
                style={{
                  maxWidth: '100%',
                  maxHeight: 400,
                  borderRadius: 8,
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
                }}
              />
              <Typography
                variant="body2"
                sx={{ mt: 2, color: '#666', textAlign: 'center' }}
              >
                {language === 'vi'
                  ? 'Gửi ảnh này để phân tích nội dung và kiểm tra dấu hiệu lừa đảo'
                  : 'Send this image to analyze content and check for fraud indicators'}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ padding: 2 }}>
          <Button onClick={handleCloseImageDialog} sx={{ color: '#666' }}>
            {t.cancel}
          </Button>
          <Button
            variant="contained"
            onClick={handleAnalyzeImage}
            sx={{
              bgcolor: '#FF8DAD',
              '&:hover': { bgcolor: '#FF6B99' }
            }}
          >
            {t.send}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default ChatPage;

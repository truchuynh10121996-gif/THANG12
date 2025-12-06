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
import ImageIcon from '@mui/icons-material/Image';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import CloseIcon from '@mui/icons-material/Close';
import toast, { Toaster } from 'react-hot-toast';
import { sendMessage, synthesizeSpeech, analyzeImageForFraud } from '../services/api';

const translations = {
  vi: {
    title: 'Agribank Digital Guard',
    subtitle: 'Trợ lý AI chống lừa đảo',
    placeholder: 'Nhập câu hỏi của bạn...',
    fraudAlert: 'Cảnh báo lừa đảo',
    errorConnection: 'Gặp sự cố kết nối. Vui lòng thử lại.',
    welcome: 'Xin chào! Tôi là trợ lý AI của Agribank. Tôi có thể giúp gì cho bạn?',
    uploadImage: 'Tải ảnh lên',
    analyzing: 'Đang phân tích ảnh...',
    extractedText: 'Văn bản trích xuất từ ảnh',
    imageUploaded: 'Đã tải ảnh lên để phân tích',
    noTextExtracted: 'Không thể trích xuất văn bản từ ảnh. Vui lòng thử ảnh khác.',
    imageTooLarge: 'Ảnh quá lớn. Vui lòng chọn ảnh nhỏ hơn 10MB.',
    invalidImageType: 'Định dạng ảnh không hợp lệ. Vui lòng chọn ảnh JPG, PNG hoặc GIF.'
  },
  en: {
    title: 'Agribank Digital Guard',
    subtitle: 'AI Anti-Fraud Assistant',
    placeholder: 'Type your question...',
    fraudAlert: 'Fraud Alert',
    errorConnection: 'Connection error. Please try again.',
    welcome: 'Hello! I am Agribank AI assistant. How can I help you?',
    uploadImage: 'Upload Image',
    analyzing: 'Analyzing image...',
    extractedText: 'Text extracted from image',
    imageUploaded: 'Image uploaded for analysis',
    noTextExtracted: 'Could not extract text from image. Please try another image.',
    imageTooLarge: 'Image too large. Please select an image smaller than 10MB.',
    invalidImageType: 'Invalid image format. Please select JPG, PNG or GIF.'
  },
  km: {
    title: 'Agribank Digital Guard',
    subtitle: 'ជំនួយការ AI ប្រឆាំងការក្លែងបន្លំ',
    placeholder: 'វាយសំណួររបស់អ្នក...',
    fraudAlert: 'ការជូនដំណឹងក្លែងបន្លំ',
    errorConnection: 'បញ្ហាការតភ្ជាប់។ សូមព្យាយាមម្តងទៀត។',
    welcome: 'សួស្តី! ខ្ញុំជាជំនួយការ AI របស់ Agribank ។ តើខ្ញុំអាចជួយអ្នកបានដូចម្តេច?',
    uploadImage: 'បង្ហោះរូបភាព',
    analyzing: 'កំពុងវិភាគរូបភាព...',
    extractedText: 'អត្ថបទដកស្រង់ពីរូបភាព',
    imageUploaded: 'រូបភាពត្រូវបានបង្ហោះសម្រាប់ការវិភាគ',
    noTextExtracted: 'មិនអាចដកស្រង់អត្ថបទពីរូបភាពបានទេ។ សូមសាកល្បងរូបភាពផ្សេង។',
    imageTooLarge: 'រូបភាពធំពេក។ សូមជ្រើសរើសរូបភាពតូចជាង 10MB។',
    invalidImageType: 'ទម្រង់រូបភាពមិនត្រឹមត្រូវ។ សូមជ្រើសរើស JPG, PNG ឬ GIF។'
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
  const fileInputRef = useRef(null);
  const [playingAudio, setPlayingAudio] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

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

  // Xử lý chọn ảnh
  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      toast.error(t.invalidImageType);
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      toast.error(t.imageTooLarge);
      return;
    }

    setSelectedImage(file);

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
  };

  // Xóa ảnh đã chọn
  const handleRemoveImage = () => {
    setSelectedImage(null);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
      setImagePreview(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Gửi ảnh để phân tích
  const handleSendImage = async () => {
    if (!selectedImage || loading) return;

    // Thêm tin nhắn người dùng với ảnh
    const userMessage = {
      id: Date.now(),
      text: t.imageUploaded,
      isBot: false,
      timestamp: new Date(),
      imageUrl: imagePreview
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    // Giữ lại preview URL trước khi xóa
    const currentPreview = imagePreview;

    try {
      const response = await analyzeImageForFraud({
        imageFile: selectedImage,
        language,
        conversationId
      });

      if (!conversationId) {
        setConversationId(response.conversationId);
      }

      // Tạo tin nhắn bot với kết quả phân tích
      let botText = response.response;

      // Thêm thông tin văn bản trích xuất nếu có
      if (response.extractedText && response.extractedText.length > 0) {
        botText = `📝 **${t.extractedText}:**\n"${response.extractedText.substring(0, 200)}${response.extractedText.length > 200 ? '...' : ''}"\n\n---\n\n${botText}`;
      }

      const botMessage = {
        id: Date.now() + 1,
        text: botText,
        isBot: true,
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        analysis: response.analysis
      };

      setMessages((prev) => [...prev, botMessage]);

      if (response.isFraudAlert) {
        toast.error(t.fraudAlert, {
          icon: '⚠️',
          duration: 4000
        });
      }

      // Clear image after successful upload
      handleRemoveImage();

    } catch (error) {
      console.error('Error analyzing image:', error);
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
                    {message.imageUrl && (
                      <Box sx={{ mb: 1 }}>
                        <img
                          src={message.imageUrl}
                          alt="Uploaded"
                          style={{
                            maxWidth: '100%',
                            maxHeight: '200px',
                            borderRadius: '8px',
                            objectFit: 'contain'
                          }}
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
          {/* Hidden file input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageSelect}
            accept="image/jpeg,image/jpg,image/png,image/gif,image/webp"
            style={{ display: 'none' }}
          />

          {/* Image Preview */}
          {imagePreview && (
            <Box
              sx={{
                mb: 2,
                p: 1,
                bgcolor: '#FFF5F8',
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 2
              }}
            >
              <img
                src={imagePreview}
                alt="Preview"
                style={{
                  width: 80,
                  height: 80,
                  objectFit: 'cover',
                  borderRadius: 8
                }}
              />
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  {selectedImage?.name}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {(selectedImage?.size / 1024).toFixed(1)} KB
                </Typography>
              </Box>
              <IconButton
                size="small"
                onClick={handleRemoveImage}
                sx={{ color: '#FF6B99' }}
              >
                <CloseIcon />
              </IconButton>
            </Box>
          )}

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {/* Upload Image Button */}
            <Tooltip title={t.uploadImage}>
              <IconButton
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                sx={{
                  bgcolor: '#FFF5F8',
                  color: '#FF8DAD',
                  width: 56,
                  height: 56,
                  '&:hover': {
                    bgcolor: '#FFE6F0'
                  },
                  '&.Mui-disabled': {
                    bgcolor: '#FFF5F8',
                    color: '#FFD6E6'
                  }
                }}
              >
                <PhotoCameraIcon />
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
              disabled={loading || selectedImage}
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

            {/* Send Button - changes based on whether there's an image or text */}
            <IconButton
              color="primary"
              onClick={selectedImage ? handleSendImage : handleSendMessage}
              disabled={(!inputText.trim() && !selectedImage) || loading}
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
              {selectedImage ? <ImageIcon /> : <SendIcon />}
            </IconButton>
          </Box>
        </Container>
      </Paper>
    </Box>
  );
}

export default ChatPage;

import axios from 'axios';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

// Cấu hình API base URL
// Tự động detect địa chỉ backend dựa trên platform
const getApiUrl = () => {
  // Nếu chạy trên web (Expo web), dùng localhost
  if (Platform.OS === 'web') {
    return 'http://localhost:5000/api';
  }

  // Nếu chạy trên thiết bị thật hoặc emulator
  // Lấy IP từ Expo Dev Server
  const { expoConfig } = Constants;
  const host = expoConfig?.hostUri?.split(':')[0];

  if (host) {
    return `http://${host}:5000/api`;
  }

  // Fallback về localhost (cho trường hợp production hoặc không detect được)
  return 'http://localhost:5000/api';
};

const API_BASE_URL = getApiUrl();

// Tạo axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Gửi tin nhắn đến chatbot
 */
export const sendMessage = async ({ message, conversationId, language }) => {
  try {
    const response = await api.post('/chatbot/message', {
      message,
      conversationId,
      language
    });

    return response.data.data;
  } catch (error) {
    console.error('API sendMessage error:', error);
    throw error;
  }
};

/**
 * Chuyển text thành giọng nói (TTS)
 */
export const synthesizeSpeech = async ({ text, language, gender = 'FEMALE' }) => {
  try {
    const response = await api.post('/tts/synthesize', {
      text,
      language,
      gender
    });

    return response.data.data;
  } catch (error) {
    console.error('API synthesizeSpeech error:', error);
    throw error;
  }
};

/**
 * Chuyển giọng nói thành text (STT)
 */
export const transcribeAudio = async (formData) => {
  try {
    const response = await api.post('/stt/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    return response.data.data;
  } catch (error) {
    console.error('API transcribeAudio error:', error);
    throw error;
  }
};

/**
 * Lấy lịch sử hội thoại
 */
export const getConversation = async (conversationId) => {
  try {
    const response = await api.get(`/chatbot/conversation/${conversationId}`);
    return response.data.data;
  } catch (error) {
    console.error('API getConversation error:', error);
    throw error;
  }
};

/**
 * Health check
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('API health check error:', error);
    throw error;
  }
};

/**
 * Phân tích ảnh và phát hiện lừa đảo (OCR)
 * @param {Object} params - Parameters
 * @param {string} params.imageUri - URI của ảnh
 * @param {string} params.language - Ngôn ngữ (vi, en, km)
 * @param {string} params.conversationId - ID hội thoại (optional)
 */
export const analyzeImageForFraud = async ({ imageUri, language, conversationId }) => {
  try {
    const formData = new FormData();

    // Lấy tên file từ URI
    const uriParts = imageUri.split('/');
    const fileName = uriParts[uriParts.length - 1];

    // Xác định MIME type
    const fileExtension = fileName.split('.').pop().toLowerCase();
    const mimeTypes = {
      jpg: 'image/jpeg',
      jpeg: 'image/jpeg',
      png: 'image/png',
      gif: 'image/gif',
      webp: 'image/webp',
      bmp: 'image/bmp'
    };
    const mimeType = mimeTypes[fileExtension] || 'image/jpeg';

    formData.append('image', {
      uri: imageUri,
      name: fileName,
      type: mimeType
    });
    formData.append('language', language || 'vi');
    if (conversationId) {
      formData.append('conversationId', conversationId);
    }

    const response = await api.post('/ocr/analyze-chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000 // 60 seconds timeout for OCR processing
    });

    return response.data.data;
  } catch (error) {
    console.error('API analyzeImageForFraud error:', error);
    throw error;
  }
};

/**
 * Trích xuất văn bản từ ảnh (OCR only)
 * @param {Object} params - Parameters
 * @param {string} params.imageUri - URI của ảnh
 * @param {string} params.language - Ngôn ngữ (vi, en, km)
 */
export const extractTextFromImage = async ({ imageUri, language }) => {
  try {
    const formData = new FormData();

    const uriParts = imageUri.split('/');
    const fileName = uriParts[uriParts.length - 1];
    const fileExtension = fileName.split('.').pop().toLowerCase();
    const mimeTypes = {
      jpg: 'image/jpeg',
      jpeg: 'image/jpeg',
      png: 'image/png',
      gif: 'image/gif',
      webp: 'image/webp'
    };
    const mimeType = mimeTypes[fileExtension] || 'image/jpeg';

    formData.append('image', {
      uri: imageUri,
      name: fileName,
      type: mimeType
    });
    formData.append('language', language || 'vi');

    const response = await api.post('/ocr/extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000
    });

    return response.data.data;
  } catch (error) {
    console.error('API extractTextFromImage error:', error);
    throw error;
  }
};

export default api;

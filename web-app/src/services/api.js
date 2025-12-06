import axios from 'axios';

// API Base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

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
 * @param {File} imageFile - File ảnh
 * @param {string} language - Ngôn ngữ (vi, en, km)
 * @param {string} conversationId - ID hội thoại (optional)
 */
export const analyzeImageForFraud = async ({ imageFile, language, conversationId }) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
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
 * @param {File} imageFile - File ảnh
 * @param {string} language - Ngôn ngữ (vi, en, km)
 */
export const extractTextFromImage = async ({ imageFile, language }) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
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

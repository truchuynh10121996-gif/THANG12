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

export default api;

const TextToSpeech = require('@google-cloud/text-to-speech');
const fs = require('fs');
const util = require('util');

// Khởi tạo TTS client
let ttsClient;

try {
  if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    ttsClient = new TextToSpeech.TextToSpeechClient();
  }
} catch (error) {
  console.warn('Google Cloud TTS not configured. Using fallback.');
}

/**
 * Chuyển text thành giọng nói
 * @param {string} text - Văn bản cần chuyển đổi
 * @param {string} language - Mã ngôn ngữ (vi, en, km)
 * @param {string} gender - Giới tính giọng nói (FEMALE, MALE)
 * @returns {Promise<Buffer>} - Audio buffer
 */
async function textToSpeech(text, language = 'vi', gender = 'FEMALE') {
  try {
    if (!ttsClient) {
      return createFallbackAudio(text);
    }

    // Map language codes
    const languageCodes = {
      vi: 'vi-VN',
      en: 'en-US',
      km: 'km-KH'
    };

    const languageCode = languageCodes[language] || 'vi-VN';

    // Cấu hình request
    const request = {
      input: { text },
      voice: {
        languageCode,
        ssmlGender: gender
      },
      audioConfig: {
        audioEncoding: 'MP3',
        speakingRate: 1.0,
        pitch: 0.0,
        volumeGainDb: 0.0
      }
    };

    // Gọi API
    const [response] = await ttsClient.synthesizeSpeech(request);

    return response.audioContent;

  } catch (error) {
    console.error('TTS error:', error);
    return createFallbackAudio(text);
  }
}

/**
 * Tạo audio dự phòng (placeholder)
 */
function createFallbackAudio(text) {
  // Trả về một silent audio file nhỏ hoặc thông báo lỗi
  const errorMessage = {
    error: 'TTS service not configured',
    message: 'Please configure Google Cloud TTS credentials',
    text: text
  };

  return Buffer.from(JSON.stringify(errorMessage));
}

module.exports = {
  textToSpeech
};

const speech = require('@google-cloud/speech');

// Khởi tạo STT client
let sttClient;

try {
  if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    sttClient = new speech.SpeechClient();
  }
} catch (error) {
  console.warn('Google Cloud STT not configured. Using fallback.');
}

/**
 * Chuyển giọng nói thành text
 * @param {Buffer} audioBuffer - Audio buffer
 * @param {string} language - Mã ngôn ngữ (vi, en, km)
 * @returns {Promise<object>} - Kết quả transcription
 */
async function speechToText(audioBuffer, language = 'vi') {
  try {
    if (!sttClient) {
      return {
        text: '[STT service not configured. Please add Google Cloud credentials]',
        confidence: 0
      };
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
      audio: {
        content: audioBuffer.toString('base64')
      },
      config: {
        encoding: 'WEBM_OPUS', // hoặc 'LINEAR16', 'MP3'
        sampleRateHertz: 48000,
        languageCode: languageCode,
        alternativeLanguageCodes: ['vi-VN', 'en-US', 'km-KH'],
        enableAutomaticPunctuation: true,
        model: 'default'
      }
    };

    // Gọi API
    const [response] = await sttClient.recognize(request);

    if (!response.results || response.results.length === 0) {
      return {
        text: '',
        confidence: 0
      };
    }

    const transcription = response.results
      .map(result => result.alternatives[0].transcript)
      .join('\n');

    const confidence = response.results[0].alternatives[0].confidence || 0;

    return {
      text: transcription,
      confidence: confidence
    };

  } catch (error) {
    console.error('STT error:', error);

    return {
      text: '[Error transcribing audio. Please try again]',
      confidence: 0,
      error: error.message
    };
  }
}

module.exports = {
  speechToText
};

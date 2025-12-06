const ttsService = require('../services/tts.service');

/**
 * Chuyển text thành giọng nói
 */
exports.synthesizeSpeech = async (req, res) => {
  try {
    const { text, language = 'vi', gender = 'FEMALE' } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({
        status: 'error',
        message: 'Text is required'
      });
    }

    const audioContent = await ttsService.textToSpeech(text, language, gender);

    // Trả về audio dưới dạng base64
    res.status(200).json({
      status: 'success',
      data: {
        audioContent: audioContent.toString('base64'),
        language,
        gender,
        format: 'mp3'
      }
    });

  } catch (error) {
    console.error('TTS synthesize error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to synthesize speech',
      error: error.message
    });
  }
};

/**
 * Lấy danh sách giọng nói có sẵn
 */
exports.getAvailableVoices = async (req, res) => {
  try {
    const voices = {
      vi: [
        { name: 'vi-VN-Standard-A', gender: 'FEMALE', language: 'Vietnamese' },
        { name: 'vi-VN-Standard-B', gender: 'MALE', language: 'Vietnamese' },
        { name: 'vi-VN-Standard-C', gender: 'FEMALE', language: 'Vietnamese' },
        { name: 'vi-VN-Standard-D', gender: 'MALE', language: 'Vietnamese' }
      ],
      en: [
        { name: 'en-US-Standard-A', gender: 'FEMALE', language: 'English (US)' },
        { name: 'en-US-Standard-B', gender: 'MALE', language: 'English (US)' },
        { name: 'en-US-Standard-C', gender: 'FEMALE', language: 'English (US)' },
        { name: 'en-US-Standard-D', gender: 'MALE', language: 'English (US)' }
      ],
      km: [
        { name: 'km-KH-Standard-A', gender: 'FEMALE', language: 'Khmer' },
        { name: 'km-KH-Standard-B', gender: 'MALE', language: 'Khmer' }
      ]
    };

    res.status(200).json({
      status: 'success',
      data: voices
    });

  } catch (error) {
    console.error('Get voices error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to get available voices',
      error: error.message
    });
  }
};

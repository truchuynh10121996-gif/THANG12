const sttService = require('../services/stt.service');

/**
 * Chuyển giọng nói thành text
 */
exports.transcribeAudio = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'Audio file is required'
      });
    }

    const { language = 'vi' } = req.body;
    const audioBuffer = req.file.buffer;

    // Chuyển audio thành text
    const transcription = await sttService.speechToText(audioBuffer, language);

    res.status(200).json({
      status: 'success',
      data: {
        transcription,
        language,
        confidence: transcription.confidence || 0,
        duration: req.file.size
      }
    });

  } catch (error) {
    console.error('STT transcribe error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to transcribe audio',
      error: error.message
    });
  }
};

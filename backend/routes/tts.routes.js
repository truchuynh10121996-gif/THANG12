const express = require('express');
const router = express.Router();
const ttsController = require('../controllers/tts.controller');

// POST /api/tts/synthesize - Chuyển text thành giọng nói
router.post('/synthesize', ttsController.synthesizeSpeech);

// GET /api/tts/voices - Lấy danh sách giọng nói có sẵn
router.get('/voices', ttsController.getAvailableVoices);

module.exports = router;

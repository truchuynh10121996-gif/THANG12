const express = require('express');
const router = express.Router();
const multer = require('multer');
const sttController = require('../controllers/stt.controller');

// Cấu hình multer để upload audio files
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB max
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg'];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only audio files are allowed.'));
    }
  }
});

// POST /api/stt/transcribe - Chuyển giọng nói thành text
router.post('/transcribe', upload.single('audio'), sttController.transcribeAudio);

module.exports = router;

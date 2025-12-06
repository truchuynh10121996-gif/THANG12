const express = require('express');
const router = express.Router();
const multer = require('multer');
const ocrController = require('../controllers/ocr.controller');

// Cấu hình multer để lưu file trong memory
const storage = multer.memoryStorage();

// Middleware validate file upload
const fileFilter = (req, file, cb) => {
  // Chấp nhận các định dạng ảnh phổ biến
  const allowedMimeTypes = [
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/gif',
    'image/webp',
    'image/bmp'
  ];

  if (allowedMimeTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type: ${file.mimetype}. Allowed types: JPEG, PNG, GIF, WebP, BMP`), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB max file size
  }
});

// Error handler middleware for multer
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        status: 'error',
        message: 'File too large. Maximum size is 10MB'
      });
    }
    return res.status(400).json({
      status: 'error',
      message: err.message
    });
  } else if (err) {
    return res.status(400).json({
      status: 'error',
      message: err.message
    });
  }
  next();
};

/**
 * @route   POST /api/ocr/extract
 * @desc    Trích xuất văn bản từ ảnh (OCR only)
 * @access  Public
 */
router.post('/extract', upload.single('image'), handleMulterError, ocrController.extractText);

/**
 * @route   POST /api/ocr/analyze
 * @desc    Phân tích ảnh để phát hiện lừa đảo (không dùng AI)
 * @access  Public
 */
router.post('/analyze', upload.single('image'), handleMulterError, ocrController.analyzeImage);

/**
 * @route   POST /api/ocr/analyze-chat
 * @desc    Phân tích ảnh và gửi đến chatbot để phát hiện lừa đảo (Endpoint chính)
 * @access  Public
 *
 * Request body (multipart/form-data):
 * - image: File ảnh (JPEG, PNG, GIF, WebP, BMP)
 * - language: Ngôn ngữ (vi, en, km) - default: vi
 * - conversationId: ID hội thoại (optional)
 *
 * Response:
 * - conversationId: ID hội thoại
 * - extractedText: Văn bản trích xuất từ ảnh
 * - confidence: Độ tin cậy OCR (0-100)
 * - analysis: Phân tích lừa đảo
 *   - indicators: Các dấu hiệu phát hiện
 *   - riskScore: Điểm rủi ro (0-100)
 *   - riskLevel: Mức độ rủi ro (safe/low/medium/high)
 *   - isFraudSuspected: true/false
 * - response: Phản hồi từ AI chatbot
 * - isFraudAlert: true/false
 */
router.post('/analyze-chat', upload.single('image'), handleMulterError, ocrController.analyzeAndChat);

module.exports = router;

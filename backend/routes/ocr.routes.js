/**
 * OCR Routes - Các endpoint liên quan đến trích xuất văn bản từ ảnh
 */

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const ocrController = require('../controllers/ocr.controller');

// Đảm bảo thư mục uploads tồn tại
const uploadsDir = path.join(__dirname, '../uploads/ocr');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Cấu hình multer cho upload ảnh
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, `ocr-${uniqueSuffix}${ext}`);
  }
});

const fileFilter = (req, file, cb) => {
  // Chỉ chấp nhận file ảnh
  const allowedMimes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'];
  if (allowedMimes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Chỉ chấp nhận file ảnh (JPEG, PNG, GIF, WebP, BMP)'), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 // Giới hạn 10MB
  }
});

// Middleware xử lý lỗi multer
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: 'File quá lớn. Kích thước tối đa là 10MB'
      });
    }
    return res.status(400).json({
      success: false,
      error: `Lỗi upload: ${err.message}`
    });
  } else if (err) {
    return res.status(400).json({
      success: false,
      error: err.message
    });
  }
  next();
};

/**
 * POST /api/ocr/extract
 * Trích xuất văn bản từ ảnh upload
 * Body: multipart/form-data với field 'image' và optional 'language'
 */
router.post('/extract', upload.single('image'), handleMulterError, ocrController.extractText);

/**
 * POST /api/ocr/analyze
 * Trích xuất văn bản và phân tích lừa đảo
 * Body: multipart/form-data với field 'image' và optional 'language'
 */
router.post('/analyze', upload.single('image'), handleMulterError, ocrController.analyzeImage);

/**
 * POST /api/ocr/extract-base64
 * Trích xuất văn bản từ ảnh dạng base64
 * Body: { imageBase64: string, language?: string }
 */
router.post('/extract-base64', ocrController.extractTextFromBase64);

/**
 * POST /api/ocr/analyze-base64
 * Phân tích lừa đảo từ ảnh dạng base64
 * Body: { imageBase64: string, language?: string }
 */
router.post('/analyze-base64', ocrController.analyzeImageFromBase64);

/**
 * POST /api/ocr/analyze-text
 * Phân tích văn bản thuần (không cần OCR)
 * Body: { text: string, language?: string }
 */
router.post('/analyze-text', ocrController.analyzeText);

module.exports = router;

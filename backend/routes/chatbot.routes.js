const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const chatbotController = require('../controllers/chatbot.controller');

// Đảm bảo thư mục uploads tồn tại
const uploadsDir = path.join(__dirname, '../uploads/chatbot');
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
    cb(null, `chatbot-${uniqueSuffix}${ext}`);
  }
});

const fileFilter = (req, file, cb) => {
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
    fileSize: 10 * 1024 * 1024 // 10MB
  }
});

// Middleware xử lý lỗi multer
const handleMulterError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        status: 'error',
        message: 'File quá lớn. Kích thước tối đa là 10MB'
      });
    }
    return res.status(400).json({
      status: 'error',
      message: `Lỗi upload: ${err.message}`
    });
  } else if (err) {
    return res.status(400).json({
      status: 'error',
      message: err.message
    });
  }
  next();
};

// POST /api/chatbot/message - Gửi tin nhắn đến chatbot
router.post('/message', chatbotController.sendMessage);

// GET /api/chatbot/conversation/:conversationId - Lấy lịch sử hội thoại
router.get('/conversation/:conversationId', chatbotController.getConversation);

// DELETE /api/chatbot/conversation/:conversationId - Xóa hội thoại
router.delete('/conversation/:conversationId', chatbotController.deleteConversation);

// POST /api/chatbot/detect-language - Phát hiện ngôn ngữ
router.post('/detect-language', chatbotController.detectLanguage);

// POST /api/chatbot/analyze-image - Phân tích ảnh (base64)
router.post('/analyze-image', chatbotController.analyzeImage);

// POST /api/chatbot/analyze-image-upload - Phân tích ảnh (multipart/form-data)
router.post('/analyze-image-upload', upload.single('image'), handleMulterError, chatbotController.analyzeImageUpload);

module.exports = router;

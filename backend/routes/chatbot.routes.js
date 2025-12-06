const express = require('express');
const router = express.Router();
const chatbotController = require('../controllers/chatbot.controller');

// POST /api/chatbot/message - Gửi tin nhắn đến chatbot
router.post('/message', chatbotController.sendMessage);

// GET /api/chatbot/conversation/:conversationId - Lấy lịch sử hội thoại
router.get('/conversation/:conversationId', chatbotController.getConversation);

// DELETE /api/chatbot/conversation/:conversationId - Xóa hội thoại
router.delete('/conversation/:conversationId', chatbotController.deleteConversation);

// POST /api/chatbot/detect-language - Phát hiện ngôn ngữ
router.post('/detect-language', chatbotController.detectLanguage);

module.exports = router;

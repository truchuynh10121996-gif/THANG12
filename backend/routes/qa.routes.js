const express = require('express');
const router = express.Router();
const qaController = require('../controllers/qa.controller');

// GET /api/qa - Lấy tất cả Q&A scenarios
router.get('/', qaController.getAllQA);

// GET /api/qa/:id - Lấy một Q&A scenario theo ID
router.get('/:id', qaController.getQAById);

// POST /api/qa - Tạo Q&A scenario mới
router.post('/', qaController.createQA);

// PUT /api/qa/:id - Cập nhật Q&A scenario
router.put('/:id', qaController.updateQA);

// DELETE /api/qa/:id - Xóa Q&A scenario
router.delete('/:id', qaController.deleteQA);

// POST /api/qa/train - Huấn luyện lại chatbot với dữ liệu mới
router.post('/train', qaController.trainChatbot);

// GET /api/qa/export - Export Q&A scenarios
router.get('/export/data', qaController.exportQA);

// POST /api/qa/import - Import Q&A scenarios
router.post('/import', qaController.importQA);

module.exports = router;

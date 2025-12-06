const ocrService = require('../services/ocr.service');
const geminiService = require('../services/gemini.service');
const qaService = require('../services/qa.service');
const conversationService = require('../services/conversation.service');

/**
 * Trích xuất văn bản từ ảnh (OCR only)
 */
exports.extractText = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'No image file provided'
      });
    }

    const { language = 'vi' } = req.body;

    // Trích xuất văn bản từ ảnh
    const result = await ocrService.extractText(req.file.buffer, language);

    res.status(200).json({
      status: 'success',
      data: {
        text: result.text,
        confidence: result.confidence,
        wordCount: result.words
      }
    });

  } catch (error) {
    console.error('OCR extractText error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to extract text from image',
      error: error.message
    });
  }
};

/**
 * Phân tích ảnh để phát hiện lừa đảo
 */
exports.analyzeImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'No image file provided'
      });
    }

    const { language = 'vi' } = req.body;

    // Xử lý ảnh và phân tích
    const result = await ocrService.processImageForFraud(req.file.buffer, language);

    res.status(200).json({
      status: 'success',
      data: result
    });

  } catch (error) {
    console.error('OCR analyzeImage error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to analyze image',
      error: error.message
    });
  }
};

/**
 * Phân tích ảnh và gửi đến chatbot để phát hiện lừa đảo
 * Đây là endpoint chính để tích hợp với chatbot
 */
exports.analyzeAndChat = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'No image file provided'
      });
    }

    const { language = 'vi', conversationId } = req.body;

    // Bước 1: Xử lý ảnh và phân tích
    console.log('Processing image for OCR...');
    const ocrResult = await ocrService.processImageForFraud(req.file.buffer, language);

    if (!ocrResult.success) {
      return res.status(400).json({
        status: 'error',
        message: 'Failed to process image',
        data: ocrResult
      });
    }

    // Nếu không trích xuất được văn bản
    if (!ocrResult.extractedText || ocrResult.extractedText.length < 10) {
      return res.status(200).json({
        status: 'success',
        data: {
          conversationId: conversationId || generateConversationId(),
          extractedText: ocrResult.extractedText || '',
          confidence: ocrResult.confidence,
          analysis: ocrResult.analysis,
          response: getNoTextResponse(language),
          isFraudAlert: false,
          timestamp: new Date().toISOString()
        }
      });
    }

    // Bước 2: Tạo prompt cho AI
    const aiPrompt = ocrService.generateAIPrompt(
      ocrResult.extractedText,
      ocrResult.analysis,
      language
    );

    // Bước 3: Lấy lịch sử hội thoại nếu có
    let conversationHistory = [];
    if (conversationId) {
      const existingConversation = await conversationService.getConversation(conversationId);
      if (existingConversation && existingConversation.messages) {
        conversationHistory = existingConversation.messages;
      }
    }

    // Bước 4: Lấy context từ Q&A scenarios
    const qaContext = await qaService.getRelevantQA(ocrResult.extractedText);

    // Bước 5: Tạo system prompt
    const systemPrompt = generateSystemPrompt(language, qaContext);

    // Bước 6: Gọi Gemini AI
    console.log('Calling Gemini AI for analysis...');
    const aiResponse = await geminiService.generateResponse(aiPrompt, systemPrompt, conversationHistory);

    // Bước 7: Lưu hội thoại
    const newConversationId = conversationId || generateConversationId();
    const conversation = await conversationService.saveMessage({
      conversationId: newConversationId,
      userMessage: `[Phân tích ảnh]\n${ocrResult.extractedText}`,
      botResponse: aiResponse,
      language,
      timestamp: new Date()
    });

    // Bước 8: Trả về kết quả
    res.status(200).json({
      status: 'success',
      data: {
        conversationId: conversation.conversationId,
        extractedText: ocrResult.extractedText,
        confidence: ocrResult.confidence,
        wordCount: ocrResult.wordCount,
        analysis: ocrResult.analysis,
        response: aiResponse,
        isFraudAlert: ocrResult.analysis.isFraudSuspected || checkFraudKeywords(aiResponse),
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('OCR analyzeAndChat error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to analyze image and generate response',
      error: error.message
    });
  }
};

// Helper Functions

function generateConversationId() {
  return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getNoTextResponse(language) {
  const responses = {
    vi: 'Xin lỗi, tôi không thể trích xuất văn bản từ ảnh này. Vui lòng thử lại với ảnh rõ hơn, hoặc bạn có thể nhập trực tiếp nội dung tin nhắn cần kiểm tra.',
    en: 'Sorry, I could not extract text from this image. Please try again with a clearer image, or you can directly type the message content to check.',
    km: 'សូមអភ័យទោស ខ្ញុំមិនអាចទាញយកអក្សរពីរូបភាពនេះបានទេ។ សូមព្យាយាមម្តងទៀតជាមួយរូបភាពច្បាស់ជាងនេះ ឬអ្នកអាចវាយផ្ org org org org org org org org org org org org org org org org org org org org។'
  };
  return responses[language] || responses.vi;
}

function generateSystemPrompt(language, qaContext) {
  return `Bạn là Chatbot cảnh báo lừa đảo cho khách hàng ngân hàng Agribank.

NHIỆM VỤ:
Phân tích nội dung tin nhắn/email mà người dùng gửi (đã trích xuất từ ảnh chụp màn hình), phát hiện dấu hiệu rủi ro, cảnh báo rõ ràng và đưa 3 bước xử lý cụ thể.

PHONG CÁCH:
Ngắn gọn, dứt khoát, thân thiện, ưu tiên an toàn.

DỮ LIỆU THAM KHẢO TỪ HỆ THỐNG:
${qaContext}

NGUYÊN TẮC:
- Phân tích kỹ nội dung được trích xuất từ ảnh
- Khi phát hiện từ khóa nghi ngờ (OTP, link lạ, đóng phí trước, dọa khóa tài khoản, dọa khóa thẻ, app từ xa, cuộc gọi công an, cuộc gọi cơ quan chức năng, cuộc gọi từ nước ngoài, cuộc gọi chuyển tiền cho người lạ), hãy kết luận rủi ro và trả lời theo mẫu:
  1) Việc KHÔNG làm.
  2) Việc CẦN làm ngay.
  3) "Trong trường hợp khẩn cấp nếu bạn đã cung cấp thông tin hoặc đã chuyển tiền, vui lòng liên hệ tổng đài khẩn cấp 1900558818 của Agribank để được hỗ trợ"
- Nếu nội dung không có dấu hiệu lừa đảo, hãy xác nhận đây là nội dung an toàn
- Luôn cung cấp số điện thoại khẩn cấp của Agribank: 1900558818

QUY TẮC NGÔN NGỮ:
1) Trả lời bằng TIẾNG VIỆT nếu language = 'vi'
2) Trả lời bằng TIẾNG ANH nếu language = 'en'
3) Trả lời SONG NGỮ (Khmer + Tiếng Việt) nếu language = 'km'

HÃY TRẢ LỜI THEO NGÔN NGỮ MÀ NGƯỜI DÙNG SỬ DỤNG (${language}).`;
}

function checkFraudKeywords(response) {
  const fraudKeywords = [
    'lừa đảo', 'cảnh báo', 'nguy hiểm', 'fraud', 'alert', 'warning',
    'scam', 'suspicious', 'không tin tưởng', 'đừng', 'stop', 'rủi ro',
    'nghi ngờ', 'không nên', 'cẩn thận', 'không cung cấp', 'không click'
  ];

  const lowerResponse = response.toLowerCase();
  return fraudKeywords.some(keyword => lowerResponse.includes(keyword));
}

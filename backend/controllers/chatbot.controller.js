const geminiService = require('../services/gemini.service');
const qaService = require('../services/qa.service');
const conversationService = require('../services/conversation.service');
const ocrService = require('../services/ocr.service');

/**
 * Gửi tin nhắn đến chatbot và nhận phản hồi
 */
exports.sendMessage = async (req, res) => {
  try {
    const { message, conversationId, language } = req.body;

    if (!message || !message.trim()) {
      return res.status(400).json({
        status: 'error',
        message: 'Message is required'
      });
    }

    // Phát hiện ngôn ngữ nếu không được cung cấp
    const detectedLanguage = language || await detectLanguageFromText(message);

    // Lấy lịch sử hội thoại nếu có conversationId
    let conversationHistory = [];
    if (conversationId) {
      const existingConversation = await conversationService.getConversation(conversationId);
      if (existingConversation && existingConversation.messages) {
        conversationHistory = existingConversation.messages;
      }
    }

    // Lấy context từ Q&A scenarios
    const qaContext = await qaService.getRelevantQA(message);

    // Tạo prompt với context
    const systemPrompt = generateSystemPrompt(detectedLanguage, qaContext);

    // Gọi Gemini API với lịch sử hội thoại
    const response = await geminiService.generateResponse(message, systemPrompt, conversationHistory);

    // Lưu hội thoại
    const conversation = await conversationService.saveMessage({
      conversationId: conversationId || generateConversationId(),
      userMessage: message,
      botResponse: response,
      language: detectedLanguage,
      timestamp: new Date()
    });

    res.status(200).json({
      status: 'success',
      data: {
        conversationId: conversation.conversationId,
        response: response,
        language: detectedLanguage,
        isFraudAlert: checkFraudKeywords(response),
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Chatbot sendMessage error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to process message',
      error: error.message
    });
  }
};

/**
 * Lấy lịch sử hội thoại
 */
exports.getConversation = async (req, res) => {
  try {
    const { conversationId } = req.params;
    const conversation = await conversationService.getConversation(conversationId);

    if (!conversation) {
      return res.status(404).json({
        status: 'error',
        message: 'Conversation not found'
      });
    }

    res.status(200).json({
      status: 'success',
      data: conversation
    });

  } catch (error) {
    console.error('Get conversation error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to get conversation',
      error: error.message
    });
  }
};

/**
 * Xóa hội thoại
 */
exports.deleteConversation = async (req, res) => {
  try {
    const { conversationId } = req.params;
    await conversationService.deleteConversation(conversationId);

    res.status(200).json({
      status: 'success',
      message: 'Conversation deleted successfully'
    });

  } catch (error) {
    console.error('Delete conversation error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to delete conversation',
      error: error.message
    });
  }
};

/**
 * Phát hiện ngôn ngữ
 */
exports.detectLanguage = async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        status: 'error',
        message: 'Text is required'
      });
    }

    const language = await detectLanguageFromText(text);

    res.status(200).json({
      status: 'success',
      data: { language }
    });

  } catch (error) {
    console.error('Detect language error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to detect language',
      error: error.message
    });
  }
};

/**
 * Xử lý ảnh chụp màn hình tin nhắn/email để phát hiện lừa đảo
 * POST /api/chatbot/analyze-image
 */
exports.analyzeImage = async (req, res) => {
  try {
    const { imageBase64, conversationId, language = 'vi' } = req.body;

    if (!imageBase64) {
      return res.status(400).json({
        status: 'error',
        message: 'Image is required (base64 format)'
      });
    }

    console.log('[Chatbot] Nhận yêu cầu phân tích ảnh');

    // Bước 1: Chuyển base64 thành buffer và xử lý OCR
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    const ocrResult = await ocrService.processImageForFraudDetection(imageBuffer, language);

    if (!ocrResult.success) {
      return res.status(400).json({
        status: 'error',
        message: 'Không thể trích xuất văn bản từ ảnh. Vui lòng thử lại với ảnh rõ hơn.',
        ocrError: ocrResult.error
      });
    }

    // Bước 2: Lấy lịch sử hội thoại nếu có
    let conversationHistory = [];
    if (conversationId) {
      const existingConversation = await conversationService.getConversation(conversationId);
      if (existingConversation && existingConversation.messages) {
        conversationHistory = existingConversation.messages;
      }
    }

    // Bước 3: Tạo message cho chatbot với context từ OCR
    const userMessage = generateImageAnalysisMessage(ocrResult, language);

    // Bước 4: Lấy context từ Q&A scenarios
    const qaContext = await qaService.getRelevantQA(ocrResult.extractedText);

    // Bước 5: Tạo system prompt đặc biệt cho phân tích ảnh
    const systemPrompt = generateImageAnalysisPrompt(language, qaContext, ocrResult);

    // Bước 6: Gọi Gemini API
    const botResponse = await geminiService.generateResponse(userMessage, systemPrompt, conversationHistory);

    // Bước 7: Lưu hội thoại
    const conversation = await conversationService.saveMessage({
      conversationId: conversationId || generateConversationId(),
      userMessage: `[Ảnh chụp màn hình]\n${ocrResult.extractedText.substring(0, 500)}...`,
      botResponse: botResponse,
      language: language,
      timestamp: new Date()
    });

    // Trả về kết quả
    res.status(200).json({
      status: 'success',
      data: {
        conversationId: conversation.conversationId,
        response: botResponse,
        language: language,
        isFraudAlert: ocrResult.fraudAnalysis.isFraudulent || checkFraudKeywords(botResponse),
        ocrResult: {
          extractedText: ocrResult.extractedText,
          confidence: ocrResult.confidence,
          fraudAnalysis: ocrResult.fraudAnalysis
        },
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Chatbot analyzeImage error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to analyze image',
      error: error.message
    });
  }
};

/**
 * Xử lý ảnh upload (multipart/form-data)
 * POST /api/chatbot/analyze-image-upload
 */
exports.analyzeImageUpload = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'Vui lòng upload một ảnh'
      });
    }

    const { conversationId, language = 'vi' } = req.body;
    const imagePath = req.file.path;

    console.log(`[Chatbot] Nhận yêu cầu phân tích ảnh upload: ${req.file.originalname}`);

    // Bước 1: Xử lý OCR
    const ocrResult = await ocrService.processImageForFraudDetection(imagePath, language);

    // Xóa file tạm sau khi xử lý
    ocrService.cleanupTempFile(imagePath);

    if (!ocrResult.success) {
      return res.status(400).json({
        status: 'error',
        message: 'Không thể trích xuất văn bản từ ảnh. Vui lòng thử lại với ảnh rõ hơn.',
        ocrError: ocrResult.error
      });
    }

    // Bước 2: Lấy lịch sử hội thoại nếu có
    let conversationHistory = [];
    if (conversationId) {
      const existingConversation = await conversationService.getConversation(conversationId);
      if (existingConversation && existingConversation.messages) {
        conversationHistory = existingConversation.messages;
      }
    }

    // Bước 3: Tạo message cho chatbot với context từ OCR
    const userMessage = generateImageAnalysisMessage(ocrResult, language);

    // Bước 4: Lấy context từ Q&A scenarios
    const qaContext = await qaService.getRelevantQA(ocrResult.extractedText);

    // Bước 5: Tạo system prompt đặc biệt cho phân tích ảnh
    const systemPrompt = generateImageAnalysisPrompt(language, qaContext, ocrResult);

    // Bước 6: Gọi Gemini API
    const botResponse = await geminiService.generateResponse(userMessage, systemPrompt, conversationHistory);

    // Bước 7: Lưu hội thoại
    const conversation = await conversationService.saveMessage({
      conversationId: conversationId || generateConversationId(),
      userMessage: `[Ảnh chụp màn hình]\n${ocrResult.extractedText.substring(0, 500)}...`,
      botResponse: botResponse,
      language: language,
      timestamp: new Date()
    });

    // Trả về kết quả
    res.status(200).json({
      status: 'success',
      data: {
        conversationId: conversation.conversationId,
        response: botResponse,
        language: language,
        isFraudAlert: ocrResult.fraudAnalysis.isFraudulent || checkFraudKeywords(botResponse),
        ocrResult: {
          extractedText: ocrResult.extractedText,
          confidence: ocrResult.confidence,
          fraudAnalysis: ocrResult.fraudAnalysis
        },
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Chatbot analyzeImageUpload error:', error);

    // Xóa file tạm nếu có lỗi
    if (req.file?.path) {
      ocrService.cleanupTempFile(req.file.path);
    }

    res.status(500).json({
      status: 'error',
      message: 'Failed to analyze image',
      error: error.message
    });
  }
};

// Helper Functions

function generateConversationId() {
  return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

async function detectLanguageFromText(text) {
  // Phát hiện ngôn ngữ đơn giản dựa trên ký tự và từ khóa
  const vietnameseChars = /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]/i;
  const khmerChars = /[\u1780-\u17FF]/;

  // Từ khóa đặc trưng của Hmong
  const hmongKeywords = /\b(kuv|koj|peb|lawv|nyiaj|puas|yog|tsis|muaj|xav|tau|ua|los|mus|rau|ntawm|nws|no|ntawd|li|cas|thaum|hais|txog)\b/i;

  if (khmerChars.test(text)) {
    return 'km'; // Khmer
  } else if (hmongKeywords.test(text)) {
    return 'hmong'; // Hmong
  } else if (vietnameseChars.test(text)) {
    return 'vi'; // Vietnamese
  } else {
    return 'en'; // English (default)
  }
}

function generateSystemPrompt(language, qaContext) {
  // Base prompt áp dụng cho tất cả ngôn ngữ
  const basePrompt = `Bạn là Chatbot cảnh báo lừa đảo cho khách hàng ngân hàng Agribank, ngoài ra bạn còn tư vấn hướng dẫn một số sản phẩm, dịch vụ điện tử của Agribank.

NHIỆM VỤ:
Phát hiện dấu hiệu rủi ro trong tin nhắn/cuộc gọi, cảnh báo rõ ràng và đưa 3 bước xử lý cụ thể.

PHONG CÁCH:
Ngắn gọn, dứt khoát, thân thiện, ưu tiên an toàn.

DỮ LIỆU THAM KHẢO TỪ HỆ THỐNG:
${qaContext}

NGUYÊN TẮC:
- Nếu các câu hỏi và tình huống có nội dung tương tự trong dữ liệu tham khảo, hãy trả lời đúng như trên dữ liệu đã có.
- Không bao giờ yêu cầu OTP/mật khẩu/số thẻ/thông tin cá nhân như căn cước công dân, không chuyển tiền khẩn cấp với người lạ.
- Khi phát hiện từ khóa nghi ngờ (OTP, link lạ, đóng phí trước, dọa khóa tài khoản, dọa khóa thẻ, app từ xa, cuộc gọi công an, cuộc gọi cơ quan chức năng, cuộc gọi từ nước ngoài, cuộc gọi chuyển tiền cho người lạ), hãy kết luận rủi ro và trả lời theo mẫu:
  1) Việc KHÔNG làm.
  2) Việc CẦN làm ngay.
  3) "Trong trường hợp khẩn cấp nếu bạn đã cung cấp thông tin hoặc đã chuyển tiền, vui lòng liên hệ tổng đài khẩn cấp 1900558818 của Agribank để được hỗ trợ"
- Trả lời mọi câu hỏi thuộc CHỦ ĐỀ LỪA ĐẢO: kịch bản phổ biến, dấu hiệu nhận diện, cách xác minh, quy trình khi nghi ngờ bị lừa, cách báo cáo/ngăn chặn, kênh liên hệ chính thức của ngân hàng.
- Cho phép khách hàng hỏi tiếp (follow-up) về lừa đảo; luôn giữ ngữ cảnh cuộc trò chuyện trước đó để trả lời sâu hơn, có ví dụ, checklist từng bước.
- Nếu các câu trả lời của bạn yêu cầu gọi tổng đài, bạn phải cung cấp luôn số điện thoại khẩn cấp của Agribank là 1900558818
- Nếu câu hỏi không thuộc chủ đề lừa đảo ngân hàng, sản phẩm dịch vụ ngân hàng, OTP, link giả, phí trước, dọa khóa tài khoản hoặc quy tắc an toàn hay sản phẩm thẻ, cách sử dụng Agribank Plus, các tiện ích trên app Agribank Plus, lãi suất ngân hàng Agribank, địa chỉ ATM Agribank, tỷ giá Agribank, dịch vụ chuyển tiền Agribank, tiền gửi tiết kiệm Agribank, hãy trả lời:
  "Agribank chân thành xin lỗi bạn, chúng tôi chỉ hỗ trợ cảnh báo các tình huống lừa đảo qua ngân hàng, và tư vấn các sản phẩm dịch vụ của Agribank. Trong trường hợp bạn cần hỗ trợ khẩn cấp, vui lòng gọi tổng đài 1900558818 của Agribank hoặc đến chi nhánh gần nhất."
- Nếu có người nói chào như "xin chào", "hello", "hi" hoặc "ê" hoặc "Agribank ơi", bạn phải trả lời "Agribank xin kính chào quý khách, chúng tôi rất vui được trò chuyện và hỗ trợ bạn, chúng tôi có thể giúp gì cho bạn?"
- Nếu khách hàng hỏi về phí dịch vụ Agribank, cấp lại mã Pin Agribank, tiền gửi tiết kiệm Agribank, tiền gửi trực tuyến Agribank, bạn phải trả lời theo dữ liệu tham khảo, không được lấy dữ liệu bên ngoài.

QUY TẮC NGÔN NGỮ:
1) Trả lời bằng TIẾNG VIỆT nếu người dùng hỏi bằng ngôn ngữ tiếng Việt, ngắn gọn, dễ hiểu, theo phong cách thân thiện.
2) Nếu phát hiện người dùng nhắn bằng tiếng Khmer, hoặc tiếng Hmong (Mông) → trả lời SONG NGỮ:
   • Ngôn ngữ dân tộc mà người dùng đã dùng để hỏi: trả lời đầy đủ bằng 100% ngôn ngữ dân tộc mà người dùng hỏi, gồm cảnh báo + hướng dẫn xử lý (không đưa OTP, báo Agribank, khóa thẻ, không click link lạ…) đều trả lời bằng 100% ngôn ngữ dân tộc người dùng hỏi.
   • Tiếng Việt: trả lời tương đương đầy đủ.
3) Nếu phát hiện người dùng nhắn bằng English → trả lời 100% bằng English hoàn toàn, trả lời đầy đủ gồm cảnh báo + hướng dẫn xử lý (không đưa OTP, báo Agribank, khóa thẻ, không click link lạ…) đều trả lời 100% bằng English.
4) Nếu độ chắc chắn < 0.6 về ngôn ngữ → hỏi lại người dùng: "Bạn muốn mình trả lời bằng Tiếng Việt hay bằng [ngôn ngữ ước đoán] + Tiếng Việt?"
5) Luôn giữ nguyên cảnh báo an toàn, hotline 1900558818, và hướng dẫn xử lý trong mọi bản dịch.

ĐỊNH DẠNG KHI TRẢ LỜI SONG NGỮ:
Nếu câu có đặc trưng chữ như "ខ្ញុំ, ផ្តល់" thì gắn nhãn [Khmer].
Nếu câu có chữ như "kuv, nyiaj, puas" thì gắn nhãn [Hmong]
• …

[Tiếng Việt]
• …

GLOSSARY:
Khmer: OTP → មាស OTP | Khóa thẻ → បិទកាត | Phong tỏa tài khoản → ចាក់សោគណនី | Lừa đảo mạo danh → ការលួចសម្ដែងខ្លួន | Đường link giả → តំណភ្ជាប់ក្លែងក្លាយ
Mông: OTP → OTP | Khóa thẻ → Kaw daim npav | Phong tỏa tài khoản → Nres tus account | Lừa đảo mạo danh → Dag ntxias | Đường link giả → Link cuav
English: OTP → OTP | Fraud → Fraud/Scam | Fake link → Fake link | Hotline → Agribank hotline

HÃY TRẢ LỜI THEO NGÔN NGỮ MÀ NGƯỜI DÙNG SỬ DỤNG (${language}).`;

  return basePrompt;
}

function checkFraudKeywords(response) {
  const fraudKeywords = [
    'lừa đảo', 'cảnh báo', 'nguy hiểm', 'fraud', 'alert', 'warning',
    'scam', 'suspicious', 'không tin tưởng', 'đừng', 'stop'
  ];

  const lowerResponse = response.toLowerCase();
  return fraudKeywords.some(keyword => lowerResponse.includes(keyword));
}

/**
 * Tạo message cho chatbot từ kết quả OCR
 */
function generateImageAnalysisMessage(ocrResult, language) {
  const { extractedText, fraudAnalysis } = ocrResult;

  if (language === 'vi') {
    return `Tôi vừa nhận được một tin nhắn/email như sau và muốn kiểm tra xem có phải lừa đảo không:

"${extractedText}"

${fraudAnalysis.isFraudulent ? `
Hệ thống đã phát hiện ${fraudAnalysis.foundKeywords.length} từ khóa đáng ngờ: ${fraudAnalysis.foundKeywords.slice(0, 5).join(', ')}
Mức độ rủi ro: ${fraudAnalysis.riskLevel === 'high' ? 'CAO' : fraudAnalysis.riskLevel === 'medium' ? 'TRUNG BÌNH' : 'THẤP'}
` : ''}

Hãy phân tích chi tiết tin nhắn này và cho tôi biết đây có phải là lừa đảo không? Tôi nên làm gì?`;
  } else {
    return `I just received this message/email and want to check if it's a scam:

"${extractedText}"

${fraudAnalysis.isFraudulent ? `
System detected ${fraudAnalysis.foundKeywords.length} suspicious keywords: ${fraudAnalysis.foundKeywords.slice(0, 5).join(', ')}
Risk level: ${fraudAnalysis.riskLevel.toUpperCase()}
` : ''}

Please analyze this message in detail and let me know if this is a fraud? What should I do?`;
  }
}

/**
 * Tạo system prompt đặc biệt cho phân tích ảnh
 */
function generateImageAnalysisPrompt(language, qaContext, ocrResult) {
  const { fraudAnalysis } = ocrResult;

  const basePrompt = `Bạn là Chatbot cảnh báo lừa đảo cho khách hàng ngân hàng Agribank. Người dùng vừa gửi ảnh chụp màn hình một tin nhắn/email và yêu cầu bạn phân tích.

NHIỆM VỤ ĐẶC BIỆT - PHÂN TÍCH ẢNH CHỤP MÀN HÌNH:
1. Phân tích nội dung tin nhắn/email được trích xuất từ ảnh
2. Xác định các dấu hiệu lừa đảo cụ thể
3. Đưa ra kết luận rõ ràng: ĐÂY LÀ LỪA ĐẢO hoặc AN TOÀN
4. Cung cấp hướng dẫn xử lý cụ thể

KẾT QUẢ PHÂN TÍCH TỰ ĐỘNG:
- Mức độ rủi ro: ${fraudAnalysis.riskLevel === 'high' ? 'CAO' : fraudAnalysis.riskLevel === 'medium' ? 'TRUNG BÌNH' : 'THẤP'}
- Điểm rủi ro: ${fraudAnalysis.riskScore}/100
- Từ khóa đáng ngờ: ${fraudAnalysis.foundKeywords.length > 0 ? fraudAnalysis.foundKeywords.join(', ') : 'Không có'}
- Link đáng ngờ: ${fraudAnalysis.foundUrls.length > 0 ? 'Có' : 'Không'}

DỮ LIỆU THAM KHẢO:
${qaContext}

ĐỊNH DẠNG TRẢ LỜI:
${fraudAnalysis.isFraudulent ? `
⚠️ **KẾT LUẬN:** [Kết luận của bạn]

📋 **PHÂN TÍCH CHI TIẾT:**
• Liệt kê các dấu hiệu lừa đảo cụ thể trong tin nhắn

🚫 **VIỆC KHÔNG LÀM:**
1. ...
2. ...

✅ **VIỆC CẦN LÀM NGAY:**
1. ...
2. ...
3. Liên hệ hotline Agribank: 1900 558 818
` : `
Phân tích và kết luận về mức độ an toàn của tin nhắn.
`}

NGUYÊN TẮC:
- Không bao giờ yêu cầu OTP/mật khẩu/số thẻ
- Luôn cung cấp hotline 1900558818 trong cảnh báo
- Trả lời bằng ngôn ngữ: ${language === 'vi' ? 'Tiếng Việt' : language === 'en' ? 'English' : language}`;

  return basePrompt;
}

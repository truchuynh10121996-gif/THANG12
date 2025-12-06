const geminiService = require('../services/gemini.service');
const qaService = require('../services/qa.service');
const conversationService = require('../services/conversation.service');

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

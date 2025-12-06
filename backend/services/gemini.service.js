const { GoogleGenerativeAI } = require('@google/generative-ai');

// Khởi tạo Gemini AI với SDK chính thức
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || 'demo-key');

/**
 * Tạo phản hồi từ Gemini AI
 * @param {string} userMessage - Tin nhắn từ người dùng
 * @param {string} systemPrompt - System prompt với context
 * @param {Array} conversationHistory - Lịch sử hội thoại (optional)
 * @returns {Promise<string>} - Phản hồi từ AI
 */
async function generateResponse(userMessage, systemPrompt, conversationHistory = []) {
  try {
    // Sử dụng Gemini 2.0 Flash - model mới nhất, nhanh và chất lượng cao
    // Model name phải chính xác theo Google AI API documentation
    const model = genAI.getGenerativeModel({
      model: 'gemini-2.0-flash',
      systemInstruction: systemPrompt
    });

    // Nếu có lịch sử hội thoại, sử dụng chat mode
    if (conversationHistory && conversationHistory.length > 0) {
      // Format history cho Gemini API
      const formattedHistory = conversationHistory.map(msg => ({
        role: msg.role === 'user' ? 'user' : 'model',
        parts: [{ text: msg.content }]
      }));

      // Khởi tạo chat với lịch sử
      const chat = model.startChat({
        history: formattedHistory,
        generationConfig: {
          maxOutputTokens: 2048,
          temperature: 0.7,
          topP: 0.8,
          topK: 40
        }
      });

      // Gửi tin nhắn mới
      const result = await chat.sendMessage(userMessage);
      const response = await result.response;
      return response.text();

    } else {
      // Không có lịch sử, gọi trực tiếp
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: userMessage }] }],
        generationConfig: {
          maxOutputTokens: 2048,
          temperature: 0.7,
          topP: 0.8,
          topK: 40
        }
      });

      const response = await result.response;
      return response.text();
    }

  } catch (error) {
    console.error('Gemini API error:', error);

    // Fallback response nếu API fail
    if (error.message && error.message.includes('API key')) {
      throw new Error('Gemini API key is invalid or missing. Please configure GEMINI_API_KEY in .env file');
    }

    // Trả về phản hồi mặc định
    return getFallbackResponse(userMessage);
  }
}

/**
 * Phản hồi dự phòng khi API fail
 */
function getFallbackResponse(message) {
  const lowerMessage = message.toLowerCase();

  // Phát hiện các từ khóa lừa đảo phổ biến
  const fraudKeywords = [
    'chuyển tiền', 'transfer money', 'otp', 'mã xác thực',
    'cập nhật thông tin', 'update info', 'tài khoản bị khóa',
    'account locked', 'trúng thưởng', 'win prize', 'link',
    'nhấn vào', 'click here', 'cài đặt ứng dụng', 'install app'
  ];

  const hasFraudKeyword = fraudKeywords.some(keyword =>
    lowerMessage.includes(keyword)
  );

  if (hasFraudKeyword) {
    return `⚠️ CẢNH BÁO LỪA ĐẢO

Tình huống bạn mô tả có nhiều dấu hiệu của thủ đoạn lừa đảo phổ biến.

HƯỚNG DẪN AN TOÀN:
1. ❌ KHÔNG cung cấp thông tin cá nhân, OTP, mật khẩu
2. ❌ KHÔNG chuyển tiền theo yêu cầu
3. ❌ KHÔNG nhấn vào link lạ hoặc cài đặt ứng dụng
4. ✅ Liên hệ ngay hotline Agribank: 1900 5555 88
5. ✅ Đến chi nhánh Agribank gần nhất để xác minh

LƯU Ý: Agribank KHÔNG BAO GIỜ yêu cầu khách hàng cung cấp OTP, mật khẩu qua điện thoại, tin nhắn hay email.

Bạn cần hỗ trợ thêm về tình huống này không?`;
  }

  return `Xin chào! Tôi là Agribank Digital Guard, trợ lý AI phòng chống lừa đảo.

Tôi sẵn sàng hỗ trợ bạn phân tích các tình huống đáng ngờ. Hãy mô tả chi tiết tình huống bạn gặp phải để tôi có thể đưa ra cảnh báo và hướng dẫn cụ thể.

Nếu bạn cần hỗ trợ khẩn cấp, vui lòng liên hệ:
📞 Hotline Agribank: 1900 5555 88`;
}

module.exports = {
  generateResponse
};

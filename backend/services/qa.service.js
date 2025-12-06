const QA = require('../models/qa.model');

/**
 * Tìm các Q&A scenarios phù hợp với câu hỏi
 * @param {string} userMessage - Câu hỏi từ người dùng
 * @returns {Promise<string>} - Context từ Q&A scenarios
 */
async function getRelevantQA(userMessage) {
  try {
    // Tìm kiếm các Q&A scenarios liên quan
    const keywords = extractKeywords(userMessage);

    const relevantQAs = await QA.find({
      $or: [
        { keywords: { $in: keywords } },
        { question: { $regex: keywords.join('|'), $options: 'i' } }
      ]
    }).limit(5);

    if (relevantQAs.length === 0) {
      return 'Không có dữ liệu tham khảo phù hợp.';
    }

    // Format context
    const context = relevantQAs.map((qa, index) => {
      return `Kịch bản ${index + 1}:
Câu hỏi: ${qa.question}
Trả lời: ${qa.answer}
${qa.isFraudScenario ? '⚠️ Đây là tình huống lừa đảo' : ''}
`;
    }).join('\n---\n');

    return context;

  } catch (error) {
    console.error('Get relevant QA error:', error);
    return '';
  }
}

/**
 * Trích xuất từ khóa từ câu hỏi
 */
function extractKeywords(text) {
  // Loại bỏ stopwords tiếng Việt
  const stopwords = [
    'là', 'của', 'và', 'có', 'được', 'trong', 'cho', 'với',
    'một', 'các', 'này', 'đó', 'để', 'thì', 'không', 'tôi',
    'bạn', 'họ', 'nó', 'như', 'đã', 'sẽ', 'bị', 'làm'
  ];

  const words = text
    .toLowerCase()
    .replace(/[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopwords.includes(word));

  return [...new Set(words)]; // Remove duplicates
}

module.exports = {
  getRelevantQA
};

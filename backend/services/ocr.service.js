const Tesseract = require('tesseract.js');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs').promises;

/**
 * OCR Service - Trích xuất văn bản từ ảnh sử dụng Tesseract.js
 */
class OCRService {
  constructor() {
    this.supportedLanguages = {
      vi: 'vie',      // Vietnamese
      en: 'eng',      // English
      km: 'khm',      // Khmer
      vie: 'vie',
      eng: 'eng',
      khm: 'khm'
    };
  }

  /**
   * Tiền xử lý ảnh để cải thiện chất lượng OCR
   */
  async preprocessImage(imageBuffer) {
    try {
      // Sử dụng sharp để tiền xử lý ảnh
      const processedBuffer = await sharp(imageBuffer)
        .grayscale()                    // Chuyển sang grayscale
        .normalize()                     // Cân bằng histogram
        .sharpen()                       // Làm sắc nét
        .toBuffer();

      return processedBuffer;
    } catch (error) {
      console.warn('Image preprocessing failed, using original:', error.message);
      return imageBuffer;
    }
  }

  /**
   * Trích xuất văn bản từ ảnh
   */
  async extractText(imageBuffer, language = 'vi') {
    try {
      // Tiền xử lý ảnh
      const processedBuffer = await this.preprocessImage(imageBuffer);

      // Chuyển đổi mã ngôn ngữ
      const tessLang = this.supportedLanguages[language] || 'vie+eng';

      // Sử dụng Tesseract để trích xuất văn bản
      // Hỗ trợ cả tiếng Việt và tiếng Anh để xử lý tin nhắn hỗn hợp
      const { data } = await Tesseract.recognize(
        processedBuffer,
        `${tessLang}+eng`,
        {
          logger: m => {
            if (m.status === 'recognizing text') {
              console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
            }
          }
        }
      );

      return {
        text: data.text.trim(),
        confidence: data.confidence,
        words: data.words?.length || 0
      };
    } catch (error) {
      console.error('OCR extraction error:', error);
      throw new Error(`OCR extraction failed: ${error.message}`);
    }
  }

  /**
   * Phân tích văn bản trích xuất để phát hiện lừa đảo
   */
  analyzeFraudIndicators(text) {
    const lowerText = text.toLowerCase();

    // Các từ khóa/cụm từ nghi ngờ lừa đảo
    const fraudKeywords = {
      otp: [
        'otp', 'mã xác thực', 'mã xác minh', 'verification code',
        'mã bảo mật', 'security code', 'one time password'
      ],
      urgency: [
        'khẩn cấp', 'ngay lập tức', 'urgent', 'immediately', 'gấp',
        'trong vòng', 'hết hạn', 'expired', 'còn lại', 'remaining'
      ],
      threat: [
        'khóa tài khoản', 'lock account', 'đóng băng', 'freeze',
        'tạm ngưng', 'suspend', 'chặn', 'block', 'vô hiệu hóa',
        'công an', 'police', 'viện kiểm sát', 'tòa án', 'court'
      ],
      money: [
        'chuyển tiền', 'transfer money', 'chuyển khoản', 'bank transfer',
        'đặt cọc', 'deposit', 'thanh toán', 'payment', 'phí',
        'trúng thưởng', 'winning', 'giải thưởng', 'prize', 'lottery'
      ],
      links: [
        'click', 'nhấp vào', 'truy cập', 'access', 'link',
        'http', 'www', '.com', '.vn', 'tải app', 'download'
      ],
      impersonation: [
        'ngân hàng', 'bank', 'agribank', 'nhân viên', 'staff',
        'hỗ trợ', 'support', 'tổng đài', 'hotline', 'bộ phận'
      ],
      personalInfo: [
        'cmnd', 'cccd', 'căn cước', 'id card', 'số thẻ', 'card number',
        'mật khẩu', 'password', 'pin', 'số tài khoản', 'account number'
      ]
    };

    const detectedIndicators = [];
    let riskScore = 0;

    // Kiểm tra từng loại từ khóa
    for (const [category, keywords] of Object.entries(fraudKeywords)) {
      const found = keywords.filter(kw => lowerText.includes(kw.toLowerCase()));
      if (found.length > 0) {
        detectedIndicators.push({
          category,
          keywords: found,
          count: found.length
        });
        riskScore += found.length * 10;
      }
    }

    // Tăng điểm rủi ro nếu có nhiều loại indicator
    if (detectedIndicators.length >= 3) {
      riskScore += 20;
    }
    if (detectedIndicators.length >= 5) {
      riskScore += 30;
    }

    // Giới hạn điểm rủi ro trong khoảng 0-100
    riskScore = Math.min(100, riskScore);

    return {
      indicators: detectedIndicators,
      riskScore,
      riskLevel: this.getRiskLevel(riskScore),
      isFraudSuspected: riskScore >= 30
    };
  }

  /**
   * Xác định mức độ rủi ro
   */
  getRiskLevel(score) {
    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    if (score >= 20) return 'low';
    return 'safe';
  }

  /**
   * Xử lý ảnh và phân tích lừa đảo
   */
  async processImageForFraud(imageBuffer, language = 'vi') {
    // Trích xuất văn bản
    const ocrResult = await this.extractText(imageBuffer, language);

    if (!ocrResult.text || ocrResult.text.length < 10) {
      return {
        success: true,
        extractedText: ocrResult.text,
        confidence: ocrResult.confidence,
        analysis: {
          indicators: [],
          riskScore: 0,
          riskLevel: 'unknown',
          isFraudSuspected: false,
          message: 'Không thể trích xuất đủ văn bản từ ảnh. Vui lòng thử lại với ảnh rõ hơn.'
        }
      };
    }

    // Phân tích văn bản
    const analysis = this.analyzeFraudIndicators(ocrResult.text);

    return {
      success: true,
      extractedText: ocrResult.text,
      confidence: ocrResult.confidence,
      wordCount: ocrResult.words,
      analysis
    };
  }

  /**
   * Tạo prompt cho AI dựa trên kết quả OCR
   */
  generateAIPrompt(extractedText, analysis, language = 'vi') {
    const languagePrompts = {
      vi: {
        intro: 'Tôi vừa nhận được một tin nhắn/email với nội dung sau:',
        question: 'Đây có phải là tin nhắn lừa đảo không? Hãy phân tích và đưa ra cảnh báo nếu cần.',
        riskWarning: `⚠️ Hệ thống đã phát hiện ${analysis.indicators.length} dấu hiệu đáng ngờ với điểm rủi ro ${analysis.riskScore}/100.`
      },
      en: {
        intro: 'I just received a message/email with the following content:',
        question: 'Is this a scam message? Please analyze and provide warnings if needed.',
        riskWarning: `⚠️ The system detected ${analysis.indicators.length} suspicious indicators with a risk score of ${analysis.riskScore}/100.`
      },
      km: {
        intro: 'ខ្ញុំទើបតែទទួលបានសារ/អ៊ីមែលដែលមានខ្លឹមសារដូចខាងក្រោម:',
        question: 'តើនេះជាសារក្លែងបន្លំទេ? សូមវិភាគ និងផ្តល់ការព្រមានប្រសិនបើចាំបាច់។',
        riskWarning: `⚠️ ប្រព័ org្ org org org org org org រួចពី ${analysis.indicators.length} សញ org org org org org org org org org org org org org org org org ${analysis.riskScore}/100។`
      }
    };

    const prompt = languagePrompts[language] || languagePrompts.vi;

    let message = `${prompt.intro}\n\n"${extractedText}"\n\n`;

    if (analysis.isFraudSuspected) {
      message += `${prompt.riskWarning}\n\n`;
    }

    message += prompt.question;

    return message;
  }
}

module.exports = new OCRService();

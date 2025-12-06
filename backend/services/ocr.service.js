/**
 * OCR Service - Trích xuất văn bản từ ảnh sử dụng Tesseract.js
 * Hỗ trợ nhận diện văn bản tiếng Việt, Anh, và Khmer
 */

const Tesseract = require('tesseract.js');
const path = require('path');
const fs = require('fs');

// Fraud detection keywords cho phân tích tin nhắn
const FRAUD_KEYWORDS = {
  vi: [
    // OTP và xác thực
    'otp', 'mã xác thực', 'mã xác nhận', 'mã bảo mật', 'ma xac thuc',
    // Yêu cầu chuyển tiền
    'chuyển tiền', 'chuyen tien', 'chuyển khoản', 'chuyen khoan',
    'chuyển ngay', 'chuyen ngay', 'gấp', 'khẩn cấp', 'khan cap',
    // Tài khoản bị khóa
    'tài khoản bị khóa', 'tai khoan bi khoa', 'khóa tài khoản',
    'đóng băng', 'dong bang', 'tạm khóa', 'tam khoa',
    // Giả mạo
    'trúng thưởng', 'trung thuong', 'nhận thưởng', 'nhan thuong',
    'quà tặng', 'qua tang', 'miễn phí', 'mien phi',
    // Link đáng ngờ
    'click vào', 'bấm vào', 'nhấn vào', 'truy cập', 'truy cap',
    'xác minh ngay', 'xac minh ngay', 'đăng nhập', 'dang nhap',
    // Thông tin cá nhân
    'số cmnd', 'cccd', 'số thẻ', 'so the', 'mật khẩu', 'mat khau',
    'password', 'pin', 'cvv', 'cvc',
    // Cảnh báo lừa đảo
    'lừa đảo', 'lua dao', 'cảnh báo', 'canh bao', 'nguy hiểm',
    // Giả danh
    'ngân hàng', 'ngan hang', 'agribank', 'vietcombank', 'techcombank',
    'bidv', 'mb bank', 'công an', 'cong an', 'tòa án', 'toa an',
    'viện kiểm sát', 'vien kiem sat', 'thuế', 'thue',
    // Số tiền lớn
    'triệu', 'trieu', 'tỷ', 'ty', 'dollar', 'usd',
    // Thời gian gấp
    '24 giờ', '24h', 'trong ngày', 'hết hạn', 'het han'
  ],
  en: [
    // OTP and authentication
    'otp', 'verification code', 'security code', 'pin code',
    // Money transfer
    'transfer money', 'wire transfer', 'urgent transfer', 'send money',
    // Account locked
    'account locked', 'account suspended', 'frozen account', 'temporarily locked',
    // Scam indicators
    'you won', 'winner', 'lottery', 'prize', 'free gift', 'congratulations',
    // Suspicious links
    'click here', 'verify now', 'login now', 'act now', 'confirm identity',
    // Personal info
    'ssn', 'social security', 'credit card', 'card number', 'password',
    'pin', 'cvv', 'cvc', 'bank account',
    // Impersonation
    'bank', 'irs', 'tax', 'police', 'court', 'government',
    // Urgency
    'urgent', 'immediately', 'expire', '24 hours', 'deadline'
  ],
  km: [
    // Khmer fraud keywords
    'លេខកូដ', 'ផ្ទេរប្រាក់', 'គណនី', 'ពាក្យសម្ងាត់',
    'ប្រាក់រង្វាន់', 'ឈ្នះ', 'ធនាគារ', 'ប៉ូលីស'
  ]
};

// URL pattern để phát hiện link đáng ngờ
const SUSPICIOUS_URL_PATTERNS = [
  /bit\.ly/i,
  /tinyurl/i,
  /goo\.gl/i,
  /t\.co/i,
  /short\.link/i,
  /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/,  // IP address
  /[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\.(com|net|org)/i,  // Random domain
  /agr[i1!]bank/i,  // Typosquatting
  /v[i1!]etcombank/i,
  /\.(xyz|top|club|work|loan|click)/i  // Suspicious TLDs
];

/**
 * Trích xuất văn bản từ ảnh sử dụng Tesseract OCR
 * @param {string|Buffer} imageInput - Đường dẫn file hoặc buffer của ảnh
 * @param {string} language - Ngôn ngữ nhận diện (vi, en, km)
 * @returns {Promise<Object>} - Kết quả OCR bao gồm text và metadata
 */
async function extractTextFromImage(imageInput, language = 'vi') {
  try {
    // Map ngôn ngữ ứng dụng sang mã Tesseract
    const langMap = {
      'vi': 'vie+eng',  // Tiếng Việt + Anh (vì tin nhắn thường có cả 2)
      'en': 'eng',
      'km': 'khm+eng'   // Khmer + Anh
    };

    const tesseractLang = langMap[language] || 'vie+eng';

    console.log(`[OCR] Bắt đầu trích xuất văn bản, ngôn ngữ: ${tesseractLang}`);

    const result = await Tesseract.recognize(
      imageInput,
      tesseractLang,
      {
        logger: m => {
          if (m.status === 'recognizing text') {
            console.log(`[OCR] Tiến độ: ${Math.round(m.progress * 100)}%`);
          }
        }
      }
    );

    const extractedText = result.data.text.trim();

    console.log(`[OCR] Hoàn tất. Độ tin cậy: ${result.data.confidence}%`);
    console.log(`[OCR] Văn bản trích xuất: ${extractedText.substring(0, 200)}...`);

    return {
      success: true,
      text: extractedText,
      confidence: result.data.confidence,
      words: result.data.words?.length || 0,
      lines: result.data.lines?.length || 0
    };

  } catch (error) {
    console.error('[OCR] Lỗi trích xuất văn bản:', error);
    throw new Error(`Không thể trích xuất văn bản từ ảnh: ${error.message}`);
  }
}

/**
 * Phân tích văn bản để phát hiện dấu hiệu lừa đảo
 * @param {string} text - Văn bản cần phân tích
 * @param {string} language - Ngôn ngữ
 * @returns {Object} - Kết quả phân tích lừa đảo
 */
function analyzeFraudIndicators(text, language = 'vi') {
  const normalizedText = text.toLowerCase();
  const foundKeywords = [];
  const foundUrls = [];
  let riskScore = 0;

  // Kiểm tra keywords lừa đảo theo ngôn ngữ
  const keywords = [...(FRAUD_KEYWORDS[language] || []), ...FRAUD_KEYWORDS.vi, ...FRAUD_KEYWORDS.en];

  for (const keyword of keywords) {
    if (normalizedText.includes(keyword.toLowerCase())) {
      if (!foundKeywords.includes(keyword)) {
        foundKeywords.push(keyword);
        riskScore += 10;
      }
    }
  }

  // Kiểm tra URL đáng ngờ
  for (const pattern of SUSPICIOUS_URL_PATTERNS) {
    const matches = text.match(pattern);
    if (matches) {
      foundUrls.push(...matches);
      riskScore += 20;
    }
  }

  // Kiểm tra các mẫu tin nhắn lừa đảo phổ biến
  const fraudPatterns = [
    /nhập.*(otp|mã)/i,
    /xác (minh|nhận).*(tài khoản|thông tin)/i,
    /tài khoản.*(khóa|đóng băng)/i,
    /chuyển.*(tiền|khoản).*gấp/i,
    /trúng.*(thưởng|giải)/i,
    /click.*(link|đường dẫn)/i,
    /verify.*account/i,
    /account.*(locked|suspended)/i,
    /won.*prize/i,
    /transfer.*urgent/i
  ];

  for (const pattern of fraudPatterns) {
    if (pattern.test(normalizedText)) {
      riskScore += 15;
    }
  }

  // Giới hạn riskScore tối đa 100
  riskScore = Math.min(riskScore, 100);

  // Xác định mức độ rủi ro
  let riskLevel;
  if (riskScore >= 60) {
    riskLevel = 'high';
  } else if (riskScore >= 30) {
    riskLevel = 'medium';
  } else {
    riskLevel = 'low';
  }

  return {
    isFraudulent: riskScore >= 30,
    riskScore,
    riskLevel,
    foundKeywords,
    foundUrls,
    analysisDetails: {
      keywordsCount: foundKeywords.length,
      suspiciousUrlsCount: foundUrls.length
    }
  };
}

/**
 * Xử lý ảnh và phân tích lừa đảo
 * @param {string|Buffer} imageInput - Đường dẫn file hoặc buffer
 * @param {string} language - Ngôn ngữ
 * @returns {Promise<Object>} - Kết quả đầy đủ
 */
async function processImageForFraudDetection(imageInput, language = 'vi') {
  // Bước 1: Trích xuất văn bản
  const ocrResult = await extractTextFromImage(imageInput, language);

  if (!ocrResult.success || !ocrResult.text) {
    return {
      success: false,
      error: 'Không thể trích xuất văn bản từ ảnh',
      ocr: ocrResult
    };
  }

  // Bước 2: Phân tích lừa đảo
  const fraudAnalysis = analyzeFraudIndicators(ocrResult.text, language);

  // Bước 3: Tạo thông điệp cảnh báo
  let warningMessage = '';
  if (fraudAnalysis.isFraudulent) {
    if (language === 'vi') {
      warningMessage = generateVietnameseWarning(fraudAnalysis);
    } else if (language === 'en') {
      warningMessage = generateEnglishWarning(fraudAnalysis);
    } else {
      warningMessage = generateVietnameseWarning(fraudAnalysis); // Default
    }
  }

  return {
    success: true,
    extractedText: ocrResult.text,
    confidence: ocrResult.confidence,
    fraudAnalysis,
    warningMessage,
    metadata: {
      words: ocrResult.words,
      lines: ocrResult.lines,
      language
    }
  };
}

/**
 * Tạo cảnh báo tiếng Việt
 */
function generateVietnameseWarning(analysis) {
  let warning = '⚠️ CẢNH BÁO LỪA ĐẢO!\n\n';

  if (analysis.riskLevel === 'high') {
    warning += '🔴 Mức độ rủi ro: CAO\n\n';
  } else if (analysis.riskLevel === 'medium') {
    warning += '🟡 Mức độ rủi ro: TRUNG BÌNH\n\n';
  }

  warning += 'Phát hiện dấu hiệu lừa đảo:\n';

  if (analysis.foundKeywords.length > 0) {
    warning += `• Từ khóa đáng ngờ: ${analysis.foundKeywords.slice(0, 5).join(', ')}\n`;
  }

  if (analysis.foundUrls.length > 0) {
    warning += `• Link đáng ngờ được phát hiện\n`;
  }

  warning += '\n📋 HƯỚNG DẪN XỬ LÝ:\n';
  warning += '1. KHÔNG nhập OTP, mật khẩu hay thông tin cá nhân\n';
  warning += '2. KHÔNG click vào bất kỳ link nào trong tin nhắn\n';
  warning += '3. KHÔNG chuyển tiền theo yêu cầu\n';
  warning += '4. Liên hệ ngay hotline Agribank: 1900 558 818\n';

  return warning;
}

/**
 * Tạo cảnh báo tiếng Anh
 */
function generateEnglishWarning(analysis) {
  let warning = '⚠️ FRAUD ALERT!\n\n';

  if (analysis.riskLevel === 'high') {
    warning += '🔴 Risk Level: HIGH\n\n';
  } else if (analysis.riskLevel === 'medium') {
    warning += '🟡 Risk Level: MEDIUM\n\n';
  }

  warning += 'Fraud indicators detected:\n';

  if (analysis.foundKeywords.length > 0) {
    warning += `• Suspicious keywords: ${analysis.foundKeywords.slice(0, 5).join(', ')}\n`;
  }

  if (analysis.foundUrls.length > 0) {
    warning += `• Suspicious links detected\n`;
  }

  warning += '\n📋 RECOMMENDED ACTIONS:\n';
  warning += '1. DO NOT enter OTP, password or personal information\n';
  warning += '2. DO NOT click any links in the message\n';
  warning += '3. DO NOT transfer money as requested\n';
  warning += '4. Contact Agribank hotline: 1900 558 818\n';

  return warning;
}

/**
 * Xóa file tạm sau khi xử lý
 */
function cleanupTempFile(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`[OCR] Đã xóa file tạm: ${filePath}`);
    }
  } catch (error) {
    console.error(`[OCR] Lỗi xóa file tạm: ${error.message}`);
  }
}

module.exports = {
  extractTextFromImage,
  analyzeFraudIndicators,
  processImageForFraudDetection,
  cleanupTempFile,
  FRAUD_KEYWORDS
};

/**
 * OCR Controller - Xử lý các request liên quan đến trích xuất văn bản từ ảnh
 */

const ocrService = require('../services/ocr.service');
const path = require('path');
const fs = require('fs');

/**
 * Trích xuất văn bản từ ảnh được upload
 * POST /api/ocr/extract
 */
async function extractText(req, res) {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng upload một ảnh'
      });
    }

    const language = req.body.language || 'vi';
    const imagePath = req.file.path;

    console.log(`[OCR Controller] Nhận yêu cầu trích xuất văn bản, file: ${req.file.originalname}`);

    const result = await ocrService.extractTextFromImage(imagePath, language);

    // Xóa file tạm sau khi xử lý
    ocrService.cleanupTempFile(imagePath);

    return res.json({
      success: true,
      data: {
        text: result.text,
        confidence: result.confidence,
        words: result.words,
        lines: result.lines
      }
    });

  } catch (error) {
    console.error('[OCR Controller] Lỗi trích xuất văn bản:', error);

    // Xóa file tạm nếu có lỗi
    if (req.file?.path) {
      ocrService.cleanupTempFile(req.file.path);
    }

    return res.status(500).json({
      success: false,
      error: error.message || 'Lỗi khi trích xuất văn bản từ ảnh'
    });
  }
}

/**
 * Trích xuất văn bản và phân tích lừa đảo
 * POST /api/ocr/analyze
 */
async function analyzeImage(req, res) {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng upload một ảnh'
      });
    }

    const language = req.body.language || 'vi';
    const imagePath = req.file.path;

    console.log(`[OCR Controller] Nhận yêu cầu phân tích lừa đảo, file: ${req.file.originalname}`);

    const result = await ocrService.processImageForFraudDetection(imagePath, language);

    // Xóa file tạm sau khi xử lý
    ocrService.cleanupTempFile(imagePath);

    if (!result.success) {
      return res.status(400).json({
        success: false,
        error: result.error
      });
    }

    return res.json({
      success: true,
      data: {
        extractedText: result.extractedText,
        confidence: result.confidence,
        fraudAnalysis: result.fraudAnalysis,
        warningMessage: result.warningMessage,
        metadata: result.metadata
      }
    });

  } catch (error) {
    console.error('[OCR Controller] Lỗi phân tích ảnh:', error);

    // Xóa file tạm nếu có lỗi
    if (req.file?.path) {
      ocrService.cleanupTempFile(req.file.path);
    }

    return res.status(500).json({
      success: false,
      error: error.message || 'Lỗi khi phân tích ảnh'
    });
  }
}

/**
 * Trích xuất văn bản từ base64 image
 * POST /api/ocr/extract-base64
 */
async function extractTextFromBase64(req, res) {
  try {
    const { imageBase64, language = 'vi' } = req.body;

    if (!imageBase64) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng cung cấp ảnh dạng base64'
      });
    }

    console.log('[OCR Controller] Nhận yêu cầu trích xuất từ base64');

    // Chuyển base64 thành buffer
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    const result = await ocrService.extractTextFromImage(imageBuffer, language);

    return res.json({
      success: true,
      data: {
        text: result.text,
        confidence: result.confidence,
        words: result.words,
        lines: result.lines
      }
    });

  } catch (error) {
    console.error('[OCR Controller] Lỗi trích xuất từ base64:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Lỗi khi trích xuất văn bản từ ảnh'
    });
  }
}

/**
 * Phân tích lừa đảo từ base64 image
 * POST /api/ocr/analyze-base64
 */
async function analyzeImageFromBase64(req, res) {
  try {
    const { imageBase64, language = 'vi' } = req.body;

    if (!imageBase64) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng cung cấp ảnh dạng base64'
      });
    }

    console.log('[OCR Controller] Nhận yêu cầu phân tích lừa đảo từ base64');

    // Chuyển base64 thành buffer
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Data, 'base64');

    const result = await ocrService.processImageForFraudDetection(imageBuffer, language);

    if (!result.success) {
      return res.status(400).json({
        success: false,
        error: result.error
      });
    }

    return res.json({
      success: true,
      data: {
        extractedText: result.extractedText,
        confidence: result.confidence,
        fraudAnalysis: result.fraudAnalysis,
        warningMessage: result.warningMessage,
        metadata: result.metadata
      }
    });

  } catch (error) {
    console.error('[OCR Controller] Lỗi phân tích từ base64:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Lỗi khi phân tích ảnh'
    });
  }
}

/**
 * Phân tích văn bản thuần (không cần OCR)
 * POST /api/ocr/analyze-text
 */
async function analyzeText(req, res) {
  try {
    const { text, language = 'vi' } = req.body;

    if (!text) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng cung cấp văn bản cần phân tích'
      });
    }

    console.log('[OCR Controller] Nhận yêu cầu phân tích văn bản');

    const fraudAnalysis = ocrService.analyzeFraudIndicators(text, language);

    let warningMessage = '';
    if (fraudAnalysis.isFraudulent) {
      // Tạo cảnh báo dựa trên ngôn ngữ
      if (language === 'vi') {
        warningMessage = generateVietnameseWarning(fraudAnalysis);
      } else {
        warningMessage = generateEnglishWarning(fraudAnalysis);
      }
    }

    return res.json({
      success: true,
      data: {
        text,
        fraudAnalysis,
        warningMessage
      }
    });

  } catch (error) {
    console.error('[OCR Controller] Lỗi phân tích văn bản:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Lỗi khi phân tích văn bản'
    });
  }
}

// Helper functions để tạo cảnh báo
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

module.exports = {
  extractText,
  analyzeImage,
  extractTextFromBase64,
  analyzeImageFromBase64,
  analyzeText
};

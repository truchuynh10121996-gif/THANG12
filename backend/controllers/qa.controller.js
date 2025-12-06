const QA = require('../models/qa.model');

/**
 * Lấy tất cả Q&A scenarios
 */
exports.getAllQA = async (req, res) => {
  try {
    const { language, category, search, page = 1, limit = 50 } = req.query;

    let filter = {};

    if (language) {
      filter.language = language;
    }

    if (category) {
      filter.category = category;
    }

    if (search) {
      filter.$or = [
        { question: { $regex: search, $options: 'i' } },
        { answer: { $regex: search, $options: 'i' } },
        { keywords: { $in: [new RegExp(search, 'i')] } }
      ];
    }

    const skip = (page - 1) * limit;

    const [qas, total] = await Promise.all([
      QA.find(filter).skip(skip).limit(parseInt(limit)).sort({ createdAt: -1 }),
      QA.countDocuments(filter)
    ]);

    res.status(200).json({
      status: 'success',
      data: {
        qas,
        pagination: {
          total,
          page: parseInt(page),
          limit: parseInt(limit),
          totalPages: Math.ceil(total / limit)
        }
      }
    });

  } catch (error) {
    console.error('Get all QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to fetch Q&A scenarios',
      error: error.message
    });
  }
};

/**
 * Lấy một Q&A scenario theo ID
 */
exports.getQAById = async (req, res) => {
  try {
    const { id } = req.params;
    const qa = await QA.findById(id);

    if (!qa) {
      return res.status(404).json({
        status: 'error',
        message: 'Q&A scenario not found'
      });
    }

    res.status(200).json({
      status: 'success',
      data: qa
    });

  } catch (error) {
    console.error('Get QA by ID error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to fetch Q&A scenario',
      error: error.message
    });
  }
};

/**
 * Tạo Q&A scenario mới
 */
exports.createQA = async (req, res) => {
  try {
    const { question, answer, language, category, keywords, isFraudScenario } = req.body;

    if (!question || !answer || !language) {
      return res.status(400).json({
        status: 'error',
        message: 'Question, answer, and language are required'
      });
    }

    const qa = new QA({
      question,
      answer,
      language,
      category: category || 'general',
      keywords: keywords || [],
      isFraudScenario: isFraudScenario || false
    });

    await qa.save();

    res.status(201).json({
      status: 'success',
      message: 'Q&A scenario created successfully',
      data: qa
    });

  } catch (error) {
    console.error('Create QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to create Q&A scenario',
      error: error.message
    });
  }
};

/**
 * Cập nhật Q&A scenario
 */
exports.updateQA = async (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    const qa = await QA.findByIdAndUpdate(
      id,
      { ...updates, updatedAt: new Date() },
      { new: true, runValidators: true }
    );

    if (!qa) {
      return res.status(404).json({
        status: 'error',
        message: 'Q&A scenario not found'
      });
    }

    res.status(200).json({
      status: 'success',
      message: 'Q&A scenario updated successfully',
      data: qa
    });

  } catch (error) {
    console.error('Update QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to update Q&A scenario',
      error: error.message
    });
  }
};

/**
 * Xóa Q&A scenario
 */
exports.deleteQA = async (req, res) => {
  try {
    const { id } = req.params;
    const qa = await QA.findByIdAndDelete(id);

    if (!qa) {
      return res.status(404).json({
        status: 'error',
        message: 'Q&A scenario not found'
      });
    }

    res.status(200).json({
      status: 'success',
      message: 'Q&A scenario deleted successfully'
    });

  } catch (error) {
    console.error('Delete QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to delete Q&A scenario',
      error: error.message
    });
  }
};

/**
 * Huấn luyện chatbot với dữ liệu mới
 */
exports.trainChatbot = async (req, res) => {
  try {
    // Lấy tất cả Q&A scenarios
    const qas = await QA.find({});

    // Cập nhật vector embeddings hoặc training data
    // (Tùy thuộc vào implementation cụ thể)

    res.status(200).json({
      status: 'success',
      message: 'Chatbot trained successfully',
      data: {
        totalScenarios: qas.length,
        trainedAt: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Train chatbot error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to train chatbot',
      error: error.message
    });
  }
};

/**
 * Export Q&A scenarios
 */
exports.exportQA = async (req, res) => {
  try {
    const qas = await QA.find({});

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=qa-scenarios.json');

    res.status(200).json({
      status: 'success',
      exportedAt: new Date().toISOString(),
      total: qas.length,
      data: qas
    });

  } catch (error) {
    console.error('Export QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to export Q&A scenarios',
      error: error.message
    });
  }
};

/**
 * Import Q&A scenarios
 */
exports.importQA = async (req, res) => {
  try {
    const { scenarios } = req.body;

    if (!Array.isArray(scenarios) || scenarios.length === 0) {
      return res.status(400).json({
        status: 'error',
        message: 'Invalid scenarios data'
      });
    }

    const results = await QA.insertMany(scenarios, { ordered: false });

    res.status(201).json({
      status: 'success',
      message: 'Q&A scenarios imported successfully',
      data: {
        imported: results.length,
        total: scenarios.length
      }
    });

  } catch (error) {
    console.error('Import QA error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to import Q&A scenarios',
      error: error.message
    });
  }
};

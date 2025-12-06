const mongoose = require('mongoose');

const qaSchema = new mongoose.Schema({
  question: {
    type: String,
    required: true,
    trim: true
  },
  answer: {
    type: String,
    required: true
  },
  language: {
    type: String,
    enum: ['vi', 'en', 'km'],
    default: 'vi'
  },
  category: {
    type: String,
    enum: ['general', 'fraud-alert', 'banking', 'security', 'otp', 'transfer', 'phishing', 'other'],
    default: 'general'
  },
  keywords: [{
    type: String
  }],
  isFraudScenario: {
    type: Boolean,
    default: false
  },
  priority: {
    type: Number,
    default: 0
  },
  isActive: {
    type: Boolean,
    default: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Indexes
qaSchema.index({ keywords: 1 });
qaSchema.index({ category: 1 });
qaSchema.index({ language: 1 });
qaSchema.index({ question: 'text', answer: 'text' });

module.exports = mongoose.model('QA', qaSchema);

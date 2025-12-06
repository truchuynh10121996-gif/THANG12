const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  role: {
    type: String,
    enum: ['user', 'bot'],
    required: true
  },
  content: {
    type: String,
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
}, { _id: false });

const conversationSchema = new mongoose.Schema({
  conversationId: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  messages: [messageSchema],
  language: {
    type: String,
    enum: ['vi', 'en', 'km'],
    default: 'vi'
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
conversationSchema.index({ conversationId: 1 });
conversationSchema.index({ updatedAt: -1 });

module.exports = mongoose.model('Conversation', conversationSchema);

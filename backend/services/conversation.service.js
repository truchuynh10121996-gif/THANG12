const Conversation = require('../models/conversation.model');

/**
 * Lưu tin nhắn vào lịch sử hội thoại
 */
async function saveMessage({ conversationId, userMessage, botResponse, language, timestamp }) {
  try {
    let conversation = await Conversation.findOne({ conversationId });

    if (!conversation) {
      // Tạo conversation mới
      conversation = new Conversation({
        conversationId,
        messages: [],
        language,
        createdAt: timestamp
      });
    }

    // Thêm message
    conversation.messages.push({
      role: 'user',
      content: userMessage,
      timestamp
    });

    conversation.messages.push({
      role: 'bot',
      content: botResponse,
      timestamp: new Date()
    });

    conversation.updatedAt = new Date();

    await conversation.save();

    return conversation;

  } catch (error) {
    console.error('Save message error:', error);
    throw error;
  }
}

/**
 * Lấy lịch sử hội thoại
 */
async function getConversation(conversationId) {
  try {
    const conversation = await Conversation.findOne({ conversationId });
    return conversation;
  } catch (error) {
    console.error('Get conversation error:', error);
    throw error;
  }
}

/**
 * Xóa hội thoại
 */
async function deleteConversation(conversationId) {
  try {
    await Conversation.deleteOne({ conversationId });
  } catch (error) {
    console.error('Delete conversation error:', error);
    throw error;
  }
}

/**
 * Lấy tất cả conversations
 */
async function getAllConversations(limit = 50) {
  try {
    const conversations = await Conversation.find({})
      .sort({ updatedAt: -1 })
      .limit(limit);
    return conversations;
  } catch (error) {
    console.error('Get all conversations error:', error);
    throw error;
  }
}

module.exports = {
  saveMessage,
  getConversation,
  deleteConversation,
  getAllConversations
};

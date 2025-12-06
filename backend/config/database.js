const mongoose = require('mongoose');

/**
 * Kết nối đến MongoDB
 */
async function connectDatabase() {
  try {
    const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/agribank-digital-guard';

    await mongoose.connect(mongoURI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });

    console.log('✅ MongoDB connected successfully');
    console.log(`📊 Database: ${mongoose.connection.name}`);

  } catch (error) {
    console.error('❌ MongoDB connection error:', error);
    console.log('⚠️  Continuing without database. Some features may be limited.');

    // Không throw error để app vẫn chạy được
    // throw error;
  }
}

/**
 * Đóng kết nối database
 */
async function disconnectDatabase() {
  try {
    await mongoose.disconnect();
    console.log('MongoDB disconnected');
  } catch (error) {
    console.error('Error disconnecting from MongoDB:', error);
  }
}

module.exports = {
  connectDatabase,
  disconnectDatabase
};

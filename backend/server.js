require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const bodyParser = require('body-parser');
const path = require('path');
const { connectDatabase } = require('./config/database');

// Import routes
const chatbotRoutes = require('./routes/chatbot.routes');
const qaRoutes = require('./routes/qa.routes');
const ttsRoutes = require('./routes/tts.routes');
const sttRoutes = require('./routes/stt.routes');
const ocrRoutes = require('./routes/ocr.routes');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Security & Performance Middleware
app.use(helmet());
app.use(compression());
app.use(morgan('dev'));

// CORS Configuration
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['http://localhost:3000', 'http://localhost:19006'];

app.use(cors({
  origin: function(origin, callback) {
    // Cho phép requests không có origin (mobile apps, Postman, etc.)
    if (!origin) {
      return callback(null, true);
    }

    // Development mode: cho phép tất cả origins từ localhost và local network
    const isDevelopment = process.env.NODE_ENV !== 'production';
    if (isDevelopment) {
      // Cho phép localhost, 127.0.0.1, và IP trong mạng local (192.168.x.x, 10.x.x.x)
      if (origin.match(/^http:\/\/(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$/)) {
        return callback(null, true);
      }
    }

    // Kiểm tra allowed origins
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      // Development: log warning nhưng vẫn cho phép
      if (isDevelopment) {
        console.warn(`CORS Warning: Origin ${origin} not in allowed list, but allowing in development mode`);
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    }
  },
  credentials: true
}));

// Body Parser Middleware
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

// Static files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// API Routes
app.use('/api/chatbot', chatbotRoutes);
app.use('/api/qa', qaRoutes);
app.use('/api/tts', ttsRoutes);
app.use('/api/stt', sttRoutes);
app.use('/api/ocr', ocrRoutes);

// Health Check Endpoint
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'success',
    message: 'Agribank Digital Guard API is running',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Root Endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Agribank Digital Guard API',
    version: '1.0.0',
    description: 'Enterprise Anti-Fraud Chatbot Backend',
    endpoints: {
      health: '/api/health',
      chatbot: '/api/chatbot',
      qa: '/api/qa',
      tts: '/api/tts',
      stt: '/api/stt',
      ocr: '/api/ocr'
    }
  });
});

// Error Handling Middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({
    status: 'error',
    message: err.message || 'Internal Server Error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Route not found'
  });
});

// Start Server
async function startServer() {
  try {
    // Connect to database
    await connectDatabase();

    // Start Express server
    app.listen(PORT, () => {
      console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🏦 AGRIBANK DIGITAL GUARD - Backend API 🛡️           ║
║                                                           ║
║     Server running on: http://localhost:${PORT}           ║
║     Environment: ${process.env.NODE_ENV || 'development'}                      ║
║     Time: ${new Date().toLocaleString('vi-VN')}                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
      `);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

module.exports = app;

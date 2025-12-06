# Agribank Digital Guard - Web Application

Phiên bản Web cho người dùng của Agribank Digital Guard - Trợ lý AI chống lừa đảo.

## Tính năng

- 💬 Chat với AI chatbot thông minh
- 🌍 Hỗ trợ đa ngôn ngữ (Tiếng Việt, English, ភាសាខ្មែរ)
- 🔊 Text-to-Speech (TTS) - Đọc tin nhắn
- 🛡️ Phát hiện và cảnh báo lừa đảo
- 🎨 Giao diện đẹp mắt với màu pastel

## Cài đặt

```bash
# Cài đặt dependencies
npm install

# Copy file .env
cp .env.example .env
```

## Chạy ứng dụng

```bash
# Development mode
npm start

# Ứng dụng sẽ chạy tại http://localhost:3001
```

## Build cho production

```bash
npm run build
```

## Cấu trúc dự án

```
web-app/
├── public/
│   └── index.html
├── src/
│   ├── pages/
│   │   ├── LanguagePage.js    # Trang chọn ngôn ngữ
│   │   ├── HomePage.js         # Trang chủ/giới thiệu
│   │   └── ChatPage.js         # Trang chat chính
│   ├── services/
│   │   └── api.js             # API service
│   ├── App.js                 # Main App với routing
│   ├── index.js               # Entry point
│   └── index.css              # Global styles
└── package.json
```

## Công nghệ sử dụng

- React 18
- Material-UI (MUI) v5
- React Router v6
- Axios
- React Hot Toast

## API Backend

Ứng dụng kết nối với backend API tại `http://localhost:5000/api`

Xem file `.env.example` để cấu hình API URL.

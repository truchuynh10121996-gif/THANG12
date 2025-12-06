# Changelog - Agribank Digital Guard

## [Update] - 2025-11-30

### ✨ Tính năng mới

#### 🌐 Web User App - Phiên bản Website cho người dùng
Tạo mới ứng dụng web đầy đủ tính năng cho người dùng laptop/desktop:
- **Trang chọn ngôn ngữ**: Hỗ trợ 3 ngôn ngữ (Tiếng Việt, English, ភាសាខ្មែរ)
- **Trang giới thiệu**: Hiển thị các tính năng nổi bật
  - Phát hiện lừa đảo
  - Hỗ trợ 24/7
  - Đa ngôn ngữ
  - Bảo mật cao
- **Trang chat**: Giao diện chat đẹp mắt với:
  - Chat real-time với AI chatbot
  - Text-to-Speech (TTS) - Đọc tin nhắn
  - Cảnh báo lừa đảo
  - Responsive design
- **Công nghệ**: React 18, Material-UI v5, React Router v6, Axios

### 🐛 Sửa lỗi

#### Mobile Chatbot - Lỗi kết nối
**Vấn đề**: Chatbot trên mobile báo "gặp sự cố kết nối" khi nhập tin nhắn

**Nguyên nhân**:
- API URL hardcode `localhost:5000` không hoạt động trên thiết bị thật
- CORS chỉ chấp nhận requests từ localhost

**Giải pháp**:
1. **Mobile App** (`mobile-app/src/services/api.js`):
   - Tự động detect IP của máy host bằng `expo-constants`
   - Sử dụng `expoConfig.hostUri` để lấy địa chỉ Expo Dev Server
   - Hỗ trợ cả web (localhost) và native (IP address)

2. **Backend** (`backend/server.js`):
   - Cập nhật CORS để chấp nhận:
     - Requests không có origin (mobile apps)
     - Localhost và 127.0.0.1
     - IP trong mạng local (192.168.x.x, 10.x.x.x)
   - Development mode: Cho phép tất cả origins từ local network

### 🎨 Thay đổi giao diện

#### Đổi Theme - Từ xanh lá sang hồng đỏ pastel
Thay đổi toàn bộ màu sắc ứng dụng:

**Màu cũ**:
- Primary: #2E7D32 (Xanh lá Agribank)
- Dark: #1B5E20
- Light: #4CAF50

**Màu mới**:
- Primary: #FF8DAD (Hồng đỏ pastel)
- Dark: #FF6B99 (Hồng đậm)
- Light: #FFB3C6 (Hồng nhạt)
- Secondary: #FFC9DD

**Phạm vi thay đổi**:
1. **Mobile App**:
   - Tất cả screens: Splash, Language, Home, Chat
   - Tất cả components: ChatBubble, VoiceRecorder
   - Buttons, headers, icons, text colors

2. **Web Admin**:
   - Theme MUI configuration
   - Layout: AppBar, Sidebar, Menu
   - Dashboard: Stats cards, charts
   - QA Management: Tables, buttons, dialogs
   - Chatbot Preview: Chat interface

3. **Web User App**:
   - Sử dụng màu hồng pastel ngay từ đầu
   - Consistent với mobile và web admin

### 📝 Files thay đổi

```
backend/
└── server.js                           # CORS configuration

mobile-app/src/
├── services/
│   └── api.js                         # Auto-detect API URL
├── screens/
│   ├── SplashScreen.js                # Pink theme
│   ├── LanguageScreen.js              # Pink theme
│   ├── HomeScreen.js                  # Pink theme
│   └── ChatScreen.js                  # Pink theme
└── components/
    ├── ChatBubble.js                  # Pink theme
    └── VoiceRecorder.js               # Pink theme

web-admin/src/
├── App.js                             # MUI theme config
├── index.css                          # Scrollbar colors
├── components/
│   └── Layout.js                      # AppBar, Sidebar colors
└── pages/
    ├── Dashboard.js                   # Pink theme
    ├── QAManagement.js                # Pink theme
    └── ChatbotPreview.js              # Pink theme

web-app/                               # NEW - Web User App
├── package.json
├── .env.example
├── .gitignore
├── README.md
├── public/
│   └── index.html
└── src/
    ├── App.js                         # Router & Theme
    ├── index.js
    ├── index.css
    ├── services/
    │   └── api.js
    └── pages/
        ├── LanguagePage.js
        ├── HomePage.js
        └── ChatPage.js
```

### 🚀 Hướng dẫn chạy

#### Web User App (Mới)
```bash
cd web-app
npm install
npm start
# Mở http://localhost:3001
```

#### Mobile App
```bash
cd mobile-app
npm start
# Quét QR code bằng Expo Go
```

#### Web Admin
```bash
cd web-admin
npm start
# Mở http://localhost:3000
```

#### Backend
```bash
cd backend
npm start
# Server chạy tại http://localhost:5000
```

### 📊 Thống kê thay đổi
- **Files thay đổi**: 26 files
- **Lines thêm**: 1,374 lines
- **Lines xóa**: 66 lines
- **Files mới**: 12 files (web-app)

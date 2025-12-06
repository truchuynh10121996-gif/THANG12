# 🏦 AGRIBANK DIGITAL GUARD 🛡️

**Chatbot Phòng Chống Lừa Đảo Cấp Quốc Gia**

Dự án enterprise chatbot AI giúp người dùng nhận diện và phòng tránh các thủ đoạn lừa đảo trong lĩnh vực ngân hàng.

---

## 📋 MỤC LỤC

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt-chi-tiết)
- [Hướng dẫn chạy dự án](#-hướng-dẫn-chạy-dự-án)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 GIỚI THIỆU

**Agribank Digital Guard** là hệ thống chatbot AI tiên tiến được thiết kế để:

- ✅ Phát hiện và cảnh báo các thủ đoạn lừa đảo
- ✅ Hỗ trợ đa ngôn ngữ: Tiếng Việt, English, Khmer
- ✅ Tích hợp AI (Google Gemini) cho phản hồi tự nhiên
- ✅ Hỗ trợ ghi âm giọng nói (STT) và phát âm thanh (TTS)
- ✅ Dashboard admin để quản lý kịch bản Q&A

---

## ⚡ TÍNH NĂNG

### Mobile App (React Native + Expo)
- 📱 Giao diện gradient pastel nhẹ nhàng (#FBD6E3 + #A9EDE9)
- 🎤 Ghi âm và chuyển giọng nói thành văn bản (STT)
- 🔊 Phát âm thanh câu trả lời (TTS)
- 🌍 Hỗ trợ 3 ngôn ngữ với tự động nhận diện
- ⚠️ Cảnh báo rõ ràng khi phát hiện lừa đảo

### Web Admin (React)
- 📊 Dashboard với thống kê trực quan
- 📝 Quản lý Q&A: Thêm, sửa, xóa kịch bản
- 🤖 Huấn luyện chatbot với dữ liệu mới
- 💬 Xem trước và test chatbot

### Backend API (Node.js + Express)
- 🔗 RESTful API hoàn chỉnh
- 🧠 Tích hợp Google Gemini AI
- 💾 MongoDB để lưu trữ dữ liệu
- 🔒 Bảo mật với CORS, Helmet
- 📡 Hỗ trợ STT/TTS với Google Cloud

---

## 📂 CẤU TRÚC DỰ ÁN

```
SANGKIENTG/
│
├── backend/                    # Backend API (Node.js + Express)
│   ├── config/                # Cấu hình database
│   ├── controllers/           # Controllers xử lý logic
│   ├── models/                # Models (MongoDB Schema)
│   ├── routes/                # API Routes
│   ├── services/              # Services (Gemini, STT, TTS)
│   ├── data/                  # Dữ liệu seed Q&A
│   ├── scripts/               # Scripts tiện ích
│   ├── .env                   # Environment variables
│   └── server.js              # Server chính
│
├── mobile-app/                # Mobile App (React Native + Expo)
│   ├── assets/                # Hình ảnh, logo
│   ├── src/
│   │   ├── screens/          # Các màn hình
│   │   ├── components/       # Components tái sử dụng
│   │   └── services/         # API services
│   ├── App.js                # App chính
│   └── package.json          # Dependencies
│
├── web-admin/                 # Web Admin Dashboard (React)
│   ├── public/               # Static files
│   ├── src/
│   │   ├── pages/           # Các trang chính
│   │   ├── components/      # Components
│   │   └── services/        # API services
│   └── package.json         # Dependencies
│
├── assets/                    # Assets chung (logo)
└── README.md                 # Tài liệu này
```

---

## 💻 YÊU CẦU HỆ THỐNG

### Phần mềm cần cài đặt:

1. **Node.js** (phiên bản 18.x hoặc mới hơn)
   - Tải tại: https://nodejs.org/

2. **MongoDB** (nếu chạy local)
   - Tải tại: https://www.mongodb.com/try/download/community
   - Hoặc dùng MongoDB Atlas (cloud): https://www.mongodb.com/atlas

3. **Git**
   - Tải tại: https://git-scm.com/

4. **Expo CLI** (cho mobile app)
   ```bash
   npm install -g expo-cli
   ```

5. **Code Editor** (khuyên dùng VS Code)
   - Tải tại: https://code.visualstudio.com/

### API Keys cần thiết:

1. **Google Gemini API Key** (BẮT BUỘC)
   - Đăng ký miễn phí tại: https://makersuite.google.com/app/apikey

2. **Google Cloud Credentials** (TÙY CHỌN - cho STT/TTS)
   - Tạo project tại: https://console.cloud.google.com/
   - Enable: Cloud Speech-to-Text API & Text-to-Speech API

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT CHI TIẾT

### Bước 1: Clone dự án

```bash
cd SANGKIENTG
# Dự án đã có sẵn trong thư mục này
```

### Bước 2: Cài đặt Backend

```bash
# Di chuyển vào thư mục backend
cd backend

# Cài đặt dependencies
npm install

# Cấu hình environment variables
# File .env đã được tạo sẵn, bạn cần cập nhật:
# 1. GEMINI_API_KEY: Thay bằng API key của bạn
# 2. MONGODB_URI: Giữ nguyên nếu dùng MongoDB local

# Khởi động MongoDB (nếu dùng local)
# Windows: Mở MongoDB Compass hoặc
# mongod --dbpath "C:\data\db"

# Seed dữ liệu mẫu Q&A
npm run seed
# Hoặc: node scripts/seed.js

# Khởi động server
npm start
# Server sẽ chạy tại: http://localhost:5000
```

### Bước 3: Cài đặt Mobile App

```bash
# Mở terminal mới, di chuyển vào thư mục mobile-app
cd mobile-app

# Cài đặt dependencies
npm install

# QUAN TRỌNG: Cập nhật API URL trong src/services/api.js
# Thay localhost bằng IP máy tính của bạn nếu test trên điện thoại thật
# Ví dụ: http://192.168.1.100:5000/api

# Khởi động Expo
npm start

# Expo Dev Tools sẽ mở tại: http://localhost:19002
```

### Bước 4: Cài đặt Web Admin

```bash
# Mở terminal mới, di chuyển vào thư mục web-admin
cd web-admin

# Cài đặt dependencies
npm install

# Khởi động web app
npm start

# Web app sẽ chạy tại: http://localhost:3000
```

---

## 🎯 HƯỚNG DẪN CHẠY DỰ ÁN

### Cách 1: Chạy từng phần riêng biệt

#### Terminal 1 - Backend:
```bash
cd backend
npm start
```

#### Terminal 2 - Mobile App:
```bash
cd mobile-app
npm start
```

#### Terminal 3 - Web Admin:
```bash
cd web-admin
npm start
```

### Cách 2: Chạy Mobile App trên máy tính (PC)

#### Option A: Sử dụng Expo Go trên điện thoại
1. Tải ứng dụng **Expo Go** từ:
   - iOS: App Store
   - Android: Google Play Store

2. Kết nối điện thoại và máy tính cùng WiFi

3. Trong Expo Dev Tools, quét QR code bằng:
   - iOS: Camera app
   - Android: Expo Go app

#### Option B: Chạy trên Emulator/Simulator

**Android (Windows/Mac/Linux):**
```bash
# Cài đặt Android Studio
# Download: https://developer.android.com/studio

# Tạo Android Virtual Device (AVD)
# Trong Android Studio: Tools > AVD Manager > Create Virtual Device

# Khởi động emulator
# Trong terminal mobile-app:
npm run android
```

**iOS (chỉ trên Mac):**
```bash
# Cài đặt Xcode từ App Store

# Khởi động simulator
npm run ios
```

#### Option C: Chạy trên Web (Đơn giản nhất)
```bash
# Trong terminal mobile-app:
npm run web

# App sẽ mở tại: http://localhost:19006
# Lưu ý: Một số tính năng như STT/TTS có thể không hoạt động trên web
```

---

## 📱 HƯỚNG DẪN SỬ DỤNG

### Sử dụng Mobile App

1. **Chọn ngôn ngữ:**
   - Khi mở app lần đầu, chọn ngôn ngữ bạn muốn sử dụng
   - Có thể thay đổi ngôn ngữ sau trong màn hình Home

2. **Trò chuyện với chatbot:**
   - Nhập văn bản vào ô input
   - Hoặc nhấn nút 🎤 để ghi âm giọng nói
   - Chatbot sẽ phân tích và đưa ra cảnh báo nếu phát hiện lừa đảo

3. **Nghe phản hồi:**
   - Nhấn nút 🔊 trên tin nhắn của chatbot để nghe giọng đọc

### Sử dụng Web Admin

1. **Dashboard:**
   - Xem thống kê tổng quan
   - Theo dõi số lượng Q&A, kịch bản lừa đảo

2. **Quản lý Q&A:**
   - Click "Thêm Q&A mới" để tạo kịch bản mới
   - Điền đầy đủ: Câu hỏi, Câu trả lời, Ngôn ngữ, Danh mục
   - Thêm từ khóa để chatbot dễ tìm kiếm
   - Đánh dấu "Đây là kịch bản lừa đảo" nếu cần

3. **Huấn luyện Chatbot:**
   - Sau khi thêm/sửa Q&A, click "Huấn luyện Chatbot"
   - Dữ liệu mới sẽ được cập nhật vào hệ thống

4. **Xem trước Chatbot:**
   - Test chatbot trực tiếp trên web
   - Kiểm tra các phản hồi trước khi deploy

---

## 📚 API DOCUMENTATION

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Chatbot
```
POST /chatbot/message
Body: {
  "message": "Tôi nhận được tin nhắn yêu cầu OTP",
  "conversationId": "optional",
  "language": "vi"
}
```

#### 2. Q&A Management
```
GET    /qa              # Lấy tất cả Q&A
POST   /qa              # Tạo Q&A mới
PUT    /qa/:id          # Cập nhật Q&A
DELETE /qa/:id          # Xóa Q&A
POST   /qa/train        # Huấn luyện chatbot
```

#### 3. Text-to-Speech
```
POST /tts/synthesize
Body: {
  "text": "Xin chào",
  "language": "vi",
  "gender": "FEMALE"
}
```

#### 4. Speech-to-Text
```
POST /stt/transcribe
Content-Type: multipart/form-data
Body: {
  "audio": <audio file>,
  "language": "vi"
}
```

---

## 🔧 TROUBLESHOOTING

### Lỗi thường gặp:

#### 1. "Cannot connect to MongoDB"
**Giải pháp:**
```bash
# Kiểm tra MongoDB đã chạy chưa
# Windows: Mở Task Manager > Services > MongoDB
# Mac/Linux:
ps aux | grep mongod

# Hoặc dùng MongoDB Atlas (cloud) - Miễn phí
# Cập nhật MONGODB_URI trong backend/.env
```

#### 2. "Gemini API Error"
**Giải pháp:**
- Kiểm tra GEMINI_API_KEY trong backend/.env
- Đảm bảo API key còn hạn và có quota
- Lấy API key mới tại: https://makersuite.google.com/app/apikey

#### 3. "Cannot connect to backend from mobile"
**Giải pháp:**
```bash
# 1. Kiểm tra backend đã chạy: http://localhost:5000
# 2. Nếu test trên điện thoại thật:
#    - Tìm IP máy tính:
#      Windows: ipconfig
#      Mac/Linux: ifconfig
#    - Cập nhật trong mobile-app/src/services/api.js:
#      const API_BASE_URL = 'http://192.168.1.xxx:5000/api';
# 3. Tắt firewall tạm thời để test
```

#### 4. "Expo error: Unable to resolve module"
**Giải pháp:**
```bash
cd mobile-app
rm -rf node_modules
rm package-lock.json
npm install
npm start -- --clear
```

#### 5. "Port 5000 already in use"
**Giải pháp:**
```bash
# Thay đổi PORT trong backend/.env
# Ví dụ: PORT=5001
# Nhớ cập nhật lại API_BASE_URL ở mobile-app và web-admin
```

---

## 📞 LIÊN HỆ & HỖ TRỢ

- **Email hỗ trợ:** support@agribank.com.vn
- **Hotline:** 1900 5555 88

---

## 📝 LƯU Ý QUAN TRỌNG

### Cho buổi trình bày:

1. **Chuẩn bị trước:**
   - Đảm bảo backend đã chạy
   - Test mobile app trước 30 phút
   - Chuẩn bị sẵn các kịch bản demo

2. **Demo scenarios:**
   - Kịch bản 1: Nhận tin nhắn yêu cầu OTP
   - Kịch bản 2: Cuộc gọi mạo danh ngân hàng
   - Kịch bản 3: Email thông báo trúng thưởng
   - Kịch bản 4: Link cập nhật app lạ

3. **Tính năng nổi bật cần nhấn mạnh:**
   - ✨ Hỗ trợ 3 ngôn ngữ tự động
   - ✨ AI phát hiện lừa đảo thông minh
   - ✨ Ghi âm và phát giọng nói
   - ✨ Dashboard admin chuyên nghiệp

4. **Nếu gặp lỗi trong buổi demo:**
   - Giữ bình tĩnh
   - Dùng web preview làm backup
   - Giải thích rằng đây là môi trường development

---

## 🎓 KIẾN THỨC BỔ SUNG

### Hiểu về Expo SDK 54
- Expo là framework giúp build React Native app dễ dàng
- SDK 54 tương thích với React Native 0.76.5
- Không cần Android Studio/Xcode để test ban đầu

### Hiểu về cấu trúc Backend
- **Routes**: Định nghĩa các API endpoints
- **Controllers**: Xử lý logic nghiệp vụ
- **Services**: Tích hợp services bên ngoài (Gemini, Google Cloud)
- **Models**: Định nghĩa cấu trúc dữ liệu

### Tùy chỉnh màu sắc
Tất cả màu gradient có thể thay đổi tại:
- Mobile: Các file trong `mobile-app/src/screens/`
- Web: `web-admin/src/index.css` và các components

---

## 🎉 KẾT LUẬN

Bạn đã hoàn thành cài đặt **AGRIBANK DIGITAL GUARD**!

Dự án này bao gồm:
- ✅ Backend API hoàn chỉnh với Gemini AI
- ✅ Mobile App với Expo SDK 54
- ✅ Web Admin Dashboard
- ✅ Tài liệu hướng dẫn chi tiết

**Chúc bạn trình bày thành công! 🚀**

---

**Version:** 1.0.0
**Last Updated:** 2024
**License:** MIT
**Author:** Agribank Digital Guard Team

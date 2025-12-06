# 📚 HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN - CHO NGƯỜI MỚI BẮT ĐẦU

> 🎯 **Mục đích:** Hướng dẫn từng bước chi tiết để cài đặt và chạy dự án AGRIBANK DIGITAL GUARD cho người không có kinh nghiệm lập trình.

---

## 📑 MỤC LỤC

1. [Chuẩn bị máy tính](#1-chuẩn-bị-máy-tính)
2. [Cài đặt phần mềm cần thiết](#2-cài-đặt-phần-mềm-cần-thiết)
3. [Lấy API Keys](#3-lấy-api-keys)
4. [Cài đặt và chạy Backend](#4-cài-đặt-và-chạy-backend)
5. [Cài đặt và chạy Mobile App](#5-cài-đặt-và-chạy-mobile-app)
6. [Cài đặt và chạy Web Admin](#6-cài-đặt-và-chạy-web-admin)
7. [Kiểm tra dự án hoạt động](#7-kiểm-tra-dự-án-hoạt-động)
8. [Xử lý lỗi thường gặp](#8-xử-lý-lỗi-thường-gặp)

---

## 1. CHUẨN BỊ MÁY TÍNH

### ✅ Yêu cầu tối thiểu:
- **Hệ điều hành:** Windows 10/11, macOS 10.15+, hoặc Linux
- **RAM:** Tối thiểu 4GB (khuyên dùng 8GB)
- **Ổ cứng trống:** Ít nhất 5GB
- **Kết nối internet:** Ổn định để tải các package

### 📋 Checklist trước khi bắt đầu:
- [ ] Máy tính đã kết nối internet
- [ ] Có quyền admin để cài đặt phần mềm
- [ ] Đã tắt antivirus tạm thời (nếu bị chặn khi cài đặt)

---

## 2. CÀI ĐẶT PHẦN MỀM CẦN THIẾT

### 🔧 Bước 2.1: Cài đặt Node.js

**Node.js là gì?** Đây là môi trường để chạy mã JavaScript trên máy tính.

#### Windows:
1. Truy cập: https://nodejs.org/
2. Tải phiên bản **LTS** (Long Term Support) - nút màu xanh lá
3. Chạy file cài đặt `.msi` đã tải về
4. Nhấn "Next" → "Next" → "Install" (giữ tất cả mặc định)
5. Chờ cài đặt xong (khoảng 2-3 phút)

#### macOS:
1. Truy cập: https://nodejs.org/
2. Tải phiên bản **LTS** cho macOS
3. Mở file `.pkg` đã tải về
4. Làm theo hướng dẫn trên màn hình
5. Nhập mật khẩu Mac khi được yêu cầu

#### Kiểm tra cài đặt thành công:
Mở **Command Prompt** (Windows) hoặc **Terminal** (Mac/Linux):

```bash
# Kiểm tra Node.js
node --version
# Kết quả mong đợi: v18.x.x hoặc cao hơn

# Kiểm tra npm (đi kèm với Node.js)
npm --version
# Kết quả mong đợi: 9.x.x hoặc cao hơn
```

**❌ Lỗi "command not found"?**
- Khởi động lại máy tính và thử lại
- Hoặc cài đặt lại Node.js, chọn "Add to PATH" khi cài đặt

---

### 🗄️ Bước 2.2: Cài đặt MongoDB

**MongoDB là gì?** Đây là cơ sở dữ liệu để lưu trữ thông tin chatbot.

#### Cách 1: Dùng MongoDB Cloud (KHUYÊN DÙNG - dễ nhất):

1. Truy cập: https://www.mongodb.com/cloud/atlas/register
2. Đăng ký tài khoản miễn phí (dùng email)
3. Chọn plan **FREE** (M0 Sandbox)
4. Chọn region gần nhất (Singapore hoặc US)
5. Đặt tên cluster (ví dụ: "AgribankDB")
6. Chờ 3-5 phút để cluster được tạo

7. **Tạo Database User:**
   - Click "Database Access" ở menu bên trái
   - Click "Add New Database User"
   - Username: `admin`
   - Password: `admin123` (hoặc mật khẩu bạn muốn)
   - User Privileges: **Atlas Admin**
   - Click "Add User"

8. **Cho phép kết nối từ mọi IP:**
   - Click "Network Access" ở menu bên trái
   - Click "Add IP Address"
   - Click "Allow Access from Anywhere" (0.0.0.0/0)
   - Click "Confirm"

9. **Lấy Connection String:**
   - Click "Database" ở menu bên trái
   - Click nút "Connect" ở cluster của bạn
   - Chọn "Drivers"
   - Chọn Driver: **Node.js**, Version: **4.1 or later**
   - Copy connection string (dạng: `mongodb+srv://admin:<password>@...`)
   - **LƯU Ý:** Thay `<password>` bằng mật khẩu thật (ví dụ: `admin123`)
   - Lưu string này lại, sẽ dùng ở bước sau

#### Cách 2: Cài MongoDB Local (cho người có kinh nghiệm):

**Windows:**
1. Tải MongoDB Community Server: https://www.mongodb.com/try/download/community
2. Chạy file `.msi` và cài đặt
3. Chọn "Complete" installation
4. Tick "Install MongoDB as a Service"
5. Để "Run service as Network Service user"
6. Nhấn "Next" và hoàn tất cài đặt

**macOS:**
```bash
# Cài đặt Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Cài MongoDB
brew tap mongodb/brew
brew install mongodb-community

# Chạy MongoDB
brew services start mongodb-community
```

---

### 📝 Bước 2.3: Cài đặt Git

**Git là gì?** Công cụ quản lý mã nguồn (bạn đã có dự án trong máy rồi nên có thể bỏ qua bước này).

#### Windows:
1. Tải Git: https://git-scm.com/download/win
2. Chạy file cài đặt
3. Nhấn "Next" cho tất cả (giữ mặc định)
4. Hoàn tất cài đặt

#### macOS:
```bash
# Cài Git qua Homebrew
brew install git
```

#### Kiểm tra:
```bash
git --version
# Kết quả: git version 2.x.x
```

---

### 📱 Bước 2.4: Cài đặt Expo CLI (cho Mobile App)

**Expo là gì?** Framework giúp chạy ứng dụng di động dễ dàng mà không cần Android Studio.

Mở Command Prompt/Terminal:

```bash
# Cài đặt Expo CLI global
npm install -g expo-cli

# Kiểm tra
expo --version
# Kết quả: x.x.x
```

**❌ Lỗi "permission denied" trên Mac/Linux?**
```bash
sudo npm install -g expo-cli
# Nhập mật khẩu máy tính khi được yêu cầu
```

---

## 3. LẤY API KEYS

### 🔑 Bước 3.1: Lấy Google Gemini API Key (BẮT BUỘC)

**Gemini API là gì?** Đây là AI của Google dùng để chatbot trả lời thông minh.

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập bằng tài khoản Google (Gmail)
3. Click "Create API Key"
4. Chọn project (hoặc tạo mới project)
5. Copy API Key (dạng: `AIzaSy...`)
6. **LƯU LẠI** API key này, sẽ dùng ngay sau

**🆓 Miễn phí:** Gemini API free có 60 requests/phút, đủ để test.

---

### ☁️ Bước 3.2: Google Cloud (TÙY CHỌN - có thể bỏ qua)

**Chức năng:** Chuyển giọng nói thành chữ (STT) và chữ thành giọng nói (TTS).

**Lưu ý:** Nếu bỏ qua, chatbot vẫn chạy bình thường, chỉ không có tính năng ghi âm/phát âm.

---

## 4. CÀI ĐẶT VÀ CHẠY BACKEND

### 📂 Bước 4.1: Mở thư mục backend

**Windows:**
1. Mở File Explorer
2. Vào thư mục `C:\Users\ADMIN\SANGKIENTG\backend`
3. Nhấn chuột phải vào thư mục trống
4. Chọn "Open in Terminal" hoặc "Git Bash Here"

**Hoặc dùng Command Prompt:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\backend
```

**macOS/Linux:**
```bash
cd /đường/dẫn/tới/SANGKIENTG/backend
```

---

### 📦 Bước 4.2: Cài đặt dependencies

Trong terminal/command prompt ở thư mục `backend`:

```bash
# Cài đặt tất cả packages cần thiết
npm install
```

**⏳ Thời gian:** Khoảng 2-5 phút tùy tốc độ internet.

**📊 Bạn sẽ thấy:**
- Nhiều dòng text chạy
- Progress bar tải packages
- Thư mục `node_modules` được tạo ra

**❌ Lỗi thường gặp:**

**Lỗi: "EACCES: permission denied"**
```bash
# Windows: Chạy Command Prompt as Administrator
# Mac/Linux: Thêm sudo
sudo npm install
```

**Lỗi: "network timeout"**
- Kiểm tra kết nối internet
- Thử lại: `npm install`

---

### ⚙️ Bước 4.3: Cấu hình file .env

File `.env` chứa các thông tin cấu hình như API key, database URL.

**Bước 1: Mở file .env**

Trong thư mục `backend`, tìm file `.env` (đã có sẵn).

**Windows:**
- Chuột phải → "Open with" → chọn "Notepad" hoặc "VS Code"

**macOS/Linux:**
```bash
nano .env
# Hoặc
code .env
```

**Bước 2: Cập nhật các giá trị**

File `.env` sẽ có dạng:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/agribank-digital-guard

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Google Cloud (Optional)
GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:19006
```

**✏️ Cần sửa:**

1. **GEMINI_API_KEY:**
   ```env
   GEMINI_API_KEY=AIzaSy... (dán API key bạn lấy ở Bước 3.1)
   ```

2. **MONGODB_URI:** (chọn 1 trong 2 cách)

   **Cách A: Dùng MongoDB Cloud (khuyên dùng):**
   ```env
   MONGODB_URI=mongodb+srv://admin:admin123@cluster0.xxxxx.mongodb.net/agribank-digital-guard?retryWrites=true&w=majority
   ```
   (Thay bằng connection string bạn lấy ở Bước 2.2)

   **Cách B: Dùng MongoDB Local:**
   ```env
   MONGODB_URI=mongodb://localhost:27017/agribank-digital-guard
   ```
   (Giữ nguyên nếu cài MongoDB local)

**Bước 3: Lưu file**
- Notepad: File → Save
- VS Code: Ctrl+S (Windows) hoặc Cmd+S (Mac)
- Nano: Ctrl+X → Y → Enter

---

### 🌱 Bước 4.4: Seed dữ liệu mẫu (tùy chọn nhưng nên làm)

**Seed là gì?** Thêm dữ liệu Q&A mẫu vào database để chatbot có thể trả lời ngay.

```bash
# Chạy script seed
node scripts/seed.js
```

**✅ Kết quả mong đợi:**
```
✅ Connected to MongoDB
✅ Cleared old data
✅ Seeded 50 Q&A scenarios
✅ Seeding completed!
```

**❌ Lỗi "Cannot connect to MongoDB"?**
- Kiểm tra lại MONGODB_URI trong .env
- Nếu dùng MongoDB Cloud, kiểm tra:
  - Username/password đúng chưa
  - Network Access đã allow 0.0.0.0/0 chưa

---

### 🚀 Bước 4.5: Chạy Backend Server

```bash
npm start
```

**✅ Thành công khi thấy:**
```
🚀 Server is running on port 5000
✅ MongoDB connected successfully
✅ Gemini AI initialized
⚠️  Google Cloud TTS not configured. Using fallback.
⚠️  Google Cloud STT not configured. Using fallback.
```

**Giải thích các dòng log:**
- ✅ Màu xanh: Thành công
- ⚠️ Màu vàng: Cảnh báo (không ảnh hưởng chính)
- ❌ Màu đỏ: Lỗi (cần sửa)

**🌐 Test Backend:**

Mở trình duyệt, vào: http://localhost:5000

Bạn sẽ thấy:
```json
{
  "message": "Agribank Digital Guard API is running",
  "version": "1.0.0",
  "status": "healthy"
}
```

**✅ Backend đã chạy thành công!**

---

### 🔧 Bước 4.6: Xử lý lỗi khi chạy backend

#### Lỗi: "SyntaxError: Identifier 'textToSpeech' has already been declared"

**✅ Đã sửa!** Lỗi này đã được khắc phục trong file `backend/services/tts.service.js`.

Nếu vẫn gặp, chạy lại:
```bash
# Dừng server (Ctrl+C)
# Xóa cache
rm -rf node_modules
npm install
npm start
```

#### Lỗi: "Port 5000 already in use"

**Giải pháp 1:** Tắt ứng dụng đang dùng port 5000

**Windows:**
```bash
# Tìm process dùng port 5000
netstat -ano | findstr :5000

# Kill process (thay PID)
taskkill /PID <số_PID> /F
```

**Mac/Linux:**
```bash
# Tìm và kill process
lsof -ti:5000 | xargs kill -9
```

**Giải pháp 2:** Đổi port trong .env
```env
PORT=5001
```

Nhớ cập nhật port ở mobile app và web admin sau.

---

## 5. CÀI ĐẶT VÀ CHẠY MOBILE APP

### 📂 Bước 5.1: Mở thư mục mobile-app

**MỞ TERMINAL MỚI** (giữ terminal backend đang chạy):

**Windows:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\mobile-app
```

**macOS/Linux:**
```bash
cd /đường/dẫn/tới/SANGKIENTG/mobile-app
```

---

### 📦 Bước 5.2: Cài đặt dependencies

```bash
npm install
```

**⏳ Thời gian:** 3-7 phút.

---

### ⚙️ Bước 5.3: Cấu hình API URL

**Mở file:** `mobile-app/src/services/api.js`

**Tìm dòng:**
```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

**Nếu test trên máy tính:** Giữ nguyên

**Nếu test trên điện thoại thật:**

1. **Tìm IP máy tính:**

   **Windows:**
   ```bash
   ipconfig
   # Tìm dòng "IPv4 Address": 192.168.x.x
   ```

   **Mac/Linux:**
   ```bash
   ifconfig
   # Hoặc
   ip addr show
   # Tìm IP dạng 192.168.x.x
   ```

2. **Sửa API URL:**
   ```javascript
   const API_BASE_URL = 'http://192.168.1.100:5000/api';
   // Thay 192.168.1.100 bằng IP thật của bạn
   ```

3. **Lưu file** (Ctrl+S / Cmd+S)

---

### 🚀 Bước 5.4: Chạy Mobile App

```bash
npm start
```

**✅ Thành công khi thấy:**
```
› Metro waiting on exp://192.168.x.x:8081
› Scan the QR code above with Expo Go (Android) or Camera (iOS)
› Press a │ open Android
› Press i │ open iOS simulator
› Press w │ open web
› Press r │ reload app
```

**🌐 Expo Dev Tools sẽ tự động mở tại:** http://localhost:19002

---

### 📱 Bước 5.5: Chạy app trên thiết bị

#### Cách 1: Chạy trên Web (Dễ nhất - cho người mới)

Trong terminal, nhấn phím: **w**

Trình duyệt sẽ mở app tại: http://localhost:19006

**Lưu ý:** Tính năng ghi âm/STT không hoạt động trên web.

---

#### Cách 2: Chạy trên điện thoại thật (Khuyên dùng)

**Bước 1: Tải Expo Go**

**Android:**
- Mở Google Play Store
- Tìm "Expo Go"
- Cài đặt

**iOS:**
- Mở App Store
- Tìm "Expo Go"
- Cài đặt

**Bước 2: Quét QR Code**

1. Đảm bảo điện thoại và máy tính **cùng mạng WiFi**
2. Mở Expo Go app
3. **Android:** Nhấn "Scan QR Code" trong app
4. **iOS:** Mở Camera app và quét QR code
5. App sẽ tự động tải và mở

**❌ Lỗi "Could not connect to development server"?**

**Giải pháp:**
1. Kiểm tra cùng WiFi
2. Tắt firewall tạm thời
3. Sửa API URL đúng IP (xem Bước 5.3)
4. Restart expo: Ctrl+C → `npm start` lại

---

#### Cách 3: Chạy trên Android Emulator (Nâng cao)

**Yêu cầu:** Đã cài Android Studio và tạo AVD (Android Virtual Device)

```bash
# Trong terminal mobile-app
npm run android
```

---

#### Cách 4: Chạy trên iOS Simulator (chỉ Mac)

**Yêu cầu:** Đã cài Xcode

```bash
# Trong terminal mobile-app
npm run ios
```

---

## 6. CÀI ĐẶT VÀ CHẠY WEB ADMIN

### 📂 Bước 6.1: Mở thư mục web-admin

**MỞ TERMINAL MỚI** (giữ backend và mobile-app đang chạy):

**Windows:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\web-admin
```

**macOS/Linux:**
```bash
cd /đường/dẫn/tới/SANGKIENTG/web-admin
```

---

### 📦 Bước 6.2: Cài đặt dependencies

```bash
npm install
```

**⏳ Thời gian:** 2-5 phút.

---

### ⚙️ Bước 6.3: Cấu hình API URL (nếu cần)

Nếu bạn đổi PORT backend (khác 5000), cần sửa file:

**Mở:** `web-admin/src/services/api.js`

**Tìm:**
```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

**Sửa nếu cần:**
```javascript
const API_BASE_URL = 'http://localhost:5001/api'; // Nếu đổi port
```

---

### 🚀 Bước 6.4: Chạy Web Admin

```bash
npm start
```

**✅ Thành công:**
```
Compiled successfully!

You can now view web-admin in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

**🌐 Trình duyệt tự động mở:** http://localhost:3000

**✅ Web Admin đã chạy!**

---

## 7. KIỂM TRA DỰ ÁN HOẠT ĐỘNG

### ✅ Checklist hoàn chỉnh:

- [ ] **Backend chạy:** http://localhost:5000 hiển thị JSON
- [ ] **Mobile App chạy:** Expo Dev Tools mở tại http://localhost:19002
- [ ] **Web Admin chạy:** Dashboard mở tại http://localhost:3000

### 🧪 Test các tính năng:

#### Test 1: Test Backend API
```bash
# Mở terminal mới
curl http://localhost:5000/api/qa

# Hoặc mở trình duyệt:
# http://localhost:5000/api/qa
# Sẽ thấy danh sách Q&A dạng JSON
```

#### Test 2: Test Mobile App
1. Mở app trên điện thoại/web
2. Chọn ngôn ngữ "Tiếng Việt"
3. Gõ tin nhắn: "Tôi nhận được tin nhắn yêu cầu OTP"
4. Nhấn gửi
5. **Kết quả mong đợi:** Chatbot cảnh báo đây là lừa đảo

#### Test 3: Test Web Admin
1. Mở http://localhost:3000
2. Click "Quản lý Q&A" ở sidebar
3. Click "Thêm Q&A mới"
4. Điền thông tin:
   - Câu hỏi: "Test question"
   - Câu trả lời: "Test answer"
   - Ngôn ngữ: Tiếng Việt
5. Click "Lưu"
6. **Kết quả:** Q&A mới xuất hiện trong danh sách

---

## 8. XỬ LÝ LỖI THƯỜNG GẶP

### ❌ Lỗi 1: "Cannot connect to MongoDB"

**Nguyên nhân:** MongoDB chưa chạy hoặc connection string sai.

**Giải pháp:**

**Nếu dùng MongoDB Cloud:**
1. Kiểm tra connection string trong .env
2. Thay `<password>` bằng password thật
3. Kiểm tra Network Access đã allow 0.0.0.0/0

**Nếu dùng MongoDB Local:**
1. **Windows:**
   - Mở Task Manager (Ctrl+Shift+Esc)
   - Tab "Services"
   - Tìm "MongoDB" → Start

2. **Mac:**
   ```bash
   brew services start mongodb-community
   ```

3. **Linux:**
   ```bash
   sudo systemctl start mongod
   ```

---

### ❌ Lỗi 2: "Gemini API Error" hoặc "API key not valid"

**Nguyên nhân:** API key sai hoặc hết quota.

**Giải pháp:**
1. Kiểm tra GEMINI_API_KEY trong backend/.env
2. Đảm bảo không có khoảng trắng thừa
3. Tạo API key mới: https://makersuite.google.com/app/apikey
4. Copy lại key mới vào .env
5. Restart backend (Ctrl+C → npm start)

---

### ❌ Lỗi 3: Mobile app không kết nối được backend

**Biểu hiện:** App báo "Network error" hoặc không load được chatbot.

**Giải pháp:**

**Bước 1:** Kiểm tra backend đã chạy
```bash
# Mở trình duyệt
http://localhost:5000
# Phải thấy JSON response
```

**Bước 2:** Kiểm tra API URL trong mobile-app

**Nếu test trên web:**
- API_BASE_URL phải là: `http://localhost:5000/api`

**Nếu test trên điện thoại:**
- Tìm IP máy tính:
  ```bash
  # Windows
  ipconfig

  # Mac/Linux
  ifconfig
  ```
- Sửa API_BASE_URL: `http://192.168.x.x:5000/api`
- Restart Expo (Ctrl+C → npm start)

**Bước 3:** Tắt firewall tạm thời
- **Windows:** Settings → Firewall → Turn off
- **Mac:** System Preferences → Security → Firewall → Turn off

**Bước 4:** Đảm bảo cùng mạng WiFi
- Máy tính và điện thoại phải cùng WiFi

---

### ❌ Lỗi 4: "Port already in use"

**Lỗi:** EADDRINUSE: address already in use :::5000

**Giải pháp 1: Kill process đang dùng port**

**Windows:**
```bash
# Tìm PID
netstat -ano | findstr :5000

# Kill (thay 1234 bằng PID thực tế)
taskkill /PID 1234 /F
```

**Mac/Linux:**
```bash
# Kill process port 5000
lsof -ti:5000 | xargs kill -9

# Kill process port 3000
lsof -ti:3000 | xargs kill -9
```

**Giải pháp 2: Đổi port**

**Backend:** Sửa .env
```env
PORT=5001
```

**Web Admin:** Không cần sửa gì (React tự động dùng port khác nếu 3000 bị chiếm)

Nhớ cập nhật API_BASE_URL ở mobile-app và web-admin nếu đổi port backend.

---

### ❌ Lỗi 5: "npm install" bị lỗi hoặc treo

**Giải pháp:**

```bash
# Xóa cache npm
npm cache clean --force

# Xóa node_modules và package-lock.json
rm -rf node_modules package-lock.json

# Cài lại
npm install
```

**Windows (dùng Command Prompt):**
```bash
npm cache clean --force
rmdir /s /q node_modules
del package-lock.json
npm install
```

---

### ❌ Lỗi 6: Expo "Unable to resolve module"

**Giải pháp:**

```bash
# Trong thư mục mobile-app
rm -rf node_modules
rm package-lock.json
npm install

# Clear Expo cache
npm start -- --clear
```

---

### ❌ Lỗi 7: "SyntaxError: Identifier 'textToSpeech' has already been declared"

**✅ ĐÃ SỬA!**

Lỗi này đã được khắc phục tự động trong file `backend/services/tts.service.js`.

Nếu vẫn gặp, pull code mới nhất:
```bash
cd backend
git pull origin main
npm install
npm start
```

---

## 🎉 HOÀN TẤT CÀI ĐẶT!

### 📋 Tóm tắt các lệnh chạy dự án:

**Terminal 1 - Backend:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\backend
npm start
```

**Terminal 2 - Mobile App:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\mobile-app
npm start
# Nhấn 'w' để mở web, hoặc quét QR trên điện thoại
```

**Terminal 3 - Web Admin:**
```bash
cd C:\Users\ADMIN\SANGKIENTG\web-admin
npm start
```

### 🌐 Các URL quan trọng:

| Thành phần | URL | Mô tả |
|------------|-----|-------|
| Backend API | http://localhost:5000 | API server |
| Web Admin | http://localhost:3000 | Dashboard quản trị |
| Mobile Web | http://localhost:19006 | App chạy trên web |
| Expo DevTools | http://localhost:19002 | Công cụ Expo |

---

## 📞 HỖ TRỢ

### Khi gặp vấn đề:

1. **Đọc lại phần "Xử lý lỗi thường gặp"** (Mục 8)
2. **Kiểm tra logs/errors** trong terminal
3. **Google error message** để tìm giải pháp
4. **Restart tất cả:**
   - Tắt tất cả terminal (Ctrl+C)
   - Khởi động lại theo thứ tự: Backend → Mobile → Web

### Log files để debug:

- Backend logs: Xem trong terminal backend
- Expo logs: Xem trong Expo DevTools
- Browser console: F12 trong trình duyệt

---

## 🎯 CHECKLIST TRƯỚC KHI DEMO/TRÌNH BÀY

- [ ] Backend chạy ổn định > 5 phút không lỗi
- [ ] Test chatbot trả lời đúng với ít nhất 3 câu hỏi khác nhau
- [ ] Mobile app load được và gửi tin nhắn thành công
- [ ] Web admin mở được và hiển thị danh sách Q&A
- [ ] Chuẩn bị sẵn 3-5 kịch bản demo (ví dụ: lừa đảo OTP, mạo danh ngân hàng)
- [ ] Chụp screenshot các màn hình đề phòng demo bị lỗi
- [ ] Backup plan: Nếu app lỗi, dùng web preview (npm run web)

---

## 📚 TÀI LIỆU THAM KHẢO

- **Node.js Tutorial:** https://nodejs.org/en/docs/guides/
- **MongoDB Tutorial:** https://www.mongodb.com/docs/manual/tutorial/
- **Expo Documentation:** https://docs.expo.dev/
- **React Tutorial:** https://react.dev/learn
- **Gemini API Docs:** https://ai.google.dev/docs

---

**📌 LƯU Ý QUAN TRỌNG:**

1. **Luôn chạy Backend trước**, sau đó mới chạy Mobile/Web
2. **Không tắt terminal** khi các app đang chạy
3. **Ctrl+C** để dừng một app trong terminal
4. **npm start** để chạy lại app
5. **Kiên nhẫn** khi lần đầu cài đặt, mọi thứ sẽ dễ dàng hơn sau lần đầu

---

**🎓 Chúc bạn thành công với dự án AGRIBANK DIGITAL GUARD!**

**Version:** 1.0.0
**Last Updated:** 2024
**Author:** Agribank Digital Guard Team

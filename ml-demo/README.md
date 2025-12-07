# ML Fraud Detection Demo

Dashboard demo cho hệ thống ML Fraud Detection.

## Tính năng

- **Dashboard**: Tổng quan metrics và biểu đồ
- **Real-time Monitor**: Giám sát giao dịch real-time
- **Transaction Test**: Test giao dịch đơn lẻ
- **Batch Analysis**: Phân tích nhiều giao dịch
- **Model Training**: Train và đánh giá models
- **Data Explorer**: Khám phá và tạo dữ liệu
- **Reports**: Báo cáo chi tiết

## Cài đặt

```bash
cd ml-demo
npm install
```

## Chạy ứng dụng

```bash
npm start
```

Ứng dụng sẽ chạy tại: http://localhost:3001

## Kết nối với ML Service

Mặc định kết nối đến: http://localhost:5001

Có thể thay đổi bằng biến môi trường:
```
REACT_APP_ML_SERVICE_URL=http://your-ml-service:5001
```

## Màu sắc Theme

- Primary: #1976d2 (xanh dương)
- Secondary: #dc004e (hồng đậm)
- Success: #4caf50 (xanh lá)
- Warning: #ff9800 (cam)
- Error: #f44336 (đỏ)

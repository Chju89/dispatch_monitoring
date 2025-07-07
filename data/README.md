# 📁 data/

Thư mục `data/` dùng để tổ chức toàn bộ dữ liệu của pipeline, theo chuẩn MLOps:

## 📂 Cấu trúc

```
data/
├── raw/              # Dữ liệu gốc (video, annotation)
│   ├── frames/       # Các frame trích xuất từ video
│   └── Dataset/      
│       ├── Detection/        # Dữ liệu dùng để train YOLOv8
│       └── Classification/   # Dữ liệu phân loại dish/tray
├── processed/        
│   └── tracking/     # Kết quả tracking (video sau khi gán ID)
├── feedback/         # Dữ liệu thu thập lại từ người dùng, gắn nhãn lại
```

## 🔍 Ghi chú

- `raw/`: không chỉnh sửa, chứa dữ liệu gốc.
- `processed/`: kết quả xử lý tự động như video tracking, ảnh inference.
- `feedback/`: sẽ dùng cho quá trình retrain/active learning.

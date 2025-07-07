# 📁 src/

Thư mục chứa toàn bộ source code chính của pipeline.

## 📄 Các thành phần chính

- `tracking.py`: chạy YOLOv8 + DeepSORT → xuất video có ID
- `detect.py`: nếu cần detect ảnh riêng lẻ
- `classify.py`: phân loại dish/tray nếu cần (giai đoạn sau)
- `retrain.py`: hỗ trợ retrain từ dữ liệu feedback (nếu có)

## 🔄 Mục tiêu

Modular hoá, dễ bảo trì, rõ ràng.

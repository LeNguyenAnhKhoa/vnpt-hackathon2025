# Hướng dẫn Build và Push Docker Image

Tài liệu này hướng dẫn cách build Docker image cho dự án và push lên Docker Hub.

## 1. Chuẩn bị

Đảm bảo bạn đã cài đặt Docker Desktop và đã đăng nhập vào Docker Hub.
```bash
docker login
```

## 2. Build Docker Image

Mở terminal tại thư mục gốc của dự án và chạy lệnh sau:

```bash
# Thay thế 'your_username' bằng tên tài khoản Docker Hub của bạn
# Thay thế 'vnpt-hackathon-submission' bằng tên image bạn muốn đặt
docker build -t your_username/vnpt-hackathon-submission:latest .
```

Ví dụ:
```bash
docker build -t nguyenvana/vnpt-hackathon-2025:v1 .
```

## 3. Kiểm tra Image (Chạy thử)

Trước khi push, hãy chạy thử image để đảm bảo mọi thứ hoạt động đúng.

```bash
# Chạy container
docker run --rm -it your_username/vnpt-hackathon-submission:latest
```

Nếu bạn muốn mount thư mục output ra ngoài để xem kết quả:
```bash
docker run --rm -v ${PWD}/output:/app/output your_username/vnpt-hackathon-submission:latest
```
(Trên Windows PowerShell, thay `${PWD}` bằng `${PWD}` hoặc đường dẫn tuyệt đối).

Container sẽ:
1. Khởi động Qdrant (sử dụng dữ liệu từ `vectorDB/qdrant_storage` đã được đóng gói trong image).
2. Chạy `predict.py`.
3. Tắt Qdrant và kết thúc.

## 4. Push lên Docker Hub

Sau khi kiểm tra thành công, push image lên Docker Hub:

```bash
docker push your_username/vnpt-hackathon-submission:latest
```

## Lưu ý quan trọng

- **Dữ liệu**: Image này bao gồm toàn bộ dữ liệu trong `vectorDB/qdrant_storage`. Điều này làm cho image có thể hoạt động độc lập mà không cần mount volume dữ liệu bên ngoài.
- **API Keys**: File `api-keys.json` được copy vào trong image. Hãy đảm bảo bạn không push image này lên public repository nếu key là bí mật quan trọng (đối với bài thi Hackathon thì thường chấp nhận được nếu nộp private).
- **Cấu trúc**:
    - `Dockerfile`: Cấu hình build image (Multi-stage build: lấy Qdrant binary + Python env).
    - `entrypoint.sh`: Script khởi động Qdrant background và chạy `predict.py`.
    - `.dockerignore`: Loại bỏ các file không cần thiết (logs, img, data gốc...) để giảm kích thước image.

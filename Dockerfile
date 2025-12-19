# Stage 1: Get Qdrant binary
FROM qdrant/qdrant:latest AS qdrant-source

# Stage 2: Final image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libunwind8 \
    && rm -rf /var/lib/apt/lists/*

# Copy Qdrant binary and config
COPY --from=qdrant-source /qdrant/qdrant /usr/local/bin/qdrant
RUN mkdir -p /qdrant/config
COPY --from=qdrant-source /qdrant/config /qdrant/config

# Set working directory
WORKDIR /code

# 1. Cài đặt thư viện trước (để tận dụng cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY CÁC FILE CODE QUAN TRỌNG (Tránh copy folder qdrant_storage ở đây)
COPY predict.py .
COPY entrypoint.sh .
COPY api-keys.json .
# Nếu bạn có dùng các file phụ trong vectorDB (trừ storage), hãy copy riêng:
COPY vectorDB/main_async.py ./vectorDB/
# COPY thêm các file/folder code khác nếu cần (ví dụ folder img, utils...)

# 3. SETUP DATABASE (Đây là dòng quan trọng nhất đảm bảo có Data)
RUN mkdir -p /qdrant/storage
# Dòng này đưa dữ liệu vào đúng chỗ Qdrant đọc
COPY vectorDB/qdrant_storage /qdrant/storage

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose ports
EXPOSE 6333 6334

# Set entrypoint
ENTRYPOINT ["/code/entrypoint.sh"]
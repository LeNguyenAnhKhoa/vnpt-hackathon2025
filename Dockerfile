# Stage 1: Get Qdrant binary
FROM qdrant/qdrant:latest AS qdrant-source

# Stage 2: Final image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Qdrant binary and config
COPY --from=qdrant-source /qdrant/qdrant /usr/local/bin/qdrant
# Create config directory
RUN mkdir -p /qdrant/config
# Copy config if available, otherwise Qdrant uses defaults
COPY --from=qdrant-source /qdrant/config /qdrant/config

# Set working directory
WORKDIR /code

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Setup Qdrant storage
# Create directory and copy local storage
RUN mkdir -p /qdrant/storage
COPY vectorDB/qdrant_storage /qdrant/storage

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose ports (Qdrant ports)
EXPOSE 6333 6334

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
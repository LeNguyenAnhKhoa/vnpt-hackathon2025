#!/bin/bash
set -e

# Start Qdrant in background
# We assume Qdrant binary is in PATH or /usr/local/bin
# and storage is mapped to /qdrant/storage

echo "Starting Qdrant..."
# Set Qdrant environment variables if not set in Dockerfile
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334

# Run Qdrant in background
qdrant &
QDRANT_PID=$!

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
MAX_RETRIES=30
count=0
while ! curl -s http://localhost:6333/healthz > /dev/null; do
    sleep 1
    count=$((count+1))
    if [ $count -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for Qdrant to start"
        exit 1
    fi
done
echo "Qdrant is ready."

# Run the prediction script
echo "Running prediction..."
# Pass all arguments to the script
python predict.py "$@"

# Capture exit code of python script
EXIT_CODE=$?

# Kill Qdrant
echo "Stopping Qdrant..."
kill $QDRANT_PID

exit $EXIT_CODE

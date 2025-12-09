# VNPT Hackathon 2025

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Qdrant with Docker

```bash
docker-compose up -d
```

Qdrant will be available at:

- HTTP API: `http://localhost:6333`
- gRPC: `localhost:6334`
- **Dashboard**: `http://localhost:6333/dashboard`

### 3. Run Pipeline
* Create vector database:
```bash
python vectorDB\main_async.py
```

* Make prediction:
```bash
python predict.py
```

### 4. Data
* Folder vectorDB\data:
```
https://drive.google.com/drive/u/0/folders/1dUnodqUE3Ea0ESjACpEEGARFgtiua-KR
```
---

## API Keys Configuration

The `api-keys.json` file contains API credentials for VNPT AI Services:

```json
[
  {
    "authorization": "Bearer <JWT_TOKEN>",
    "tokenKey": "<PUBLIC_KEY_BASE64>",
    "llmApiName": "<MODEL_NAME>",
    "tokenId": "<TOKEN_UUID>"
  }
]
```

| Field | Description |
|-------|-------------|
| `authorization` | JWT Bearer token for API authentication |
| `tokenKey` | RSA public key (Base64) for encryption |
| `llmApiName` | Model name: `LLM large`, `LLM small`, or `LLM embedings` |
| `tokenId` | Token UUID identifier |

> **Note:** Configure `api-keys.json` with valid credentials before running `predict.py`.

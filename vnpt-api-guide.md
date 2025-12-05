# VNPT AI API - Hướng dẫn sử dụng

## 1. Giới thiệu

Tài liệu hướng dẫn tích hợp API LLM/Embedding Track 2 cho cuộc thi VNPT AI - Age of AInicorns - Track 2 - The Builder.

## 2. Cấu hình

### Lấy API Key
- Đăng nhập tại portal VNPT AI - Age of AInicorns
- Download API key tại tab **Instruction**

### Thiết lập credentials
```python
import requests

# Thông tin xác thực
AUTHORIZATION = "Bearer your_access_token"
TOKEN_ID = "your_token_id"
TOKEN_KEY = "your_token_key"
BASE_URL = "https://api.idg.vnpt.vn/data-service"
```

## 3. API Endpoints

### 3.1. LLM Small - Sinh câu trả lời

**Endpoint:** `/v1/chat/completions/vnptai-hackathon-small`

**Quota:** 1000 req/ngày, 60 req/giờ

**Tham số chính:**
- `model`: `vnptai_hackathon_small`
- `messages`: Lịch sử hội thoại (role: system/user/assistant)
- `temperature`: Độ ngẫu nhiên (0-2, mặc định 1.0)
- `top_p`: Ngưỡng xác suất tích lũy
- `top_k`: Số token xem xét
- `n`: Số câu trả lời
- `max_completion_tokens`: Giới hạn token đầu ra
- `presence_penalty`: Phạt token đã xuất hiện (-2.0 đến 2.0)
- `frequency_penalty`: Phạt token lặp lại (-2.0 đến 2.0)
- `response_format`: Định dạng đầu ra (vd: `{"type": "json_object"}`)
- `seed`: Tái tạo kết quả (deterministic)
- `tools`: Danh sách functions
- `tool_choice`: Kiểm soát gọi tool (auto/none/tên hàm)
- `logprobs`: Trả về log probabilities
- `top_logprobs`: Số token có xác suất cao nhất (0-20)

**Code mẫu:**
```python
import requests

headers = {
    'Authorization': 'Bearer #Authorization',
    'Token-id': '#TokenID',
    'Token-key': '#TokenKey',
    'Content-Type': 'application/json',
}

json_data = {
    'model': 'vnptai_hackathon_small',
    'messages': [
        {
            'role': 'user',
            'content': 'Hi, VNPT AI.',
        },
    ],
    'temperature': 1.0,
    'top_p': 1.0,
    'top_k': 20,
    'n': 1,
    'max_completion_tokens': 10,
}

response = requests.post(
    'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small',
    headers=headers,
    json=json_data
)

result = response.json()
print(result)
```

**Response mẫu:**
```json
{
  "id": "chatcmpl-2f2c048b7af24514b3cd6c4cfa0ec97d",
  "object": "chat.completion",
  "created": 1764754595,
  "model": "vnptai_hackathon_large",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Chào bạn! Tôi là VNPT AI.",
        "refusal": null,
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": null,
    "total_tokens": null,
    "completion_tokens": null
  }
}
```

### 3.2. LLM Large - Sinh câu trả lời

**Endpoint:** `/v1/chat/completions/vnptai-hackathon-large`

**Quota:** 500 req/ngày, 40 req/giờ

**Tham số:** Giống LLM Small

**Code mẫu:**
```python
import requests

headers = {
    'Authorization': 'Bearer #Authorization',
    'Token-id': '#TokenID',
    'Token-key': '#TokenKey',
    'Content-Type': 'application/json',
}

json_data = {
    'model': 'vnptai_hackathon_large',
    'messages': [
        {
            'role': 'user',
            'content': 'Hi, VNPT AI.',
        },
    ],
    'temperature': 1.0,
    'top_p': 1.0,
    'top_k': 20,
    'n': 1,
    'max_completion_tokens': 10,
}

response = requests.post(
    'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large',
    headers=headers,
    json=json_data
)

result = response.json()
print(result)
```

### 3.3. Embedding - Vector hóa văn bản

**Endpoint:** `/vnptai-hackathon-embedding`

**Quota:** 500 req/phút

**Tham số:**
- `model`: `vnptai_hackathon_embedding`
- `input`: Văn bản cần vector hóa
- `encoding_format`: Định dạng encoding (float/base64)

**Code mẫu:**
```python
import requests

headers = {
    'Authorization': 'Bearer #Authorization',
    'Token-id': '#TokenID',
    'Token-key': '#TokenKey',
    'Content-Type': 'application/json',
}

json_data = {
    'model': 'vnptai_hackathon_embedding',
    'input': 'Xin chào, mình là VNPT AI.',
    'encoding_format': 'float',
}

response = requests.post(
    'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
    headers=headers,
    json=json_data
)

result = response.json()
print(result)
```

**Response mẫu:**
```json
{
  "data": [
    {
      "index": 0,
      "embedding": [-0.044116780161857605, -0.021570704877376556, ...]
    }
  ],
  "model": "vnptai_hackathon_embedding",
  "id": "embd-7db936b0881543cea0b95b5c182abd09",
  "object": "list"
}
```

## 4. Lưu ý

- Tuân thủ quota của từng API
- Sử dụng đúng model name cho từng endpoint
- Kiểm tra response status code và xử lý lỗi phù hợp
- Bảo mật thông tin API key

---

**Tài liệu:** VNPT IT - LLM API Compatible Track 2
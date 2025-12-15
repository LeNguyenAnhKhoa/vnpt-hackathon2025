import argparse
import json
import requests
from pathlib import Path
import time
import logging
import re
from datetime import datetime

TARGET_QIDS = [
    "test_0056",
]

FILE_CONTEXT = 'output/predict_val.json'
FILE_ORIGINAL = 'data/test2.json'

# ===================== CẤU HÌNH API =====================
API_URL_LARGE = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'
LLM_API_NAME_LARGE = 'vnptai_hackathon_large'

API_URL_SMALL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'
LLM_API_NAME_SMALL = 'vnptai_hackathon_small'

MAX_RETRIES = 5
RETRY_DELAY = 5

# ===================== LOGGING =====================
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f'two_stage_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== HELPER: JSON PARSER (QUAN TRỌNG) =====================
def clean_and_parse_json(content):
    """Làm sạch và parse JSON từ response của LLM"""
    try:
        # Trường hợp 1: LLM trả về đúng JSON thuần
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Trường hợp 2: LLM bọc trong markdown ```json ... ```
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            
            # Trường hợp 3: Cố gắng tìm cặp ngoặc {} đầu tiên và cuối cùng
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
                
        except Exception as e:
            logger.error(f"Failed to parse JSON content: {e}")
            return None
    return None

# ===================== LOAD CREDENTIALS =====================
def load_credentials(json_path='./api-keys.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        large = next((item for item in keys if item.get('llmApiName') == 'LLM large'), None)
        small = next((item for item in keys if item.get('llmApiName') == 'LLM small'), None)
        return large, small
    except Exception as e:
        logger.error(f"Lỗi load key: {e}")
        exit(1)

config_large, config_small = load_credentials()
if not config_large or not config_small:
    logger.error("Thiếu cấu hình API Key")
    exit(1)

AUTHORIZATION_LARGE = config_large['authorization']
TOKEN_ID_LARGE = config_large['tokenId']
TOKEN_KEY_LARGE = config_large['tokenKey']

AUTHORIZATION_SMALL = config_small['authorization']
TOKEN_ID_SMALL = config_small['tokenId']
TOKEN_KEY_SMALL = config_small['tokenKey']

# ===================== PROMPT STAGE 1: LARGE MODEL =====================
def create_prompt_large(question: str, ref_documents: list) -> tuple:
    context_text = ""
    # for i, doc in enumerate(ref_documents):
    #     text = doc.get('text') or doc.get('content') or ""
    #     context_text += f"\n[Tài liệu tham khảo {i+1}]\n{text}\n"

    system_prompt = """Bạn là một Chuyên gia Khoa học & Logic (Expert Reasoner).

Nhiệm vụ DUY NHẤT:
- Phân tích bài toán
- Thiết lập công thức / phương trình đúng
- Thay số đầy đủ
- Viết biểu thức số TƯỜNG MINH để phục vụ kiểm tra

QUY TẮC BẮT BUỘC:
1. Nếu bài toán yêu cầu kết quả số:
   - BẮT BUỘC phải tạo ra một biểu thức chỉ gồm số và phép toán.
   - Không được kết luận chỉ bằng công thức chữ.

2. Không được gộp phép tính:
   - Mỗi bước biến đổi phải rõ ràng.
   - Không viết tắt nhiều phép toán trong một dòng.

3. KHÔNG chọn đáp án A/B/C/D. Chỉ đưa ra lời giải.

ĐẦU RA (JSON BẮT BUỘC):
{
  "problem_identification": "Bài toán yêu cầu tính gì",
  "formula": "Công thức tổng quát",
  "numeric_expression": "Biểu thức số đã thay đầy đủ (Ví dụ: '0.8 * 0.15 + 0.2 * -0.1')",
  "step_by_step_evaluation": [
    "Bước 1: ...",
    "Bước 2: ..."
  ],
  "intermediate_result": "Giá trị số thu được"
}
"""
    user_prompt = f"""## Tài Liệu Tham Khảo:
{context_text}

## Câu Hỏi:
{question}

Hãy đưa ra lời giải chi tiết dưới dạng JSON."""
    return system_prompt, user_prompt

# ===================== PROMPT STAGE 2: SMALL MODEL =====================
def create_prompt_small(question: str, choices: list, large_output: dict) -> tuple:
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
    
    # Trích xuất thông tin an toàn từ Large output
    if not isinstance(large_output, dict):
        large_output = {}
        
    numeric_expression = large_output.get('numeric_expression', 'Không có biểu thức số')
    derived_result = large_output.get('intermediate_result', 'N/A')
    steps = json.dumps(large_output.get('step_by_step_evaluation', []), ensure_ascii=False)

    system_prompt = """Bạn là một Kiểm toán viên Toán học & Logic.

Nhiệm vụ DUY NHẤT:
1. Kiểm tra lời giải đề xuất từ mô hình khác.
2. TÍNH TOÁN LẠI biểu thức số (numeric_expression) một cách độc lập.
3. Đối chiếu kết quả tính lại với các lựa chọn A, B, C, D.

QUY TẮC BẮT BUỘC:
1. Nếu numeric_expression hợp lệ: Hãy tin vào kết quả tính toán lại của bạn.
2. Chọn đáp án có giá trị gần đúng nhất với kết quả tính toán.
3. Nếu không có thông tin tính toán, hãy tự suy luận để chọn đáp án.

## Định Dạng Đầu Ra (JSON)
{
  "verification_process": "Tôi đã tính lại [biểu thức] ra kết quả [giá trị]. So sánh với các lựa chọn...",
  "answer": "X"
}
Trong đó "X" là chữ cái của đáp án bạn chọn (A, B, C, D...).
"""

    user_prompt = f"""## Đề Bài:
{question}

## Các Lựa Chọn:
{formatted_choices}

## Lời Giải Đề Xuất (Cần kiểm tra):
- Biểu thức số cần tính lại: {numeric_expression}
- Kết quả đề xuất ban đầu: {derived_result}
- Các bước giải: {steps}

Hãy kiểm tra, tính toán lại và đưa ra quyết định cuối cùng."""
    return system_prompt, user_prompt

# ===================== API CALL FUNCTION =====================
def call_llm(url, token_id, token_key, auth, model_name, system, user, temp=0.1):
    headers = {
        'Authorization': auth,
        'Token-id': token_id,
        'Token-key': token_key,
        'Content-Type': 'application/json',
    }
    json_data = {
        'model': model_name,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'temperature': temp,
        'response_format': {'type': 'json_object'},
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=json_data, timeout=120)
            
            if response.status_code == 429:
                logger.warning(f"Rate limit 429 ({model_name}). Sleeping...")
                time.sleep(RETRY_DELAY * 2)
                continue
                
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Sử dụng hàm parser an toàn
            parsed_json = clean_and_parse_json(content)
            if parsed_json:
                return parsed_json
            else:
                logger.warning(f"Invalid JSON format from {model_name}. Content: {content[:50]}...")
                
        except Exception as e:
            logger.error(f"API Error ({model_name}) attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY)
            
    return None

# ===================== MAIN PROCESS =====================
def load_json_as_dict(filepath, key_field='qid'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item[key_field]: item for item in data}
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        exit(1)

def main():
    logger.info("Starting Two-Stage Pipeline...")
    
    context_map = load_json_as_dict(FILE_CONTEXT)
    original_map = load_json_as_dict(FILE_ORIGINAL)
    
    results = []
    
    for qid in TARGET_QIDS:
        logger.info(f"\nProcessing {qid}...")
        
        if qid not in original_map:
            logger.error(f"Skipping {qid}: Not found.")
            continue
            
        question = original_map[qid].get('question', '')
        choices = original_map[qid].get('choices', [])
        ref_docs = context_map.get(qid, {}).get('reference_docs', [])
        
        # --- STAGE 1: LARGE (REASONING) ---
        logger.info("  -> Stage 1: Reasoning...")
        sys_1, user_1 = create_prompt_large(question, ref_docs)
        large_out = call_llm(
            API_URL_LARGE, TOKEN_ID_LARGE, TOKEN_KEY_LARGE, AUTHORIZATION_LARGE, 
            LLM_API_NAME_LARGE, sys_1, user_1, temp=0.2
        )
        
        if not large_out:
            logger.error("     Stage 1 Failed. Using empty reasoning.")
            large_out = {} # Fallback rỗng để Stage 2 vẫn chạy

        # --- STAGE 2: SMALL (VERIFYING) ---
        logger.info("  -> Stage 2: Verifying...")
        sys_2, user_2 = create_prompt_small(question, choices, large_out)
        small_out = call_llm(
            API_URL_SMALL, TOKEN_ID_SMALL, TOKEN_KEY_SMALL, AUTHORIZATION_SMALL,
            LLM_API_NAME_SMALL, sys_2, user_2, temp=0.0
        )
        
        final_answer = 'A'
        final_reason = ""
        
        if small_out:
            final_answer = small_out.get('answer', 'A').strip().upper()
            
            # Gộp lý do để debug
            final_reason = json.dumps({
                "large_step": large_out,
                "small_check": small_out.get('verification_process', '')
            }, ensure_ascii=False)
            
            logger.info(f"     Answer: {final_answer}")
        else:
            logger.error("     Stage 2 Failed. Default to A.")
            final_reason = json.dumps(large_out, ensure_ascii=False)

        results.append({
            'qid': qid,
            'predict': final_answer,
            'reason': final_reason,
        })
        
        time.sleep(2)

    # Save Results
    output_dir = Path('output_stem')
    output_dir.mkdir(exist_ok=True)
    json_out = output_dir / 'predict_two_stage.json'
    
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Done. Saved to {json_out}")

if __name__ == '__main__':
    main()
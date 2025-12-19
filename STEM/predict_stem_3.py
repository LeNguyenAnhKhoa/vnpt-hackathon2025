import argparse
import json
import requests
from pathlib import Path
import time
import logging
import re
from datetime import datetime

TARGET_QIDS = [
    "val_0024",
    "val_0025",
    "val_0027",
    "val_0071",
    "val_0076",
    "val_0087"
]

FILE_CONTEXT = 'output/predict_val.json'
FILE_ORIGINAL = 'data/val.json'

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

# ===================== STAGE 0: CLASSIFICATION (LLM SMALL) =====================
def create_classification_prompt(question: str) -> tuple:
    system = """Bạn là bộ lọc phân loại câu hỏi.
Nhiệm vụ: Xác định xem câu hỏi có thuộc nhóm STEM hoặc Tính toán hoặc Logic định lượng hay không.

NHÓM STEM (True):
- Toán học, Vật lý, Hóa học, Sinh học.
- Câu hỏi yêu cầu tính toán số liệu cụ thể.
- Câu hỏi về công thức, phương trình, định luật khoa học.

NHÓM KHÁC (False):
- Văn học, Lịch sử, Địa lý, Xã hội.
- Câu hỏi chỉ yêu cầu nhớ kiến thức (không cần suy luận logic).

OUTPUT JSON:
{
  "is_stem": true/false,
  "reason": "Lý do ngắn gọn"
}
"""
    user = f"Câu hỏi: {question}\n\nHãy phân loại."
    return system, user

# ===================== PROMPT STAGE 1: LARGE MODEL =====================
def create_prompt_large(question: str) -> tuple:
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

# ===================== PROCESSING LOGIC =====================
def process_test_file(input_path, output_dir, start_idx=None, end_idx=None):
    # 1. Load Data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Not found: {input_path}")
        return

    # 2. Slice Data
    if start_idx is None: start_idx = 0
    if end_idx is None: end_idx = len(data)
    
    batch_data = data[start_idx:end_idx]
    logger.info(f"Processing {len(batch_data)} samples (Index {start_idx} to {end_idx})")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for idx, item in enumerate(batch_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        logger.info(f"\nProcessing [{start_idx + idx}] QID: {qid}")

        # === HEURISTIC RULE: RAG BYPASS ===
        # Kiểm tra điều kiện: Bắt đầu bằng "Đoạn thông tin" VÀ > 200 từ
        word_count = len(question.split())
        is_rag_start = question.strip().startswith("Đoạn thông tin")

        if is_rag_start and word_count > 200:
            logger.info(f"  -> HEURISTIC SKIP: RAG Question detected ({word_count} words). Defaulting to 'A'.")
            
            results.append({
                'qid': qid,
                'predict': 'A',
                'reason': f"Heuristic: Starts with 'Đoạn thông tin' & Length {word_count} > 200"
            })
            
            # Lưu checkpoint nếu cần và chuyển sang câu tiếp theo luôn
            if (idx + 1) % 10 == 0:
                temp_path = Path(output_dir) / f'temp_results_{start_idx}.json'
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            continue 

        # --- STEP 0: CLASSIFICATION (SMALL) ---
        # Chỉ chạy xuống đây nếu KHÔNG thỏa mãn điều kiện RAG ở trên
        sys_0, user_0 = create_classification_prompt(question)
        class_out = call_llm(API_URL_SMALL, TOKEN_ID_SMALL, TOKEN_KEY_SMALL, AUTHORIZATION_SMALL,
                             LLM_API_NAME_SMALL, sys_0, user_0, temp=0.0)
        
        is_stem = False
        if class_out and class_out.get('is_stem') is True:
            is_stem = True
            logger.info(f"  -> Type: STEM/Calculation ({class_out.get('reason')})")
        else:
            logger.info(f"  -> Type: Non-STEM (Defaulting to A)")

        final_answer = 'A'
        final_reason = "Non-STEM Default"

        # --- BRANCHING ---
        if not is_stem:
            # Non-STEM -> Default A
            final_answer = 'A'
        
        else:
            # STEM -> Run Two-Stage Pipeline
            
            # Stage 1: Large Reasoner
            logger.info("  -> Stage 1: Large Reasoning...")
            sys_1, user_1 = create_prompt_large(question)
            large_out = call_llm(API_URL_LARGE, TOKEN_ID_LARGE, TOKEN_KEY_LARGE, AUTHORIZATION_LARGE,
                                 LLM_API_NAME_LARGE, sys_1, user_1, temp=0.2)
            
            if not large_out: large_out = {}

            # Stage 2: Small Verifier
            logger.info("  -> Stage 2: Small Verifying...")
            sys_2, user_2 = create_prompt_small(question, choices, large_out)
            small_out = call_llm(API_URL_SMALL, TOKEN_ID_SMALL, TOKEN_KEY_SMALL, AUTHORIZATION_SMALL,
                                 LLM_API_NAME_SMALL, sys_2, user_2, temp=0.0)
            
            if small_out:
                final_answer = small_out.get('answer', 'A').strip().upper()
                final_reason = json.dumps({
                    "large": large_out,
                    "small": small_out.get('check_calc')
                }, ensure_ascii=False)
            else:
                final_reason = "Error in Stage 2"

        logger.info(f"  => Final Answer: {final_answer}")

        # Save Result
        results.append({
            'qid': qid,
            'predict': final_answer,
            'reason': final_reason
        })
        
        # Save check-point every 10 items (optional but good practice)
        if (idx + 1) % 10 == 0:
            temp_path = Path(output_dir) / f'temp_results_{start_idx}.json'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Final Save
    final_path = Path(output_dir) / f'predict_final_{start_idx}_{end_idx}.json'
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Done! Saved to {final_path}")

# ===================== MAIN PARSER =====================
def main():
    parser = argparse.ArgumentParser(description="STEM Two-Stage Predictor")
    parser.add_argument('--input', type=str, default='data/test2.json', help='Path to input JSON')
    parser.add_argument('--output-dir', type=str, default='output_stem', help='Output directory')
    parser.add_argument('--start', type=int, default=None, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index')
    
    args = parser.parse_args()
    
    process_test_file(args.input, args.output_dir, args.start, args.end)

if __name__ == '__main__':
    main()
import argparse
import json
import requests
from pathlib import Path
import time
import logging
import re
from datetime import datetime

# ===================== CẤU HÌNH API =====================
API_URL_LARGE = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'
LLM_API_NAME_LARGE = 'vnptai_hackathon_large'

API_URL_SMALL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'
LLM_API_NAME_SMALL = 'vnptai_hackathon_small'

MAX_RETRIES = 5
RETRY_DELAY = 5

# ===================== LOGGING SETUP =====================
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f'target_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== HELPER: JSON PARSER =====================
def clean_and_parse_json(content):
    """Làm sạch và parse JSON từ response của LLM"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Thử tìm block json
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
            # Thử tìm cặp ngoặc {}
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
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
            parsed = clean_and_parse_json(content)
            if parsed: return parsed
        except Exception as e:
            logger.error(f"API Error ({model_name}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY)
    return None

# ===================== STAGE 1: REASONING (LLM LARGE) =====================
def create_prompt_large(question: str) -> tuple:
    system = """Bạn là Chuyên gia Khoa học Tự nhiên (Expert STEM Reasoner).

Nhiệm vụ: Giải quyết bài toán một cách chi tiết để tìm ra kết quả chính xác.

QUY TRÌNH BẮT BUỘC:
1. Xác định vấn đề.
2. Thiết lập công thức HOẶC Phương trình phản ứng HOẶC Luận điểm logic.
3. THAY SỐ (nếu có) hoặc CÂN BẰNG (nếu là hóa học).
4. Đưa ra kết luận cuối cùng (Kết quả số hoặc Mệnh đề đúng).

LƯU Ý QUAN TRỌNG: 
- Nếu là bài toán TÍNH TOÁN: Phải viết `key_expression` là biểu thức số tường minh (VD: "10 * 2.5 / 100").
- Nếu là bài toán HÓA HỌC/LÝ THUYẾT: Phải viết `key_expression` là phương trình/luận điểm chính.
- Nếu bài toán không liên quan đến sử dụng công thức, hãy mô tả cách giải thích hợp và đưa ra kết luận cuối cùng trong `final_result`.

OUTPUT JSON:
{
  "method": "Công thức / Định luật sử dụng",
  "key_expression": "Biểu thức số hoặc Phương trình quan trọng nhất",
  "step_by_step": ["Bước 1...", "Bước 2..."],
  "final_result": "Kết quả cuối cùng (Số hoặc Đáp án chữ)"
}
"""
    user = f"Câu hỏi: {question}\n\nHãy giải chi tiết."
    return system, user

# ===================== STAGE 2: VERIFICATION (LLM SMALL) =====================
def create_prompt_small(question: str, choices: list, large_output: dict) -> tuple:
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
    
    key_expr = large_output.get('key_expression', 'N/A')
    result = large_output.get('final_result', 'N/A')
    steps = json.dumps(large_output.get('step_by_step', []), ensure_ascii=False)

    system = """Bạn là Kiểm toán viên. Nhiệm vụ:
1. Kiểm tra lời giải của Chuyên gia.
2. Nếu `key_expression` là biểu thức SỐ: Hãy TÍNH TOÁN LẠI.
3. Nếu `key_expression` là Phương trình Hóa học/Logic: Hãy kiểm tra xem nó ĐÚNG hay SAI (cân bằng chưa, logic đúng không).
4. Chọn đáp án A, B, C, D khớp nhất với kết quả kiểm tra.

OUTPUT JSON:
{
  "check_process": "Tôi đã tính/kiểm tra lại [key_expression]...",
  "answer": "X"
}
(X là chữ cái in hoa).
"""
    user = f"""Câu hỏi: {question}
Lựa chọn:
{formatted_choices}

Lời giải chuyên gia:
- Biểu thức/Phương trình chính: {key_expr}
- Kết quả họ tìm ra: {result}
- Các bước: {steps}

Hãy kiểm tra và chọn đáp án."""
    return system, user

# ===================== DATA LOADING =====================
def load_target_qids(filepath):
    """Đọc file text, mỗi dòng là 1 QID"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Lọc bỏ dòng trống và khoảng trắng thừa
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file target: {filepath}")
        exit(1)

def load_json_data(filepath):
    """Load toàn bộ dữ liệu câu hỏi vào Dictionary để tra cứu nhanh"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Tạo map {qid: item}
        return {item['qid']: item for item in data}
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file data: {filepath}")
        exit(1)

# ===================== PROCESSING LOGIC =====================
def process_targets(data_path, target_path, output_dir):
    # 1. Load Data
    logger.info("Loading data...")
    full_data_map = load_json_data(data_path)
    target_qids = load_target_qids(target_path)
    
    logger.info(f"Found {len(target_qids)} QIDs in {target_path}")
    
    # 2. Setup Output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    
    # 3. Process Loop
    for idx, qid in enumerate(target_qids):
        logger.info(f"\n[{idx+1}/{len(target_qids)}] Processing QID: {qid}")
        
        # Kiểm tra QID có trong data gốc không
        if qid not in full_data_map:
            logger.warning(f"  -> QID {qid} not found in {data_path}. Skipping.")
            continue
            
        item = full_data_map[qid]
        question = item['question']
        choices = item['choices']

        # --- STAGE 1: LARGE REASONING ---
        logger.info("  -> Stage 1: Large Reasoning...")
        sys_1, user_1 = create_prompt_large(question)
        large_out = call_llm(API_URL_LARGE, TOKEN_ID_LARGE, TOKEN_KEY_LARGE, AUTHORIZATION_LARGE,
                             LLM_API_NAME_LARGE, sys_1, user_1, temp=0.1) # Temp thấp để ổn định
        
        if not large_out:
            logger.error("     Stage 1 Failed.")
            large_out = {}

        # --- STAGE 2: SMALL VERIFICATION ---
        logger.info("  -> Stage 2: Small Verifying...")
        sys_2, user_2 = create_prompt_small(question, choices, large_out)
        small_out = call_llm(API_URL_SMALL, TOKEN_ID_SMALL, TOKEN_KEY_SMALL, AUTHORIZATION_SMALL,
                             LLM_API_NAME_SMALL, sys_2, user_2, temp=0.0)
        
        final_answer = 'A'
        final_reason = "Pipeline Failed"
        
        if small_out:
            final_answer = small_out.get('answer', 'A').strip().upper()
            final_reason = json.dumps({
                "large": large_out,
                "small": small_out.get('check_process')
            }, ensure_ascii=False)
            logger.info(f"  => Final Answer: {final_answer}")
        else:
            logger.error("     Stage 2 Failed. Defaulting to A.")
            final_reason = json.dumps(large_out, ensure_ascii=False)

        # Save Result
        results.append({
            'qid': qid,
            'predict': final_answer,
            'reason': final_reason
        })
        
        # Checkpoint save (mỗi 5 câu lưu 1 lần cho chắc)
        if (idx + 1) % 5 == 0:
            temp_path = Path(output_dir) / f'temp_targets_0.json'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        # Sleep to avoid rate limit
        time.sleep(2)

    # Final Save
    timestamp = datetime.now().strftime("%H%M%S")
    final_path = Path(output_dir) / f'temp_results_2.json'
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Done! All results saved to {final_path}")

# ===================== MAIN PARSER =====================
def main():
    parser = argparse.ArgumentParser(description="Target QID Pipeline")
    parser.add_argument('--input', type=str, default='data/test.json', help='Path to full data JSON')
    parser.add_argument('--targets', type=str, default='STEM/target_qid.txt', help='Path to target QID text file')
    parser.add_argument('--output-dir', type=str, default='STEM/output_stem', help='Output directory')
    
    args = parser.parse_args()
    
    process_targets(args.input, args.targets, args.output_dir)

if __name__ == '__main__':
    main()
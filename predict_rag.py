import argparse
import json
import csv
import requests
from pathlib import Path
import os
import time
import logging
from datetime import datetime

# ===================== CẤU HÌNH =====================
# Cấu hình API Endpoint
API_URL_LARGE = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'

# Cấu hình Filter
MIN_WORD_COUNT = 300  # Chỉ xử lý câu hỏi có > 300 từ

# Retry config
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds 

# ===================== LOGGING CONFIGURATION =====================
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f'predict_long_questions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== LOAD CREDENTIALS =====================
def load_credentials(json_path='./api-keys.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        config_large = next((item for item in keys if item.get('llmApiName') == 'LLM large'), None)
        
        if not config_large:
            raise ValueError("Không tìm thấy cấu hình 'LLM large' trong file api-keys.json")
            
        return config_large
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình: {e}")
        exit(1)

config_large = load_credentials()
AUTHORIZATION_LARGE = config_large['authorization']
TOKEN_ID_LARGE = config_large['tokenId']
TOKEN_KEY_LARGE = config_large['tokenKey']
LLM_API_NAME_LARGE = 'vnptai_hackathon_large'

# ===================== SYSTEM PROMPT =====================
SYSTEM_PROMPT_ADVANCED = """Bạn là một hệ thống giải đề thi trắc nghiệm và câu hỏi đọc hiểu ở mức độ thi đấu.
Nhiệm vụ của bạn là chọn đáp án CHÍNH XÁC NHẤT dựa trên dữ kiện trong câu hỏi và đoạn văn đi kèm,
ngay cả khi nội dung rất dài, lắt léo hoặc chứa nhiều thông tin gây nhiễu.

────────────────────────────────
1. NGUYÊN TẮC CHUNG
────────────────────────────────
- **Neo Trọng Tâm (Focus Anchoring):**
  Trước hết, xác định rõ đối tượng hoặc vấn đề DUY NHẤT mà câu hỏi đang yêu cầu.
  Mọi thông tin không phục vụ trực tiếp cho yêu cầu này đều phải coi là nhiễu.

- **Ưu Tiên Ngữ Cảnh Tổng Thể (Contextual Priority):**
  Không chọn đáp án chỉ vì nó xuất hiện trong văn bản.
  Chỉ chọn đáp án phản ánh nội dung được ngữ cảnh chung mô tả hoặc củng cố.

- **Phân Biệt Dạng Thông Tin (Information Type Awareness):**
  Phân biệt rõ giữa:
  (a) dữ kiện / mô tả,
  (b) lập luận / suy diễn,
  (c) ý kiến được thuật lại hoặc quan điểm riêng.
  Nếu câu hỏi không yêu cầu rõ về quan điểm, không được chọn phương án (c).

────────────────────────────────
2. XÁC ĐỊNH Ý ĐỊNH CÂU HỎI
────────────────────────────────
Trước khi đánh giá các phương án, hãy xác định câu hỏi đang yêu cầu:
- mô tả thực tế,
- kết quả logic,
- quan hệ nguyên nhân – hệ quả,
- hay đánh giá / quan điểm.

Đáp án phải khớp CHÍNH XÁC với ý định này.

────────────────────────────────
3. ĐÁNH GIÁ PHƯƠNG ÁN
────────────────────────────────
Đối với từng phương án:
- Kiểm tra xem nó có trực tiếp trả lời yêu cầu của câu hỏi không.
- Kiểm tra xem nó có mâu thuẫn với bất kỳ ràng buộc nào trong đề không.
- Loại bỏ các phương án:
  - dựa trên suy diễn không được hỗ trợ,
  - dựa trên thông tin phụ hoặc gây nhiễu,
  - hoặc chỉ phản ánh một góc nhìn riêng lẻ.

────────────────────────────────
4. TỰ KIỂM TRA CUỐI
────────────────────────────────
Trước khi chốt đáp án, hãy đọc lại câu hỏi một lần nữa và tự hỏi:
“Đáp án này có thể bị loại bởi chi tiết nào trong đề không?”

Nếu có mâu thuẫn, phải chọn lại.

────────────────────────────────
5. ĐỊNH DẠNG ĐẦU RA
────────────────────────────────
Chỉ trả về JSON hợp lệ, không trình bày suy luận chi tiết:

{
  "reason": "Giải thích ngắn gọn dựa trên ngữ cảnh và yêu cầu của câu hỏi.",
  "answer": "X"
}

Trong đó:
- "answer" là MỘT chữ cái in hoa (A, B, C, D, …)
- "reason" phải ngắn, trung lập, không kể lại các bước suy luận
"""

# ===================== PROMPT CREATION =====================
def create_prediction_prompt(question: str, choices: list) -> tuple:
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    user_prompt = f"""## Đề Bài (Bao gồm ngữ cảnh và câu hỏi):
{question}

## Các Lựa Chọn:
{formatted_choices}

Hãy suy luận từng bước theo hướng dẫn System Prompt và đưa ra đáp án JSON."""

    return SYSTEM_PROMPT_ADVANCED, user_prompt

# ===================== API CALL =====================
def call_llm_large(system_prompt: str, user_prompt: str) -> dict:
    headers = {
        'Authorization': AUTHORIZATION_LARGE,
        'Token-id': TOKEN_ID_LARGE,
        'Token-key': TOKEN_KEY_LARGE,
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'model': LLM_API_NAME_LARGE,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_LARGE, headers=headers, json=json_data, timeout=120)
            
            if response.status_code == 429:
                logger.warning("Rate limit hit (429). Sleeping longer...")
                time.sleep(RETRY_DELAY * 2)
                continue
                
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"Error calling API (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(10)
            else:
                return None

# ===================== MAIN PROCESSING =====================
def process_test_file(input_path, output_dir, start_idx=None, end_idx=None):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_path}")
        return

    # Filter range
    total_samples = len(test_data)
    if start_idx is None: start_idx = 0
    if end_idx is None: end_idx = total_samples
    
    test_data = test_data[start_idx:end_idx]
    logger.info(f"Processing samples {start_idx} to {end_idx - 1} (Total: {len(test_data)})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_results = []
    json_results = []
    
    for idx, item in enumerate(test_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        # === FILTER LOGIC ===
        word_count = len(question.split())
        
        logger.info(f"\nProcessing {idx + 1}/{len(test_data)} (ID: {qid}) - Length: {word_count} words")
        
        if word_count <= MIN_WORD_COUNT:
            # SKIP CASE
            logger.info(f"  SKIPPING: Length {word_count} <= {MIN_WORD_COUNT}. Defaulting to 'A'.")
            
            # Vẫn lưu kết quả mặc định để file submission đủ dòng
            csv_results.append({'qid': qid, 'answer': 'A'})
            json_results.append({
                'qid': qid,
                'predict': 'A',
                'reason': f"Skipped because word count ({word_count}) <= {MIN_WORD_COUNT}"
            })
            continue # Bỏ qua các bước gọi API bên dưới, sang câu tiếp theo
        
        # === PROCESS CASE (Length > MIN_WORD_COUNT) ===
        # 1. Create Prompts
        system_prompt, user_prompt = create_prediction_prompt(question, choices)
        
        # 2. Call LLM
        logger.info(f"  Length > {MIN_WORD_COUNT}. Calling LLM Large...")
        start_time = time.time()
        response = call_llm_large(system_prompt, user_prompt)
        duration = time.time() - start_time
        
        # 3. Parse Response
        answer = 'A' # Default
        reason = ""
        
        if response and 'answer' in response:
            raw_answer = response['answer'].strip().upper()
            reason = response.get('reason', '')
            
            valid_choices = [chr(ord('A') + i) for i in range(len(choices))]
            if raw_answer in valid_choices:
                answer = raw_answer
                logger.info(f"  Result: {answer} (Time: {duration:.2f}s)")
            else:
                logger.warning(f"  Invalid answer '{raw_answer}', defaulting to A")
        else:
            logger.error("  Failed to get response, defaulting to A")

        # 4. Save results
        csv_results.append({'qid': qid, 'answer': answer})
        json_results.append({
            'qid': qid,
            'predict': answer,
            'reason': reason
        })
        
        # 5. Rate Limit Sleep (Only if API was called)
        logger.info(f"  Sleeping {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)

    # Save files
    csv_path = output_dir / 'submission_long_questions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(csv_results)
    
    json_path = output_dir / 'predict_long_questions.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"\nDone! Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/val.json')
    parser.add_argument('--output-dir', type=str, default='output_long_q')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    
    args = parser.parse_args()
    process_test_file(args.input, args.output_dir, args.start, args.end)

if __name__ == '__main__':
    main()
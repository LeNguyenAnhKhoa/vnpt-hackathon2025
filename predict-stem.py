import argparse
import json
import requests
from pathlib import Path
import time
import logging
from datetime import datetime

# ===================== CẤU HÌNH NGƯỜI DÙNG =====================

# 1. DANH SÁCH CÁC QID MUỐN CHẠY LẠI
TARGET_QIDS = [
    "val_0024",
    # "val_0025",
    # "val_0027",
    # "val_0071",
    # "val_0076",
    # "val_0087"
]

# 2. ĐƯỜNG DẪN FILE
FILE_CONTEXT = 'output/predict_val.json'  # File chứa reference_docs (đã chạy trước đó)
FILE_ORIGINAL = 'data/val.json'    # File chứa question và choices gốc

# ===================== CẤU HÌNH API =====================
API_URL_LARGE = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'
LLM_API_NAME_LARGE = 'vnptai_hackathon_large'
MAX_RETRIES = 5
RETRY_DELAY = 5

# ===================== LOGGING =====================
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f'repredict_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== LOAD CREDENTIALS =====================
def load_credentials(json_path='./api-keys.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        return next((item for item in keys if item.get('llmApiName') == 'LLM large'), None)
    except Exception as e:
        logger.error(f"Lỗi load key: {e}")
        exit(1)

def load_small_credentials(json_path='./api-keys.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        return next((item for item in keys if item.get('llmApiName') == 'LLM small'), None)
    except Exception as e:
        logger.error(f"Lỗi load key: {e}")
        exit(1)
config = load_credentials()
config_small = load_small_credentials()
if not config: exit(1)
AUTHORIZATION_LARGE = config['authorization']
TOKEN_ID_LARGE = config['tokenId']
TOKEN_KEY_LARGE = config['tokenKey']

AUTHORIZATION_SMALL = config_small['authorization']
TOKEN_ID_SMALL = config_small['tokenId']
TOKEN_KEY_SMALL = config_small['tokenKey']

# ===================== PROMPT FUNCTION =====================
def create_prompts_with_context(question: str, choices: list, ref_documents: list) -> tuple:
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
    
    context_text = ""
    # for i, doc in enumerate(ref_documents):
    #     # Ưu tiên lấy text, nếu không có thì lấy content (tuỳ format file json)
    #     text = doc.get('text') or doc.get('content') or ""
    #     context_text += f"\n[Tài liệu tham khảo {i+1}]\n{text}\n"
    
    system_prompt = f"""
Bạn là hệ thống suy luận chuyên giải câu hỏi trắc nghiệm có yếu tố tính toán
(toán học, STEM, lý, hóa, logic định lượng).

Mục tiêu duy nhất: chọn đáp án ĐÚNG NHẤT. Không được đoán.

QUY TẮC CƯỠNG BỨC:

1. Với mọi câu hỏi yêu cầu kết quả số:
- LỜI GIẢI PHẢI chứa ít nhất một biểu thức CHỈ GỒM SỐ.
- Nếu không tồn tại biểu thức số → KHÔNG ĐƯỢC chọn đáp án.

2. CẤM chuyển trực tiếp từ:
- công thức / suy luận chữ → kết quả.
Bắt buộc theo chuỗi:
(công thức hoặc suy luận)
→ (biểu thức số cụ thể)
→ (kết quả trung gian)
→ (kết quả cuối).

3. PHÉP TÍNH:
- Không gộp nhiều phép toán trong một dòng.
- Mỗi phép nhân, chia, cộng, trừ phải có kết quả riêng.
- Nếu có cả nhân/chia và cộng/trừ → nhân/chia trước.

4. KIỂM TRA TÍNH TOÁN (BẮT BUỘC):
- Sau khi có kết quả cuối, PHẢI viết lại phép tính số đó theo cách khác.
- Nếu không thể viết lại → coi là CHƯA KIỂM TRA → không được chọn đáp án.
- Nếu hai kết quả khác nhau → PHẢI sửa, không được kết luận.

5. KIỂM TRA ĐIỀU KIỆN:
- Phải nêu rõ điều kiện cần của bài toán.
- Phải xác minh kết quả thỏa điều kiện đó.
- Nếu không chứng minh được điều kiện đủ → không được kết luận.

6. Với bài toán thuần toán học (phương trình, biểu thức, hình học):
- “Thay số” = thay giá trị cụ thể vào phương trình / biểu thức / điều kiện.
- Không được bỏ qua bước này nếu đề bài có số.

7. ĐIỀU KIỆN DỪNG:
- Nếu thiếu bước thay số
- Hoặc thiếu phép tính cụ thể
- Hoặc thiếu kiểm tra lại
→ LỜI GIẢI KHÔNG HỢP LỆ, KHÔNG ĐƯỢC CHỌN ĐÁP ÁN.

ĐẦU RA DUY NHẤT (JSON):

{{
  "reason": "Chuỗi suy luận có biểu thức số, tính từng bước, kiểm tra điều kiện và kiểm tra lại phép tính.",
  "answer": "X"
}}

"""

    user_prompt = f"""## Tài Liệu Tham Khảo
Dưới đây là TOP 5 tài liệu được xem là liên quan nhất đến câu hỏi, đã được hệ thống truy xuất và xếp hạng.
Lưu ý: Nếu tài liệu KHÔNG liên quan trực tiếp đến câu hỏi, hãy bỏ qua hoàn toàn và không sử dụng.

{context_text}
Question: {question}
Choices:
{formatted_choices}
"""

    return system_prompt, user_prompt

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
        'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
        'temperature': 0.05,
        'response_format': {'type': 'json_object'},
        'top_p': 0.95
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_LARGE, headers=headers, json=json_data, timeout=120)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            logger.error(f"API Error ({attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY)
            else: return None

# ===================== MAIN LOGIC =====================
def load_json_as_dict(filepath, key_field='qid'):
    """Load json và chuyển thành dictionary {qid: item} để tra cứu nhanh"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item[key_field]: item for item in data}
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file {filepath}")
        exit(1)

def main():
    logger.info("Đang tải dữ liệu...")
    
    # 1. Load dữ liệu Context (predict_val.json)
    context_map = load_json_as_dict(FILE_CONTEXT)
    
    # 2. Load dữ liệu Gốc (data/val.json)
    original_map = load_json_as_dict(FILE_ORIGINAL)
    
    results = []
    
    # 3. Chạy vòng lặp qua các QID mục tiêu
    for qid in TARGET_QIDS:
        logger.info(f"Processing {qid}...")
        
        # Kiểm tra dữ liệu có đủ không
        if qid not in context_map:
            logger.error(f"  -> QID {qid} không có trong file context ({FILE_CONTEXT}). Bỏ qua.")
            continue
        if qid not in original_map:
            logger.error(f"  -> QID {qid} không có trong file gốc ({FILE_ORIGINAL}). Bỏ qua.")
            continue
            
        # Lấy các thành phần cần thiết
        ref_docs = context_map[qid].get('reference_docs', [])
        question = original_map[qid].get('question', '')
        choices = original_map[qid].get('choices', [])
        
        if not ref_docs:
            logger.warning(f"  -> QID {qid} có list reference_docs rỗng!")

        # Tạo prompt và gọi LLM
        system_prompt, user_prompt = create_prompts_with_context(question, choices, ref_docs)
        response = call_llm_large(system_prompt, user_prompt)
        
        # Xử lý kết quả
        answer = 'A'
        reason = "Failed"
        if response and 'answer' in response:
            answer = response['answer'].strip().upper()
            reason = response.get('reason', '')
            logger.info(f"  -> Answer: {answer}")
        else:
            logger.error(f"  -> Failed to get answer.")

        results.append({
            'qid': qid,
            'predict': answer,
            'reason': reason,
            # 'reference_docs': ref_docs # Lưu lại docs cũ để tiện check
        })
        
        time.sleep(2) # Delay nhẹ

    # 4. Lưu kết quả
    output_dir = Path('output_stem')
    output_dir.mkdir(exist_ok=True)
    json_out = output_dir / f'predict_stem.json'
    
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Hoàn thành! Kết quả tại: {json_out}")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
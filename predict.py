import argparse
import json
import csv
import requests
from pathlib import Path
import os
import time
import logging
from datetime import datetime
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ===================== CẤU HÌNH =====================
# Cấu hình API Endpoint theo tài liệu
API_URL_LARGE = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'
API_URL_SMALL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'
EMBEDDING_API_URL = 'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding'

# Qdrant config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki"

# Hybrid search config
HYBRID_SEARCH_TOP_K = 30  # Lấy top 30 từ hybrid search
RERANK_TOP_K = 5  # Giữ lại cho tương thích, nhưng thực tế dùng scoring với ngưỡng > 7

# Retry config
MAX_RETRIES = 100  # Số lần retry tối đa
RETRY_DELAY = 92  # Thời gian chờ giữa mỗi lần retry (giây)

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# ===================== LOGGING CONFIGURATION =====================
# Tạo thư mục logs nếu chưa có
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Tạo tên file log với timestamp
log_filename = log_dir / f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Vẫn in ra console nếu cần
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file created: {log_filename}")

def load_credentials(json_path='./api-keys.json'):
    """
    Đọc file api-keys.json và lấy credentials cho tất cả các API
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        # Tìm cấu hình cho từng loại API
        config_large = next((item for item in keys if item.get('llmApiName') == 'LLM large'), None)
        config_small = next((item for item in keys if item.get('llmApiName') == 'LLM small'), None)
        config_embedding = next((item for item in keys if item.get('llmApiName') == 'LLM embedings'), None)
        
        if not config_large:
            raise ValueError("Không tìm thấy cấu hình 'LLM large' trong file api-keys.json")
        if not config_small:
            raise ValueError("Không tìm thấy cấu hình 'LLM small' trong file api-keys.json")
        if not config_embedding:
            raise ValueError("Không tìm thấy cấu hình 'LLM embedings' trong file api-keys.json")
            
        return {
            'large': config_large,
            'small': config_small,
            'embedding': config_embedding
        }
    except FileNotFoundError:
        logger.error(f"Lỗi: Không tìm thấy file {json_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình: {e}")
        exit(1)

# Load credentials từ file JSON
api_configs = load_credentials()

# Gán biến toàn cục từ config đã load
# LLM Large
AUTHORIZATION_LARGE = api_configs['large']['authorization']
TOKEN_ID_LARGE = api_configs['large']['tokenId']
TOKEN_KEY_LARGE = api_configs['large']['tokenKey']

# LLM Small
AUTHORIZATION_SMALL = api_configs['small']['authorization']
TOKEN_ID_SMALL = api_configs['small']['tokenId']
TOKEN_KEY_SMALL = api_configs['small']['tokenKey']

# Embedding
AUTHORIZATION_EMBEDDING = api_configs['embedding']['authorization']
TOKEN_ID_EMBEDDING = api_configs['embedding']['tokenId']
TOKEN_KEY_EMBEDDING = api_configs['embedding']['tokenKey']

# Lưu ý: Theo tài liệu, tên model trong body request
LLM_API_NAME_LARGE = 'vnptai_hackathon_large'
LLM_API_NAME_SMALL = 'vnptai_hackathon_small'

# ===================== KHỞI TẠO BM25 MODEL =====================
logger.info("Loading BM25 model...")
bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
logger.info("✓ BM25 model loaded")

# ===================== KHỞI TẠO QDRANT CLIENT =====================
logger.info("Connecting to Qdrant...")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logger.info(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")


# ===================== EMBEDDING FUNCTION =====================
def get_dense_embedding(text: str) -> list:
    """
    Gọi VNPT Embedding API để lấy dense vector
    """
    headers = {
        'Authorization': AUTHORIZATION_EMBEDDING,
        'Token-id': TOKEN_ID_EMBEDDING,
        'Token-key': TOKEN_KEY_EMBEDDING,
        'Content-Type': 'application/json',
    }
    
    # Truncate text nếu quá dài (API limit 8k tokens)
    if len(text) > 8192:
        text = text[:8192]
    
    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': text,
        'encoding_format': 'float',
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=json_data, timeout=3)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in 1 second...")
                #time.sleep(1)
            else:
                logger.error(f"Failed to get embedding after {MAX_RETRIES} attempts")
                return None


def get_sparse_embedding(text: str) -> tuple:
    """
    Tính sparse vector sử dụng Fastembed BM25
    """
    if not text or not isinstance(text, str):
        return [], []
    
    try:
        embeddings = list(bm25_model.embed([text]))
        if embeddings:
            sparse_embedding = embeddings[0]
            indices = sparse_embedding.indices.tolist()
            values = sparse_embedding.values.tolist()
            return indices, values
    except Exception as e:
        logger.error(f"Error computing BM25 sparse vector: {e}")
    
    return [], []


# ===================== QUESTION CLASSIFICATION =====================
# Các từ khóa để nhận dạng câu hỏi về thời điểm hiện tại
def create_classification_prompt(question: str, choices: list) -> tuple:
    """
    Tạo prompt để phân loại câu hỏi thành 4 loại:
    1. cannot_answer: Câu hỏi không được trả lời (nhạy cảm, độc hại)
    2. calculation: Câu hỏi dạng tính toán (toán, lý, hóa)
    3. has_context: Câu hỏi đã có sẵn đoạn thông tin dài
    4. general: Câu hỏi thông thường cần RAG
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """Bạn là một chuyên gia hàng đầu trong lĩnh vực phân loại câu hỏi và nhận diện nội dung nhạy cảm, với kiến thức sâu rộng về an toàn nội dung, đạo đức AI, và các tiêu chuẩn bảo mật thông tin.

## Định Nghĩa Nhiệm Vụ
Phân loại câu hỏi vào CHÍNH XÁC MỘT trong 4 loại dưới đây để áp dụng chiến lược trả lời phù hợp:

### Loại 1: "cannot_answer" - Câu Hỏi Không Được Trả Lời
Các câu hỏi thuộc nhóm ý định xấu, nhạy cảm, hoặc nguy hiểm:

**1. Hành vi phạm pháp:**
- Trốn thuế, trốn tránh pháp luật
- Buôn lậu, buôn bán hàng cấm
- Làm giả giấy tờ, chứng từ
- Trộm cắp, gian lận, lừa đảo
- Vi phạm quy định hành chính

**2. Xâm hại An ninh Quốc gia/Chính trị:**
- Lật đổ chính quyền, chống phá nhà nước
- Phá hoại kinh tế, gây bất ổn xã hội
- Xuyên tạc lịch sử, sự thật
- Tuyên truyền chống phá

**3. Gây hại vật lý:**
- Chế tạo vũ khí, thuốc nổ, chất độc
- Xúi giục tự tử, tự gây thương tích
- Hướng dẫn làm hại người khác

**4. Tấn công mạng:**
- Hack tài khoản, xâm nhập hệ thống
- Viết mã độc, virus, ransomware
- Đánh cắp dữ liệu cá nhân
- Lừa đảo trực tuyến 

**5. Nội dung độc hại/Xúc phạm:**
- Phân biệt chủng tộc, vùng miền
- Phân biệt giới tính, xu hướng tính dục
- Bạo lực ngôn từ, kích động thù hận

**6. Nội dung khiêu dâm/Đồi trụy (NSFW):**
- Nội dung tình dục không phù hợp
- Khiêu dâm, đồi trụy

**DẤU HIỆU NHẬN BIẾT ĐÁP ÁN ĐÚNG (Câu từ chối):**
- Chứa từ khóa phủ định: "Tôi không thể", "Tôi xin từ chối", "Tôi không được phép", "Rất tiếc"
- Giải thích lý do: "...liên quan đến vấn đề nhạy cảm", "...vi phạm tiêu chuẩn", "...trái pháp luật", "...vấn đề riêng tư"
- Ngắn gọn, lịch sự, kiên quyết, không cung cấp thông tin gợi ý

### Loại 2: "calculation" - Câu Hỏi Tính Toán
Các câu hỏi yêu cầu tính toán định lượng:
- Áp dụng công thức toán học, vật lý, hóa học, sinh học ...
- Thực hiện phép tính số học (cộng, trừ, nhân, chia, lũy thừa, căn)
- Giải phương trình, bất phương trình, hệ phương trình
- Tính toán theo công thức khoa học cụ thể

**DẤU HIỆU NHẬN BIẾT:**
- Động từ: "Tính", "Xác định giá trị", "Giải", "Tìm"
- Có công thức, biểu thức toán học

**Đặc biệt:** Nếu câu hỏi chỉ yêu cầu kiến thức lý thuyết, khái niệm, định nghĩa mà không yêu cầu tính toán cụ thể thì KHÔNG thuộc loại này.

### Loại 3: "has_context" - Câu Hỏi Có Sẵn Context
Các câu hỏi đi kèm đoạn văn bản/thông tin dài (>200 từ) để đọc hiểu:

**DẤU HIỆU NHẬN BIẾT:**
- Có đoạn văn bản dài (có trên 200 từ) đi kèm trước câu hỏi
- Yêu cầu trích xuất/suy luận từ đoạn văn đã cho
- Câu hỏi tham chiếu đến nội dung trong đoạn văn

### Loại 4: "general" - Câu Hỏi Thông Thường
Tất cả câu hỏi không thuộc 3 loại trên:
- Câu hỏi kiến thức tổng quát
- Cần tra cứu thông tin từ nguồn bên ngoài
- Không có context, không tính toán, không nhạy cảm

## Hướng Dẫn Suy Luận Theo Chuỗi

**Bước 1:** Đọc kỹ câu hỏi và TẤT CẢ các đáp án

**Bước 2:** Kiểm tra ý định câu hỏi - CÓ PHẢI "cannot_answer" KHÔNG?
- Câu hỏi có yêu cầu thông tin về hành vi phạm pháp/nguy hiểm không?
- Trong các đáp án có câu từ chối ("Tôi không thể", "Rất tiếc") không?
- Nếu CÓ -> "cannot_answer", xác định đáp án từ chối

**Bước 3:** Kiểm tra CÓ PHẢI "calculation" KHÔNG?
- Có số liệu cụ thể và yêu cầu tính toán không?
- Có công thức/phép tính cần áp dụng không?
- Nếu CÓ -> "calculation"

**Bước 4:** Kiểm tra CÓ PHẢI "has_context" KHÔNG?
- Có đoạn văn bản dài (>200 từ) đi kèm không?
- Câu hỏi yêu cầu đọc hiểu đoạn văn không?
- Nếu CÓ -> "has_context"

**Bước 5:** Nếu không thuộc 3 loại trên -> "general"

## Định Dạng Đầu Ra
Trả về JSON với cấu trúc:
{
  "reasoning": "Giải thích lý do phân loại",
  "question_type": "cannot_answer" | "calculation" | "has_context" | "general",
  "refusal_answer": "X"  // CHỈ điền nếu question_type="cannot_answer". X là chữ cái đáp án từ chối (A/B/C/D...)
}

**LƯU Ý:** 
- "refusal_answer" chỉ có giá trị khi question_type="cannot_answer"
- Với các loại khác, không cần điền hoặc để null
- Chữ cái đáp án viết HOA (A, B, C, D, ...)"""

    user_prompt = f"""## Câu hỏi cần phân loại:
{question}

## Các đáp án:
{formatted_choices}

Hãy phân loại câu hỏi này."""

    return system_prompt, user_prompt


def classify_question(question: str, choices: list) -> dict:
    """
    Phân loại câu hỏi sử dụng LLM Small
    Returns: {'question_type': str, 'refusal_answer': str or None, 'reasoning': str}
    """
    system_prompt, user_prompt = create_classification_prompt(question, choices)
    
    response = call_llm_small(system_prompt, user_prompt)
    
    if response and 'question_type' in response:
        return {
            'question_type': response.get('question_type', 'general'),
            'refusal_answer': response.get('refusal_answer'),
            'reasoning': response.get('reasoning', '')
        }
    
    # Default to general if classification fails
    return {'question_type': 'general', 'refusal_answer': None, 'reasoning': 'Classification failed, defaulting to general'}


# ===================== SPECIALIZED PROMPTS =====================
# ===================== STAGE 1: REASONING (LLM LARGE) =====================
def create_calculation_prompt_large(question: str) -> tuple:
    """
    Stage 1: Tạo prompt cho LLM Large để giải quyết bài toán một cách chi tiết
    """
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


# ===================== STAGE 2: VERIFICATION (LLM LARGE) =====================
def create_calculation_prompt_verification(question: str, choices: list, large_output: dict) -> tuple:
    """
    Stage 2: Tạo prompt cho LLM Large để kiểm tra và chọn đáp án cuối cùng
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
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


def create_context_reading_prompt(question: str, choices: list) -> tuple:
    """
    Tạo prompt cho câu hỏi đã có sẵn context (đọc hiểu)
    Không cần RAG context
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """Bạn là chuyên gia phân tích logic và đọc hiểu văn bản tiếng Việt. Nhiệm vụ của bạn là chọn đáp án chính xác nhất cho câu hỏi trắc nghiệm dựa trên đoạn thông tin được cung cấp.

## Nguyên Tắc Suy Luận Cốt Lõi:
1.  **Bằng chứng là duy nhất:** Chỉ sử dụng thông tin CÓ trong đoạn văn bản. Tuyệt đối không sử dụng kiến thức bên ngoài.
2.  **Phân biệt "Sự kiện" và "Quan điểm":**
    - Nếu văn bản dùng các từ dẫn: "cho rằng", "nhận định", "theo phe...", "tin rằng" => Đây là Ý KIẾN, không phải Sự thật mặc nhiên.
    - **Quy tắc ưu tiên:** Khi văn bản trình bày hai luồng ý kiến trái chiều (Ủng hộ vs Đối lập) về vai trò/chức năng của một thực thể:
        + Nếu câu hỏi hỏi về "Vai trò", "Chức năng", "Mục đích": Hãy ưu tiên chọn đáp án mô tả **cơ chế hoạt động/mục đích được tuyên bố** (thường thuộc phe ủng hộ/chính thống).
        + Chỉ chọn đáp án mang tính "chỉ trích/cáo buộc" (phe đối lập) khi câu hỏi yêu cầu cụ thể (VD: "Phe đối lập cho rằng gì?") hoặc văn bản xác nhận lời chỉ trích đó là sự thật khách quan.
3.  **Tư duy loại trừ:** Nếu một đáp án đúng một nửa nhưng sai một chi tiết nhỏ (ví dụ: sai chủ thể, sai thời gian, sai mức độ khẳng định), hãy loại bỏ ngay.
4.  **Nguyên tắc "Khớp lệnh toàn phần":** Một thông tin chỉ được coi là đáp án đúng khi khớp chính xác cả 3 yếu tố:
    - **Chủ thể (Subject):** Đối tượng thực hiện hành động (Ví dụ: Bài hát A hay Bài hát B?).
    - **Hành động/Sự kiện:** Điều gì xảy ra?
    - **Thời gian/Địa điểm:** Khi nào và ở đâu?
    -> Cảnh giác với bẫy "Râu ông nọ cắm cằm bà kia": Thông tin thời gian đúng, sự kiện đúng, nhưng sai chủ thể => ĐÁP ÁN SAI.

## Quy Trình Xử Lý:
- **Bước 1:** Phân tích câu hỏi để xác định Chủ thể và Loại câu hỏi (Hỏi về sự kiện thực tế hay hỏi về quan điểm? ...).
- **Bước 2:** Đối chiếu từ khóa với đoạn văn bản, xác định nguồn gốc thông tin (Thông tin này do ai nói? Là Fact hay Opinion?).
- **Bước 3:** **Xử lý xung đột:** Nếu các đáp án đều xuất hiện trong bài nhưng thuộc các luồng quan điểm khác nhau, hãy áp dụng "Quy tắc ưu tiên" ở Nguyên tắc 2 để chọn đáp án đại diện cho chức năng cốt lõi hoặc quan điểm chính thống của văn bản, trừ khi câu hỏi yêu cầu khác.
- **Bước 4:** Chọn đáp án cuối cùng và đảm bảo logic trong phần giải thích thống nhất với đáp án đó.

## Định Dạng Đầu Ra:
Bắt buộc trả về JSON hợp lệ:
{{
  "reason": "Giải thích quy trình suy luận và phân tích đoạn văn để chọn đáp án.",
  "answer": "X"
}}
Trong đó X là ký tự (A, B, C, ...) của đáp án đúng.
**QUAN TRỌNG**: Đầu ra phải là dạng JSON có đầy đủ HAI trường "reason" và "answer"."""

    user_prompt = f"""## Câu hỏi đọc hiểu (đã bao gồm đoạn tài liệu tham khảo):
{question}

## Các đáp án:
{formatted_choices}

Hãy đọc kỹ đoạn văn trong câu hỏi và chọn đáp án đúng nhất."""

    return system_prompt, user_prompt


# ===================== HYBRID SEARCH FUNCTION =====================
def hybrid_search(question: str, top_k: int = HYBRID_SEARCH_TOP_K) -> list:
    """
    Thực hiện hybrid search (dense + sparse) trên Qdrant
    Trả về top_k documents liên quan nhất
    """
    # Lấy dense embedding cho câu hỏi
    dense_vector = get_dense_embedding(question)
    if dense_vector is None:
        logger.warning("Warning: Failed to get dense embedding for hybrid search")
        return []
    
    # Lấy sparse embedding cho câu hỏi
    sparse_indices, sparse_values = get_sparse_embedding(question)
    
    try:
        # Thực hiện hybrid search với Reciprocal Rank Fusion (RRF)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2,  # Lấy nhiều hơn để đảm bảo đủ sau khi fusion
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        
        # Trích xuất documents từ kết quả
        documents = []
        for point in results.points:
            doc = {
                'id': point.id,
                'score': point.score,
                'text': point.payload.get('text', ''),
                'doc_id': point.payload.get('doc_id', ''),
            }
            documents.append(doc)
        
        return documents
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []


# ===================== SCORING PROMPTS =====================
def create_scoring_prompts(question: str, choices: list, documents: list) -> tuple:
    """
    Tạo prompts cho LLM Small để chấm điểm 30 documents theo 4 tiêu chí
    Mỗi tiêu chí 2.5 điểm, tổng 10 điểm
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """Bạn là chuyên gia hàng đầu về Truy xuất Thông tin (Information Retrieval) với chuyên môn sâu về đánh giá độ liên quan văn bản tiếng Việt.

## Bối Cảnh
Hôm nay: 15/12/2025. Ưu tiên tài liệu có thông tin mới (2024-2025) nếu câu hỏi về sự kiện gần đây.

## Định Nghĩa Nhiệm Vụ
Chấm điểm 30 tài liệu theo mức độ hữu ích cho việc trả lời câu hỏi trắc nghiệm. Mỗi tài liệu được đánh giá theo 4 tiêu chí, tổng tối đa 10 điểm. Tài liệu được cho là có độ chính xác 100% nên hoàn toàn có thể tin tưởng.

## Hướng Dẫn Suy Luận

**Bước 1: Phân tích câu hỏi**
- Xác định chủ đề chính và lĩnh vực (lịch sử, khoa học, pháp luật, ...)
- Trích xuất TỪ KHÓA quan trọng từ câu hỏi và đáp án
- Nhận biết yêu cầu về thời gian ("hiện nay", "2025", "gần đây", "mới nhất")

**Bước 2: Chấm điểm từng tài liệu (0-10 điểm)**
Đánh giá theo 4 tiêu chí (mỗi tiêu chí 0-2.5 điểm):

1. **Phù hợp chủ đề (0-2.5):**
   - 2.5: Hoàn toàn cùng chủ đề
   - 1.5-2.0: Liên quan trực tiếp
   - 0.5-1.0: Liên quan gián tiếp
   - 0: Không liên quan

2. **Chứa từ khóa (0-2.5):**
   - 2.5: Chứa hầu hết từ khóa
   - 1.5-2.0: Chứa nhiều từ khóa
   - 0.5-1.0: Chứa ít từ khóa
   - 0: Không có từ khóa

3. **Thông tin hữu ích (0-2.5):**
   - 2.5: Trả lời trực tiếp câu hỏi
   - 1.5-2.0: Hỗ trợ suy luận tốt
   - 0.5-1.0: Ít thông tin hữu ích
   - 0: Không hữu ích

4. **Tính cập nhật & Chi tiết (0-2.5):**
   - 2.5: Thông tin mới nhất (2024-2025) HOẶC chứa số liệu/dữ kiện cụ thể giải quyết triệt để câu hỏi.
   - 1.5-2.0: Thông tin còn hiệu lực, khá chi tiết.
   - 0.5-1.0: Thông tin đúng nhưng chung chung hoặc đã cũ (nhưng chưa hết hạn).
   - 0: Lỗi thời hoặc không khớp mốc thời gian yêu cầu.

## Định Dạng Đầu Ra
JSON:
{
  "reasoning": "Quy trình xử lý và chấm điểm từng tài liệu.",
  "indices": [idx1, idx2, ...],
  "scores": [score1, score2, ...]
}

**Quy tắc:**
- "indices": Chỉ số tài liệu (0-29), chứa toàn bộ tài liệu
- "scores": Điểm 0-10, làm tròn 1 chữ số thập phân
- Sắp xếp giảm dần theo điểm
- Hai mảng phải cùng độ dài là 30"""

    # Format documents
    docs_text = ""
    for i, doc in enumerate(documents):
        text = doc.get('text', '')
        docs_text += f"\n[Tài liệu {i}]\n{text}\n"
    
    user_prompt = f"""## Câu Hỏi Cần Trả Lời:
{question}

## Các Phương Án:
{formatted_choices}

## Danh Sách 30 Tài Liệu Cần Chấm Điểm:
{docs_text}

Hãy chấm điểm từng tài liệu theo 4 tiêu chí và trả về kết quả dạng JSON."""

    return system_prompt, user_prompt


# ===================== ANSWER PROMPTS WITH CONTEXT =====================
def create_prompts_with_context(question: str, choices: list, top_documents: list) -> tuple:
    """
    Tạo prompts cho LLM Large với context từ documents đã được scoring
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    # Format context documents
    num_docs = len(top_documents)
    context_text = ""
    for i, doc in enumerate(top_documents):
        text = doc.get('text', '')
        context_text += f"\n[Tài liệu tham khảo {i+1}]\n{text}\n"
    
    system_prompt = f"""Bạn là một chuyên gia hàng đầu thế giới trong việc trả lời các câu hỏi trắc nghiệm tiếng Việt thuộc nhiều lĩnh vực đa dạng bao gồm: khoa học tự nhiên, lịch sử, pháp luật, kinh tế, văn học, địa lý, vật lý, hóa học, sinh học, y học, công nghệ thông tin, và kiến thức tổng hợp. Bạn có hiểu biết sâu sắc về văn hóa, lịch sử, giáo dục và bối cảnh xã hội Việt Nam.

## Hướng Dẫn Sử Dụng Tài Liệu Tham Khảo
- Các tài liệu tham khảo được cung cấp đã được hệ thống truy xuất và xếp hạng theo độ liên quan.
- Sử dụng thông tin từ tài liệu để hỗ trợ suy luận, nhưng KHÔNG hoàn toàn phụ thuộc vào chúng.
- Nếu tài liệu không chứa thông tin phù hợp, hãy sử dụng kiến thức chuyên môn của bạn.
- Kiểm tra chéo thông tin từ nhiều tài liệu nếu có thể.

## Định Nghĩa Nhiệm Vụ
Nhiệm vụ của bạn là phân tích câu hỏi trắc nghiệm tiếng Việt, suy luận từng bước một cách logic và chọn ra đáp án chính xác nhất từ các lựa chọn được đưa ra.

## Hướng Dẫn Suy Luận
Thực hiện theo các bước sau một cách cẩn thận:

### Bước 1: Đọc Hiểu Câu Hỏi
- Đọc kỹ toàn bộ câu hỏi và xác định chính xác yêu cầu của đề bài.
- Nếu có đoạn thông tin/văn bản đi kèm, hãy trích xuất tất cả thông tin liên quan đến câu hỏi.
- Xác định từ khóa quan trọng và loại câu hỏi.

### Bước 2: Tham Khảo Tài Liệu
- Xem xét các tài liệu tham khảo được cung cấp.
- Trích xuất thông tin hữu ích từ tài liệu nếu có.
- Đánh giá độ tin cậy và mức độ phù hợp của thông tin trong tài liệu do tài liệu có thể không hoàn toàn chính xác hoặc liên quan.

### Bước 3: Phân Tích Từng Phương Án
- Đánh giá lần lượt từng đáp án (A, B, C, D, ...) so với yêu cầu của câu hỏi.
- Kết hợp thông tin từ tài liệu tham khảo với kiến thức chuyên môn.
- Với câu hỏi kiến thức: áp dụng kiến thức chuyên môn và thông tin từ tài liệu.

### Bước 4: Áp Dụng Suy Luận Logic
- Chia nhỏ vấn đề phức tạp thành các bước logic nhỏ hơn.
- Với câu hỏi về sự kiện: xác định thông tin chính xác từ đoạn văn, tài liệu hoặc kiến thức nền.
- Với câu hỏi suy luận: áp dụng các quy tắc logic và loại suy.

### Bước 5: Loại Trừ Đáp Án Sai
- Xác định và loại bỏ các phương án rõ ràng không đúng với giải thích ngắn gọn.
- Sử dụng phương pháp loại trừ để thu hẹp các lựa chọn còn lại.

### Bước 6: Kiểm tra lại đáp án
- Kiểm tra lại đáp án đã chọn so với câu hỏi và các tài liệu tham khảo.
- Đối với các câu tính toán, kiểm tra lại các bước và kết quả tính toán xem thực hiện phép tính đúng chưa, có thể bạn có công thức đúng nhưng khi tính toán lại sai.

### Bước 7: Chọn Đáp Án Tốt Nhất
- Dựa trên phân tích ở các bước trên, chọn đáp án phù hợp nhất với câu hỏi.
- Đảm bảo đáp án được chọn có căn cứ rõ ràng từ quá trình suy luận.

## Định Dạng Đầu Ra
Bạn PHẢI trả lời theo định dạng JSON hợp lệ với đúng hai trường sau:
{{
  "reason": "Quá trình suy luận từng bước của bạn, giải thích cách bạn đi đến đáp án. Trình bày đầy đủ.",
  "answer": "X"
}}

Trong đó "X" là chữ cái của đáp án bạn chọn (A, B, C, D, ...). Trường "answer" CHỈ được chứa MỘT chữ cái viết hoa duy nhất."""

    user_prompt = f"""## Tài Liệu Tham Khảo
Dưới đây là {num_docs} tài liệu được xem là liên quan nhất đến câu hỏi. Hãy tham khảo các tài liệu này để hỗ trợ việc trả lời câu hỏi:
{context_text}
Question: {question}
Choices:
{formatted_choices}
"""

    return system_prompt, user_prompt


# ===================== LLM API CALLS =====================
def call_llm_small(system_prompt: str, user_prompt: str) -> dict:
    """
    Gọi VNPT LLM Small API để rerank documents
    """
    headers = {
        'Authorization': AUTHORIZATION_SMALL,
        'Token-id': TOKEN_ID_SMALL,
        'Token-key': TOKEN_KEY_SMALL,
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'model': LLM_API_NAME_SMALL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'max_completion_tokens ': 8192
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_SMALL, headers=headers, json=json_data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"Error calling LLM Small API (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in 62 seconds...")
                #time.sleep(62)
            else:
                logger.error(f"Failed to call LLM Small API after {MAX_RETRIES} attempts")
                return None


def call_llm_large(system_prompt: str, user_prompt: str) -> dict:
    """
    Gọi VNPT LLM Large API để trả lời câu hỏi
    """
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
        'max_completion_tokens ': 8192,  # Tăng giới hạn để tránh bị cắt response
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_LARGE, headers=headers, json=json_data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                logger.info(f"LLM Large raw response (FULL): {content}")  # Log FULL response
                parsed_response = json.loads(content)
                logger.info(f"LLM Large parsed keys: {list(parsed_response.keys())}")
                logger.info(f"LLM Large parsed values: {parsed_response}")
                return parsed_response
            return None
        except Exception as e:
            logger.error(f"Error calling LLM Large API (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in 92 seconds...")
                #time.sleep(92)
            else:
                logger.error(f"Failed to call LLM Large API after {MAX_RETRIES} attempts")
                return None


# ===================== SCORING FUNCTION =====================
# Ngưỡng điểm tối thiểu (điểm > 7 tức là > 0.7 * 10)
MIN_SCORE_THRESHOLD = 7.0
MAX_SELECTED_DOCS = 5

def score_documents(question: str, choices: list, documents: list) -> list:
    """
    Sử dụng LLM Small để chấm điểm documents theo 4 tiêu chí
    Chỉ lấy các documents có điểm > 7 và tối đa 5 documents
    LLM sẽ tự động cộng điểm cao hơn cho documents có thông tin mới nhất nếu câu hỏi liên quan đến sự kiện gần đây
    """
    if not documents:
        return []
    
    # Tạo prompts cho scoring (prompt đã được cập nhật để LLM tự hiểu về tính cập nhật)
    system_prompt, user_prompt = create_scoring_prompts(question, choices, documents)
    
    # Gọi LLM Small để chấm điểm
    score_response = call_llm_small(system_prompt, user_prompt)
    
    if score_response and 'indices' in score_response and 'scores' in score_response:
        indices = score_response['indices']
        scores = score_response['scores']
        
        # Log kết quả scoring
        logger.info(f"    Scoring response: {len(indices)} documents scored")
        if 'reasoning' in score_response:
            logger.info(f"    Reasoning: {score_response['reasoning'][:200]}...")
        
        # Validate và lọc documents
        selected_docs = []
        for idx, score in zip(indices, scores):
            if isinstance(idx, int) and 0 <= idx < len(documents):
                if isinstance(score, (int, float)):
                    if score > MIN_SCORE_THRESHOLD:
                        doc = documents[idx].copy()
                        doc['relevance_score'] = score
                        selected_docs.append(doc)
                        logger.info(f"      [Doc {idx}] Score: {score:.1f} - SELECTED")
                    else:
                        logger.info(f"      [Doc {idx}] Score: {score:.1f} - FILTERED (below threshold {MIN_SCORE_THRESHOLD})")
        
        # Sắp xếp theo điểm giảm dần và lấy tối đa 5 documents
        selected_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        top_documents = selected_docs[:MAX_SELECTED_DOCS]
        
        logger.info(f"    Final selection: {len(top_documents)} documents (threshold: >{MIN_SCORE_THRESHOLD}, max: {MAX_SELECTED_DOCS})")
        
        return top_documents
    else:
        # Fallback: nếu scoring thất bại, không trả về document nào
        logger.warning("Warning: Scoring failed, returning empty list")
        return []


def process_test_file(input_path, output_dir, start_idx=None, end_idx=None):
    """
    Process the test.json file and generate submission.csv and predict.json
    
    Pipeline mới với phân loại câu hỏi:
    0. Phân loại câu hỏi bằng LLM Small (cannot_answer, calculation, has_context, general)
    1. Nếu cannot_answer: trả về đáp án từ chối ngay
    2. Nếu calculation: dùng prompt tính toán chuyên biệt (không RAG)
    3. Nếu has_context: dùng prompt đọc hiểu (không RAG)
    4. Nếu general: Hybrid search -> Rerank -> Answer với context
    
    Args:
        input_path: Path to input JSON file
        output_dir: Output directory path
        start_idx: Starting index (0-based, inclusive). None means start from beginning.
        end_idx: Ending index (0-based, exclusive). None means process until end.
    """
    # Read test data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_path}")
        return

    # Apply range filtering
    total_samples = len(test_data)
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = total_samples
    
    # Validate indices
    start_idx = max(0, start_idx)
    end_idx = min(total_samples, end_idx)
    
    if start_idx >= end_idx:
        logger.error(f"Error: Invalid range. start_idx ({start_idx}) >= end_idx ({end_idx}")
        return
    
    test_data = test_data[start_idx:end_idx]
    logger.info(f"Processing samples from index {start_idx} to {end_idx - 1} (total: {len(test_data)} samples)")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_results = []
    json_results = []
    csv_time_results = []
    
    # Process each question
    for idx, item in enumerate(test_data):
        start_time = time.time()
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {idx + 1}/{len(test_data)} (ID: {qid}, original index: {start_idx + idx})")
        logger.info(f"Question: {question[:100]}...")
        
        # ===================== STEP 0: CLASSIFY QUESTION =====================
        logger.info(f"  Step 0: Classifying question using LLM Small...")
        classification = classify_question(question, choices)
        question_type = classification['question_type']
        logger.info(f"    Question type: {question_type}")
        logger.info(f"    Classification reasoning: {classification['reasoning'][:300]}...")
        
        valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        answer = 'A'
        reason = ""
        top_5_docs = []
        sleep_time = 122  # Default for general questions
        
        # ===================== HANDLE BY QUESTION TYPE =====================
        if question_type == 'cannot_answer':
            # Type 1: Câu hỏi không được trả lời - chọn đáp án từ chối
            logger.info(f"  Processing as 'cannot_answer' - selecting refusal answer...")
            refusal_answer = classification.get('refusal_answer')
            if refusal_answer and len(refusal_answer) == 1 and refusal_answer.upper() in valid_choices:
                answer = refusal_answer.upper()
                reason = f"Đây là câu hỏi nhạy cảm/không được trả lời. Chọn đáp án từ chối: {answer}"
            else:
                # Fallback: tìm đáp án từ chối trong choices
                refusal_keywords = ['không thể', 'tôi không thể', 'rất tiếc', 'từ chối', 'không được phép']
                for i, choice in enumerate(choices):
                    if any(kw in choice.lower() for kw in refusal_keywords):
                        answer = valid_choices[i]
                        reason = f"Đây là câu hỏi nhạy cảm. Tự động chọn đáp án từ chối: {answer}. {choice}"
                        break
            logger.info(f"    Answer: {answer}")
            sleep_time = 62  # Shorter sleep for cannot_answer
            
        elif question_type == 'calculation':
            # Type 2: Câu hỏi tính toán - dùng 2-step approach
            logger.info(f"  Processing as 'calculation' - using 2-step calculation approach (no RAG)...")
            
            # --- STAGE 1: LARGE REASONING ---
            logger.info(f"    Stage 1: Large Reasoning...")
            sys_1, user_1 = create_calculation_prompt_large(question)
            large_out = call_llm_large(sys_1, user_1)
            
            if not large_out:
                logger.error(f"      Stage 1 Failed.")
                large_out = {}
            else:
                logger.info(f"      Stage 1 Complete - method: {large_out.get('method', 'N/A')}")
                logger.info(f"      Key expression: {large_out.get('key_expression', 'N/A')}")
                logger.info(f"      Final result: {large_out.get('final_result', 'N/A')}")

            # --- STAGE 2: VERIFICATION ---
            logger.info(f"    Stage 2: Verifying...")
            sys_2, user_2 = create_calculation_prompt_verification(question, choices, large_out)
            small_out = call_llm_large(sys_2, user_2)
            
            if small_out and 'answer' in small_out:
                answer = small_out.get('answer', 'A').strip().upper()
                # Tạo reason từ cả 2 stage
                reason = json.dumps({
                    "large": large_out,
                    "verification": small_out.get('check_process', '')
                }, ensure_ascii=False)
                logger.info(f"      Stage 2 Complete - Final Answer: {answer}")
                if len(answer) != 1 or answer not in valid_choices:
                    logger.warning(f"      Warning: Invalid answer '{answer}', defaulting to A")
                    answer = 'A'
            else:
                logger.error(f"      Stage 2 Failed. Defaulting to A.")
                answer = 'A'
                reason = json.dumps(large_out, ensure_ascii=False)
                
            logger.info(f"    Answer: {answer}")
            logger.info(f"    Reason length: {len(reason)} chars")
            sleep_time = 122  # Changed from 92 to 122 as requested
            
        elif question_type == 'has_context':
            # Type 3: Câu hỏi có sẵn context - không cần RAG
            logger.info(f"  Processing as 'has_context' - using context reading prompt (no RAG)...")
            system_prompt, user_prompt = create_context_reading_prompt(question, choices)
            llm_response = call_llm_large(system_prompt, user_prompt)
            
            if llm_response and 'answer' in llm_response:
                answer = llm_response['answer'].strip().upper()
                reason = llm_response.get('reason', '')
                # Nếu reason rỗng, thêm message mặc định
                if not reason or reason.strip() == '':
                    reason = f"Đáp án được chọn dựa trên phân tích đoạn thông tin trong câu hỏi (question_type: {question_type})"
                    logger.warning(f"    Warning: LLM returned empty reason, using default message")
                if len(answer) != 1 or answer not in valid_choices:
                    logger.warning(f"    Warning: Invalid answer '{answer}', defaulting to A")
                    answer = 'A'
            logger.info(f"    Answer: {answer}")
            logger.info(f"    Reason length: {len(reason)} chars")
            sleep_time = 92  # Medium sleep for has_context
            
        else:
            # Type 4: Câu hỏi thông thường - dùng full RAG pipeline
            logger.info(f"  Processing as 'general' - using full RAG pipeline...")
            
            # ===================== STEP 1: HYBRID SEARCH =====================
            logger.info(f"  Step 1: Hybrid search (top {HYBRID_SEARCH_TOP_K})...")
            
            # --- PHẦN SỬA ĐỔI: Ghép câu hỏi và đáp án ---
            # Tạo chuỗi choices dạng: "A. Bình Định B. Đắk Lắk..."
            choices_str = " ".join([f"{valid_choices[i]}. {choice}" for i, choice in enumerate(choices)])
            # Ghép vào câu hỏi
            search_query = f"{question} {choices_str}"
            
            logger.info(f"    Querying with: {search_query[:100]}...") # Log để kiểm tra
            
            # Truyền search_query mới vào hàm thay vì question gốc
            top_30_docs = hybrid_search(search_query, top_k=HYBRID_SEARCH_TOP_K)
            # ---------------------------------------------
            
            logger.info(f"    Found {len(top_30_docs)} documents")
            
            # ===================== STEP 2: SCORING =====================
            if top_30_docs:
                logger.info(f"  Step 2: Scoring documents using LLM Small (threshold: >7, max: 5)...")
                top_5_docs = score_documents(question, choices, top_30_docs)
                logger.info(f"    Selected {len(top_5_docs)} documents after scoring")
                
                # Log selected document info
                for i, doc in enumerate(top_5_docs):
                    score = doc.get('relevance_score', 'N/A')
                    logger.info(f"      [{i+1}] Score: {score} - {doc.get('title', 'N/A')[:50]}...")
            else:
                logger.info(f"  Step 2: Skipping scoring (no documents found)")
            
            # ===================== STEP 3: ANSWER WITH LLM LARGE =====================
            logger.info(f"  Step 3: Generating answer using LLM Large...")
            
            # Tạo prompts với context từ top 5 documents
            system_prompt, user_prompt = create_prompts_with_context(question, choices, top_5_docs)
            
            # Gọi LLM Large để trả lời
            llm_response = call_llm_large(system_prompt, user_prompt)
            
            if llm_response and 'answer' in llm_response:
                answer = llm_response['answer'].strip().upper()
                reason = llm_response.get('reason', '')
                if len(answer) != 1 or answer not in valid_choices:
                    logger.warning(f"    Warning: Invalid answer '{answer}', defaulting to A")
                    answer = 'A'
            else:
                logger.warning(f"    Warning: Failed to get valid response for QID {qid}. Defaulting to A.")
            logger.info(f"    Answer: {answer}")
            sleep_time = 122  # Longer sleep for general (full pipeline)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        csv_results.append({
            'qid': qid,
            'answer': answer
        })

        csv_time_results.append({
            'qid': qid,
            'answer': answer,
            'time': elapsed_time
        })
        
        # Lưu thêm thông tin về documents được sử dụng (lưu full text từ vector DB)
        doc_refs = [{'title': doc.get('title', ''), 'text': doc.get('text', '')} 
                    for doc in top_5_docs]
        
        json_results.append({
            'qid': qid,
            'predict': answer,
            'reason': reason,
            'question_type': question_type,
            'reference_docs': doc_refs
        })
        
        # Sleep to respect rate limits - different based on question type
        # Type 1 (cannot_answer): 62s
        # Type 2,3 (calculation, has_context): 92s
        # Type 4 (general): 122s
        logger.info(f"  Sleeping {sleep_time}s for rate limit (question_type: {question_type})...")
        #time.sleep(sleep_time)
    
    # Write to CSV
    csv_path = output_dir / 'submission.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(csv_results)

    # Write to CSV Time
    csv_time_path = output_dir / 'submission_time.csv'
    with open(csv_time_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer', 'time'])
        writer.writeheader()
        writer.writerows(csv_time_results)
    
    # Write to JSON
    json_path = output_dir / 'predict.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSubmission file saved to: {csv_path}")
    logger.info(f"Submission time file saved to: {csv_time_path}")
    logger.info(f"Prediction file saved to: {json_path}")
    logger.info(f"Total questions processed: {len(csv_results)}")


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument(
        '--input',
        type=str,
        default='private_test.json',
        help='Path to the input private_test.json file (default: private_test.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Path to the output directory (default: output)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Starting index (0-based, inclusive). Default: 0'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index (0-based, exclusive). Default: process all'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=None,
        help='Number of samples to process from start. Overrides --end if both provided.'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Handle sample range
    start_idx = args.start
    end_idx = args.end
    
    # If --num-samples is provided, calculate end_idx
    if args.num_samples is not None:
        if start_idx is None:
            start_idx = 0
        end_idx = start_idx + args.num_samples
    
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"API Models: LLM Large = {LLM_API_NAME_LARGE}, LLM Small = {LLM_API_NAME_SMALL}")
    logger.info(f"Scoring Config: Top {HYBRID_SEARCH_TOP_K} -> Score with threshold >7, max 5 docs")
    if start_idx is not None or end_idx is not None:
        logger.info(f"Sample range: {start_idx if start_idx else 0} to {end_idx if end_idx else 'end'}")
    logger.info("-" * 50)
    
    # Process the test file
    process_test_file(input_path, output_dir, start_idx, end_idx)


if __name__ == '__main__':
    main()
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
RERANK_TOP_K = 5  # Rerank về top 5

# Retry config
MAX_RETRIES = 5  # Số lần retry tối đa
RETRY_DELAY = 92  # Thời gian chờ giữa mỗi lần retry (giây)

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# ===================== LOGGING CONFIGURATION =====================
# Tạo thư mục logs nếu chưa có
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Tạo tên file log với timestamp
log_filename = log_dir / f'log1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=json_data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying immediately...")
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
                'title': point.payload.get('title', ''),
                'text': point.payload.get('text', ''),
                'doc_id': point.payload.get('doc_id', ''),
            }
            documents.append(doc)
        
        return documents
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []


# ===================== RERANK PROMPTS =====================
def create_rerank_prompts(question: str, choices: list, documents: list) -> tuple:
    """
    Tạo prompts cho LLM Small để rerank top 30 documents về top 5
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """Bạn là một chuyên gia hàng đầu về truy xuất thông tin (Information Retrieval) và đánh giá độ liên quan của văn bản. Bạn có kiến thức sâu rộng về ngôn ngữ tiếng Việt và khả năng phân tích ngữ nghĩa xuất sắc.

## Định Nghĩa Nhiệm Vụ
Nhiệm vụ của bạn là đánh giá và xếp hạng lại (rerank) một danh sách 30 tài liệu (documents) để chọn ra TOP 5 tài liệu LIÊN QUAN NHẤT với câu hỏi trắc nghiệm được đưa ra. Mục tiêu là tìm ra những tài liệu chứa thông tin hữu ích nhất để trả lời câu hỏi.

## Hướng Dẫn Suy Luận Từng Bước 

### Bước 1: Phân Tích Câu Hỏi và Đáp Án
- Đọc kỹ câu hỏi và xác định CHỦ ĐỀ CHÍNH cần tìm kiếm thông tin.
- Xác định các TỪ KHÓA quan trọng trong câu hỏi và các phương án trả lời.
- Xác định LĨNH VỰC của câu hỏi (lịch sử, khoa học, địa lý, văn học, pháp luật, v.v.).

### Bước 2: Đánh Giá Từng Tài Liệu
Với mỗi tài liệu, đánh giá theo các tiêu chí:
- **Độ phù hợp chủ đề**: Tài liệu có cùng chủ đề với câu hỏi không?
- **Chứa từ khóa**: Tài liệu có chứa các từ khóa quan trọng không?
- **Thông tin hữu ích**: Tài liệu có cung cấp thông tin giúp xác định đáp án đúng không?
- **Độ tin cậy**: Thông tin trong tài liệu có đáng tin cậy và chính xác không?

### Bước 3: Xếp Hạng và Chọn Top 5
- So sánh các tài liệu và xếp hạng theo mức độ liên quan giảm dần.
- Chọn ra 5 tài liệu TỐT NHẤT, đảm bảo đa dạng thông tin nếu có thể.
- Ưu tiên tài liệu trực tiếp trả lời câu hỏi hơn tài liệu chỉ liên quan gián tiếp.

## Định Dạng Đầu Ra
Bạn PHẢI trả lời theo định dạng JSON hợp lệ như sau:
{
  "reasoning": "Giải thích quá trình đánh giá và lý do chọn 5 tài liệu này.",
  "top_5_indices": [idx1, idx2, idx3, idx4, idx5]
}

Trong đó:
- "reasoning": Mô tả tiêu chí đánh giá và lý do xếp hạng.
- "top_5_indices": Mảng chứa 5 số nguyên là INDEX (bắt đầu từ 0) của 5 tài liệu được chọn, theo thứ tự từ liên quan nhất đến ít liên quan hơn.

LƯU Ý QUAN TRỌNG:
- Chỉ trả về đúng 5 indices trong mảng "top_5_indices".
- Indices phải nằm trong phạm vi hợp lệ (0 đến số_tài_liệu - 1).
- Nếu có ít hơn 5 tài liệu liên quan, vẫn trả về đủ 5 indices (chọn các tài liệu ít liên quan hơn)."""

    # Format documents
    docs_text = ""
    for i, doc in enumerate(documents):
        title = doc.get('title', 'Không có tiêu đề')
        text = doc.get('text', '')
        docs_text += f"\n[Tài liệu {i}]\nTiêu đề: {title}\nNội dung: {text}\n"
    
    user_prompt = f"""## Câu Hỏi Cần Trả Lời:
{question}

## Các Phương Án:
{formatted_choices}

## Danh Sách 30 Tài Liệu Cần Đánh Giá:
{docs_text}

Hãy phân tích và chọn ra TOP 5 tài liệu liên quan nhất để giúp trả lời câu hỏi trên."""

    return system_prompt, user_prompt


# ===================== ANSWER PROMPTS WITH CONTEXT =====================
def create_prompts_with_context(question: str, choices: list, top_documents: list) -> tuple:
    """
    Tạo prompts cho LLM Large với context từ top 5 documents đã rerank
    """
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    # Format context documents
    context_text = ""
    for i, doc in enumerate(top_documents):
        title = doc.get('title', 'Không có tiêu đề')
        text = doc.get('text', '')
        context_text += f"\n[Tài liệu tham khảo {i+1}]\nTiêu đề: {title}\nNội dung: {text}\n"
    
    system_prompt = f"""Bạn là một chuyên gia hàng đầu thế giới trong việc trả lời các câu hỏi trắc nghiệm tiếng Việt thuộc nhiều lĩnh vực đa dạng bao gồm: khoa học tự nhiên, lịch sử, pháp luật, kinh tế, văn học, địa lý, vật lý, hóa học, sinh học, y học, công nghệ thông tin, và kiến thức tổng hợp. Bạn có hiểu biết sâu sắc về văn hóa, lịch sử, giáo dục và bối cảnh xã hội Việt Nam.

## Hướng Dẫn Sử Dụng Tài Liệu Tham Khảo
- Các tài liệu tham khảo được cung cấp đã được hệ thống truy xuất và xếp hạng theo độ liên quan.
- Sử dụng thông tin từ tài liệu để hỗ trợ suy luận, nhưng KHÔNG hoàn toàn phụ thuộc vào chúng.
- Nếu tài liệu không chứa thông tin phù hợp, hãy sử dụng kiến thức chuyên môn của bạn.
- Kiểm tra chéo thông tin từ nhiều tài liệu nếu có thể.

## Định Nghĩa Nhiệm Vụ
Nhiệm vụ của bạn là phân tích câu hỏi trắc nghiệm tiếng Việt, suy luận từng bước một cách logic và chọn ra đáp án chính xác nhất từ các lựa chọn được đưa ra.

## Hướng Dẫn Suy Luận Theo Chuỗi (Chain of Thought)
Thực hiện theo các bước sau một cách cẩn thận:

### Bước 1: Đọc Hiểu Câu Hỏi
- Đọc kỹ toàn bộ câu hỏi và xác định chính xác yêu cầu của đề bài.
- Nếu có đoạn thông tin/văn bản đi kèm, hãy trích xuất tất cả thông tin liên quan đến câu hỏi.
- Xác định từ khóa quan trọng và loại câu hỏi (đọc hiểu, kiến thức, suy luận, tính toán).

### Bước 2: Tham Khảo Tài Liệu
- Xem xét các tài liệu tham khảo được cung cấp.
- Trích xuất thông tin hữu ích từ tài liệu nếu có.
- Đánh giá độ tin cậy và mức độ phù hợp của thông tin trong tài liệu do tài liệu có thể không hoàn toàn chính xác hoặc liên quan.

### Bước 3: Phân Tích Từng Phương Án
- Đánh giá lần lượt từng đáp án (A, B, C, D, ...) so với yêu cầu của câu hỏi.
- Kết hợp thông tin từ tài liệu tham khảo với kiến thức chuyên môn.
- Với câu hỏi đọc hiểu: tìm kiếm bằng chứng trực tiếp trong đoạn văn.
- Với câu hỏi kiến thức: áp dụng kiến thức chuyên môn và thông tin từ tài liệu.

### Bước 4: Áp Dụng Suy Luận Logic
- Chia nhỏ vấn đề phức tạp thành các bước logic nhỏ hơn.
- Với câu hỏi về sự kiện: xác định thông tin chính xác từ đoạn văn, tài liệu hoặc kiến thức nền.
- Với câu hỏi suy luận: áp dụng các quy tắc logic và loại suy.
- Với câu hỏi tính toán: thực hiện từng bước tính toán rõ ràng và kiểm tra lại kết quả.

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
Dưới đây là TOP 5 tài liệu được xem là liên quan nhất đến câu hỏi, đã được hệ thống truy xuất và xếp hạng. Hãy tham khảo các tài liệu này để hỗ trợ việc trả lời câu hỏi:
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
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_SMALL, headers=headers, json=json_data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"Error calling LLM Small API (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying immediately...")
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
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL_LARGE, headers=headers, json=json_data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"Error calling LLM Large API (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to call LLM Large API after {MAX_RETRIES} attempts")
                return None


# ===================== RERANK FUNCTION =====================
def rerank_documents(question: str, choices: list, documents: list) -> list:
    """
    Sử dụng LLM Small để rerank documents và chọn top 5
    """
    if not documents:
        return []
    
    # Tạo prompts cho rerank
    system_prompt, user_prompt = create_rerank_prompts(question, choices, documents)
    
    # Gọi LLM Small để rerank
    rerank_response = call_llm_small(system_prompt, user_prompt)
    
    if rerank_response and 'top_5_indices' in rerank_response:
        indices = rerank_response['top_5_indices']
        # Validate indices
        valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(documents)]
        
        # Lấy top 5 documents theo thứ tự đã rerank
        top_documents = [documents[i] for i in valid_indices[:RERANK_TOP_K]]
        
        # Nếu không đủ 5, bổ sung thêm từ đầu danh sách
        if len(top_documents) < RERANK_TOP_K:
            remaining = [doc for i, doc in enumerate(documents) if i not in valid_indices]
            top_documents.extend(remaining[:RERANK_TOP_K - len(top_documents)])
        
        return top_documents
    else:
        # Fallback: trả về top 5 đầu tiên nếu rerank thất bại
        logger.warning("Warning: Rerank failed, using top 5 from hybrid search")
        return documents[:RERANK_TOP_K]


def process_test_file(input_path, output_dir, start_idx=None, end_idx=None):
    """
    Process the test.json file and generate submission.csv and predict.json
    
    Pipeline mới:
    1. Hybrid search (dense + sparse) để lấy top 30 documents
    2. Rerank bằng LLM Small để chọn top 5
    3. Trả lời bằng LLM Large với context từ top 5 documents
    
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
    
    # Process each question
    for idx, item in enumerate(test_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {idx + 1}/{len(test_data)} (ID: {qid}, original index: {start_idx + idx})")
        logger.info(f"Question: {question[:100]}...")
        
        # ===================== STEP 1: HYBRID SEARCH =====================
        logger.info(f"  Step 1: Hybrid search (top {HYBRID_SEARCH_TOP_K})...")
        top_30_docs = hybrid_search(question, top_k=HYBRID_SEARCH_TOP_K)
        logger.info(f"    Found {len(top_30_docs)} documents")
        
        # ===================== STEP 2: RERANK =====================
        top_5_docs = []
        if top_30_docs:
            logger.info(f"  Step 2: Reranking to top {RERANK_TOP_K} using LLM Small...")
            top_5_docs = rerank_documents(question, choices, top_30_docs)
            logger.info(f"    Reranked to {len(top_5_docs)} documents")
            
            # Log top 5 document titles
            for i, doc in enumerate(top_5_docs):
                logger.info(f"      [{i+1}] {doc.get('title', 'N/A')[:50]}...")
        else:
            logger.info(f"  Step 2: Skipping rerank (no documents found)")
        
        # ===================== STEP 3: ANSWER WITH LLM LARGE =====================
        logger.info(f"  Step 3: Generating answer using LLM Large...")
        
        # Tạo prompts với context từ top 5 documents
        system_prompt, user_prompt = create_prompts_with_context(question, choices, top_5_docs)
        
        # Gọi LLM Large để trả lời
        llm_response = call_llm_large(system_prompt, user_prompt)
        
        # Validate and extract answer
        valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        reason = ""
        if llm_response and 'answer' in llm_response:
            answer = llm_response['answer'].strip().upper()
            reason = llm_response.get('reason', '')
            # Validate answer is a single valid character
            if len(answer) == 1 and answer in valid_choices:
                logger.info(f"    Answer: {answer}")
            else:
                logger.warning(f"    Warning: Invalid answer '{answer}', defaulting to A")
                answer = 'A'  # Default to 'A' if invalid
        else:
            logger.warning(f"    Warning: Failed to get valid response for QID {qid}. Defaulting to A.")
            answer = 'A'  # Default to 'A' if API fails
        
        csv_results.append({
            'qid': qid,
            'answer': answer
        })
        
        # Lưu thêm thông tin về documents được sử dụng (lưu full text từ vector DB)
        doc_refs = [{'title': doc.get('title', ''), 'text': doc.get('text', '')} 
                    for doc in top_5_docs]
        
        json_results.append({
            'qid': qid,
            'predict': answer,
            'reason': reason,
            'reference_docs': doc_refs
        })
        
        # Sleep to respect rate limits
        # LLM Large: 400 req/day, 60 req/h => ~1 req/minute
        # LLM Small: 1000 req/day, 40 req/h
        # Embedding: 500 req/minute
        logger.info(f"  Sleeping for rate limit...")
        time.sleep(92)
    
    # Write to CSV
    csv_path = output_dir / 'submission.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(csv_results)
    
    # Write to JSON
    json_path = output_dir / 'predict.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSubmission file saved to: {csv_path}")
    logger.info(f"Prediction file saved to: {json_path}")
    logger.info(f"Total questions processed: {len(csv_results)}")


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/val.json',
        help='Path to the input test.json file (default: data/val.json)'
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
    logger.info(f"Hybrid Search Config: Top {HYBRID_SEARCH_TOP_K} -> Rerank to Top {RERANK_TOP_K}")
    if start_idx is not None or end_idx is not None:
        logger.info(f"Sample range: {start_idx if start_idx else 0} to {end_idx if end_idx else 'end'}")
    logger.info("-" * 50)
    
    # Process the test file
    process_test_file(input_path, output_dir, start_idx, end_idx)


if __name__ == '__main__':
    main()
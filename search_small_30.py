import json
import requests
from pathlib import Path
import os
import time
import logging
from datetime import datetime
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ===================== INPUT VARIABLES =====================
# Chỉ cần nhập ID (số thứ tự): ví dụ id = 1 -> qid = "test_0001"
TEST_ID = 3  # Thay đổi giá trị này để chọn câu hỏi khác

def load_test_data(test_id: int) -> tuple:
    """
    Load dữ liệu từ test.json theo ID
    Args:
        test_id: ID của câu hỏi (1-N)
    Returns:
        (qid, query, choices_with_labels)
    """
    try:
        # Convert ID to qid format: 1 -> "val_0001"
        qid = f"val_{test_id:04d}"
        
        # Load val.json
        test_file = Path('./data/val.json')
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Find the corresponding item
        item = next((q for q in test_data if q.get('qid') == qid), None)
        if not item:
            raise ValueError(f"Không tìm thấy câu hỏi với id {test_id} (qid: {qid})")
        
        query = item.get('question', '')
        raw_choices = item.get('choices', [])
        
        # Add A, B, C, D... labels to choices
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        choices_with_labels = [f"{choice_labels[i]}. {choice}" for i, choice in enumerate(raw_choices)]
        
        return qid, query, choices_with_labels
    except Exception as e:
        logger.error(f"Lỗi load test data: {e}")
        exit(1)

# Load dữ liệu từ test.json
qid, query, choices = load_test_data(TEST_ID)

# ===================== CẤU HÌNH (GIỐNG PREDICT.PY) =====================
API_URL_SMALL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'
EMBEDDING_API_URL = 'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding'

# Qdrant config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki"

# Hybrid search config
HYBRID_SEARCH_TOP_K = 30
BM25_MODEL_NAME = "Qdrant/bm25"

# Retry config
MAX_RETRIES = 5
RETRY_DELAY = 92
LLM_API_NAME_SMALL = 'vnptai_hackathon_small'

# ===================== LOGGING CONFIGURATION =====================
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== LOAD CREDENTIALS =====================
def load_credentials(json_path='./vectorDB/api-keys1.json'):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        config_small = next((item for item in keys if item.get('llmApiName') == 'LLM small'), None)
        config_embedding = next((item for item in keys if item.get('llmApiName') == 'LLM embedings'), None)
        
        if not config_small or not config_embedding:
            raise ValueError("Thiếu cấu hình API trong file api-keys.json")
            
        return {'small': config_small, 'embedding': config_embedding}
    except Exception as e:
        logger.error(f"Lỗi đọc credentials: {e}")
        exit(1)

api_configs = load_credentials()

AUTHORIZATION_SMALL = api_configs['small']['authorization']
TOKEN_ID_SMALL = api_configs['small']['tokenId']
TOKEN_KEY_SMALL = api_configs['small']['tokenKey']

AUTHORIZATION_EMBEDDING = api_configs['embedding']['authorization']
TOKEN_ID_EMBEDDING = api_configs['embedding']['tokenId']
TOKEN_KEY_EMBEDDING = api_configs['embedding']['tokenKey']

# ===================== INIT MODELS =====================
logger.info("Loading BM25 model...")
bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

logger.info("Connecting to Qdrant...")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ===================== CORE FUNCTIONS (FROM PREDICT.PY) =====================
def get_dense_embedding(text: str) -> list:
    headers = {
        'Authorization': AUTHORIZATION_EMBEDDING,
        'Token-id': TOKEN_ID_EMBEDDING,
        'Token-key': TOKEN_KEY_EMBEDDING,
        'Content-Type': 'application/json',
    }
    if len(text) > 8192: text = text[:8192]
    json_data = {'model': 'vnptai_hackathon_embedding', 'input': text, 'encoding_format': 'float'}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=json_data, timeout=180)
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Embedding error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(1)
            else: return None

def get_sparse_embedding(text: str) -> tuple:
    if not text or not isinstance(text, str): return [], []
    try:
        embeddings = list(bm25_model.embed([text]))
        if embeddings:
            return embeddings[0].indices.tolist(), embeddings[0].values.tolist()
    except Exception as e:
        logger.error(f"BM25 error: {e}")
    return [], []

def hybrid_search(question: str, top_k: int = HYBRID_SEARCH_TOP_K) -> list:
    dense_vector = get_dense_embedding(question)
    if dense_vector is None: return []
    sparse_indices, sparse_values = get_sparse_embedding(question)
    
    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                models.Prefetch(query=models.SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=top_k * 2),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        
        documents = []
        for point in results.points:
            documents.append({
                'id': point.id,
                'hybrid_score': point.score, # Score từ RRF
                'text': point.payload.get('text', ''),
                'title': point.payload.get('title', ''), # Lấy thêm title để dễ debug
                'doc_id': point.payload.get('doc_id', ''),
            })
        return documents
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return []

def create_scoring_prompts(question: str, choices: list, documents: list) -> tuple:
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
    
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

    docs_text = ""
    for i, doc in enumerate(documents):
        docs_text += f"\n[Tài liệu {i}]\n{doc.get('text', '')}\n"
    
    user_prompt = f"""## Câu Hỏi Cần Trả Lời:
{question}

## Các Phương Án:
{formatted_choices}

## Danh Sách 30 Tài Liệu Cần Chấm Điểm:
{docs_text}

Hãy chấm điểm từng tài liệu theo 4 tiêu chí và trả về kết quả dạng JSON."""

    return system_prompt, user_prompt

def call_llm_small(system_prompt: str, user_prompt: str) -> dict:
    headers = {
        'Authorization': AUTHORIZATION_SMALL,
        'Token-id': TOKEN_ID_SMALL,
        'Token-key': TOKEN_KEY_SMALL,
        'Content-Type': 'application/json',
    }
    json_data = {
        'model': LLM_API_NAME_SMALL,
        'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
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
                logger.info(f"LLM Small raw response (FULL): {content}")
                parsed_response = json.loads(content)
                logger.info(f"LLM Small parsed keys: {list(parsed_response.keys())}")
                logger.info(f"LLM Small parsed values: {parsed_response}")
                return parsed_response
            return None
        except Exception as e:
            logger.error(f"LLM Small Error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else: return None

# ===================== MAIN EXECUTION =====================
def main():
    logger.info(f"STARTING SEARCH TEST")
    logger.info(f"QID: {qid}")
    logger.info(f"Query: {query}")
    logger.info(f"Choices: {choices}")
    
    # 1. Hybrid Search (Lấy top 20)
    logger.info("Step 1: Executing Hybrid Search...")
    
    # --- PHẦN SỬA ĐỔI: Ghép query và choices ---
    # Nối các lựa chọn lại thành chuỗi, cách nhau bởi dấu cách
    choices_str = " ".join(choices) 
    # Tạo search_query bao gồm câu hỏi + các lựa chọn
    search_query = f"{query} {choices_str}"
    
    logger.info(f"Search Query combined: {search_query}")
    
    # Truyền search_query (đã gộp) vào hàm tìm kiếm
    documents = hybrid_search(search_query, top_k=HYBRID_SEARCH_TOP_K)
    # -------------------------------------------
    
    logger.info(f"Found {len(documents)} documents via Hybrid Search.")

    # 2. Scoring with LLM Small
    if documents:
        logger.info("Step 2: Scoring documents with LLM Small...")
        system_prompt, user_prompt = create_scoring_prompts(query, choices, documents)
        score_response = call_llm_small(system_prompt, user_prompt)
        
        # 3. Process Scores
        # Khởi tạo điểm mặc định là 0 cho tất cả docs
        for doc in documents:
            doc['llm_score'] = 0.0
            
        llm_reasoning = ""
        
        if score_response and 'indices' in score_response and 'scores' in score_response:
            indices = score_response['indices']
            scores = score_response['scores']
            llm_reasoning = score_response.get('reasoning', '')
            
            logger.info(f"LLM Reasoning: {llm_reasoning[:200]}...")
            
            # Map điểm từ LLM về document tương ứng
            count_scored = 0
            for idx, score in zip(indices, scores):
                if isinstance(idx, int) and 0 <= idx < len(documents):
                    if isinstance(score, (int, float)):
                        documents[idx]['llm_score'] = float(score)
                        count_scored += 1
            logger.info(f"LLM returned scores for {count_scored} documents.")
        else:
            logger.warning("Failed to get valid scoring response from LLM.")

        # 4. Sort documents by LLM Score Descending
        documents.sort(key=lambda x: x['llm_score'], reverse=True)

    # 5. Output to JSON
    output_dir = Path('./output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'search_results.json'

    result_data = {
        'qid': qid,
        'query': query,
        'choices': choices,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'llm_reasoning': llm_reasoning if 'llm_reasoning' in locals() else "",
        'documents': documents  # Đã sort theo llm_score
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSearch results saved to: {output_path}")
    
    # In ra terminal top 5 để kiểm tra nhanh
    logger.info("\n=== TOP 5 DOCUMENTS BY LLM SCORE ===")
    for i, doc in enumerate(documents[:5]):
        logger.info(f"Rank {i+1}: LLM Score {doc['llm_score']} | Hybrid Score {doc['hybrid_score']:.4f}")
        logger.info(f"Title: {doc.get('title', 'No title')}")
        logger.info(f"Snippet: {doc['text'][:100]}...")
        logger.info("-" * 30)
    
    # 6. Filter documents with llm_score > 7 and take max 5
    filtered_docs = [doc for doc in documents if doc['llm_score'] > 7][:5]
    logger.info(f"\nFiltered {len(filtered_docs)} documents with llm_score > 7")
    
    if filtered_docs:
        # 7. Create context text from filtered documents
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
        
        context_parts = []
        for i, doc in enumerate(filtered_docs, 1):
            context_parts.append(f"[Tài liệu {i}] {doc.get('title', 'No title')}")
            context_parts.append(doc['text'])
            context_parts.append("")
        
        context_text = "\n".join(context_parts)
        
        # 8. Create output text file
        output_txt_path = output_dir / 'small.txt'
        
        txt_content = f"""Tài Liệu Tham Khảo
Dưới đây là {len(filtered_docs)} tài liệu được xem là liên quan nhất đến câu hỏi. Hãy tham khảo các tài liệu này để hỗ trợ việc trả lời câu hỏi:

{context_text}

Question: {query}

Choices:
{formatted_choices}
"""
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        logger.info(f"Reference documents saved to: {output_txt_path}")

if __name__ == '__main__':
    main()
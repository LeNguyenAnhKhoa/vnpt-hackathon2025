"""
Main script để tạo Vector Database từ Vietnamese Wikipedia dataset
- Sử dụng Qdrant cho vector storage
- VNPT AI Embedding API cho dense vectors
- Fastembed BM25 (Qdrant/bm25) cho sparse vectors
- Chunking theo câu với llama-index SentenceSplitter (chunk_size=2048, overlap=128)
"""
import json
import requests
import pandas as pd
import time
import logging
from datetime import datetime
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, Modifier
from llama_index.core.node_parser import SentenceSplitter
from fastembed import SparseTextEmbedding

# ===================== LOGGING CONFIG =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectordb_creation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== CẤU HÌNH =====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki"

# Data config
CSV_PATH = "data/vie_wiki_dataset.csv"
NUM_ROWS = 10000  # Chỉ lấy 10000 dòng đầu tiên

# Chunking config
MAX_CHUNK_LENGTH = 2048
CHUNK_OVERLAP = 128

# Embedding API config
EMBEDDING_API_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
EMBEDDING_DIM = 1024  # Đã xác nhận từ test

# Rate limiting: 500 req/minute = ~8.3 req/second
# Để an toàn, sử dụng 7 req/second = ~0.15s delay
REQUESTS_PER_MINUTE = 500
REQUEST_DELAY = 60 / REQUESTS_PER_MINUTE + 0.02  # ~0.14s delay giữa các request

# Batch config
BATCH_SIZE = 100  # Số points để upsert mỗi batch

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# ===================== LOAD API KEYS =====================
def load_embedding_credentials(json_path='../api-keys.json'):
    """Đọc file api-keys.json và lấy credentials cho Embedding"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        config = next((item for item in keys if item.get('llmApiName') == 'LLM embedings'), None)
        
        if not config:
            raise ValueError("Không tìm thấy cấu hình 'LLM embedings' trong file api-keys.json")
            
        return config
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file {json_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình: {e}")
        exit(1)

# ===================== CHUNKING FUNCTIONS =====================
# Khởi tạo SentenceSplitter từ llama-index
sentence_splitter = SentenceSplitter(
    chunk_size=MAX_CHUNK_LENGTH,
    chunk_overlap=CHUNK_OVERLAP,
)

def chunk_by_sentences(text: str, max_length: int = 2048) -> list:
    """
    Chunking theo câu sử dụng llama-index SentenceSplitter
    với chunk_size=2048 và chunk_overlap=128
    """
    if not text or not isinstance(text, str):
        return []
    
    try:
        # Sử dụng SentenceSplitter để tách văn bản thành chunks
        chunks = sentence_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.warning(f"Error chunking text: {e}")
        # Fallback: cắt theo ký tự nếu có lỗi
        if len(text) <= max_length:
            return [text]
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# ===================== EMBEDDING FUNCTION =====================
def get_embedding(text: str, credentials: dict, max_retries: int = 3) -> list:
    """
    Gọi VNPT Embedding API để lấy vector
    Có retry logic để xử lý lỗi tạm thời
    """
    headers = {
        'Authorization': credentials['authorization'],
        'Token-id': credentials['tokenId'],
        'Token-key': credentials['tokenKey'],
        'Content-Type': 'application/json',
    }
    
    # Truncate text nếu quá dài (API limit 8k tokens)
    if len(text) > 8000:
        text = text[:8000]
    
    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': text,
        'encoding_format': 'float',
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=json_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']
            elif response.status_code == 429:  # Rate limited
                wait_time = (attempt + 1) * 5  # Exponential backoff
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

# ===================== BM25 SPARSE VECTOR (FASTEMBED) =====================
# Khởi tạo Fastembed BM25 model
bm25_model = None

def get_bm25_model():
    """Lazy initialization của BM25 model"""
    global bm25_model
    if bm25_model is None:
        logger.info(f"Loading BM25 model: {BM25_MODEL_NAME}")
        bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
        logger.info("BM25 model loaded successfully")
    return bm25_model

def compute_bm25_sparse_vector(text: str) -> tuple:
    """
    Tính sparse vector sử dụng Fastembed BM25 (Qdrant/bm25)
    
    Trả về (indices, values)
    
    Args:
        text: Văn bản cần tính sparse vector
    """
    if not text or not isinstance(text, str):
        return [], []
    
    try:
        model = get_bm25_model()
        embeddings = list(model.embed([text]))
        
        if embeddings:
            sparse_embedding = embeddings[0]
            indices = sparse_embedding.indices.tolist()
            values = sparse_embedding.values.tolist()
            return indices, values
        else:
            return [], []
    except Exception as e:
        logger.warning(f"Error computing BM25 sparse vector: {e}")
        return [], []

# ===================== QDRANT SETUP =====================
def setup_qdrant_collection(client: QdrantClient, collection_name: str, recreate: bool = False):
    """
    Tạo hoặc kiểm tra collection trong Qdrant
    """
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists, resuming...")
            return False  # Không cần tạo mới
    
    logger.info(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            # Sử dụng Modifier.IDF để Qdrant tự động apply IDF weights khi search
            # Điều này giúp BM25 hoạt động chính xác hơn
            "sparse": SparseVectorParams(
                modifier=Modifier.IDF
            )
        }
    )
    
    return True

def get_existing_point_ids(client: QdrantClient, collection_name: str) -> set:
    """
    Lấy danh sách các point ID đã có trong collection để resume
    """
    try:
        # Scroll qua tất cả points để lấy ID
        existing_ids = set()
        offset = None
        
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            points, next_offset = result
            
            for point in points:
                existing_ids.add(point.id)
            
            if next_offset is None:
                break
            offset = next_offset
        
        return existing_ids
    except Exception as e:
        logger.warning(f"Could not get existing IDs: {e}")
        return set()

# ===================== MAIN PROCESSING =====================
def process_documents(
    df: pd.DataFrame,
    client: QdrantClient,
    credentials: dict,
    collection_name: str,
    resume: bool = True
):
    """
    Xử lý tất cả documents: chunking, embedding, và lưu vào Qdrant
    """
    # Lấy danh sách point ID đã xử lý (nếu resume)
    existing_ids = set()
    if resume:
        existing_ids = get_existing_point_ids(client, collection_name)
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing points, will skip them")
    
    # Đếm số chunk cần xử lý
    total_chunks = 0
    doc_chunks_info = []
    
    logger.info("Calculating total chunks...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing documents"):
        text = row['text']
        if pd.isna(text):
            continue
        chunks = chunk_by_sentences(str(text), max_length=MAX_CHUNK_LENGTH)
        doc_chunks_info.append({
            'doc_idx': idx,
            'doc_id': row['id'],
            'title': row['title'],
            'chunks': chunks
        })
        total_chunks += len(chunks)
    
    logger.info(f"Total documents: {len(doc_chunks_info)}")
    logger.info(f"Total chunks to process: {total_chunks}")
    
    # Xử lý từng chunk
    points_batch = []
    global_chunk_idx = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    start_time = datetime.now()
    
    progress_bar = tqdm(total=total_chunks, desc="Processing chunks")
    
    for doc_info in doc_chunks_info:
        doc_id = doc_info['doc_id']
        title = doc_info['title']
        chunks = doc_info['chunks']
        
        for chunk_idx, chunk in enumerate(chunks):
            point_id = global_chunk_idx
            global_chunk_idx += 1
            progress_bar.update(1)
            
            # Skip nếu đã xử lý
            if point_id in existing_ids:
                skipped_count += 1
                continue
            
            # Get embedding
            embedding = get_embedding(chunk, credentials)
            
            if embedding is None:
                error_count += 1
                logger.warning(f"Failed to get embedding for point {point_id}")
                continue
            
            # Compute sparse vector
            sparse_indices, sparse_values = compute_bm25_sparse_vector(chunk)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": embedding,
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "doc_id": int(doc_id) if not pd.isna(doc_id) else 0,
                    "title": str(title) if not pd.isna(title) else "",
                    "chunk_index": chunk_idx,
                    "text": chunk
                }
            )
            
            points_batch.append(point)
            processed_count += 1
            
            # Upsert batch
            if len(points_batch) >= BATCH_SIZE:
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=points_batch
                    )
                    logger.info(f"Upserted batch of {len(points_batch)} points. Total processed: {processed_count}")
                except Exception as e:
                    logger.error(f"Error upserting batch: {e}")
                    error_count += len(points_batch)
                
                points_batch = []
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
            
            # Log progress every 500 chunks
            if processed_count % 500 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_chunks - global_chunk_idx) / rate if rate > 0 else 0
                logger.info(f"Progress: {processed_count}/{total_chunks} chunks, "
                           f"Rate: {rate:.2f} chunks/s, "
                           f"ETA: {remaining/60:.1f} minutes")
    
    progress_bar.close()
    
    # Upsert remaining points
    if points_batch:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points_batch
            )
            logger.info(f"Upserted final batch of {len(points_batch)} points")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")
            error_count += len(points_batch)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Time elapsed: {elapsed/60:.2f} minutes")
    logger.info("=" * 60)

def main():
    logger.info("=" * 60)
    logger.info("STARTING VECTOR DATABASE CREATION")
    logger.info("=" * 60)
    
    # 1. Load credentials
    logger.info("Loading API credentials...")
    credentials = load_embedding_credentials()
    logger.info("✓ Loaded credentials")
    
    # 2. Connect to Qdrant
    logger.info("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 3. Load data
    logger.info(f"Loading data from {CSV_PATH} (first {NUM_ROWS} rows)...")
    df = pd.read_csv(CSV_PATH, nrows=NUM_ROWS)
    logger.info(f"✓ Loaded {len(df)} documents")
    
    # 4. Setup collection (set recreate=True để tạo mới, False để resume)
    setup_qdrant_collection(client, COLLECTION_NAME, recreate=False)
    
    # 5. Process documents
    process_documents(df, client, credentials, COLLECTION_NAME, resume=True)
    
    # 6. Final stats
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Final collection stats:")
    logger.info(f"  Collection: {COLLECTION_NAME}")
    logger.info(f"  Points count: {collection_info.points_count}")
    logger.info(f"  Status: {collection_info.status}")

if __name__ == "__main__":
    main()
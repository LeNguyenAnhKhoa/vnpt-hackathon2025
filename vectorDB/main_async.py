"""
Main script để tạo Vector Database từ Vietnamese Wikipedia dataset (ASYNC VERSION)
- Sử dụng Qdrant cho vector storage
- VNPT AI Embedding API cho dense vectors (async với aiohttp)
- Fastembed BM25 (Qdrant/bm25) cho sparse vectors
- Chunking theo câu với llama-index SentenceSplitter (chunk_size=2048, overlap=128)
- Async processing để tăng tốc ~5-8x so với sync version
"""
import json
import asyncio
import aiohttp
import pandas as pd
import logging
from datetime import datetime
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from qdrant_client import QdrantClient, models, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, Modifier
from llama_index.core.node_parser import SentenceSplitter
from fastembed import SparseTextEmbedding
from concurrent.futures import ThreadPoolExecutor
import functools

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
CSV_PATH = "data/data.csv"
NUM_ROWS = 100000000  # Chỉ lấy 10000 dòng đầu tiên

# Chunking config
MAX_CHUNK_LENGTH = 512
CHUNK_OVERLAP = 32

# Embedding API config
EMBEDDING_API_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
EMBEDDING_DIM = 1024  # Đã xác nhận từ test

# Rate limiting: 500 req/minute = ~8.3 req/second
# Sử dụng semaphore để kiểm soát concurrent requests
REQUESTS_PER_MINUTE = 500
MAX_CONCURRENT_REQUESTS = 8  # Số request đồng thời tối đa
# Tính delay giữa các batch để đảm bảo không vượt quá rate limit
BATCH_DELAY = 1.1  # ~0.96s cho mỗi batch 8 requests

# Batch config  
EMBEDDING_BATCH_SIZE = 8  # Số chunks để embed đồng thời
UPSERT_BATCH_SIZE = 500  # Số points để upsert mỗi batch

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# Point ID offset để tránh conflict khi chạy nhiều file cùng lúc
POINT_ID_OFFSET = 0  # File 1: bắt đầu từ 4,000,000

# ===================== LOAD API KEYS =====================
def load_embedding_credentials(json_path='./api-keys1.json'):
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

def chunk_by_sentences(text: str, max_length: int = 512) -> list:
    """
    Chunking theo câu sử dụng llama-index SentenceSplitter
    với chunk_size=512 và chunk_overlap=32
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

# ===================== ASYNC EMBEDDING FUNCTION =====================
async def get_embedding_async(
    session: aiohttp.ClientSession,
    text: str,
    credentials: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5
) -> list:
    """
    Gọi VNPT Embedding API để lấy vector (async version)
    Sử dụng semaphore để kiểm soát rate limit
    """
    headers = {
        'Authorization': credentials['authorization'],
        'Token-id': credentials['tokenId'],
        'Token-key': credentials['tokenKey'],
        'Content-Type': 'application/json',
    }
    
    # Truncate text nếu quá dài (API limit 8k tokens)
    # Với tiếng Việt, ước tính ~3 ký tự = 1 token, giới hạn an toàn là 512 ký tự
    # để đảm bảo không vượt quá limit của API
    max_chars = 1024
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Validate text không rỗng
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text, skipping embedding")
        return None
    
    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': text.strip(),
        'encoding_format': 'float',
    }
    
    async with semaphore:  # Rate limiting với semaphore
        for attempt in range(max_retries):
            try:
                async with session.post(
                    EMBEDDING_API_URL,
                    headers=headers,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['data'][0]['embedding']
                    elif response.status == 429:  # Rate limited
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Log chi tiết đầy đủ về lỗi
                        text_response = await response.text()
                        logger.error(f"="*80)
                        logger.error(f"API ERROR DETAILS (Attempt {attempt + 1}/{max_retries}):")
                        logger.error(f"  Status Code: {response.status}")
                        logger.error(f"  URL: {EMBEDDING_API_URL}")
                        logger.error(f"  Request Headers:")
                        for key, value in headers.items():
                            if key.lower() in ['authorization', 'token-id', 'token-key']:
                                logger.error(f"    {key}: {value[:20]}...{value[-10:] if len(value) > 30 else value[20:]}")
                            else:
                                logger.error(f"    {key}: {value}")
                        logger.error(f"  Request Body:")
                        logger.error(f"    model: {json_data['model']}")
                        logger.error(f"    input: {json_data['input'][:100]}..." if len(json_data['input']) > 100 else f"    input: {json_data['input']}")
                        logger.error(f"    encoding_format: {json_data['encoding_format']}")
                        logger.error(f"  Response Headers:")
                        for key, value in response.headers.items():
                            logger.error(f"    {key}: {value}")
                        logger.error(f"  Response Body:")
                        logger.error(f"    {text_response}")
                        logger.error(f"="*80)
                        
                        if attempt < max_retries - 1:
                            # Exponential backoff: 1s, 2s, 4s
                            wait_time = 2 ** attempt
                            logger.warning(f"Retrying after {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrying after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                logger.error(f"Request error: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrying after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
    
    return None

async def get_embeddings_batch_async(
    session: aiohttp.ClientSession,
    texts: list,
    credentials: dict,
    semaphore: asyncio.Semaphore
) -> list:
    """
    Lấy embeddings cho một batch texts đồng thời
    """
    tasks = [
        get_embedding_async(session, text, credentials, semaphore)
        for text in texts
    ]
    return await asyncio.gather(*tasks)

# ===================== BM25 SPARSE VECTOR (FASTEMBED) =====================
# Khởi tạo Fastembed BM25 model (sẽ được load trong main_async)
bm25_model = None
bm25_executor = ThreadPoolExecutor(max_workers=4)

def compute_bm25_sparse_vector(text: str) -> tuple:
    """
    Tính sparse vector sử dụng Fastembed BM25 (Qdrant/bm25)
    
    Trả về (indices, values)
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
        else:
            return [], []
    except Exception as e:
        logger.warning(f"Error computing BM25 sparse vector: {e}")
        return [], []

def compute_bm25_batch(texts: list) -> list:
    """
    Tính sparse vectors cho một batch texts
    """
    results = []
    
    try:
        # Fastembed hỗ trợ batch embedding
        embeddings = list(bm25_model.embed(texts))
        for emb in embeddings:
            results.append((emb.indices.tolist(), emb.values.tolist()))
    except Exception as e:
        logger.warning(f"Error in batch BM25: {e}")
        # Fallback to individual processing
        for text in texts:
            results.append(compute_bm25_sparse_vector(text))
    
    return results

async def compute_bm25_batch_async(texts: list, loop=None) -> list:
    """
    Async wrapper cho BM25 batch computation (chạy trong thread pool)
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    
    return await loop.run_in_executor(
        bm25_executor,
        compute_bm25_batch,
        texts
    )

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

# ===================== ASYNC MAIN PROCESSING =====================
async def process_documents_async(
    df: pd.DataFrame,
    sync_client: QdrantClient,
    credentials: dict,
    collection_name: str,
    resume: bool = True
):
    """
    Xử lý tất cả documents: chunking, embedding, và lưu vào Qdrant (async version)
    Tối ưu: BM25 và API embedding CHẠY SONG SONG để tăng tốc
    """
    # Lấy danh sách point ID đã xử lý (nếu resume)
    existing_ids = set()
    if resume:
        existing_ids = get_existing_point_ids(sync_client, collection_name)
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing points, will skip them")
    
    # Đếm số chunk cần xử lý và chuẩn bị data
    logger.info("Calculating total chunks and preparing data...")
    all_chunks_data = []  # List of (point_id, doc_id, title, chunk_idx, chunk_text)
    
    # Sử dụng POINT_ID_OFFSET làm điểm bắt đầu
    global_chunk_idx = POINT_ID_OFFSET
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing documents"):
        text = row['text']
        if pd.isna(text):
            continue
        chunks = chunk_by_sentences(str(text), max_length=MAX_CHUNK_LENGTH)
        
        for chunk_idx, chunk in enumerate(chunks):
            point_id = global_chunk_idx
            global_chunk_idx += 1
            
            # Skip nếu đã xử lý
            if point_id in existing_ids:
                continue
            
            all_chunks_data.append({
                'point_id': point_id,
                'doc_id': row['id'],
                'title': row['title'],
                'chunk_idx': chunk_idx,
                'text': chunk
            })
    
    total_to_process = len(all_chunks_data)
    skipped_count = (global_chunk_idx - POINT_ID_OFFSET) - total_to_process
    logger.info(f"Total chunks to process: {total_to_process}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    if all_chunks_data:
        logger.info(f"First point_id: {all_chunks_data[0]['point_id']}")
    
    if total_to_process == 0:
        logger.info("No chunks to process!")
        return
    
    # Setup async resources
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS + 2)
    
    processed_count = 0
    error_count = 0
    points_batch = []
    
    start_time = datetime.now()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches of EMBEDDING_BATCH_SIZE
        with tqdm(total=total_to_process, desc="Processing chunks") as pbar:
            for batch_start in range(0, total_to_process, EMBEDDING_BATCH_SIZE):
                batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_to_process)
                batch_data = all_chunks_data[batch_start:batch_end]
                batch_texts = [item['text'] for item in batch_data]
                
                # ✨ OPTIMIZED: Run API embedding and BM25 in PARALLEL
                embeddings, sparse_vectors = await asyncio.gather(
                    get_embeddings_batch_async(session, batch_texts, credentials, semaphore),
                    compute_bm25_batch_async(batch_texts),
                )
                
                # Create points
                for i, (item, embedding, sparse) in enumerate(zip(batch_data, embeddings, sparse_vectors)):
                    if embedding is None:
                        error_count += 1
                        logger.warning(f"Failed to get embedding for point {item['point_id']}")
                        continue
                    
                    sparse_indices, sparse_values = sparse
                    
                    point = PointStruct(
                        id=item['point_id'],
                        vector={
                            "dense": embedding,
                            "sparse": models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        },
                        payload={
                            "doc_id": int(item['doc_id']) if not pd.isna(item['doc_id']) else 0,
                            "title": str(item['title']) if not pd.isna(item['title']) else "",
                            "chunk_index": item['chunk_idx'],
                            "text": item['text']
                        }
                    )
                    
                    points_batch.append(point)
                    processed_count += 1
                
                pbar.update(len(batch_data))
                
                # Upsert batch khi đủ UPSERT_BATCH_SIZE
                if len(points_batch) >= UPSERT_BATCH_SIZE:
                    try:
                        sync_client.upsert(
                            collection_name=collection_name,
                            points=points_batch
                        )
                        logger.info(f"Upserted batch of {len(points_batch)} points. Total processed: {processed_count}")
                    except Exception as e:
                        logger.error(f"Error upserting batch: {e}")
                        error_count += len(points_batch)
                    
                    points_batch = []
                
                # Rate limiting delay between batches
                await asyncio.sleep(BATCH_DELAY)
                
                # Log progress every ~500 chunks
                if processed_count > 0 and processed_count % 500 < EMBEDDING_BATCH_SIZE:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_to_process - batch_end) / rate if rate > 0 else 0
                    logger.info(f"Progress: {processed_count}/{total_to_process} chunks, "
                               f"Rate: {rate:.2f} chunks/s, "
                               f"ETA: {remaining/60:.1f} minutes")
    
    # Upsert remaining points
    if points_batch:
        try:
            sync_client.upsert(
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
    logger.info(f"Total chunks: {global_chunk_idx - POINT_ID_OFFSET}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (already exists): {(global_chunk_idx - POINT_ID_OFFSET) - total_to_process}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Time elapsed: {elapsed/60:.2f} minutes")
    logger.info(f"Average rate: {processed_count/elapsed:.2f} chunks/s")
    logger.info("=" * 60)

async def main_async():
    global bm25_model
    
    logger.info("=" * 60)
    logger.info("STARTING VECTOR DATABASE CREATION (ASYNC VERSION)")
    logger.info("=" * 60)
    
    # 1. Load credentials
    logger.info("Loading API credentials...")
    credentials = load_embedding_credentials()
    logger.info("✓ Loaded credentials")
    
    # 2. Load BM25 model (eager loading)
    logger.info(f"Loading BM25 model: {BM25_MODEL_NAME}")
    bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    logger.info("✓ BM25 model loaded successfully")
    
    # 3. Connect to Qdrant (sync client for setup)
    logger.info("Connecting to Qdrant...")
    sync_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 4. Load data
    logger.info(f"Loading data from {CSV_PATH} (first {NUM_ROWS} rows)...")
    df = pd.read_csv(CSV_PATH, nrows=NUM_ROWS)
    logger.info(f"✓ Loaded {len(df)} documents")
    
    # 5. Setup collection
    setup_qdrant_collection(sync_client, COLLECTION_NAME, recreate=False)
    
    # 6. Process documents async
    await process_documents_async(df, sync_client, credentials, COLLECTION_NAME, resume=True)
    
    # 7. Final stats
    collection_info = sync_client.get_collection(COLLECTION_NAME)
    logger.info(f"Final collection stats:")
    logger.info(f"  Collection: {COLLECTION_NAME}")
    logger.info(f"  Points count: {collection_info.points_count}")
    logger.info(f"  Status: {collection_info.status}")

def main():
    """Entry point - chạy async main"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

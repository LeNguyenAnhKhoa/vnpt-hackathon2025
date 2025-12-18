"""
Main script để tạo Vector Database từ Vietnamese Wikipedia dataset (ASYNC VERSION - FULL TEXT)
- Sử dụng Qdrant cho vector storage
- VNPT AI Embedding API cho dense vectors (async với aiohttp)
- Fastembed BM25 (Qdrant/bm25) cho sparse vectors
- KHÔNG Chunking: Sử dụng Full Text của bài viết (Input embedding sẽ bị truncate nếu quá dài, nhưng lưu full text)
- Async processing để tăng tốc
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
from fastembed import SparseTextEmbedding
from concurrent.futures import ThreadPoolExecutor
import functools

# ===================== LOGGING CONFIG =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectordb_creation_fulltext.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== CẤU HÌNH =====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki_fulltext" # Đổi tên collection để tránh lẫn với bản chunk

# Data config
CSV_PATH = "data/.csv"
NUM_ROWS = 100000  # Chỉ lấy 10000 dòng đầu tiên

# Embedding API config
EMBEDDING_API_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
EMBEDDING_DIM = 1024

# Rate limiting
REQUESTS_PER_MINUTE = 500
MAX_CONCURRENT_REQUESTS = 8
BATCH_DELAY = 1.1

# Batch config  
EMBEDDING_BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 300

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# Point ID offset
POINT_ID_OFFSET = 100000000

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
    """
    headers = {
        'Authorization': credentials['authorization'],
        'Token-id': credentials['tokenId'],
        'Token-key': credentials['tokenKey'],
        'Content-Type': 'application/json',
    }
    
    # Truncate text cho việc EMBEDDING để tránh lỗi API (giới hạn token)
    # Lưu ý: Việc này chỉ ảnh hưởng đến vector, không ảnh hưởng đến text lưu trong DB
    max_chars = 8192
    embedding_text = text
    if len(embedding_text) > max_chars:
        embedding_text = embedding_text[:max_chars]
    
    if not embedding_text or not embedding_text.strip():
        logger.warning("Empty or whitespace-only text, skipping embedding")
        return None
    
    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': embedding_text.strip(),
        'encoding_format': 'float',
    }
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    EMBEDDING_API_URL,
                    headers=headers,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['data'][0]['embedding']
                    elif response.status == 429:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        text_response = await response.text()
                        logger.error(f"API ERROR: {response.status} - {text_response}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
    
    return None

async def get_embeddings_batch_async(
    session: aiohttp.ClientSession,
    texts: list,
    credentials: dict,
    semaphore: asyncio.Semaphore
) -> list:
    tasks = [
        get_embedding_async(session, text, credentials, semaphore)
        for text in texts
    ]
    return await asyncio.gather(*tasks)

# ===================== BM25 SPARSE VECTOR (FASTEMBED) =====================
bm25_model = None
bm25_executor = ThreadPoolExecutor(max_workers=4)

def compute_bm25_batch(texts: list) -> list:
    results = []
    try:
        embeddings = list(bm25_model.embed(texts))
        for emb in embeddings:
            results.append((emb.indices.tolist(), emb.values.tolist()))
    except Exception as e:
        logger.warning(f"Error in batch BM25: {e}")
        # Fallback empty
        for _ in texts:
            results.append(([], []))
    return results

async def compute_bm25_batch_async(texts: list, loop=None) -> list:
    if loop is None:
        loop = asyncio.get_event_loop()
    return await loop.run_in_executor(bm25_executor, compute_bm25_batch, texts)

# ===================== QDRANT SETUP =====================
def setup_qdrant_collection(client: QdrantClient, collection_name: str, recreate: bool = False):
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists, resuming...")
            return False
    
    logger.info(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        }
    )
    return True

def get_existing_point_ids(client: QdrantClient, collection_name: str) -> set:
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
    Xử lý documents: KHÔNG CHUNK, lấy full text, embed và lưu vào Qdrant
    """
    existing_ids = set()
    if resume:
        existing_ids = get_existing_point_ids(sync_client, collection_name)
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing points, will skip them")
    
    logger.info("Preparing data list...")
    all_docs_data = [] 
    
    # Sử dụng global counter cho ID
    global_idx = POINT_ID_OFFSET
    
    # Lặp qua từng dòng và lấy FULL TEXT
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing documents"):
        text = row['text']
        if pd.isna(text) or str(text).strip() == "":
            continue
            
        point_id = global_idx
        global_idx += 1
        
        if point_id in existing_ids:
            continue
        
        # Lưu ý: chunk_idx luôn là 0 vì không cắt nhỏ
        all_docs_data.append({
            'point_id': point_id,
            'doc_id': row['id'],
            'title': row['title'],
            'chunk_idx': 0, 
            'text': str(text) # Full text
        })
    
    total_to_process = len(all_docs_data)
    logger.info(f"Total documents to process: {total_to_process}")
    
    if total_to_process == 0:
        logger.info("No documents to process!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS + 2)
    
    processed_count = 0
    error_count = 0
    points_batch = []
    
    start_time = datetime.now()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=total_to_process, desc="Processing full-text docs") as pbar:
            for batch_start in range(0, total_to_process, EMBEDDING_BATCH_SIZE):
                batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_to_process)
                batch_data = all_docs_data[batch_start:batch_end]
                batch_texts = [item['text'] for item in batch_data]
                
                # Chạy song song Embedding và BM25
                embeddings, sparse_vectors = await asyncio.gather(
                    get_embeddings_batch_async(session, batch_texts, credentials, semaphore),
                    compute_bm25_batch_async(batch_texts),
                )
                
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
                            "chunk_index": 0,
                            "text": item['text'] # Full text gốc
                        }
                    )
                    
                    points_batch.append(point)
                    processed_count += 1
                
                pbar.update(len(batch_data))
                
                if len(points_batch) >= UPSERT_BATCH_SIZE:
                    try:
                        sync_client.upsert(
                            collection_name=collection_name,
                            points=points_batch
                        )
                        logger.info(f"Upserted batch of {len(points_batch)} points.")
                    except Exception as e:
                        logger.error(f"Error upserting batch: {e}")
                        error_count += len(points_batch)
                    points_batch = []
                
                await asyncio.sleep(BATCH_DELAY)
    
    if points_batch:
        try:
            sync_client.upsert(
                collection_name=collection_name,
                points=points_batch
            )
            logger.info(f"Upserted final batch of {len(points_batch)} points")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE (FULL TEXT)")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("=" * 60)

async def main_async():
    global bm25_model
    logger.info("=" * 60)
    logger.info("STARTING VECTOR DATABASE CREATION (ASYNC - FULL TEXT)")
    logger.info("=" * 60)
    
    credentials = load_embedding_credentials()
    
    logger.info(f"Loading BM25 model: {BM25_MODEL_NAME}")
    bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    
    logger.info("Connecting to Qdrant...")
    sync_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    logger.info(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, nrows=NUM_ROWS)
    
    setup_qdrant_collection(sync_client, COLLECTION_NAME, recreate=False)
    
    await process_documents_async(df, sync_client, credentials, COLLECTION_NAME, resume=True)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
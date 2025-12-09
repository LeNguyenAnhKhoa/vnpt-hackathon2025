"""
Test script để thử nghiệm embedding API và lưu vào Qdrant với 1 mẫu
"""
import json
import requests
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, Modifier
from llama_index.core.node_parser import SentenceSplitter
from fastembed import SparseTextEmbedding

# ===================== CẤU HÌNH =====================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki_test"

# Embedding API config
EMBEDDING_API_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"

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
        print(f"Lỗi: Không tìm thấy file {json_path}")
        exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file cấu hình: {e}")
        exit(1)

# ===================== CHUNKING CONFIG =====================
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 128

# Fastembed BM25 model
BM25_MODEL_NAME = "Qdrant/bm25"

# ===================== CHUNKING FUNCTIONS =====================
# Khởi tạo SentenceSplitter từ llama-index
sentence_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
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
        print(f"Error chunking text: {e}")
        # Fallback: cắt theo ký tự nếu có lỗi
        if len(text) <= max_length:
            return [text]
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# ===================== EMBEDDING FUNCTION =====================
def get_embedding(text: str, credentials: dict) -> list:
    """
    Gọi VNPT Embedding API để lấy vector
    """
    headers = {
        'Authorization': credentials['authorization'],
        'Token-id': credentials['tokenId'],
        'Token-key': credentials['tokenKey'],
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': text,
        'encoding_format': 'float',
    }
    
    response = requests.post(EMBEDDING_API_URL, headers=headers, json=json_data)
    
    if response.status_code == 200:
        result = response.json()
        return result['data'][0]['embedding']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# ===================== BM25 SPARSE VECTOR (FASTEMBED) =====================
# Khởi tạo Fastembed BM25 model
bm25_model = None

def get_bm25_model():
    """Lazy initialization của BM25 model"""
    global bm25_model
    if bm25_model is None:
        print(f"Loading BM25 model: {BM25_MODEL_NAME}")
        bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
        print("BM25 model loaded successfully")
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
        print(f"Error computing BM25 sparse vector: {e}")
        return [], []

# ===================== MAIN TEST =====================
def main():
    print("=" * 60)
    print("TEST: Vector Database với 1 mẫu")
    print("=" * 60)
    
    # 1. Load credentials
    print("\n[1] Loading API credentials...")
    credentials = load_embedding_credentials()
    print("✓ Loaded credentials successfully")
    
    # 2. Connect to Qdrant
    print("\n[2] Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # 3. Load 1 sample from CSV
    print("\n[3] Loading 1 sample from CSV...")
    df = pd.read_csv('data/vie_wiki_dataset.csv', nrows=1)
    sample_text = df.iloc[0]['text']
    sample_id = df.iloc[0]['id']
    sample_title = df.iloc[0]['title']
    print(f"✓ Loaded sample - ID: {sample_id}, Title: {sample_title}")
    print(f"  Text length: {len(sample_text)} characters")
    print(f"  Text preview: {sample_text[:200]}...")
    
    # 4. Chunk the text
    print("\n[4] Chunking text by sentences (max_length=2048)...")
    chunks = chunk_by_sentences(sample_text, max_length=2048)
    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars - '{chunk[:50]}...'")
    
    # 5. Get embedding for first chunk
    print("\n[5] Getting embedding from VNPT API...")
    test_chunk = chunks[0] if chunks else sample_text[:2048]
    embedding = get_embedding(test_chunk, credentials)
    
    if embedding is None:
        print("✗ Failed to get embedding")
        return
    
    print(f"✓ Got embedding vector with dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    
    # 6. Compute sparse vector (BM25)
    print("\n[6] Computing BM25 sparse vector...")
    sparse_indices, sparse_values = compute_bm25_sparse_vector(test_chunk)
    print(f"✓ Sparse vector has {len(sparse_indices)} non-zero elements")
    
    # 7. Create/recreate collection
    print("\n[7] Creating Qdrant collection...")
    
    # Delete collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create new collection with dense and sparse vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=len(embedding),
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
    print(f"✓ Created collection: {COLLECTION_NAME}")
    
    # 8. Insert point
    print("\n[8] Inserting point into collection...")
    
    point = PointStruct(
        id=0,
        vector={
            "dense": embedding,
            "sparse": models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        },
        payload={
            "doc_id": int(sample_id),
            "title": sample_title,
            "chunk_index": 0,
            "text": test_chunk
        }
    )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print("✓ Inserted point successfully")
    
    # 9. Verify by searching
    print("\n[9] Verifying by searching...")
    
    # Search using the same embedding (qdrant-client v1.16+ uses query_points)
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        using="dense",
        limit=1
    )
    
    if search_results.points:
        result = search_results.points[0]
        print(f"✓ Search successful!")
        print(f"  Score: {result.score}")
        print(f"  Payload: {result.payload}")
    else:
        print("✗ No results found")
    
    # 10. Collection info
    print("\n[10] Collection info:")
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"  Points count: {collection_info.points_count}")
    print(f"  Status: {collection_info.status}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
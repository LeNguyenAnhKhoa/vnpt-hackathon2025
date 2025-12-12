"""
Script để kiểm tra point_id cao nhất trong Qdrant collection
"""
from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print(f"Checking collection: {COLLECTION_NAME}")
print("-" * 50)

# Lấy collection info
try:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    total_points = collection_info.points_count
    print(f"Total points in collection: {total_points}")
    print("-" * 50)
except Exception as e:
    print(f"Error getting collection info: {e}")
    exit(1)

# Scroll qua tất cả points và tìm max point_id
print("\nScanning all point IDs...")
max_point_id = -1
min_point_id = float('inf')
all_point_ids = []

offset = None
scanned = 0

while True:
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        offset=offset,
        with_payload=False,
        with_vectors=False
    )
    
    points, next_offset = result
    
    for point in points:
        point_id = point.id
        all_point_ids.append(point_id)
        if point_id > max_point_id:
            max_point_id = point_id
        if point_id < min_point_id:
            min_point_id = point_id
        scanned += 1
    
    print(f"Scanned: {scanned}/{total_points} points...", end="\r")
    
    if next_offset is None:
        break
    offset = next_offset

print("\n" + "-" * 50)
print(f"MIN point_id: {min_point_id}")
print(f"MAX point_id: {max_point_id}")
print(f"Total scanned: {len(all_point_ids)}")
print("-" * 50)

# Kiểm tra xem có gap không
if len(all_point_ids) > 0:
    all_point_ids.sort()
    gaps = []
    for i in range(len(all_point_ids) - 1):
        if all_point_ids[i+1] - all_point_ids[i] > 1:
            gaps.append((all_point_ids[i], all_point_ids[i+1]))
    
    if gaps:
        print(f"\nFound {len(gaps)} gaps in point IDs:")
        for start, end in gaps[:10]:  # Show first 10 gaps
            print(f"  Gap: {start} -> {end} (missing {end-start-1} IDs)")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps)-10} more gaps")
    else:
        print("\nNo gaps found - IDs are continuous from 0 to max")

# Kiểm tra sample payload
print("\n" + "=" * 50)
print("Sample point details:")
print("=" * 50)
sample_result = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,
    with_vectors=False
)
sample_points, _ = sample_result

for i, point in enumerate(sample_points, 1):
    print(f"\nPoint #{i}:")
    print(f"  point_id: {point.id}")
    if point.payload:
        print(f"  doc_id: {point.payload.get('doc_id', 'N/A')}")
        print(f"  title: {point.payload.get('title', 'N/A')[:50]}...")
        print(f"  chunk_index: {point.payload.get('chunk_index', 'N/A')}")

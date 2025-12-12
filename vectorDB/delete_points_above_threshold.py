"""
Script để xóa các points có point_id > 15691 từ Qdrant collection
"""
from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vnpt_wiki"
THRESHOLD = 15690

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
print(f"Collection: {COLLECTION_NAME}")
print("-" * 50)

# Lấy collection info trước
try:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    total_points_before = collection_info.points_count
    print(f"Total points before deletion: {total_points_before}")
except Exception as e:
    print(f"Error getting collection info: {e}")
    exit(1)

# Lấy tất cả points có point_id > THRESHOLD
print(f"\nScanning points with point_id > {THRESHOLD}...")
points_to_delete = []

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
        if point.id > THRESHOLD:
            points_to_delete.append(point.id)
        scanned += 1
    
    print(f"Scanned: {scanned}/{total_points_before} points...", end="\r")
    
    if next_offset is None:
        break
    offset = next_offset

print("\n" + "-" * 50)
print(f"Found {len(points_to_delete)} points to delete")

if len(points_to_delete) == 0:
    print("No points to delete. Exiting.")
    exit(0)

# Xóa các points theo batch
print(f"\nDeleting {len(points_to_delete)} points...")
batch_size = 1000
deleted_count = 0

for i in range(0, len(points_to_delete), batch_size):
    batch = points_to_delete[i:i+batch_size]
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=batch
        )
        deleted_count += len(batch)
        print(f"Deleted: {deleted_count}/{len(points_to_delete)} points...", end="\r")
    except Exception as e:
        print(f"\nError deleting batch: {e}")
        exit(1)

print("\n" + "-" * 50)
print("✓ Deletion completed!")

# Kiểm tra lại collection info sau khi xóa
try:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    total_points_after = collection_info.points_count
    print(f"Total points after deletion: {total_points_after}")
    print(f"Points deleted: {total_points_before - total_points_after}")
    print("-" * 50)
except Exception as e:
    print(f"Error getting collection info after deletion: {e}")

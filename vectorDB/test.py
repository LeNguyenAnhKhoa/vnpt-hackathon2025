import os
import csv
from pathlib import Path

# Đường dẫn đến thư mục data
data_dir = Path(__file__).parent / "data"
output_file = Path(__file__).parent / "mix.csv"

# Tìm tất cả file .txt
txt_files = sorted(data_dir.glob("*.txt"))

# Tạo danh sách dữ liệu
rows = []
id_counter = 1

for txt_file in txt_files:
    try:
        # Đọc nội dung file
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Thêm vào danh sách
        rows.append({
            'id': id_counter,
            'title': txt_file.name,
            'text': content
        })
        
        print(f"Đã đọc file: {txt_file.name} (ID: {id_counter})")
        id_counter += 1
    
    except Exception as e:
        print(f"Lỗi khi đọc file {txt_file.name}: {e}")

# Ghi vào file CSV
if rows:
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'title', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nĐã tạo file {output_file}")
    print(f"Tổng số file: {len(rows)}")
else:
    print("Không tìm thấy file .txt nào!")

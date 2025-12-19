import json
import csv
import os

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# Thư mục chứa các file JSON kết quả mới (STEM)
JSON_DIR = 'output_stem'
JSON_PATTERN = 'temp_targets_{}.json' # Format tên file
NUM_FILES = 3 # Số lượng file json (từ 0 đến 2)

# File CSV gốc (Base)
BASE_CSV_PATH = 'output/predict_test_base.csv'

# File CSV đầu ra (Kết quả sau khi merge)
OUTPUT_CSV_PATH = 'output/submission_merged.csv'

def load_updates_from_json():
    """Đọc tất cả file JSON và tạo map {qid: predict}"""
    update_map = {}
    print(f"--- Đang tải dữ liệu từ {JSON_DIR} ---")
    
    for i in range(NUM_FILES):
        file_path = os.path.join(JSON_DIR, JSON_PATTERN.format(i))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Duyệt qua từng item trong json
                for item in data:
                    qid = item.get('qid')
                    predict = item.get('predict')
                    
                    if qid and predict:
                        update_map[qid] = predict
            print(f"✓ Đã tải {file_path} ({len(data)} mẫu)")
            
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file: {file_path}")
        except Exception as e:
            print(f"❌ Lỗi khi đọc {file_path}: {e}")
            
    print(f"==> Tổng cộng có {len(update_map)} câu hỏi cần cập nhật.\n")
    return update_map

def merge_and_save(update_map):
    """Đọc CSV gốc, cập nhật dữ liệu và lưu file mới"""
    if not os.path.exists(BASE_CSV_PATH):
        print(f"❌ Lỗi: Không tìm thấy file gốc {BASE_CSV_PATH}")
        return

    print(f"--- Đang xử lý file gốc {BASE_CSV_PATH} ---")
    
    updated_rows = []
    count_updated = 0
    count_total = 0
    
    # Đọc file CSV gốc
    with open(BASE_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames # Lấy tên cột (qid, predict)
        
        for row in reader:
            count_total += 1
            qid = row['qid']
            
            # Kiểm tra xem QID này có trong danh sách update không
            if qid in update_map:
                new_predict = update_map[qid]
                
                # Chỉ đếm là update nếu kết quả khác nhau (tùy chọn)
                if row['predict'] != new_predict:
                    # print(f"Update {qid}: {row['predict']} -> {new_predict}") # Uncomment nếu muốn xem chi tiết
                    pass
                
                # CẬP NHẬT GIÁ TRỊ MỚI
                row['predict'] = new_predict
                count_updated += 1
            
            updated_rows.append(row)

    # Lưu ra file CSV mới
    with open(OUTPUT_CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"✓ Đã quét {count_total} dòng.")
    print(f"✓ Đã cập nhật {count_updated} dòng từ dữ liệu JSON.")
    print(f"🎉 Kết quả đã lưu tại: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    # 1. Lấy dữ liệu update
    updates = load_updates_from_json()
    
    # 2. Thực hiện merge
    if updates:
        merge_and_save(updates)
    else:
        print("Không có dữ liệu nào để cập nhật.")
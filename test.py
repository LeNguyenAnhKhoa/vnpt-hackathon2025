import json
import csv
import os

# Đường dẫn đến thư mục chứa file (bạn có thể sửa lại nếu cần)
folder_path = 'output'

# --- PHẦN 1: GỘP FILE JSON ---
print("Đang gộp file JSON...")
merged_json_data = []

# Duyệt qua các file từ 1 đến 3
for i in range(1, 4):
    file_name = f'predict{i}.json'
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_json_data.extend(data)
                    print(f" -> Đã thêm dữ liệu từ {file_name}")
                else:
                    print(f" -> Cảnh báo: {file_name} không phải là một danh sách (list).")
        except Exception as e:
            print(f" -> Lỗi khi đọc {file_name}: {e}")
    else:
        print(f" -> Không tìm thấy {file_name}")

# Ghi ra file predict.json tổng
output_json_path = os.path.join(folder_path, 'predict.json')
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(merged_json_data, f, ensure_ascii=False, indent=2)
print(f"Hoàn tất! File JSON tổng tại: {output_json_path}\n")


# --- PHẦN 2: GỘP FILE CSV ---
print("Đang gộp file CSV...")
output_csv_path = os.path.join(folder_path, 'submission.csv')
header_saved = False

with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    
    for i in range(1, 4):
        file_name = f'submission{i}.csv'
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    reader = csv.reader(f_in)
                    header = next(reader, None) # Đọc dòng tiêu đề
                    
                    # Nếu là file đầu tiên thì viết tiêu đề vào file tổng
                    if not header_saved and header:
                        writer.writerow(header)
                        header_saved = True
                    
                    # Viết các dòng dữ liệu còn lại
                    for row in reader:
                        writer.writerow(row)
                        
                print(f" -> Đã thêm dữ liệu từ {file_name}")
            except Exception as e:
                print(f" -> Lỗi khi đọc {file_name}: {e}")
        else:
            print(f" -> Không tìm thấy {file_name}")

print(f"Hoàn tất! File CSV tổng tại: {output_csv_path}")
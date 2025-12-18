import os
import json
import pandas as pd

def process_json_to_csv(input_folder, output_file):
    all_data = []
    
    # 1. Kiểm tra và tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Quét tất cả các file .json trong thư mục đầu vào
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except json.JSONDecodeError:
                print(f"Lỗi: Không thể đọc file {file_name}")

    # 3. Chế biến dữ liệu theo định dạng yêu cầu
    processed_list = []
    for index, item in enumerate(all_data, start=1):
        # Kết hợp dieu và content, xóa các ký tự xuống dòng thừa
        full_text = f"{item.get('dieu', '')} {item.get('content', '')}".replace('\n', ' ').strip()
        
        processed_list.append({
            "id": index,
            "title": index,
            "text": full_text
        })

    # 4. Xuất ra file CSV
    df = pd.DataFrame(processed_list)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã xử lý xong {len(processed_list)} dòng. File lưu tại: {output_file}")

# Thực thi
process_json_to_csv('./json_data', './data/phap_luat.csv')
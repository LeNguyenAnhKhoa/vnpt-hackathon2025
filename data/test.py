import pandas as pd
import json
import re

# Hàm xử lý để chuyển chuỗi choices thành mảng
def parse_choices(text):
    if not isinstance(text, str):
        return []
    # Tách dòng
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Dùng regex để bỏ 'A. ', 'B. ' ở đầu dòng
        # ^[A-Z]\. tìm 1 chữ cái in hoa đứng đầu theo sau là dấu chấm
        cleaned = re.sub(r'^[A-Z]\.\s*', '', line)
        cleaned_lines.append(cleaned)
    return cleaned_lines

# 1. Đọc file val.json để lấy danh sách qid
with open('val.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)
valid_qids = set(item['qid'] for item in val_data)

# 2. Đọc file csv
df = pd.read_csv('error_last.csv')

# 3. Lọc theo qid
filtered_df = df[df['qid'].isin(valid_qids)].copy()

# 4. Áp dụng hàm parse_choices để sửa cột choices từ chuỗi thành list
filtered_df['choices'] = filtered_df['choices'].apply(parse_choices)

# 5. Chọn cột và đổi tên label -> answer
columns_to_keep = ['qid', 'question', 'choices', 'label']
filtered_df = filtered_df[columns_to_keep]
filtered_df = filtered_df.rename(columns={'label': 'answer'})

# 6. Xuất ra file JSON
output_filename = 'filtered_error_last_fixed.json'
json_output = filtered_df.to_dict(orient='records')

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=4)

print(f"Đã lưu file: {output_filename}")
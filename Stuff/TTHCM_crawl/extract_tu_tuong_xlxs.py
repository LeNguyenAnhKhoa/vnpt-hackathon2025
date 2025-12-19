import pandas as pd
import json
import re
from pathlib import Path

# Tập hợp đáp án hợp lệ
ANSWER_SET = {"a", "b", "c", "d"}

def normalize_text(text: str) -> str:
    """Gộp nhiều dòng thành 1 dòng, xóa khoảng trắng thừa"""
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def merge_columns(val1, val2):
    """
    Hàm gộp 2 cột (dành cho Table 1 bị lỗi).
    Ưu tiên lấy giá trị không rỗng.
    """
    v1 = normalize_text(val1)
    v2 = normalize_text(val2)
    
    # Nếu v1 có dữ liệu, lấy v1. Nếu không, lấy v2.
    if v1: return v1
    if v2: return v2
    return ""

def split_question_and_choices(text: str):
    """
    Tách câu hỏi và choices từ 1 chuỗi văn bản gộp
    """
    if not text:
        return None, None

    # Tìm tất cả vị trí xuất hiện của "a." hoặc "A." đứng đầu câu hoặc sau khoảng trắng
    # Regex: (?:^|\s) nghĩa là bắt đầu dòng hoặc dấu cách
    # [aA] để bắt cả hoa thường
    matches = list(re.finditer(r'(?:^|\s)[aA]\.\s*', text))
    
    if not matches:
        return None, None

    # Lấy "a." xuất hiện cuối cùng làm điểm cắt (để tránh chữ a. trong nội dung câu hỏi)
    start_idx = matches[-1].start()
    
    # Nếu match bắt đầu bằng khoảng trắng (do \s), ta cần +1 index để cắt đúng chữ a
    if text[start_idx].isspace():
        start_idx += 1

    question = text[:start_idx].strip()
    choice_block = text[start_idx:].strip()

    # Tách các lựa chọn a. b. c. d.
    # Pattern bắt a., b., c., d. (không phân biệt hoa thường)
    parts = re.split(r'(?:^|\s)([a-dA-D])\.\s*', choice_block)
    
    choices = {}
    # re.split sẽ tạo ra mảng: ['', 'a', 'nội dung a', 'b', 'nội dung b'...]
    # Nên ta duyệt từ 1, nhảy cóc 2 bước
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].lower() # Đưa về 'a', 'b'...
        val = parts[i + 1].strip()
        choices[key] = val

    return question, choices

def extract_qa_from_excel(xlsx_path, output_json):
    result = []
    
    # Đọc toàn bộ file excel
    excel = pd.ExcelFile(xlsx_path)
    sheet_names = excel.sheet_names

    print(f"Found {len(sheet_names)} sheets: {sheet_names}")

    for sheet_idx, sheet_name in enumerate(sheet_names):
        # Đọc sheet không có header để truy cập bằng index cột (0, 1, 2...)
        df = excel.parse(sheet_name, header=None)
        
        print(f"Processing sheet: {sheet_name}...")

        for idx, row in df.iterrows():
            raw_text = ""
            ans_raw = ""

            # === XỬ LÝ SỰ KHÁC BIỆT CẤU TRÚC ===
            
            # TRƯỜNG HỢP 1: Sheet đầu tiên (Table 1 bị lỗi cột)
            # Cột B(1) và C(2) là Câu hỏi
            # Cột D(3) và E(4) là Đáp án
            if sheet_idx == 0: 
                if len(row) < 5: continue # Phải có ít nhất đến cột E
                
                # Gộp cột B và C
                raw_text = merge_columns(row[1], row[2])
                
                # Gộp cột D và E
                ans_raw = merge_columns(row[3], row[4])
            
            # TRƯỜNG HỢP 2: Các Sheet còn lại (Cấu trúc chuẩn)
            # Cột B(1) là Câu hỏi, Cột C(2) là Đáp án
            else:
                if len(row) < 3: continue
                
                raw_text = normalize_text(row[1])
                ans_raw = normalize_text(row[2])

            # === BẮT ĐẦU XỬ LÝ LOGIC CHUNG ===
            
            if not raw_text or not ans_raw:
                continue

            # Chuẩn hóa đáp án (chỉ lấy ký tự đầu tiên: 'A.' -> 'a')
            ans_clean = ans_raw[0].lower()
            if ans_clean not in ANSWER_SET:
                continue

            # Tách câu hỏi và lựa chọn
            question, choices = split_question_and_choices(raw_text)
            
            # Kiểm tra hợp lệ
            if not question or not choices:
                continue
            
            # Kiểm tra xem đáp án có nằm trong danh sách lựa chọn không (quan trọng)
            if ans_clean not in choices:
                # Fallback: Đôi khi đáp án trong excel là 'A' nhưng choices tách ra thiếu 'a' do lỗi format
                continue

            result.append({
                "sheet": sheet_name, # (Tùy chọn) Để debug xem lấy từ sheet nào
                "id": len(result) + 1,
                "question": question,
                "choices": choices,
                "answer": ans_clean
            })

    # Ghi ra JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted {len(result)} questions successfully → {output_json}")

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn
    xlsx_path = Path("Stuff/trac_nghiem_tu_tuong.xlsx")
    output_json = Path("Stuff/trac_nghiem_tu_tuong_qa.json")
    
    if xlsx_path.exists():
        extract_qa_from_excel(xlsx_path, output_json)
    else:
        print(f"❌ File not found: {xlsx_path}")
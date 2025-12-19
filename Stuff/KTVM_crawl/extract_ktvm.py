from docx import Document
import re
import json
import os

def normalize_text(text):
    """Xóa khoảng trắng thừa, đưa về 1 dòng chuẩn để Regex chạy mượt"""
    # Thay thế xuống dòng bằng dấu cách, xóa tab, xóa khoảng trắng kép
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_inline_all(docx_path, output_json):
    print(f"--- Đang xử lý: {docx_path} ---")
    
    doc = Document(docx_path)
    
    # BƯỚC 1: GỘP TOÀN BỘ TEXT LẠI
    # Dùng "\n" để nối các đoạn, giúp regex phân biệt được đâu là ranh giới
    full_text = "\n".join([p.text for p in doc.paragraphs])
    
    # BƯỚC 2: TÁCH ĐÔI FILE (ĐỀ BÀI vs LỜI GIẢI)
    # Tìm mốc "Đáp án Câu 1" đầu tiên để cắt
    split_match = re.search(r'(Đáp án\s*[:]?\s*Câu\s*1\s*[:])', full_text, re.IGNORECASE)
    
    if not split_match:
        print("❌ Lỗi: Không tìm thấy mốc phân chia 'Đáp án Câu 1'.")
        # Nếu file không có phần đáp án tách biệt, bạn có thể comment dòng return để chạy mỗi phần câu hỏi
        return

    split_idx = split_match.start()
    text_questions = full_text[:split_idx]  # Khối văn bản chứa toàn bộ câu hỏi
    text_answers = full_text[split_idx:]    # Khối văn bản chứa toàn bộ đáp án

    # ==================================================
    # XỬ LÝ KHỐI CÂU HỎI (CẮT THEO MỐC "CÂU X")
    # ==================================================
    questions_map = {}
    
    # Regex tìm mốc: "Câu 1:", "Câu 1.", "Câu 1 " (đứng đầu dòng hoặc sau khoảng trắng)
    # Group 1: Toàn bộ cụm "Câu 1:"
    # Group 2: Số ID (1)
    q_pattern = r'(?:^|\n|\s)(Câu\s+(\d+)\s*[:.])'
    
    # Tìm tất cả các mốc "Câu X"
    q_matches = list(re.finditer(q_pattern, text_questions, re.IGNORECASE))
    
    print(f"-> Tìm thấy {len(q_matches)} câu hỏi.")

    for i, match in enumerate(q_matches):
        q_id = int(match.group(2))
        
        # Mốc bắt đầu nội dung (Lấy từ vị trí bắt đầu của chữ "Câu X")
        start_idx = match.start()
        
        # Mốc kết thúc (Là vị trí bắt đầu của "Câu X+1" tiếp theo)
        if i < len(q_matches) - 1:
            end_idx = q_matches[i+1].start()
        else:
            end_idx = len(text_questions) # Câu cuối cùng lấy hết phần còn lại
            
        # Cắt chuỗi
        raw_content = text_questions[start_idx:end_idx].strip()
        
        # Làm sạch: Xóa các từ khóa rác nếu dính vào cuối (ví dụ chữ ĐÁP ÁN)
        raw_content = re.sub(r'\n?ĐÁP ÁN.*$', '', raw_content, flags=re.IGNORECASE).strip()
        
        questions_map[q_id] = raw_content

    # ==================================================
    # XỬ LÝ KHỐI ĐÁP ÁN (CẮT THEO MỐC "ĐÁP ÁN CÂU X")
    # ==================================================
    answers_map = {}
    
    # Regex tìm mốc: "Đáp án Câu 1 : a"
    a_pattern = r'(?:^|\n|\s)Đáp án\s*[:]?\s*Câu\s+(\d+)\s*[:]\s*([a-dA-D])'
    
    a_matches = list(re.finditer(a_pattern, text_answers, re.IGNORECASE))
    
    print(f"-> Tìm thấy {len(a_matches)} lời giải.")

    for i, match in enumerate(a_matches):
        q_id = int(match.group(1))
        ans_char = match.group(2).lower()
        
        # Mốc bắt đầu giải thích (ngay sau mốc tìm thấy)
        start_idx = match.end()
        
        # Mốc kết thúc (ngay trước mốc "Đáp án Câu..." tiếp theo)
        if i < len(a_matches) - 1:
            end_idx = a_matches[i+1].start()
        else:
            end_idx = len(text_answers)
            
        explanation = text_answers[start_idx:end_idx].strip()
        
        # Làm sạch text giải thích
        explanation = explanation.replace('⇨', '').strip()
        if explanation.lower().startswith("giải thích"):
             explanation = re.sub(r'^giải thích\s*[:]\s*', '', explanation, flags=re.IGNORECASE).strip()
             
        full_ans = f"Đáp án: {ans_char}\nGiải thích: {explanation}"
        answers_map[q_id] = full_ans

    # ==================================================
    # GHÉP VÀ XUẤT JSON
    # ==================================================
    final_results = []
    all_ids = sorted(questions_map.keys())
    
    for q_id in all_ids:
        final_results.append({
            "id": q_id,
            "question": questions_map.get(q_id, ""),
            "answer_explanation": answers_map.get(q_id, "Không có giải thích")
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Xong! Kiểm tra file: {output_json}")

# --- CHẠY ---
if __name__ == "__main__":
    extract_inline_all("Stuff/KTVM_crawl/ktvm_trac_nghiem.docx", "Stuff/KTVM_crawl/ktvm_qa.json")
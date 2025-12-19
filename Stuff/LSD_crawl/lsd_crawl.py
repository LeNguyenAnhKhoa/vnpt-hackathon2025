from docx import Document
from docx.oxml.ns import qn
import re
import json
import os

def is_list_paragraph(paragraph):
    """
    Kiểm tra xem dòng này có phải là List tự động (Auto Numbering/Bullet) không.
    Kiểm tra thẻ XML: <w:numPr> bên trong <w:pPr>
    """
    try:
        # Truy cập vào XML element cấp thấp của đoạn văn
        p_element = paragraph._element
        if p_element.pPr is not None and p_element.pPr.numPr is not None:
            return True
    except AttributeError:
        return False
    return False

def is_run_red(font):
    """Kiểm tra màu đỏ"""
    if not font.color or not font.color.rgb:
        return False
    return str(font.color.rgb).upper() in ['FF0000', 'C00000', 'FF3333', '990000', 'ED1C24']

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_with_auto_numbering(docx_path, output_json):
    print(f"--- Đang xử lý file (Hỗ trợ Auto Numbering): {docx_path} ---")
    doc = Document(docx_path)
    
    questions = []
    current_q = None
    
    # Mapping số thứ tự list sang a,b,c,d
    list_index_map = ['a', 'b', 'c', 'd', 'e', 'f']
    current_list_index = 0 

    re_q_start = re.compile(r'^(?:Câu|Bài)\s+\d+', re.IGNORECASE)
    
    for para in doc.paragraphs:
        text = clean_text(para.text)
        
        # Kiểm tra màu đỏ
        has_red = False
        for run in para.runs:
            if is_run_red(run.font):
                has_red = True; break

        # 1. PHÁT HIỆN CÂU HỎI
        if re_q_start.match(text):
            if current_q: questions.append(current_q)
            
            current_q = {
                "question": text,
                "choices": {},
                "correct_answer": None
            }
            current_list_index = 0 # Reset bộ đếm cho câu mới
            continue

        # 2. XỬ LÝ NỘI DUNG (CHOICE)
        if current_q:
            if not text: continue # Bỏ dòng rỗng

            # KIỂM TRA: Đây có phải là dòng đánh số tự động không?
            is_auto_list = is_list_paragraph(para)
            
            if is_auto_list:
                # Nếu là list tự động -> Tự gán a, b, c, d theo thứ tự xuất hiện
                if current_list_index < len(list_index_map):
                    char = list_index_map[current_list_index]
                    
                    current_q["choices"][char] = text
                    if has_red: current_q["correct_answer"] = char
                    
                    current_list_index += 1 # Tăng đếm lên
                    print(f" -> Phát hiện Auto-List '{char}': {text[:20]}...")
            else:
                # Nếu KHÔNG phải list tự động -> Check xem có gõ tay 'a.' không
                match_manual = re.match(r'^(?:^|\s)([a-dA-D])[\.\)]\s*(.*)', text)
                
                if match_manual:
                    # Trường hợp gõ tay
                    char = match_manual.group(1).lower()
                    content = match_manual.group(2)
                    current_q["choices"][char] = content
                    if has_red: current_q["correct_answer"] = char
                    current_list_index += 1
                else:
                    # Nếu không phải list, cũng không gõ a,b,c -> Có thể là nội dung nối dài của câu hỏi
                    # Hoặc nối dài của đáp án trước đó.
                    # Ở đây ta ưu tiên nối vào câu hỏi nếu chưa có đáp án nào
                    if not current_q["choices"]:
                        current_q["question"] += " " + text
                    else:
                        # Nếu đã có đáp án rồi mà lòi ra dòng text thừa -> Nối vào đáp án gần nhất
                        last_char = list_index_map[current_list_index-1]
                        if last_char in current_q["choices"]:
                             current_q["choices"][last_char] += " " + text

    if current_q: questions.append(current_q)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Xong! Kiểm tra file: {output_json}")

if __name__ == "__main__":
    extract_with_auto_numbering("Stuff/LSD_crawl/lsd.docx", "Stuff/LSD_crawl/lsd_qa.json")

# if __name__ == "__main__":
#     extract_smart_choices("Stuff/LSD_crawl/lsd.docx", "Stuff/LSD_crawl/lsd_qa.json")
import json
import re
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

# Danh sách nhãn chuẩn để tự điền nếu thiếu
AUTO_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

def has_auto_numbering(paragraph):
    """
    Kiểm tra xem paragraph có thuộc tính danh sách tự động (Auto-list) trong XML không
    """
    if paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None:
        return True
    return False

def get_highlighted_text(paragraph):
    """Lấy nội dung text được tô màu vàng"""
    highlighted = ""
    for run in paragraph.runs:
        if run.font.highlight_color == WD_COLOR_INDEX.YELLOW:
            highlighted += run.text
    return highlighted.strip()

def clean_text(text):
    """Xóa khoảng trắng thừa"""
    return re.sub(r'\s+', ' ', text).strip()

def process_word_final(input_path, output_path):
    doc = Document(input_path)
    questions = []
    
    current_q = None
    label_idx = 0 # Biến đếm để gán nhãn A, B, C, D tự động
    
    for para in doc.paragraphs:
        raw_text = para.text.strip()
        if not raw_text:
            continue

        # --- 1. XÁC ĐỊNH BOLD (CÂU HỎI) ---
        # Logic: Heading hoặc >60% ký tự là đậm
        is_bold = False
        if para.style.name.startswith('Heading'):
            is_bold = True
        else:
            total_char = 0
            bold_char = 0
            for run in para.runs:
                t_len = len(run.text.replace(" ", ""))
                if t_len > 0:
                    total_char += t_len
                    # Check bold từ run hoặc style cha
                    if run.bold or (run.bold is None and para.style.font.bold):
                        bold_char += t_len
            
            if total_char > 0 and (bold_char / total_char) > 0.6:
                is_bold = True

        # --- 2. XỬ LÝ LOGIC ---
        
        # === NẾU LÀ CÂU HỎI ===
        if is_bold:
            # Lưu câu trước đó nếu có
            if current_q:
                # Join danh sách choice thành 1 chuỗi duy nhất ngăn cách bằng xuống dòng
                current_q["choice"] = "\n".join(current_q["_temp_choices"])
                del current_q["_temp_choices"] # Xóa biến tạm
                questions.append(current_q)
                current_q = None
            
            # Khởi tạo câu mới
            # Clean tiêu đề: Xóa "Câu 1:", "1."
            q_content = re.sub(r"^(?:Câu\s*)?\d+[:\.]\s*", "", raw_text, flags=re.IGNORECASE).strip()
            
            # Nếu câu hỏi bị ngắt dòng bold (dòng 2 của câu hỏi), nối vào câu cũ
            # Nhưng ở đây ta giả định bold mới là câu mới, trừ khi chưa có choice nào
            current_q = {
                "question": q_content,
                "choice": "",
                "answer": "",
                "_temp_choices": [] # List tạm để gom đáp án
            }
            label_idx = 0 # Reset về A

        # === NẾU LÀ ĐÁP ÁN (KHÔNG BOLD) ===
        else:
            if current_q is None:
                continue

            # Xử lý NHÃN (LABEL): A, B, C, D
            # Kiểm tra xem text gốc đã có nhãn chưa (Ví dụ: "A. Nội dung")
            has_manual_label = re.match(r"^[A-D][\.:\)]", raw_text)
            
            final_line_text = raw_text
            
            # Nếu không có nhãn gõ tay, ta kiểm tra xem có cần chèn tự động không
            if not has_manual_label:
                # Nếu là auto-numbering của Word HOẶC là dòng đầu tiên sau câu hỏi
                # Ta sẽ chèn nhãn vào đầu chuỗi
                if has_auto_numbering(para) or label_idx < 4:
                    if label_idx < len(AUTO_LABELS):
                        prefix = f"{AUTO_LABELS[label_idx]}. "
                        final_line_text = prefix + raw_text
                        label_idx += 1
            else:
                # Nếu đã có nhãn thủ công (VD: "A."), ta parse để tăng label_idx cho đúng nhịp
                # Để đề phòng dòng tiếp theo mất nhãn
                label_idx += 1

            # Lưu vào list tạm
            current_q["_temp_choices"].append(final_line_text)

            # Xử lý HIGHLIGHT (ANSWER)
            h_text = get_highlighted_text(para)
            if h_text:
                # Nếu highlight trúng vào auto-number (ẩn), h_text sẽ rỗng hoặc chỉ là text nội dung
                # Ta cứ cộng dồn text highlight tìm thấy
                if current_q["answer"]:
                    current_q["answer"] += " " + h_text
                else:
                    current_q["answer"] = h_text

    # Lưu câu cuối cùng
    if current_q:
        current_q["choice"] = "\n".join(current_q["_temp_choices"])
        del current_q["_temp_choices"]
        questions.append(current_q)

    # --- SAVE JSON ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Xong! {len(questions)} câu hỏi đã được xử lý.")

# === RUN ===
input_file = "Stuff/CNXHKH_crawl/cnxhkh.docx"
output_file = "Stuff/CNXHKH_crawl/cnxhkh.json"

try:
    process_word_final(input_file, output_file)
except Exception as e:
    print(f"Lỗi: {e}")
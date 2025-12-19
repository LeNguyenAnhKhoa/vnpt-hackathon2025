import json
import os

# Cấu hình đường dẫn
PREDICT_PATH = 'output_stem/predict_two_stage.json'
OUT_DIR = 'output_stem'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1. Load dữ liệu
    try:
        val_data = load_json('data/val.json')
        predict_data = load_json(PREDICT_PATH)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        return

    # Tạo map để tra cứu nhanh
    predict_map = {item['qid']: item for item in predict_data}
    
    # Chỉ lấy những câu có trong file predict
    question_map = {item['qid']: item['question'] for item in val_data if item['qid'] in predict_map}
    label_map = {item['qid']: item['answer'] for item in val_data if item['qid'] in predict_map}
    choices_map = {item['qid']: item['choices'] for item in val_data if item['qid'] in predict_map}
    
    # Biến thống kê
    correct = 0
    total = 0
    errors = []
    corrects = []
    
    # 2. Vòng lặp so sánh
    for qid, label in label_map.items():
        if qid in predict_map:
            total += 1
            predict_item = predict_map[qid]
            predict = predict_item['predict']
            reason = predict_item.get('reason', '')
            question = question_map.get(qid, '')
            choices = choices_map.get(qid, [])
            
            # Format choices thành chuỗi dễ đọc (A. ... B. ...)
            if isinstance(choices, list):
                choices_str = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            else:
                choices_str = str(choices)
            
            # Trích xuất context từ reference_docs (làm phẳng để dễ nhìn)
            reference_docs = predict_item.get('reference_docs', [])
            contexts = {}
            for i in range(5):
                if i < len(reference_docs):
                    # Lấy text hoặc content tùy vào cấu trúc file predict
                    doc_content = reference_docs[i].get('text') or reference_docs[i].get('content', '')
                    contexts[f'context{i+1}'] = doc_content
                else:
                    contexts[f'context{i+1}'] = ''
            
            # Tạo object kết quả
            row = {
                'qid': qid,
                'question': question,
                'choices': choices_str,
                'label': label,       # Đáp án đúng
                'predict': predict,   # AI dự đoán
                'reason': reason,     # Lý do AI đưa ra
                **contexts            # Giải nén context1, context2... vào đây
            }
            
            # Phân loại Đúng/Sai
            if predict == label:
                correct += 1
                corrects.append(row)
            else:
                errors.append(row)
    
    # 3. In thống kê ra màn hình
    accuracy = correct / total if total > 0 else 0
    print(f"Total processed: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Errors: {len(errors)}")
    
    # 4. Xuất ra file JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Lưu file Error
    error_path = os.path.join(OUT_DIR, 'error.json')
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    # Lưu file Correct
    correct_path = os.path.join(OUT_DIR, 'correct.json')
    with open(correct_path, 'w', encoding='utf-8') as f:
        json.dump(corrects, f, ensure_ascii=False, indent=2)
    
    print(f"\nError details saved to: {error_path}")
    print(f"Correct predictions saved to: {correct_path}")

if __name__ == "__main__":
    main()
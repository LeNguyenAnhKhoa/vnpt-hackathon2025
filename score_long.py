import json
import csv
import os

PREDICT_PATH = 'output_long_q/predict_long_questions.json'
OUT_DIR = 'output_long_q'
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Load data
    val_data = load_json('data/val.json')
    predict_data = load_json(PREDICT_PATH)
    
    # Create mapping from qid to label
    label_map = {item['qid']: item['answer'] for item in val_data if len(item.get('question', '').split()) > 300}
    question_map = {item['qid']: item['question'] for item in val_data if len(item.get('question', '').split()) > 300}
    choices_map = {item['qid']: item['choices'] for item in val_data if len(item.get('question', '').split()) > 300}
    
    # Create mapping from qid to prediction, item['question'] length > 200
    predict_map = {item['qid']: item for item in predict_data if item['qid'] in question_map}
    
    # Calculate accuracy and collect errors and correct predictions
    correct = 0
    total = 0
    errors = []
    corrects = []
    
    for qid, label in label_map.items():
        if qid in predict_map:
            total += 1
            predict = predict_map[qid]['predict']
            reason = predict_map[qid].get('reason', '')
            question = question_map.get(qid, '')
            choices = choices_map.get(qid, [])
            
            # Format choices as:
            # A. choice 1
            # B. choice 2
            # ...
            if isinstance(choices, list):
                choices_str = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            else:
                choices_str = str(choices)
            
            # Extract context from reference_docs
            reference_docs = predict_map[qid].get('reference_docs', [])
            contexts = {}
            for i in range(5):
                if i < len(reference_docs):
                    contexts[f'context{i+1}'] = reference_docs[i].get('text', '')
                else:
                    contexts[f'context{i+1}'] = ''
            
            row = {
                'qid': qid,
                'question': question,
                'choices': choices_str,
                'reason': reason,
                'predict': predict,
                'label': label
            }
            row.update(contexts)
            
            if predict == label:
                correct += 1
                corrects.append(row)
            else:
                errors.append(row)
    
    # Calculate and print accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Errors: {len(errors)}")
    
    # Export errors and correct predictions to CSV
    os.makedirs('output', exist_ok=True)
    fieldnames = ['qid', 'question', 'choices', 'context1', 'context2', 'context3', 'context4', 'context5', 'reason', 'predict', 'label']
    
    with open(f'{OUT_DIR}/error.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    
    with open(f'{OUT_DIR}/correct.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrects)
    
    print(f"\nError details saved to {OUT_DIR}/error.csv")
    print(f"Correct predictions saved to {OUT_DIR}/correct.csv")

if __name__ == "__main__":
    main()

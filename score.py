import json
import csv
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Load data
    val_data = load_json('data/val.json')
    predict_data = load_json('output/predict_vi.json')
    
    # Create mapping from qid to label
    label_map = {item['qid']: item['answer'] for item in val_data}
    question_map = {item['qid']: item['question'] for item in val_data}
    choices_map = {item['qid']: item['choices'] for item in val_data}
    
    # Create mapping from qid to prediction
    predict_map = {item['qid']: item for item in predict_data}
    
    # Calculate accuracy and collect errors
    correct = 0
    total = 0
    errors = []
    
    for qid, label in label_map.items():
        if qid in predict_map:
            total += 1
            predict = predict_map[qid]['predict']
            reason = predict_map[qid].get('reason', '')
            question = question_map.get(qid, '')
            choices = choices_map.get(qid, [])
            
            if predict == label:
                correct += 1
            else:
                # Format choices as:
                # A. choice 1
                # B. choice 2
                # ...
                if isinstance(choices, list):
                    choices_str = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                else:
                    choices_str = str(choices)
                errors.append({
                    'qid': qid,
                    'question': question,
                    'choices': choices_str,
                    'reason': reason,
                    'predict': predict,
                    'label': label
                })
    
    # Calculate and print accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Errors: {len(errors)}")
    
    # Export errors to CSV
    os.makedirs('output', exist_ok=True)
    with open('output/error_vi.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'question', 'choices', 'reason', 'predict', 'label'])
        writer.writeheader()
        writer.writerows(errors)
    
    print(f"\nError details saved to output/error_vi.csv")

if __name__ == "__main__":
    main()

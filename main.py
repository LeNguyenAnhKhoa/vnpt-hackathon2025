import json
import csv
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_labels(file_path):
    label_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row['qid']] = row['answer']
    return label_map

def main():
    # Load data
    label_data = load_csv_labels('output/submission_deep.csv')
    predict_data = load_json('output/predict.json')
    test_data = load_json('data/test.json')
    
    # Create mapping from qid to label
    label_map = label_data
    
    # Create mapping from qid to question and choices
    question_map = {item['qid']: item['question'] for item in test_data}
    choices_map = {item['qid']: item['choices'] for item in test_data}
    
    # Create mapping from qid to prediction
    predict_map = {item['qid']: item for item in predict_data}
    
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
    
    with open('output/error_test.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    
    print(f"\nError details saved to output/error_qwen.csv")

if __name__ == "__main__":
    main()

import argparse
import json
import csv
import requests
from pathlib import Path
import os
import time

# Cấu hình API Endpoint theo tài liệu [cite: 21]
API_URL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large'

def load_credentials(json_path='api-keys.json'):
    """
    Đọc file api-keys.json và lấy credentials cho LLM Large
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        # Tìm cấu hình có llmApiName là 'LLM large' trong danh sách
        config = next((item for item in keys if item.get('llmApiName') == 'LLM large'), None)
        
        if not config:
            raise ValueError("Không tìm thấy cấu hình 'LLM large' trong file api-keys.json")
            
        return config
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {json_path}")
        exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file cấu hình: {e}")
        exit(1)

# Load credentials từ file JSON
api_config = load_credentials()

# Gán biến toàn cục từ config đã load
AUTHORIZATION = api_config['authorization'] # File json đã có sẵn chữ "Bearer "
TOKEN_ID = api_config['tokenId']
TOKEN_KEY = api_config['tokenKey']

# Lưu ý: Theo tài liệu, tên model trong body request phải chính xác là 'vnptai_hackathon_large'
LLM_API_NAME = 'vnptai_hackathon_large' 

def create_prompts(question, choices):
    """
    Create system and user prompts using Chain of Thought with expert prompting
    """
    # Format choices as A. B. C. D. etc.
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """You are a world-class expert in answering Vietnamese multiple-choice questions across diverse domains including science, history, law, economics, literature, geography, and general knowledge. You possess deep understanding of Vietnamese culture, education system, and contextual nuances.

## Task Definition
Your task is to analyze a Vietnamese multiple-choice question, reason through the problem step-by-step, and select the most accurate answer from the given options.

## Instructions
Follow these steps carefully:
1. **Understand the Question**: Read the question thoroughly. If there is a passage or context provided, extract all relevant information.
2. **Analyze Each Option**: Evaluate each answer choice (A, B, C, D, etc.) against the question requirements.
3. **Apply Chain of Thought Reasoning**: 
   - Break down complex problems into smaller logical steps.
   - For factual questions, recall relevant knowledge or extract information from the given passage.
   - For reasoning questions, apply logical deduction.
   - For calculation questions, show your work step by step.
4. **Eliminate Wrong Answers**: Identify and eliminate clearly incorrect options with brief justification.
5. **Select the Best Answer**: Choose the option that best answers the question based on your analysis.

## Output Format
You MUST respond in valid JSON format with exactly two fields:
{
  "reason": "Your step-by-step reasoning process explaining how you arrived at the answer. Be concise but thorough.",
  "answer": "X"
}

Where "X" is the letter of your chosen answer (A, B, C, D, etc.). The answer field must contain ONLY a single uppercase letter."""

    user_prompt = f"""Question:
{question}

Choices:
{formatted_choices}"""

    return system_prompt, user_prompt


def call_llm_api(system_prompt, user_prompt):
    """
    Call VNPT LLM Small API with the given prompts
    Reference: Tài liệu mô tả APIs LLM_Embedding Track 2.pdf
    """
    headers = {
        'Authorization': AUTHORIZATION,  #  Authorization Bearer ${access_token}
        'Token-id': TOKEN_ID,            #  Key dịch vụ được cấp
        'Token-key': TOKEN_KEY,          #  Key dịch vụ được cấp
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'model': LLM_API_NAME, #  Giá trị là: vnptai_hackathon_small
        'messages': [          #  List[Dict] role system/user/assistant
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            },
        ],
        'temperature': 0,      #  Kiểm soát độ ngẫu nhiên
        'response_format': {'type': 'json_object'}, # [cite: 35] Đảm bảo trả về JSON hợp lệ
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=json_data)
        response.raise_for_status()
        result = response.json()
        
        # Extract the content from the response
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        else:
            return None
            
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def process_test_file(input_path, output_dir, start_idx=None, end_idx=None):
    """
    Process the test.json file and generate submission.csv and predict.json
    
    Args:
        input_path: Path to input JSON file
        output_dir: Output directory path
        start_idx: Starting index (0-based, inclusive). None means start from beginning.
        end_idx: Ending index (0-based, exclusive). None means process until end.
    """
    # Read test data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Apply range filtering
    total_samples = len(test_data)
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = total_samples
    
    # Validate indices
    start_idx = max(0, start_idx)
    end_idx = min(total_samples, end_idx)
    
    if start_idx >= end_idx:
        print(f"Error: Invalid range. start_idx ({start_idx}) >= end_idx ({end_idx})")
        return
    
    test_data = test_data[start_idx:end_idx]
    print(f"Processing samples from index {start_idx} to {end_idx - 1} (total: {len(test_data)} samples)")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_results = []
    json_results = []
    
    # Process each question
    for idx, item in enumerate(test_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        print(f"Processing {idx + 1}/{len(test_data)} (ID: {qid}, original index: {start_idx + idx})")
        
        # Create prompts
        system_prompt, user_prompt = create_prompts(question, choices)
        
        # Call LLM API
        llm_response = call_llm_api(system_prompt, user_prompt)
        
        # Validate and extract answer
        valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        reason = ""
        if llm_response and 'answer' in llm_response:
            answer = llm_response['answer'].strip().upper()
            reason = llm_response.get('reason', '')
            # Validate answer is a single valid character
            if len(answer) == 1 and answer in valid_choices:
                pass  # answer is valid
            else:
                answer = 'A'  # Default to 'A' if invalid
        else:
            print(f"  Warning: Failed to get valid response for QID {qid}. Defaulting to A.")
            answer = 'A'  # Default to 'A' if API fails
        
        csv_results.append({
            'qid': qid,
            'answer': answer
        })
        
        json_results.append({
            'qid': qid,
            'predict': answer,
            'reason': reason
        })
        
        # Sleep to respect rate limits
        # Quota: 1000 req/day, 60 req/h 
        # 60 req/h = 1 req/minute. 1.2s sleep might be too fast if hitting hourly limits,
        # but okay for short bursts. Adjust if you get 429 errors.
        time.sleep(91)
    
    # Write to CSV
    csv_path = output_dir / 'submission.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(csv_results)
    
    # Write to JSON
    json_path = output_dir / 'predict.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSubmission file saved to: {csv_path}")
    print(f"Prediction file saved to: {json_path}")
    print(f"Total questions processed: {len(csv_results)}")


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/val.json',
        help='Path to the input test.json file (default: data/val.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Path to the output directory (default: output)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Starting index (0-based, inclusive). Default: 0'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index (0-based, exclusive). Default: process all'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=None,
        help='Number of samples to process from start. Overrides --end if both provided.'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Handle sample range
    start_idx = args.start
    end_idx = args.end
    
    # If --num-samples is provided, calculate end_idx
    if args.num_samples is not None:
        if start_idx is None:
            start_idx = 0
        end_idx = start_idx + args.num_samples
    
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"API Model: {LLM_API_NAME}")
    if start_idx is not None or end_idx is not None:
        print(f"Sample range: {start_idx if start_idx else 0} to {end_idx if end_idx else 'end'}")
    print("-" * 50)
    
    # Process the test file
    process_test_file(input_path, output_dir, start_idx, end_idx)


if __name__ == '__main__':
    main()
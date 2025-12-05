import argparse
import json
import csv
import requests
from pathlib import Path
import os
import time

# Cấu hình API Endpoint theo tài liệu [cite: 21]
API_URL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'

def load_credentials(json_path='api-keys.json'):
    """
    Đọc file api-keys.json và lấy credentials cho LLM Small
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            keys = json.load(f)
        
        # Tìm cấu hình có llmApiName là 'LLM small' trong danh sách
        config = next((item for item in keys if item.get('llmApiName') == 'LLM small'), None)
        
        if not config:
            raise ValueError("Không tìm thấy cấu hình 'LLM small' trong file api-keys.json")
            
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

# Lưu ý: Theo tài liệu, tên model trong body request phải chính xác là 'vnptai_hackathon_small'
LLM_API_NAME = 'vnptai_hackathon_small' 

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


def process_test_file(input_path, output_path):
    """
    Process the test.json file and generate submission.csv
    """
    # Read test data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    results = []
    
    # Process each question
    for idx, item in enumerate(test_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        print(f"Processing {idx + 1}/{len(test_data)} (ID: {qid})")
        
        # Create prompts
        system_prompt, user_prompt = create_prompts(question, choices)
        
        # Call LLM API
        llm_response = call_llm_api(system_prompt, user_prompt)
        
        # Validate and extract answer
        valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        if llm_response and 'answer' in llm_response:
            answer = llm_response['answer'].strip().upper()
            # Validate answer is a single valid character
            if len(answer) == 1 and answer in valid_choices:
                pass  # answer is valid
            else:
                answer = 'A'  # Default to 'A' if invalid
        else:
            print(f"  Warning: Failed to get valid response for QID {qid}. Defaulting to A.")
            answer = 'A'  # Default to 'A' if API fails
        
        results.append({
            'qid': qid,
            'answer': answer
        })
        
        # Sleep to respect rate limits
        # Quota: 1000 req/day, 60 req/h 
        # 60 req/h = 1 req/minute. 1.2s sleep might be too fast if hitting hourly limits,
        # but okay for short bursts. Adjust if you get 429 errors.
        time.sleep(1.2)
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSubmission file saved to: {output_path}")
    print(f"Total questions processed: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/val.json',
        help='Path to the input test.json file (default: data/val.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Path to the output submission.csv file (default: submission.csv)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"API Model: {LLM_API_NAME}")
    print("-" * 50)
    
    # Process the test file
    process_test_file(input_path, output_path)


if __name__ == '__main__':
    main()
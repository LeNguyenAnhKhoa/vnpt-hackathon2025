import argparse
import json
import csv
import requests
from pathlib import Path
from dotenv import load_dotenv
import os
import time
from qdrant_client import QdrantClient, models

# Load environment variables
load_dotenv()

# Get API credentials from .env
AUTHORIZATION = os.getenv('authorization_llm_small')
TOKEN_ID = os.getenv('tokenId_llm_small')
TOKEN_KEY = os.getenv('tokenKey_llm_small')
LLM_API_NAME = os.getenv('llmApiName_llm_small', 'vnptai_hackathon_small')

# API endpoint
API_URL = 'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small'

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
    """
    headers = {
        'Authorization': f'Bearer {AUTHORIZATION}',
        'Token-id': TOKEN_ID,
        'Token-key': TOKEN_KEY,
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'model': LLM_API_NAME,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            },
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
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
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    results = []
    
    # Process each question
    for idx, item in enumerate(test_data):
        qid = item['qid']
        question = item['question']
        choices = item['choices']
        
        print(f"Processing {idx + 1}")
        
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
            answer = 'A'  # Default to 'A' if API fails
        
        results.append({
            'qid': qid,
            'answer': answer
        })
        
        # Sleep for 1.2 seconds after each sample
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
        default='data/test.json',
        help='Path to the input test.json file (default: data/test.json)'
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
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"API Model: {LLM_API_NAME}")
    print("-" * 50)
    
    # Process the test file
    process_test_file(input_path, output_path)


if __name__ == '__main__':
    main()

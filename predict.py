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
    # Format choices as A, B, C, D, etc.
    choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted_choices = '\n'.join([f"{choice_labels[i]}: {choice}" 
                                   for i, choice in enumerate(choices)])
    
    system_prompt = """You are an expert problem solver with exceptional analytical skills. Your task is to carefully analyze multiple-choice questions and provide accurate answers with clear reasoning.

Your expertise includes:
- Critical thinking and logical reasoning
- Subject matter knowledge across various domains
- Ability to eliminate incorrect options systematically
- Clear articulation of thought process

Always approach problems methodically and justify your conclusions."""

    user_prompt = f"""Task Definition:
Analyze the following multiple-choice question and select the single best answer from the given options.

Question:
{question}

Available Options:
{formatted_choices}

Chain of Thought Instructions:
1. Carefully read and understand the question
2. Analyze each option systematically
3. Identify key concepts and requirements
4. Eliminate obviously incorrect options
5. Compare remaining options
6. Select the most appropriate answer based on logical reasoning

Output Format:
Provide your response as a JSON object with exactly two fields:
- "reason": A explanation of your reasoning process (2-3 sentences)
- "answer": The letter of your selected answer (must be one of: {', '.join(choice_labels[:len(choices)])})

Example format:
{{"reason": "Your reasoning here", "answer": "A"}}"""

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

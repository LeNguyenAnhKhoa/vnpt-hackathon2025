import os
import re

# Path to the data folder
data_folder = r"d:\vnpt-hackathon2025\vectorDB\data"

# Get all .txt files in the folder
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

print(f"Found {len(txt_files)} .txt files:")
for file in txt_files:
    print(f"  - {file}")

# Process each file
for file in txt_files:
    file_path = os.path.join(data_folder, file)
    
    print(f"\nProcessing: {file}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove lines containing "--- TRANG X ---"
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^---\s+TRANG\s+\d+\s+---$', line.strip())]
    
    # Join lines and clean up extra whitespace
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove extra newlines (multiple consecutive newlines)
    cleaned_text = re.sub(r'\n\s*\n+', ' ', cleaned_text)
    
    # Clean up extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Trim leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"  ✓ Cleaned and combined into single paragraph")
    print(f"  Length: {len(cleaned_text)} characters")

print("\nDone! All files have been processed.")

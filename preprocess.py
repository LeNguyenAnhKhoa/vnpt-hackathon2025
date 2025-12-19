import pandas as pd

# Giả sử đọc sheet có vấn đề
df = pd.read_excel("Stuff/trac_nghiem_tu_tuong.xlsx", sheet_name="Table 1", header=None)

# Gộp B+C → text, D+E → answer
df['text'] = df[1].combine_first(df[2])  # lấy giá trị không null
df['answer'] = df[3].combine_first(df[4])  # lấy giá trị không null

# Chuẩn hóa answer
df['answer'] = df['answer'].astype(str).str.strip().str.lower()
df['answer'] = df['answer'].str[0]  # chỉ lấy ký tự đầu tiên

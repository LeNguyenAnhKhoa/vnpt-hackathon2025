import requests
from bs4 import BeautifulSoup
import pandas as pd
import bs4
import time
import random

# --- CẤU HÌNH ---
INPUT_FILE = 'danh_sach_luat_full.csv'
OUTPUT_FILE = 'dataset_chi_tiet_dieu_luat.csv'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def is_strict_article_header(tag):
    """
    Kiểm tra thẻ có phải là Tiêu đề Điều luật chuẩn không.
    Điều kiện:
    1. Là thẻ <p>
    2. Chứa thẻ <a> có name bắt đầu bằng 'dieu_'
    3. name chỉ có dạng 'dieu_x' (không có dấu _ nào nữa phía sau)
    4. Text hiển thị phải nằm trong thẻ <b>
    """
    if tag.name != 'p':
        return False
    
    # Kiểm tra thẻ a
    a_tag = tag.find('a', attrs={'name': True})
    if not a_tag:
        return False
    
    name = a_tag['name']
    if not name.startswith('dieu_'):
        return False
    
    # Kiểm tra điều kiện "phía sau không có dấu '_' nào nữa"
    # Ví dụ: 'dieu_1' (ok - 1 dấu _), 'dieu_1_1' (fail - 2 dấu _)
    if name.count('_') > 1:
        return False

    # Kiểm tra điều kiện "luôn có phần text nằm trong thẻ <b>"
    if not tag.find('b'):
        return False

    return True

def is_chapter_header(tag):
    """
    Kiểm tra thẻ có phải là Tiêu đề Chương không (để ngắt nội dung điều trước đó).
    """
    if tag.name != 'p':
        return False
    
    a_tag = tag.find('a', attrs={'name': True})
    if a_tag and a_tag['name'].startswith('chuong_'):
        return True
    
    return False

def get_articles_content(url, law_name, effective_date):
    print(f"--> Đang xử lý: {law_name[:30]}...")
    try:
        time.sleep(random.uniform(0.5, 1.5)) # Delay nhẹ
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Xác định khung chứa (Container)
        # Tìm bất kỳ thẻ a nào có name='dieu_...' để định vị div cha
        anchor = soup.find('a', attrs={'name': lambda x: x and x.startswith('dieu_')})
        if not anchor:
            print("    Không tìm thấy cấu trúc luật.")
            return []
            
        header_p = anchor.find_parent('p')
        if not header_p: return []
        content_container = header_p.parent
        
        # 2. Duyệt thẻ con
        articles_data = []
        current_article_name = None
        current_content_buffer = []
        
        for child in content_container.children:
            # Bỏ qua khoảng trắng
            if isinstance(child, bs4.element.NavigableString) and not child.strip():
                continue
            
            # Chỉ xét thẻ Tag (p, table, div...)
            if isinstance(child, bs4.element.Tag):
                
                # --- TRƯỜNG HỢP 1: GẶP TIÊU ĐỀ CHƯƠNG (NGẮT) ---
                if is_chapter_header(child):
                    # Nếu đang ghi dở điều luật thì lưu lại
                    if current_article_name:
                        full_content = "\n".join(current_content_buffer).strip()
                        articles_data.append({
                            'Tên văn bản': law_name,
                            'Điều': current_article_name,
                            'Nội dung': full_content,
                            'Ngày hiệu lực': effective_date
                        })
                    # QUAN TRỌNG: Reset trạng thái về None
                    # Để các thẻ <p> mô tả chương sau đó không bị gộp vào điều luật cũ
                    current_article_name = None
                    current_content_buffer = []
                    continue

                # --- TRƯỜNG HỢP 2: GẶP TIÊU ĐỀ ĐIỀU LUẬT MỚI ---
                if is_strict_article_header(child):
                    # Lưu điều luật cũ (nếu có)
                    if current_article_name:
                        full_content = "\n".join(current_content_buffer).strip()
                        articles_data.append({
                            'Tên văn bản': law_name,
                            'Điều': current_article_name,
                            'Nội dung': full_content,
                            'Ngày hiệu lực': effective_date
                        })
                    
                    # Bắt đầu điều mới
                    # Lấy text trong thẻ <b> cho chuẩn xác
                    b_tag = child.find('b')
                    current_article_name = b_tag.get_text(strip=True) if b_tag else child.get_text(strip=True)
                    current_content_buffer = [] 
                    continue

                # --- TRƯỜNG HỢP 3: NỘI DUNG ---
                if current_article_name:
                    text = child.get_text(separator=' ', strip=True)
                
                    
                    if text:
                        current_content_buffer.append(text)
        
        # 3. Lưu điều cuối cùng (nếu còn sót lại sau khi hết vòng lặp)
        if current_article_name:
            full_content = "\n".join(current_content_buffer).strip()
            articles_data.append({
                'Tên văn bản': law_name,
                'Điều': current_article_name,
                'Nội dung': full_content,
                'Ngày hiệu lực': effective_date
            })
            
        return articles_data

    except Exception as e:
        print(f"    Lỗi: {e}")
        return []

def main():
    try:
        df_input = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Không tìm thấy file input.")
        return

    all_articles = []
    total = len(df_input)
    
    for index, row in df_input.iterrows():
        print(f"[{index+1}/{total}]", end=" ")
        articles = get_articles_content(row['Link'], row['Tên văn bản'], row['Ngày hiệu lực'])
        
        if articles:
            all_articles.extend(articles)
            print(f"-> {len(articles)} điều.")
        else:
            print("-> 0 điều.")

    if all_articles:
        df_final = pd.DataFrame(all_articles)
        df_final = df_final[['Tên văn bản', 'Điều', 'Nội dung', 'Ngày hiệu lực']]
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nHoàn tất! File lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
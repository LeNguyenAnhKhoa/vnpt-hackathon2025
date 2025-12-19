import requests
from bs4 import BeautifulSoup
import pandas as pd
import bs4
import time
import random

# --- CẤU HÌNH ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

STOP_PHRASE = "Luật này được Quốc hội"

# --- CÁC HÀM KIỂM TRA ĐIỀU KIỆN ---

def is_strict_article_header(tag):
    """
    Kiểm tra thẻ có phải là Tiêu đề Điều luật chuẩn không.
    Điều kiện:
    1. Là thẻ <p>
    2. Chứa thẻ <a> có name bắt đầu bằng 'dieu_'
    3. name chỉ có dạng 'dieu_x' (không có dấu _ thừa phía sau)
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
    
    # Kiểm tra: phía sau dieu_x không được có thêm dấu gạch dưới (để loại dieu_1_1)
    if name.count('_') > 1:
        return False

    # Kiểm tra: bắt buộc phải có thẻ <b> (in đậm tên điều)
    if not tag.find('b'):
        return False

    return True

def is_chapter_header(tag):
    """
    Kiểm tra thẻ có phải là Tiêu đề Chương/Mục không (để ngắt nội dung).
    """
    if tag.name != 'p':
        return False
    
    a_tag = tag.find('a', attrs={'name': True})
    if a_tag and a_tag['name'].startswith('chuong_'):
        return True
    
    return False

# --- HÀM CÀO DỮ LIỆU CHÍNH ---

def get_law_content_fixed(law_metadata):
    url = law_metadata['Link']
    print(f"--> Đang xử lý: {url}")
    
    try:
        # Thêm delay nhỏ
        time.sleep(random.uniform(0.5, 1.0))
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 1. TÌM KHUNG CHỨA (CONTAINER)
    # Tìm mốc neo đầu tiên để xác định div cha
    first_anchor = soup.find('a', attrs={'name': lambda x: x and x.startswith('dieu_')})
    
    if not first_anchor:
        print("Không tìm thấy cấu trúc điều luật.")
        return []
        
    # Logic: a -> p -> div (container)
    content_container = first_anchor.find_parent('p').parent 
    
    if not content_container:
        print("Không xác định được khung chứa nội dung.")
        return []

    articles_data = []
    
    # Biến lưu trạng thái
    current_article_name = None 
    current_content_buffer = [] 

    # 2. DUYỆT QUA CÁC THẺ CON TRONG CONTAINER
    for child in content_container.children:
        
        # Bỏ qua khoảng trắng vô nghĩa
        if isinstance(child, bs4.element.NavigableString) and not child.strip():
            continue
        
        # Lấy text của thẻ hiện tại để kiểm tra footer
        current_text = ""
        if isinstance(child, bs4.element.Tag):
            current_text = child.get_text(separator=' ', strip=True)
        elif isinstance(child, bs4.element.NavigableString):
            current_text = child.strip()

        
            
        # Chỉ xử lý tiếp nếu là thẻ Tag (p, div, table...)
        if isinstance(child, bs4.element.Tag):
            
            # --- TRƯỜNG HỢP 1: GẶP CHƯƠNG/MỤC (NGẮT) ---
            if is_chapter_header(child):
                # Lưu điều luật cũ nếu đang ghi
                if current_article_name:
                    full_content = "\n".join(current_content_buffer).strip()
                    articles_data.append({
                        'Tên văn bản': law_metadata['Tên văn bản'],
                        'Điều': current_article_name,
                        'Nội dung': full_content,
                        'Ngày ban hành': law_metadata['Ngày ban hành'],
                        'Ngày hiệu lực': law_metadata['Ngày hiệu lực']
                    })
                # Reset trạng thái về Null -> Để các đoạn text sau chương không bị gộp bừa bãi
                current_article_name = None
                current_content_buffer = []
                continue

            # --- TRƯỜNG HỢP 2: GẶP ĐIỀU LUẬT MỚI (CHUẨN) ---
            if is_strict_article_header(child):
                # Lưu điều luật cũ
                if current_article_name:
                    full_content = "\n".join(current_content_buffer).strip()
                    articles_data.append({
                        'Tên văn bản': law_metadata['Tên văn bản'],
                        'Điều': current_article_name,
                        'Nội dung': full_content,
                        'Ngày ban hành': law_metadata['Ngày ban hành'],
                        'Ngày hiệu lực': law_metadata['Ngày hiệu lực']
                    })
                
                # Bắt đầu điều mới (Lấy text trong thẻ <b>)
                b_tag = child.find('b')
                # Phòng hờ trường hợp có b nhưng get_text lỗi
                current_article_name = b_tag.get_text(strip=True) if b_tag else child.get_text(strip=True)
                current_content_buffer = [] # Reset bộ đệm
                continue

        # --- TRƯỜNG HỢP 3: NỘI DUNG ---
        # Nếu đang trong trạng thái ghi nhận một điều luật (current_article_name not None)
        if current_article_name:
            if current_text:
                current_content_buffer.append(current_text)

    # 3. LƯU ĐIỀU CUỐI CÙNG (Nếu vòng lặp kết thúc tự nhiên mà chưa gặp Stop Phrase)
    if current_article_name:
        full_content = "\n".join(current_content_buffer).strip()
        articles_data.append({
            'Tên văn bản': law_metadata['Tên văn bản'],
            'Điều': current_article_name,
            'Nội dung': full_content,
            'Ngày ban hành': law_metadata['Ngày ban hành'],
            'Ngày hiệu lực': law_metadata['Ngày hiệu lực']
        })

    return articles_data

# --- TEST ---
sample_input = {
    'Tên văn bản': 'Luật Đất đai 2024 (Ví dụ)',
    'Link': 'https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-quyen-tu-do-hoi-hop-1957-101-SL-L-003-36793.aspx', # Link có cấu trúc phức tạp
    'Ngày ban hành': '18/01/2024',
    'Ngày hiệu lực': '01/01/2025'
}

result = get_law_content_fixed(sample_input)

if result:
    df = pd.DataFrame(result)
    print(f"\nĐã cào được {len(df)} điều.")
    
    df.to_csv("dataset_luat_fixed_final.csv", index=False, encoding='utf-8-sig')
else:
    print("Không có dữ liệu.")
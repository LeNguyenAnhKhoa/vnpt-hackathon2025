import requests
from bs4 import BeautifulSoup
import pandas as pd

# Giả lập trình duyệt
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def scrape_sequential_data(url):
    print(f"Đang truy cập: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 1. Tìm container chính chứa tất cả nội dung
    container = soup.find('div', class_='divModelDetail')
    
    if not container:
        print("Không tìm thấy div class='divModelDetail'. Kiểm tra lại URL hoặc Class.")
        return []

    # 2. Lấy tất cả các thẻ <p> bên trong container này
    # Vì dữ liệu xếp hàng dọc trong các thẻ p
    all_paragraphs = container.find_all('p')
    
    print(f"Tìm thấy tổng cộng {len(all_paragraphs)} thẻ <p>. Bắt đầu phân loại...")

    laws_list = []
    current_law = None # Biến tạm để lưu thông tin luật đang xét

    for p in all_paragraphs:
        # --- KIỂM TRA 1: Có phải là TIÊU ĐỀ LUẬT (chứa thẻ strong > a) không? ---
        strong_tag = p.find('strong')
        if strong_tag and strong_tag.find('a'):
            # Nếu đang có một luật cũ chưa lưu, thì lưu nó vào danh sách trước
            if current_law:
                laws_list.append(current_law)
            
            # BẮT ĐẦU LUẬT MỚI
            a_tag = strong_tag.find('a')
            current_law = {
                'Tên văn bản': a_tag.get_text(strip=True),
                'Link': a_tag.get('href'),
                'Ngày ban hành': '', # Tạo sẵn key để tránh lỗi
                'Ngày hiệu lực': ''
            }
            
            # Xử lý link tương đối
            if current_law['Link'].startswith('/'):
                current_law['Link'] = f"https://thuvienphapluat.vn{current_law['Link']}"
            
            continue # Xong việc với thẻ này, nhảy sang thẻ p tiếp theo

        # --- KIỂM TRA 2: Có phải là THÔNG TIN (Ban hành/Hiệu lực) không? ---
        # Chỉ xử lý nếu chúng ta đang "nắm giữ" một luật (current_law is not None)
        if current_law:
            text = p.get_text(strip=True)
            
            if text.startswith("- Ban hành"):
                # Cắt bỏ chữ "Ban hành" và dấu hai chấm nếu có
                clean_date = text.replace("- Ban hành:", "").replace("- Ban hành", "").strip()
                current_law['Ngày ban hành'] = clean_date

            elif text.startswith("- Hiệu lực"):
                clean_date = text.replace("- Hiệu lực:", "").replace("- Hiệu lực", "").strip()
                current_law['Ngày hiệu lực'] = clean_date

    # --- QUAN TRỌNG: Lưu luật cuối cùng ---
    # Vì vòng lặp kết thúc khi hết thẻ p, luật cuối cùng chưa kịp được append
    if current_law:
        laws_list.append(current_law)

    return laws_list

# --- CHẠY THỬ ---
# Thay URL bằng link thực tế chứa cấu trúc này
target_url = "https://thuvienphapluat.vn/chinh-sach-phap-luat-moi/vn/ho-tro-phap-luat/chinh-sach-moi/84013/danh-muc-luat-bo-luat-hieu-luc-nam-2025-tai-viet-nam"

data = scrape_sequential_data(target_url)

if data:
    df = pd.DataFrame(data)
    
    # Sắp xếp cột
    cols = ['Tên văn bản', 'Ngày ban hành', 'Ngày hiệu lực', 'Link']
    df = df[cols]
    
    print(f"\nĐã cào thành công {len(df)} văn bản.")
    print(df.head(10)) # In 10 dòng đầu
    
    # Xuất file
    df.to_csv("danh_sach_luat_full.csv", index=False, encoding='utf-8-sig')
else:
    print("Không có dữ liệu.")
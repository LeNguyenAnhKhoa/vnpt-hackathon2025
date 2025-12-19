import requests
from bs4 import BeautifulSoup
import json
import re

def clean_text(text):
    """
    Hàm làm sạch văn bản:
    - Xóa khoảng trắng thừa đầu đuôi.
    - Xóa số thứ tự ở đầu (VD: "1. ", "10. ")
    """
    if not text:
        return ""
    
    # Xóa khoảng trắng 2 đầu
    text = text.strip()
    
    # Dùng Regex xóa số thứ tự đầu dòng (VD: "1.", "2)", "01.")
    # ^\d+ : Bắt đầu bằng số
    # [\.\)] : Theo sau là dấu chấm hoặc ngoặc đơn
    # \s* : Khoảng trắng bất kỳ
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    
    return text

def crawl_cadao_tucngu(url, output_file):
    print(f"Đang tải dữ liệu từ: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Báo lỗi nếu link chết
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm container chứa nội dung theo class bạn cung cấp
        # Lưu ý: Class trong HTML có thể có nhiều, ta chỉ cần tìm div có chứa class chính
        content_div = soup.find('div', class_='content-detail')
        
        if not content_div:
            print("❌ Không tìm thấy thẻ div chứa nội dung (class='content-detail').")
            return

        data_list = []
        seen_content = set() # Dùng để lọc trùng lặp
        current_id = 1

        # --- PHẦN 1: Lấy dữ liệu từ thẻ <li> (thường là tục ngữ ngắn) ---
        list_items = content_div.find_all('li')
        for li in list_items:
            text = li.get_text(strip=True)
            cleaned_text = clean_text(text)
            
            if cleaned_text and cleaned_text not in seen_content:
                data_list.append({
                    "id": current_id,
                    "content": cleaned_text,
                    "type": "li_tag" # (Tùy chọn) Để biết nguồn gốc
                })
                seen_content.add(cleaned_text)
                current_id += 1

        # --- PHẦN 2: Lấy dữ liệu từ thẻ <p> (thường là ca dao/thơ) ---
        p_items = content_div.find_all('p')
        for p in p_items:
            # Xử lý thẻ <br> thành xuống dòng \n trước khi lấy text
            for br in p.find_all("br"):
                br.replace_with("\n")
            
            text = p.get_text()
            cleaned_text = clean_text(text)
            
            # Lọc bớt các dòng rác (quá ngắn hoặc là tiêu đề bài viết)
            if cleaned_text and len(cleaned_text) > 5 and cleaned_text not in seen_content:
                # Kiểm tra thêm: Nếu thẻ p chỉ chứa thông tin metadata rác thì bỏ qua
                if "Nguồn:" in cleaned_text or "Sưu tầm" in cleaned_text:
                    continue
                    
                data_list.append({
                    "id": current_id,
                    "content": cleaned_text,
                    "type": "p_tag"
                })
                seen_content.add(cleaned_text)
                current_id += 1

        # Lưu ra file JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Đã cào thành công {len(data_list)} câu.")
        print(f"📁 Kết quả lưu tại: {output_file}")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    TARGET_URL = "http://thcamlinh.edu.vn/tin-tuc-su-kien/tin-cua-truong/500-cau-ca-dao-tuc-ngu-thanh-ngu-viet-nam-hay.html"
    OUTPUT_FILE = "./cadao_tucngu.json"
    
    crawl_cadao_tucngu(TARGET_URL, OUTPUT_FILE)
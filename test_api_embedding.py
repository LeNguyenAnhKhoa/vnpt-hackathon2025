import requests
import json
import time

def check_embedding_api():
    # 1. Cấu hình Endpoint [cite: 158]
    # Lưu ý: URL trong tài liệu mẫu Python bị ngắt dòng, cần viết liền.
    url = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"

    # 2. Cấu hình Headers (Lấy từ api-keys.json và tài liệu [cite: 167])
    headers = {
        "Content-Type": "application/json",
        # Authorization token lấy từ file api-keys.json (LLM embedings)
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6IjY2M2JlZGEwLWU3YzQtNGQ5ZC04NGIyLWU0YTcyZGIxN2FmZSIsInN1YiI6IjA5MjBhNzk4LWQzNzMtMTFmMC1hNzY5LWRmOTYwN2I5YTdjMSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ3V5ZW5uZ29jODE3OUBnbWFpbC5jb20iLCJzY29wZSI6WyJyZWFkIl0sImlzcyI6Imh0dHBzOi8vbG9jYWxob3N0IiwibmFtZSI6Im5ndXllbm5nb2M4MTc5QGdtYWlsLmNvbSIsInV1aWRfYWNjb3VudCI6IjA5MjBhNzk4LWQzNzMtMTFmMC1hNzY5LWRmOTYwN2I5YTdjMSIsImF1dGhvcml0aWVzIjpbIlVTRVIiLCJUUkFDS18yIl0sImp0aSI6Ijk1MTg2MzI3LTI2MTAtNDViNi04YmFmLWNmNDQ3ZjhkYmVkOCIsImNsaWVudF9pZCI6ImFkbWluYXBwIn0.fXXdGz0ivqhSAZ9GsB3K4s-Sbya9lVxFLTEvqm36ScW23VGLdnIgVWYV1RnD8GfHzyPVW43vlvdh80cCUV6B6ap4nb9vJ0_7qMxn_EjO_xMerbM4XyQ5O2fjGutGpdsH2ZIDWnPSoqauxbOK126EaQSJqu0gB3vaGFy1GXOq2DIykoQ6rGx2s7zrAo44vzjpBhIL6SaAY5eNE1xpT6WW1MIF8rtbV4KxAvYV6eHh2ZHEQ_8gTjOsTJahTfn_v54UepgawRfgJfHOvhOPhnbxZ_ApjprN-gwh-yVoMXEJPM2PBjyK58l22KHsXBH_4MhXrCfWdQyTpu5Hnbfgy8Yv0w",
        # Token-id lấy từ file api-keys.json
        "Token-id": "45698c95-436f-70bb-e063-63199f0a6bf2",
        # Token-key lấy từ file api-keys.json
        "Token-key": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJxrcnRj6jVUERnLVMniMnKUc1KslpcjgiodQ4UWpHn2g317YpBnaWhDKE2vX280m4dNqB1X9laiZWHIG3HQrz8CAwEAAQ=="
    }

    # 3. Cấu hình Body Request 
    # Tạo một câu dài 4096 kí tự để test API embedding
    long_input = """Việt Nam là một đất nước giàu truyền thống, nằm ở phía đông của Đông Nam Á, được biết đến với những cảnh quan đẹp, văn hóa phong phú và lịch sử lâu đời. Từ những ngọn núi hùng vĩ ở phía bắc đến những bãi cát trắng tinh khôi ở phía nam, Việt Nam mang trong mình sự đa dạng tự nhiên tuyệt vời. Nhân dân Việt Nam, với tinh thần cần cù và bất khuất, đã xây dựng nên một nền văn minh vàng son kéo dài hàng ngàn năm, để lại nhiều di sản văn hóa vô cùng quý báu cho thế hệ sau.

Công nghệ thông tin và trí tuệ nhân tạo đã trở thành những lĩnh vực quan trọng nhất trong thế kỷ 21. Các ứng dụng của trí tuệ nhân tạo trong xử lý ngôn ngữ tự nhiên, tìm kiếm thông tin, và phân tích dữ liệu lớn đã mở ra những cơ hội mới không tưởng. Embedding (nhúng) là một kỹ thuật cơ bản trong lĩnh vực này, cho phép chúng ta biểu diễn các từ, câu văn, hoặc toàn bộ tài liệu dưới dạng các vector trong không gian đa chiều. Điều này giúp máy tính có khả năng hiểu được ý nghĩa ngữ pháp và ngữ nghĩa của văn bản.

API Embedding của VNPT được thiết kế để cung cấp các dịch vụ embedding chất lượng cao, hỗ trợ xử lý các văn bản tiếng Việt. Với những mô hình được huấn luyện trên các tập dữ liệu lớn, API này có thể tạo ra các vector biểu diễn chính xác và có ý nghĩa. Những ứng dụng tiềm năng bao gồm tìm kiếm ngữ nghĩa, phân loại tài liệu, phát hiện những tài liệu tương tự, và nhiều ứng dụng khác trong lĩnh vực xử lý ngôn ngữ tự nhiên.

Để kiểm tra hiệu năng của API embedding, chúng ta cần thực hiện các bài test với các đoạn văn bản có độ dài khác nhau, từ những câu ngắn đến những đoạn văn dài. Điều này sẽ giúp chúng ta đánh giá được khả năng xử lý của API, thời gian phản hồi, và chất lượng của các vector được tạo ra. Các bài test này là vô cùng quan trọng để đảm bảo rằng API có thể hoạt động hiệu quả trong các ứng dụng thực tế.

Hackathon VNPT năm 2025 là một sự kiện lớn, tập hợp những lập trình viên, nhà khoa học dữ liệu, và những người đam mê công nghệ từ khắp nơi. Mục tiêu của hackathon này là thúc đẩy sự đổi mới và sáng tạo trong lĩnh vực trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên. Thông qua hackathon, các participant sẽ có cơ hội được làm việc với các công nghệ tiên tiến, học hỏi từ những chuyên gia hàng đầu, và tạo ra những giải pháp độc đáo để giải quyết những bài toán thực tế.

Khóa học về xử lý ngôn ngữ tự nhiên (NLP) đã trở nên ngày càng phổ biến, với hàng ngàn người lao động, sinh viên, và những người tò mò tham gia học tập. Các kỹ năng NLP không chỉ cần thiết cho các vị trí công việc chuyên biệt mà còn là những kỹ năng quan trọng cho bất kỳ ai làm việc trong lĩnh vực công nghệ. Từ chatbot thông minh đến dịch máy chất lượng cao, từ phân tích cảm xúc văn bản đến trích xuất thông tin tự động, các ứng dụng của NLP đã len lỏi vào hầu hết các khía cạnh của cuộc sống hiện đại.

Cơ sở dữ liệu vector (vector database) là một công nghệ mới đang nổi lên, cho phép lưu trữ và tìm kiếm hiệu quả các vector embedding. Những cơ sở dữ liệu này được tối ưu hóa để xử lý các truy vấn tương tự, giúp tìm kiếm nhanh chóng những vector gần nhất với một vector truy vấn nhất định. Qdrant, một cơ sở dữ liệu vector mã nguồn mở, đã được nhiều công ty và dự án sử dụng để xây dựng các ứng dụng tìm kiếm và khuyến nghị cao cấp.

Trong bối cảnh của cuộc cạnh tranh toàn cầu về công nghệ, Việt Nam đang tích cực đẩy mạnh sự phát triển của ngành công nghệ cao. Những doanh nghiệp công nghệ Việt Nam như VNPT, FPT, Viettel đang đầu tư lớn vào nghiên cứu và phát triển các giải pháp AI và NLP tiên tiến. Các sáng kiến như hackathon VNPT không chỉ giúp tìm kiếm tài năng mới mà còn thúc đẩy sự hợp tác và chia sẻ kiến thức trong cộng đồng công nghệ.

Việc test và validate các API là một phần không thể thiếu trong quá trình phát triển phần mềm. Các bài test toàn diện giúp đảm bảo rằng các API hoạt động đúng như mong đợi, có hiệu năng tốt, và xử lý các lỗi một cách hợp lý. Đối với các API embedding, việc kiểm tra với các đoạn văn bản có độ dài và nội dung khác nhau là đặc biệt quan trọng, vì chất lượng của embedding có thể phụ thuộc vào độ dài và đặc điểm của văn bản đầu vào."""
    
    payload = {
        "model": "vnptai_hackathon_embedding",
        "input": long_input,
        "encoding_format": "float"
    }

    print(f"--- Đang kiểm tra kết nối tới: {url} ---")
    print(f"Độ dài câu input: {len(long_input)} kí tự")
    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        end_time = time.time()
        duration = end_time - start_time

        # 4. Phân tích kết quả
        print(f"Status Code: {response.status_code}")
        print(f"Thời gian phản hồi: {duration:.2f} giây")
        
        if response.status_code == 200:
            data = response.json()
            # Kiểm tra xem có trường 'data' và 'embedding' vector không [cite: 175]
            if 'data' in data and len(data['data']) > 0 and 'embedding' in data['data'][0]:
                embedding_vector = data['data'][0]['embedding']
                print("\n✅ API HOẠT ĐỘNG BÌNH THƯỜNG")
                print(f"Kích thước vector nhận được: {len(embedding_vector)}")
                print(f"Mẫu vector (5 giá trị đầu): {embedding_vector[:5]}")
            else:
                print("\n⚠️ API trả về 200 nhưng cấu trúc dữ liệu không đúng mong đợi.")
                print("Response:", json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print("\n❌ API ĐANG GẶP LỖI")
            print("Chi tiết lỗi:", response.text)

    except requests.exceptions.RequestException as e:
        print("\n❌ LỖI KẾT NỐI (Network Error)")
        print(e)

if __name__ == "__main__":
    check_embedding_api()
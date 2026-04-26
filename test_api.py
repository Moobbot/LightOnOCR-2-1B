import requests
import os
import json


def test_api():
    url = "http://localhost:7861/extract"
    img_path = (
        r"d:\Work\Clients\A_Giap\extract-pdf\datasets\Quet so bang-1-5\Trang000005.jpg"
    )

    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy file {img_path}")
        return

    print(f"Đang gửi request tới API {url}...")
    print(f"Upload file: {img_path}")

    # Mở file để chuẩn bị upload
    with open(img_path, "rb") as f:
        # Chuẩn bị files và data (form-data)
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        data = {
            "page_num": 1,
            "prompt": "Extract all text and tables from this image.",
            "temperature": 0.2,
            "max_tokens": 8192,
        }

        try:
            # Gửi request POST
            response = requests.post(url, files=files, data=data)

            # Raise exception nếu HTTP status code trả về lỗi (4xx, 5xx)
            response.raise_for_status()

            # Parse kết quả JSON
            result = response.json()

            print("\n" + "=" * 50)
            print("--- TRẠNG THÁI ---")
            print(result.get("status"))

            print("\n" + "=" * 50)
            print("--- KẾT QUẢ RAW TEXT ---")
            print(result.get("raw_text"))

            print("\n" + "=" * 50)
            print(f"File JSON trên server: {result.get('json_path')}")
            print(f"File Excel trên server: {result.get('excel_path')}")
            print("=" * 50 + "\n")

        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi gọi API: {e}")
            if e.response is not None:
                print(f"Chi tiết lỗi từ server: {e.response.text}")


if __name__ == "__main__":
    test_api()

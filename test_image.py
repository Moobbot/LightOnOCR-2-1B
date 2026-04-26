import sys
import os

sys.path.append(r"d:\Work\Clients\A_Giap\extract-pdf\LightOnOCR-2-1B")

from pipeline.lightonocr_common import process_uploaded_document

def main():
    img_path = r"d:\Work\Clients\A_Giap\extract-pdf\datasets\Quet so bang-1-5\Trang000005.jpg"
    print(f"Đang xử lý ảnh: {img_path}...\n(Có thể mất một chút thời gian để tải model và chạy OCR)")
    
    try:
        loaded, bundle = process_uploaded_document(
            file_input=img_path,
            page_num=1,
            prompt="Extract all text and tables from this image.",
            temperature=0.2,
            max_tokens=8192
        )
        
        print("\n" + "="*50)
        print("--- TRẠNG THÁI ---")
        print(bundle.status)
        
        print("\n" + "="*50)
        print("--- KẾT QUẢ RAW TEXT ---")
        print(bundle.raw_text)
        
        print("\n" + "="*50)
        print(f"File JSON được lưu tại: {bundle.json_path}")
        print(f"File Excel được lưu tại: {bundle.excel_path}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")

if __name__ == "__main__":
    main()

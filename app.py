import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import io
from PIL import Image
import os

# Cấu hình trang
st.set_page_config(
    page_title="Nhận Dạng Ký Tự Viết Tay",
    page_icon="🔤",
    layout="wide"
)

# Cache model để tránh load lại mỗi lần
@st.cache_resource
def load_ocr_model():
    """Load mô hình OCR đã được train"""
    try:
        model = load_model('my_modelcnn.h5')
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None

def preprocess_char(img):
    """Tiền xử lý ảnh ký tự để phù hợp với định dạng EMNIST"""
    # Chuyển về grayscale nếu cần
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Đảo ngược nếu cần (EMNIST có chữ trắng trên nền đen)
    if np.mean(img) > 127:
        img = 255 - img
    
    # Áp dụng Otsu's thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tìm center of mass
    m = cv2.moments(img)
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    else:
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
    
    # Lấy bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Cắt theo bounding box với một chút margin
        margin = 4
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(img.shape[1], x + w + margin)
        y_max = min(img.shape[0], y + h + margin)
        img = img[y_min:y_max, x_min:x_max]
    
    # Resize về 20x20 và pad về 28x28 (chuẩn EMNIST)
    if img.size > 0:
        # Resize về 20x20 giữ tỷ lệ khung hình
        h, w = img.shape
        if h > w:
            new_h, new_w = 20, int(w * 20 / h)
        else:
            new_h, new_w = int(h * 20 / w), 20
        
        if new_h > 0 and new_w > 0:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Pad về 28x28
        h, w = img.shape
        top = (28 - h) // 2
        bottom = 28 - h - top
        left = (28 - w) // 2
        right = 28 - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        img = np.zeros((28, 28), dtype=np.uint8)
    
    return img

def segment_characters(img):
    """Phân đoạn ký tự từ ảnh có nhiều ký tự viết tay"""
    # Chuyển về grayscale nếu cần
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Áp dụng thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sắp xếp contours từ trái qua phải
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    char_images = []
    bounding_boxes = []
    
    for contour in contours:
        # Lấy bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Lọc bỏ contours quá nhỏ (nhiễu)
        if w > 5 and h > 5:
            # Trích xuất ký tự
            char_img = thresh[y:y+h, x:x+w]
            # Tiền xử lý ký tự
            processed_char = preprocess_char(char_img)
            char_images.append(processed_char)
            bounding_boxes.append((x, y, w, h))
    
    return char_images, bounding_boxes

def recognize_text(model, char_images):
    """Nhận dạng ký tự và kết hợp thành text"""
    # Map indices thành letters (A-Z)
    letters = [chr(i + ord('A')) for i in range(26)]
    
    recognized_text = ""
    confidence_scores = []
    
    for char_img in char_images:
        # Chuẩn hóa và reshape cho model input
        img = char_img.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Dự đoán
        prediction = model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Chuyển thành ký tự
        if predicted_class < len(letters):
            recognized_text += letters[predicted_class]
            confidence_scores.append(confidence)
    
    return recognized_text, confidence_scores

def main():
    # Tiêu đề
    st.title("🔤 Nhận Dạng Ký Tự Viết Tay")
    st.markdown("Upload ảnh chứa ký tự viết tay để nhận dạng thành text")
    
    # Load model
    model = load_ocr_model()
    
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file 'model.h5'")
        st.stop()
    
    # Sidebar cho cài đặt
    st.sidebar.header("Cài đặt")
    show_processed = st.sidebar.checkbox("Hiển thị ảnh đã xử lý", value=True)
    show_confidence = st.sidebar.checkbox("Hiển thị độ tin cậy", value=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Chọn ảnh chứa ký tự viết tay",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Hỗ trợ các định dạng: PNG, JPG, JPEG, BMP"
    )
    
    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, caption="Ảnh đã upload", use_column_width=True)
        
        # Xử lý nhận dạng
        with st.spinner("Đang xử lý nhận dạng..."):
            try:
                # Phân đoạn ký tự
                char_images, bounding_boxes = segment_characters(img_array)
                
                if len(char_images) == 0:
                    st.warning("Không tìm thấy ký tự nào trong ảnh. Hãy thử với ảnh khác.")
                else:
                    # Nhận dạng text
                    recognized_text, confidence_scores = recognize_text(model, char_images)
                    
                    with col2:
                        st.subheader("Kết quả nhận dạng")
                        
                        # Hiển thị kết quả
                        st.success(f"**Text nhận dạng:** {recognized_text}")
                        
                        if show_confidence and confidence_scores:
                            avg_confidence = np.mean(confidence_scores)
                            st.info(f"**Độ tin cậy trung bình:** {avg_confidence:.2%}")
                    
                    # Hiển thị ký tự đã phân đoạn
                    if show_processed and char_images:
                        st.subheader("Ký tự đã phân đoạn")
                        
                        # Tạo grid hiển thị ký tự
                        cols = st.columns(min(len(char_images), 6))
                        for i, (char_img, confidence) in enumerate(zip(char_images, confidence_scores)):
                            with cols[i % 6]:
                                st.image(
                                    char_img, 
                                    caption=f"{recognized_text[i]} ({confidence:.1%})",
                                    width=80
                                )
                    
                    # Thống kê chi tiết
                    with st.expander("Chi tiết nhận dạng"):
                        st.write(f"**Số ký tự tìm thấy:** {len(char_images)}")
                        st.write(f"**Text nhận dạng:** {recognized_text}")
                        
                        if confidence_scores:
                            for i, (char, conf) in enumerate(zip(recognized_text, confidence_scores)):
                                st.write(f"Ký tự {i+1}: **{char}** - Độ tin cậy: {conf:.2%}")
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}")
    
    # Hướng dẫn sử dụng
    with st.expander("Hướng dẫn sử dụng"):
        st.markdown("""
        ### Cách sử dụng:
        1. **Upload ảnh:** Chọn file ảnh chứa ký tự viết tay (PNG, JPG, JPEG, BMP)
        2. **Chờ xử lý:** Hệ thống sẽ tự động phân đoạn và nhận dạng ký tự
        3. **Xem kết quả:** Text được nhận dạng sẽ hiển thị cùng với độ tin cậy
        
        ### Lưu ý:
        - Ảnh nên có nền sáng, chữ tối để kết quả tốt nhất
        - Ký tự viết rõ ràng, không quá nhỏ
        - Hiện tại chỉ hỗ trợ nhận dạng chữ cái tiếng Anh (A-Z)
        - Model được train trên dataset EMNIST
        """)

if __name__ == "__main__":
    main()
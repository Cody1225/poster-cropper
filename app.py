import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="自動旋轉海報擷取器", layout="wide")
st.title("🎬 全自動旋轉海報偵測與轉正工具")

st.markdown("""
### 運作原理：
1. **邊緣偵測**：找出圖片中所有區塊。
2. **角度計算**：自動計算每個區塊的旋轉角度。
3. **無損轉正**：使用高品質演算法將海報拉直並擷取。
""")

# 側邊欄：過濾參數
st.sidebar.header("🔍 自動偵測微調")
min_area = st.sidebar.slider("最小海報面積 (像素)", 1000, 50000, 10000)
threshold = st.sidebar.slider("邊緣偵測靈敏度", 10, 255, 100)

uploaded_file = st.file_uploader("上傳含旋轉海報的圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取原圖
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    h_orig, w_orig = original_img.shape[:2]

    # 預處理
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, threshold)
    
    # 閉運算連通邊緣
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # 找輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    st.subheader("🤖 自動偵測結果")
    cols = st.columns(3)
    count = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            # --- 關鍵：獲取最小外接矩形 (包含中心、寬高、角度) ---
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # 修正 OpenCV 的角度邏輯
            if w < h:
                angle += 90
                w, h = h, w
            
            # 計算旋轉矩陣並轉正
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated_full = cv2.warpAffine(original_img, M, (w_orig, h_orig), flags=cv2.INTER_LANCZOS4)
            
            # 從轉正後的圖中切出海報
            x1, y1 = int(cx - w//2), int(cy - h//2)
            # 確保裁切座標在圖內
            x1, y1 = max(0, x1), max(0, y1)
            crop = rotated_full[y1:y1+int(h), x1:x1+int(w)]
            
            if crop.size > 0:
                with cols[count % 3]:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, use_container_width=True, caption=f"偵測到海報 {count+1}")
                    
                    # 下載按鈕
                    is_success, buffer = cv2.imencode(".png", crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    st.download_button(
                        label=f"下載特典 {count+1}",
                        data=buffer.tobytes(),
                        file_name=f"auto_poster_{count+1}.png",
                        mime="image/png",
                        key=f"dl_{i}"
                    )
                count += 1

    if count == 0:
        st.warning("沒偵測到明顯的海報區塊，請調整左側的「面積」或「靈敏度」。")

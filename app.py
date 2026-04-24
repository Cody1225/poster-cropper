import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="高品質特典擷取器", layout="wide")
st.title("🎬 高品質特典海報擷取工具 (原畫質版)")

# 側邊欄調整參數
st.sidebar.header("🔧 精準度微調")
threshold1 = st.sidebar.slider("邊緣偵測靈敏度 1", 10, 500, 100)
threshold2 = st.sidebar.slider("邊緣偵測靈敏度 2", 10, 500, 200)
st.sidebar.info("如果海報被切得太碎，請調高數值；如果偵測不到，請調低數值。")

uploaded_file = st.file_uploader("請上傳官方公告圖片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 為了保持原畫質，我們直接讀取原始位元組
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # 使用 cv2.IMREAD_UNCHANGED 保持原始通道與畫質
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # 轉為灰階做偵測，但不影響原始圖片
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用 Canny 偵測邊緣
    edged = cv2.Canny(blurred, threshold1, threshold2)
    
    # 閉運算：將細小的裂縫連起來，增加偵測完整度
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    st.subheader("擷取結果")
    cols = st.columns(3)
    
    count = 0
    # 將輪廓由上而下、由左而右排序，下載時才不會亂掉
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 過濾太小的區塊 (例如 LOGO 或文字)
        # 你可以根據海報大概的長寬比來過濾，例如海報通常是長方形
        aspect_ratio = float(w)/h
        if w > 200 and h > 200: # 提高過濾門檻，確保只抓大圖
            # **關鍵：直接對 original_img 做裁切，保證原畫質**
            roi = original_img[y:y+h, x:x+w]
            
            # 轉換為 RGB 供網頁顯示 (預覽圖)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(roi_rgb)
            
            with cols[count % 3]:
                st.image(img_pil, use_container_width=True, caption=f"尺寸: {w}x{h}")
                
                # 轉回 BGR 存成 PNG 下載，確保顏色不跑掉
                is_success, buffer = cv2.imencode(".png", roi, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) # 0 代表不壓縮
                if is_success:
                    st.download_button(
                        label=f"下載原畫質特典 {count+1}",
                        data=buffer.tobytes(),
                        file_name=f"poster_original_{count+1}.png",
                        mime="image/png",
                        key=f"dl_{i}"
                    )
            count += 1

    if count == 0:
        st.warning("目前設定下偵測不到海報，請試著調整左側靈敏度。")

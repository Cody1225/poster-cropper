import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 網頁標題
st.set_page_config(page_title="影城特典擷取器", layout="wide")
st.title("🎬 影城特典海報自動擷取工具")
st.info("上傳官方公告圖，系統會自動偵測矩形海報並切割。")

# 側邊欄設定
st.sidebar.header("調整參數")
threshold = st.sidebar.slider("邊緣偵測靈敏度 (越低越敏感)", 10, 255, 100)

uploaded_file = st.file_uploader("請上傳圖片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 處理圖片：灰階 -> 模糊 -> 邊緣偵測
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, threshold)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    st.subheader("擷取結果")
    cols = st.columns(3)
    
    count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 這裡設定過濾條件：寬度與高度都要大於 100 像素，且比例要像長方形
        if w > 100 and h > 100:
            roi = image[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(roi_rgb)
            
            with cols[count % 3]:
                st.image(img_pil, use_container_width=True)
                
                # 下載按鈕
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button(
                    label=f"下載特典 {count+1}",
                    data=buf.getvalue(),
                    file_name=f"poster_{count+1}.png",
                    mime="image/png",
                    key=f"btn_{i}"
                )
            count += 1

    if count == 0:
        st.warning("偵測不到明顯區塊，請試著拉低左側的「靈敏度」滑桿。")

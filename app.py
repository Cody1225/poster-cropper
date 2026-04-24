import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="旋轉海報擷取器", layout="wide")
st.title("🎬 旋轉海報精準擷取工具 (原畫質版)")

st.markdown("""
### 操作指南：
1. **上傳圖片**。
2. 在左側調整 **中心座標**、**裁切尺寸** 以及 **旋轉角度**。
3. 預覽圖中的紅色框即為擷取範圍，請調整至完全框住旋轉的海報。
4. 點擊 **執行旋轉擷取** 即可下載原畫質圖檔。
""")

uploaded_file = st.file_uploader("1. 上傳官方圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 讀取原始圖檔 ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    h_orig, w_orig = original_img.shape[:2]

    # --- 側邊欄控制面板 ---
    st.sidebar.header("🔧 調整擷取範圍")
    
    # 座標與尺寸調整
    center_x = st.sidebar.number_input("中心點 X", 0, w_orig, value=w_orig // 2)
    center_y = st.sidebar.number_input("中心點 Y", 0, h_orig, value=h_orig // 2)
    crop_w = st.sidebar.number_input("裁切寬度", 1, w_orig, value=300)
    crop_h = st.sidebar.number_input("裁切高度", 1, h_orig, value=400)
    
    # 旋轉角度調整
    angle = st.sidebar.slider("旋轉角度", -180.0, 180.0, 0.0, step=0.1)
    
    # --- 計算旋轉矩陣與預覽 ---
    # 定義旋轉矩形 (中心, 尺寸, 角度)
    rect = ((center_x, center_y), (crop_w, crop_h), angle)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    # 製作預覽圖
    display_max_width = 1000
    scale = display_max_width / w_orig if w_orig > display_max_width else 1.0
    disp_img = cv2.resize(original_img, (int(w_orig * scale), int(h_orig * scale)))
    
    # 在預覽圖畫出紅框
    preview_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    scaled_box = (box * scale).astype(np.int32)
    cv2.polylines(preview_img, [scaled_box], True, (255, 0, 0), 3)
    
    st.image(preview_img, use_container_width=True, caption="預覽範圍 (紅框)")

    # --- 執行裁切 ---
    if st.sidebar.button("🚀 執行旋轉擷取並轉正"):
        # 1. 取得旋轉矩陣
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # 2. 對全圖進行旋轉 (使用高品質插值)
        rotated_full = cv2.warpAffine(original_img, M, (w_orig, h_orig), flags=cv2.INTER_LANCZOS4)
        
        # 3. 從旋轉後的圖中切出正的區域
        # 計算切片範圍
        x1 = int(center_x - crop_w / 2)
        y1 = int(center_y - crop_h / 2)
        x2 = x1 + int(crop_w)
        y2 = y1 + int(crop_h)
        
        # 確保不超出邊界
        final_crop = rotated_full[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
        
        st.subheader("擷取結果")
        st.image(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))

        # 下載
        is_success, buffer = cv2.imencode(".png", final_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if is_success:
            st.download_button(
                label="📥 下載原畫質轉正海報",
                data=buffer.tobytes(),
                file_name="rotated_poster.png",
                mime="image/png"
            )

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="海報變形校正擷取器", layout="wide")
st.title("🎬 歪斜海報手動校正擷取工具 (原畫質)")
st.markdown("""
### 使用說明：
1. **上傳圖片**。
2. 在下方預覽圖中，找出目標海報的**四個角**的像素座標 (滑鼠指過去通常會顯示，或估算一下)。
3. 在左側輸入這四個角的 `(X, Y)` 座標。
4. 程式會自動將歪斜的海報**拉直**並提供原畫質下載。
""")

uploaded_file = st.file_uploader("1. 上傳官方公告圖片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 1. 高品質讀取 ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    h_orig, w_orig = original_img.shape[:2]
    
    # 網頁顯示用 (避免原圖太大卡頓)
    display_max_width = 1000
    scale_ratio = 1.0
    if w_orig > display_max_width:
        scale_ratio = display_max_width / w_orig
        h_disp = int(h_orig * scale_ratio)
        disp_img = cv2.resize(original_img, (display_max_width, h_disp))
    else:
        disp_img = original_img.copy()

    # --- 2. 側邊欄：手動輸入四點座標 ---
    st.sidebar.header("2. 定位海報四個角")
    st.sidebar.write(f"原圖尺寸: {w_orig}x{h_orig}")
    st.sidebar.info("請輸入原圖的像素座標 (非預覽圖座標)")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        pt1_x = st.number_input("左上 X", 0, w_orig, value=int(w_orig*0.1))
        pt1_y = st.number_input("左上 Y", 0, h_orig, value=int(h_orig*0.1))
        pt4_x = st.number_input("左下 X", 0, w_orig, value=int(w_orig*0.1))
        pt4_y = st.number_input("左下 Y", 0, h_orig, value=int(h_orig*0.9))
    with col2:
        pt2_x = st.number_input("右上 X", 0, w_orig, value=int(w_orig*0.9))
        pt2_y = st.number_input("右上 Y", 0, h_orig, value=int(h_orig*0.1))
        pt3_x = st.number_input("右下 X", 0, w_orig, value=int(w_orig*0.9))
        pt3_y = st.number_input("右下 Y", 0, h_orig, value=int(h_orig*0.9))

    # 在預覽圖上畫出標記點和連線 (顯示用)
    preview_pts = np.array([
        [pt1_x * scale_ratio, pt1_y * scale_ratio],
        [pt2_x * scale_ratio, pt2_y * scale_ratio],
        [pt3_x * scale_ratio, pt3_y * scale_ratio],
        [pt4_x * scale_ratio, pt4_y * scale_ratio]
    ], np.int32)
    
    preview_with_markers = disp_img.copy()
    cv2.polylines(preview_with_markers, [preview_pts], True, (0, 255, 0), 2) # 畫綠色框
    for i, pt in enumerate(preview_pts):
        cv2.circle(preview_with_markers, tuple(pt), 7, (0, 0, 255), -1) # 畫紅色點
        cv2.putText(preview_with_markers, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 顯示預覽圖
    st.subheader("原圖預覽與定位標記")
    roi_rgb = cv2.cvtColor(preview_with_markers, cv2.COLOR_BGR2RGB)
    st.image(roi_rgb, use_container_width=True)

    # --- 3. 核心技術：透視變換 (拉直海報) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("3. 執行校正並擷取"):
        # 輸入的源座標
        src_pts = np.float32([
            [pt1_x, pt1_y],
            [pt2_x, pt2_y],
            [pt3_x, pt3_y],
            [pt4_x, pt4_y]
        ])

        # 計算目標（拉直後）海報的尺寸
        # 簡單估算：取頂邊和底邊寬度的最大值，左邊和右邊高度的最大值
        width_a = np.sqrt(((pt3_x - pt4_x) ** 2) + ((pt3_y - pt4_y) ** 2))
        width_b = np.sqrt(((pt2_x - pt1_x) ** 2) + ((pt2_y - pt1_y) ** 2))
        max_width = int(max(width_a, width_b))

        height_a = np.sqrt(((pt2_x - pt3_x) ** 2) + ((pt2_y - pt3_y) ** 2))
        height_b = np.sqrt(((pt1_x - pt4_x) ** 2) + ((pt1_y - pt4_y) ** 2))
        max_height = int(max(height_a, height_b))

        # 定義目標座標 (拉直後的長方形)
        dst_pts = np.float32([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ])

        # 計算透視變換矩陣
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 應用變換：對原始高品質圖片進行拉直
        # cv2.INTER_LANCZOS4 提供極高品質的插值，確保畫質不降低
        corrected_img = cv2.warpPerspective(original_img, matrix, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

        # 顯示結果
        st.subheader("校正後的特典")
        st.image(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB), caption=f"解析度: {max_width}x{max_height}")

        # 下載按鈕 (無損 PNG)
        is_success, buffer = cv2.imencode(".png", corrected_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if is_success:
            st.download_button(
                label="下載原畫質校正特典",
                data=buffer.tobytes(),
                file_name="poster_corrected.png",
                mime="image/png"
            )
else:
    st.warning("請先上傳圖片。")

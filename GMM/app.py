import streamlit as st
import cv2
import os
from collections import deque
import time
import pandas as pd
import numpy as np

# Core modules của hệ thống
from core_pipeline import CorePipeline      
from centroid_tracker import CentroidTracker 
from yolo_detector import YOLODetector

# UTILS: GÁN BOUNDING BOX CHO TRACKING ID
def match_boxes_to_objects(boxes, objects):
    id_to_box = {}
    for object_id, (cx, cy) in objects.items():
        min_dist = float("inf")
        best_box = None
        for (x, y, w, h) in boxes:
            bx = x + w // 2
            
            # --- ĐÃ TRẢ VỀ TÂM NGƯỜI (GIỮA BỤNG) CHO MƯỢT ---
            by = y + h // 2 
            
            dist = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_box = (x, y, w, h)
        if best_box is not None:
            id_to_box[object_id] = best_box
    return id_to_box

# PAGE CONFIG & CSS
st.set_page_config(page_title="People Counting Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #F4F7F6; color: #1E293B; }
    h1, h2, h3, h4, h5, h6, p, span { color: #0F172A !important; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .stButton>button { background-color: #2563EB; color: #FFFFFF !important; border-radius: 6px; font-weight: 600; border: none; transition: all 0.2s ease-in-out; width: 100%; padding: 0.5rem; }
    .stButton>button:hover { background-color: #1D4ED8; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3); }
    .event-log { background-color: #FFFFFF; color: #334155; padding: 16px; border-radius: 8px; min-height: 380px; font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; border: 1px solid #CBD5E1; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); white-space: pre-line; }
    [data-testid="stImage"] { border-radius: 8px; overflow: hidden; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] { color: #2563EB; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True
)

# HEADER
st.markdown(
    """
    <h1 style="text-align:center;">People Counting Dashboard</h1>
    <p style="text-align:center; color:#9CA3AF;">MATH AI Project</p>
    """, unsafe_allow_html=True
)
st.divider()

# SESSION STATE – LƯU TRẠNG THÁI 
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False      
if "count_in" not in st.session_state:
    st.session_state.count_in = 0
if "count_out" not in st.session_state:
    st.session_state.count_out = 0
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "object_states" not in st.session_state:
    st.session_state.object_states = {}  
if "detector" not in st.session_state:
    st.session_state.detector = None     
if "tracker" not in st.session_state:
    st.session_state.tracker = None      
if "current_algo" not in st.session_state:
    st.session_state.current_algo = None 
if "event_log" not in st.session_state:
    st.session_state.event_log = deque(maxlen=5)
if "fps_history_yolo" not in st.session_state:
    st.session_state.fps_history_yolo = []
if "fps_history_gmm" not in st.session_state:
    st.session_state.fps_history_gmm = []

# SIDEBAR
with st.sidebar:
    st.divider()
    st.subheader("🧠 Algorithm Selection")
    algo_choice = st.radio("Choose Engine:", ["YOLOv11 (AI + ByteTrack)", "GMM (Traditional + Centroid)"])
    
    st.header("⚙ Controls")
    playback_speed = st.slider("Playback Speed (x)", min_value=0.5, max_value=7.0, value=2.0, step=0.25)
    line_y = st.slider("Counting Line (Y)", 200, 450, 250)
    roi_top = st.slider("ROI Top", 0, 480, 0)
    roi_bottom = st.slider("ROI Bottom", 0, 480, 480)
    min_ratio = st.slider("Min Height / Width Ratio (Filter Sitting)", 0.5, 2.5, 1.6, 0.1)
    show_centroids = st.checkbox("Show Centroids", True)

    if st.button("🔄 Reset Counters"):
        st.session_state.count_in = 0
        st.session_state.count_out = 0
        st.session_state.object_states = {}
        st.session_state.event_log.clear()
        st.rerun()

    st.divider()
    st.subheader("🎞 Video Preview")


# MAIN LAYOUT
tab1, tab2 = st.tabs(["📺 Live View", "📊 Summary"])
video_path = None

# TAB 1 – UPLOAD VIDEO + LIVE VIEW
with tab1:
    uploaded = st.file_uploader("📂 Upload video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded:
        os.makedirs("temp", exist_ok=True)
        video_path = f"temp/{uploaded.name}"
        
        if not os.path.exists(video_path) or "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded.name:
            with st.spinner("📥 Đang lưu file video..."):
                uploaded.seek(0)
                with open(video_path, "wb") as f:
                    f.write(uploaded.read())
            st.session_state.last_uploaded = uploaded.name 
            
        with st.sidebar:
            st.video(video_path)

    st.write("---")
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Counting", use_container_width=True):
            if video_path is None:
                st.error("❌ CẢNH BÁO: Bạn chưa Upload video! Vui lòng tải file lên trước.")
            else:
                st.session_state.is_playing = True
                st.session_state.current_frame = 0     
                st.session_state.count_in = 0 
                st.session_state.count_out = 0
                st.session_state.object_states = {}
                st.session_state.event_log.clear()
                st.rerun() 

    with col2:
        btn_label = "⏸ Pause" if st.session_state.is_playing else "⏯ Resume"
        if st.button(btn_label, use_container_width=True):
            if video_path is None:
                st.error("❌ CẢNH BÁO: Chưa có video để Pause/Resume!")
            else:
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()

    st.write("---")

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_in = kpi_col1.empty()
    kpi_out = kpi_col2.empty()
    kpi_status = kpi_col3.empty()

    col_video, col_log = st.columns([3, 1])

    with col_video:
        col_vid1, col_vid2 = st.columns(2)
        stframe_main = col_vid1.empty() 
        stframe_mask = col_vid2.empty() 

    with col_log:
        st.markdown("### 🖥️ System Log")
        log_placeholder = st.empty()

    if not st.session_state.is_playing:
        kpi_in.metric(label="🟢 TOTAL IN", value=st.session_state.count_in)
        kpi_out.metric(label="🔴 TOTAL OUT", value=st.session_state.count_out)
        kpi_status.metric(label="⚙️ STATUS", value="Paused / Ready ⏳")
        
        if st.session_state.last_frame is not None:
            stframe_main.image(st.session_state.last_frame, channels="RGB")
        else:
            stframe_main.info("Trạng thái sẵn sàng. Hãy tải video và bấm Start Counting...")
            
        log_placeholder.markdown(
            '<div class="event-log">System is ready or paused.\nWaiting for video stream...</div>',
            unsafe_allow_html=True
        )

# MAIN PROCESS – VÒNG LẶP XỬ LÝ VIDEO
if st.session_state.is_playing and video_path:
    
    if st.session_state.current_algo != algo_choice:
        with st.spinner("⏳ Đang nạp mô hình AI (Lần đầu sẽ mất 5-10 giây)..."):
            st.session_state.current_algo = algo_choice
            st.session_state.current_frame = 0
            st.session_state.count_in = 0
            st.session_state.count_out = 0
            st.session_state.object_states = {}
            st.session_state.event_log.clear()
            
            if "YOLO" in algo_choice:
                st.session_state.detector = YOLODetector()
                st.session_state.tracker = None
            else:
                from gmm_detector import GMMDetector
                st.session_state.detector = GMMDetector()
                st.session_state.tracker = CentroidTracker(max_distance=100, max_disappeared=30) # Tăng max_distance để bù cho việc GMM có thể bị lệch tâm hơn YOLO

    pipeline = CorePipeline(video_path)
    
    if not pipeline.cap.isOpened():
        st.error("❌ OpenCV không thể đọc được file video này. Bạn hãy thử upload lại file khác nhé!")
        st.session_state.is_playing = False
    else:
        if st.session_state.current_frame > 0:
            pipeline.cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            
        while pipeline.cap.isOpened():
            ret, frame = pipeline.cap.read()
            
            if not ret:
                st.session_state.is_playing = False 
                st.session_state.current_frame = 0  
                pipeline.release()
                st.rerun() 
                break

            base_delay = 0.03
            time.sleep(base_delay / playback_speed)
            
            frame = pipeline.preprocess(frame)
            roi_frame = frame[roi_top:roi_bottom, :]

            # NHẬN DIỆN & ĐO FPS
            start_time = time.time()

            if "YOLO" in algo_choice:
                raw_objects, raw_boxes = st.session_state.detector.detect(roi_frame)
                fg_mask = None 
            else: 
                boxes, fg_mask = st.session_state.detector.detect(roi_frame) 
                
                valid_boxes = []
                for (x, y, w, h) in boxes:
                    # Bỏ qua các vật nhỏ hơn kích thước người  
                    if w < 25 or h < 50: 
                        continue
            
                    aspect_ratio = float(h) / float(w) 
                    if aspect_ratio >= min_ratio:
                        valid_boxes.append((x, y, w, h))
                # --------------------------------------------------------

                # Truyền valid_boxes vào tracker thay vì boxes thô
                raw_objects = st.session_state.tracker.update(valid_boxes)
                raw_boxes = match_boxes_to_objects(valid_boxes, raw_objects)

            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            if "YOLO" in algo_choice:
                st.session_state.fps_history_yolo.append(fps)
            else:
                st.session_state.fps_history_gmm.append(fps)

            # ==========================================
            objects = {}
            object_boxes = {}
            
            for obj_id in list(raw_objects.keys()):
                if obj_id not in raw_boxes:
                    continue
                cx, cy = raw_objects[obj_id]
                x, y, w, h = raw_boxes[obj_id]
                objects[obj_id] = (cx, cy + roi_top)
                object_boxes[obj_id] = (x, y + roi_top, w, h)

            # VẼ KHUNG VÀ LINE BẰNG OPENCV
            cv2.rectangle(frame, (0, roi_top), (640, roi_bottom), (255, 255, 0), 2)
            cv2.line(frame, (0, line_y), (640, line_y), (0, 0, 255), 3)

            for object_id, (cx, cy) in objects.items():
                x, y, w, h = object_boxes[object_id]

                x, y, w, h = int(x), int(y), int(w), int(h)
                cx, cy = int(cx), int(cy)

                # Vẽ Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {object_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Vẽ Centroid
                if show_centroids:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                # KIỂM TRA ĐẾM VƯỢT VẠCH
                if object_id not in st.session_state.object_states:
                    st.session_state.object_states[object_id] = {
                        "prev_cy": cy,      
                        "counted": False
                    }
                    continue

                state = st.session_state.object_states[object_id]
                prev_cy = state["prev_cy"]
                
                if not state["counted"]:
                    # Đi từ TRÊN xuống DƯỚI (Vào)
                    if prev_cy < line_y and cy >= line_y:
                        st.session_state.count_in += 1
                        state["counted"] = True
                        st.session_state.event_log.appendleft(f"ID {object_id} IN")

                    # Đi từ DƯỚI lên TRÊN (Ra)
                    elif prev_cy > line_y and cy <= line_y:
                        st.session_state.count_out += 1
                        state["counted"] = True
                        st.session_state.event_log.appendleft(f"ID {object_id} OUT")

                # Cập nhật vị trí cũ cho vòng lặp frame tiếp theo
                state["prev_cy"] = cy

            # GHI CHỮ LÊN FRAME
            cv2.rectangle(frame, (0, 0), (260, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"IN: {st.session_state.count_in}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {st.session_state.count_out}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Algo: {'YOLO' if 'YOLO' in algo_choice else 'GMM'}", (frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # RENDER LÊN GIAO DIỆN STREAMLIT
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = rgb_frame
            
            stframe_main.image(rgb_frame, channels="RGB")

            if fg_mask is not None:
                stframe_mask.image(fg_mask, channels="GRAY", use_container_width=True)
            else:
                stframe_mask.empty() 

            # RENDER LOG & KPI TỚI CÁC PLACEHOLDER
            log_text = ""
            for msg in st.session_state.event_log:
                log_text += ("🟢 " if "IN" in msg else "🔴 ") + msg + "\n"

            log_placeholder.markdown(
                f'<div class="event-log">{log_text if log_text else "No events yet"}</div>',
                unsafe_allow_html=True
            )

            kpi_in.metric(label="🟢 TOTAL IN", value=st.session_state.count_in)
            kpi_out.metric(label="🔴 TOTAL OUT", value=st.session_state.count_out)
            kpi_status.metric(label="⚙️ STATUS", value="Running 🏃" if st.session_state.is_playing else "Stopped ⏸️")

            st.session_state.current_frame = int(pipeline.cap.get(cv2.CAP_PROP_POS_FRAMES))

        pipeline.release()


# phần này dùng so sánh thôi
with tab2:
    st.divider()
    st.subheader("📊 Phân tích Hiệu suất: Sức mạnh của GMM so với AI (YOLO)")

    def calc_metrics(fps_list):
        if not fps_list:
            return 0, 0, 0, 0
        avg_fps = np.mean(fps_list)
        max_fps = np.max(fps_list)
        latency = 1000 / avg_fps if avg_fps > 0 else 0
        return avg_fps, max_fps, latency, len(fps_list)

    g_avg, g_max, g_lat, g_frames = calc_metrics(st.session_state.fps_history_gmm)
    y_avg, y_max, y_lat, y_frames = calc_metrics(st.session_state.fps_history_yolo)

    st.markdown("<b>⚡ 1. Ưu thế Tốc độ & Tiết kiệm tài nguyên (GMM vs Deep Learning)</b>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    col1.metric("🚀 Tốc độ trung bình (GMM)", f"{g_avg:.1f} fps", help="Rất cao nhờ thuật toán xác suất tối ưu")
    col2.metric("🐌 Tốc độ trung bình (YOLO)", f"{y_avg:.1f} fps", help="Thấp hơn do tốn tài nguyên chạy mạng Nơ-ron")

    if g_avg > 0 and y_avg > 0:
        speedup = g_avg / y_avg
        col3.metric("🔥 GMM Nhanh hơn YOLO", f"{speedup:.1f} lần", delta="Siêu nhẹ cho CPU", delta_color="normal")

    st.markdown("<br><b>⏱️ 2. Thời gian xử lý mỗi khung hình (Độ trễ - Latency)</b>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns([1, 1, 1])

    col4.metric("⚡ Độ trễ GMM", f"{g_lat:.1f} ms", delta="Phản hồi tức thì", delta_color="normal")
    col5.metric("🧠 Độ trễ YOLO", f"{y_lat:.1f} ms", delta="Suy nghĩ chậm hơn", delta_color="inverse")
    col6.metric("📈 Tốc độ đỉnh (GMM Peak)", f"{g_max:.1f} fps", help="GMM có thể đạt tốc độ cực đại rất cao")

    st.caption(f"⏱️ **Dữ liệu Benchmark thu thập từ:** {g_frames} frames (GMM) | {y_frames} frames (YOLO)")

    max_len = max(y_frames, g_frames)
    if max_len > 0:
        gmm_data = st.session_state.fps_history_gmm + [np.nan] * (max_len - g_frames)
        yolo_data = st.session_state.fps_history_yolo + [np.nan] * (max_len - y_frames)
        
        df_fps = pd.DataFrame({
            "GMM (Nhanh & Nhẹ)": gmm_data,
            "YOLO (Nặng nề)": yolo_data
        })
        
        st.markdown("📈 **Biểu đồ so sánh Real-time (Đường càng cao càng tốt)**")
        st.line_chart(df_fps, color=["#00FF00", "#FF4B4B"])
import streamlit as st
import cv2
import numpy as np
import time
from haze_estimation import HazeEstimator
from aqi_classifier import AQIModel

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Air Quality Vision System", layout="wide")

# --- HEADER ---
st.title("🏙️ Real-Time Urban Air Quality Assessment")
st.markdown("Detailed analysis using **Computer Vision** and **Deep Learning**.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
input_source = st.sidebar.radio("Select Input Source:", ("Webcam", "Upload Video"))
confidence_threshold = st.sidebar.slider("Haze Sensitivity", 0.0, 1.0, 0.95)

# --- INITIALIZE MODULES ---
@st.cache_resource
def load_models():
    return HazeEstimator(), AQIModel()

estimator, ai_model = load_models()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed & Vision Analysis")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Metrics")
    aqi_metric = st.empty()
    pm25_metric = st.empty()
    haze_metric = st.empty()
    st.divider()
    st.subheader("Data Trends")
    chart_placeholder = st.empty()

# --- DATA STORAGE FOR GRAPHS ---
if "pm25_history" not in st.session_state:
    st.session_state.pm25_history = []

# --- VIDEO PROCESSING LOOP ---
def process_frame(frame):
    # Resize for performance
    frame = cv2.resize(frame, (640, 480))
    
    # 1. Optical Physics (Dark Channel)
    dark_channel = estimator.get_dark_channel(frame)
    A = estimator.estimate_atmospheric_light(frame, dark_channel)
    transmission = estimator.get_transmission_map(frame, A)
    haze_score = estimator.calculate_haze_score(transmission)
    
    # 2. AI Prediction
    aqi_label, pm25_est = ai_model.predict_aqi(frame, haze_score)
    
    # 3. Update History
    st.session_state.pm25_history.append(pm25_est)
    if len(st.session_state.pm25_history) > 50:
        st.session_state.pm25_history.pop(0)

    # 4. Visualization (Heatmap overlay)
    transmission_display = (transmission * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(transmission_display, cv2.COLORMAP_JET)
    
    # Blend original and heatmap
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    
    return overlay, aqi_label, pm25_est, haze_score

# --- INPUT HANDLING ---
cap = None
if input_source == "Webcam":
    cap = cv2.VideoCapture(0)
    stop_button = st.button("Stop Camera")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break
            
        # Process
        processed_frame, label, pm25, score = process_frame(frame)
        
        # Update UI
        video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        
        # Color Logic
        color = "normal"
        if label == "Unhealthy": color = "off"
        if label == "Hazardous": color = "inverse"

        aqi_metric.metric("AQI Status", label, delta_color=color)
        pm25_metric.metric("Est. PM2.5", f"{pm25:.1f} µg/m³")
        haze_metric.progress(int(score))
        
        # Update Chart
        chart_placeholder.line_chart(st.session_state.pm25_history)
        
        time.sleep(0.1) # Stability pause

    cap.release()

elif input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame, label, pm25, score = process_frame(frame)
            video_placeholder.image(processed_frame, channels="BGR")
            aqi_metric.metric("AQI Status", label)
            pm25_metric.metric("PM2.5", f"{pm25:.1f}")
            chart_placeholder.line_chart(st.session_state.pm25_history)
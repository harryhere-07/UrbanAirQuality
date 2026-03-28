import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import queue
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from haze_estimation import HazeEstimator
from aqi_classifier import AQIModel

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Urban Air Quality Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Custom Title */
    .main-title {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    
    .sub-title {
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 198, 255, 0.15);
        border: 1px solid rgba(0, 198, 255, 0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #A0AEC0 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A202C;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Progress bar custom colors */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
    }
    
    /* Video/Image container styling */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Divider line */
    hr {
        border-color: rgba(255,255,255,0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<h1 class="main-title">🏙️ Urban Air Quality Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced environmental analysis powered by Computer Vision and Deep Learning.</p>', unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR SETUP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3211/3211116.png", width=60)
    st.markdown("### Control Center")
    input_source = st.radio("Select Input Source", ("📹 Live Camera", "📁 Upload Media"), index=1)
    
    st.markdown("---")
    st.markdown("### Fine Tuning")
    confidence_threshold = st.slider("Haze Sensitivity", 0.0, 1.0, 0.95, help="Adjust how aggressively the system detects haze.")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("Modules Initialized")
    st.info("Models Ready")

# --- INITIALIZE MODULES ---
@st.cache_resource(show_spinner="Loading Deep Learning Models...")
def load_models():
    return HazeEstimator(), AQIModel()

estimator, ai_model = load_models()

# --- HELPER FUNCTION: GET COLOR HEX ---
def get_aqi_color(label):
    colors = {
        "Good": "#00E676", # Bright Green
        "Moderate": "#FFEA00", # Yellow
        "Unhealthy for Sensitive Groups": "#FF9100", # Orange
        "Unhealthy": "#FF3D00", # Deep Orange
        "Very Unhealthy": "#D50000", # Red
        "Hazardous": "#880E4F" # Dark Red/Purple
    }
    return colors.get(label, "#FFFFFF")

# --- MAIN LAYOUT ---
col_vision, col_data = st.columns([2.5, 1.5], gap="large")

with col_vision:
    st.markdown("### 👁️ Real-Time Vision Feed")
    video_placeholder = st.empty()
    status_text = st.empty()

with col_data:
    st.markdown("### 📊 Live Analytics")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        aqi_metric = st.empty()
    with metric_col2:
        pm25_metric = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Haze Density Index")
    haze_metric = st.empty()
    
    st.markdown("---")
    st.markdown("### 📈 PM2.5 Trends")
    chart_placeholder = st.empty()

# --- DATA STORAGE FOR GRAPHS ---
if "pm25_history" not in st.session_state:
    st.session_state.pm25_history = []

# --- VIDEO PROCESSING LOOP ---
def process_frame(frame, update_history=True):
    # Resize for performance and consistency
    frame = cv2.resize(frame, (640, 480))
    
    # 1. Optical Physics (Dark Channel)
    dark_channel = estimator.get_dark_channel(frame)
    A = estimator.estimate_atmospheric_light(frame, dark_channel)
    transmission = estimator.get_transmission_map(frame, A)
    haze_score = estimator.calculate_haze_score(transmission)
    
    # 2. AI Prediction
    aqi_label, pm25_est = ai_model.predict_aqi(frame, haze_score)
    
    # 3. Update History
    if update_history:
        st.session_state.pm25_history.append(pm25_est)
        if len(st.session_state.pm25_history) > 60: # Keep more history for nicer charts
            st.session_state.pm25_history.pop(0)

    # 4. Visualization (Heatmap overlay)
    transmission_display = (transmission * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(transmission_display, cv2.COLORMAP_TURBO) # TURBO is perceptually better than JET
    
    # Smooth blending
    overlay = cv2.addWeighted(frame, 0.65, heatmap, 0.35, 0)
    
    # Add beautiful status text directly on the frame
    overlay = cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
    
    # Convert hex color to BGR for OpenCV
    hex_color = get_aqi_color(aqi_label).lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    
    cv2.putText(overlay, f"STATUS: {aqi_label.upper()}", (20, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, bgr_color, 2, cv2.LINE_AA)
                
    cv2.putText(overlay, f"PM2.5: {pm25_est:.1f}", (450, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    return overlay, aqi_label, pm25_est, haze_score

# --- MAP CACHED QUEUE FOR WEBRTC ANALYTICS ---
@st.cache_resource
def get_analytics_queue():
    import queue
    return queue.Queue()

# --- INPUT HANDLING ---
if input_source == "📹 Live Camera":
    status_text.info("Setting up secure WebRTC camera connection...")
    video_placeholder.empty() # Clear placeholder
    
    result_queue = get_analytics_queue()
    
    # Define how to process video frames asynchronously
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_frame, label, pm25, score = process_frame(img, update_history=False)
        
        # Clear out queue to drop old frames and only push latest
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
                
        result_queue.put((label, pm25, score))
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        
    with col_vision:
        # Create WebRTC context
        webrtc_ctx = webrtc_streamer(
            key="city-cam",
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
    if webrtc_ctx.state.playing:
        status_text.success("✅ Secure Live Feed Active")
        st.info("💡 Note: Real-time analytics are dynamically updated from the live feed.")
        
        while True:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                result = None
            
            if result:
                label, pm25, score = result
                
                # Update UI Metrics
                color_delta = "normal" if label in ["Good", "Moderate"] else "off" if label in ["Unhealthy for Sensitive Groups", "Unhealthy"] else "inverse"
                
                aqi_metric.metric("Air Quality", label, delta=f"{pm25:.1f} PM2.5", delta_color=color_delta)
                pm25_metric.metric("Pollution Score", f"{score:.1f}/100")
                haze_metric.progress(min(max(int(score) / 100.0, 0.0), 1.0))
                
                # Update History
                st.session_state.pm25_history.append(pm25)
                if len(st.session_state.pm25_history) > 60:
                    st.session_state.pm25_history.pop(0)
                
                # Beautiful Area Chart
                chart_placeholder.area_chart(st.session_state.pm25_history, color="#00C6FF")
    else:
        status_text.warning("Click 'START' below to grant camera permissions and begin analysis.")

elif input_source == "📁 Upload Media":
    uploaded_file = st.sidebar.file_uploader("Upload Surveillance Footage", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"], help="Max 200MB.")
    
    if uploaded_file is None:
        video_placeholder.info("👈 Please upload a media file from the sidebar to begin analysis.")
    else:
        st.sidebar.success("File loaded successfully.")
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Image processing
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            processed_frame, label, pm25, score = process_frame(frame)
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Update UI Metrics
            color_delta = "normal" if label in ["Good", "Moderate"] else "off" if label in ["Unhealthy for Sensitive Groups", "Unhealthy"] else "inverse"
            aqi_metric.metric("Air Quality", label, delta=f"{pm25:.1f} PM2.5", delta_color=color_delta)
            pm25_metric.metric("Pollution Score", f"{score:.1f}/100")
            haze_metric.progress(min(max(int(score) / 100.0, 0.0), 1.0))
            
            chart_placeholder.area_chart(st.session_state.pm25_history, color="#00C6FF")
        else:
            # Video processing
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    video_placeholder.success("✅ Analysis Complete.")
                    break
                    
                # Process & Render
                processed_frame, label, pm25, score = process_frame(frame)
                video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Update UI Metrics
                color_delta = "normal" if label in ["Good", "Moderate"] else "off" if label in ["Unhealthy for Sensitive Groups", "Unhealthy"] else "inverse"
                
                aqi_metric.metric("Air Quality", label, delta=f"{pm25:.1f} PM2.5", delta_color=color_delta)
                pm25_metric.metric("Pollution Score", f"{score:.1f}/100")
                haze_metric.progress(min(max(int(score) / 100.0, 0.0), 1.0))
                
                # Beautiful Area Chart
                chart_placeholder.area_chart(st.session_state.pm25_history, color="#00C6FF")
                
                # Provide an abort mechanism if video is too long
                # (Streamlit loop can be aborted by stopping the app, but graceful exit is better)
                
            cap.release()
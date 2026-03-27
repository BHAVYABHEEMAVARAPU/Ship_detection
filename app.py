import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Page Configuration
st.set_page_config(page_title="ShipVision SAR Detector", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 ShipVision: SAR Maritime Monitoring")
st.write("Upload a Synthetic Aperture Radar (SAR) image to identify and count maritime vessels.")

# 2. Load the Model
@st.cache_resource
def load_model():
    # Make sure 'ship_model.pt' is in the same folder as this script
    return YOLO('ship_model.pt')

model = load_model()

# 3. Sidebar Controls
st.sidebar.header("🛠️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, help="Adjust sensitivity: Higher = fewer false alarms, Lower = find more ships.")

# 4. Image Uploader
uploaded_file = st.file_uploader("Upload SAR Data (JPG, PNG, or TIF)...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    # Processing the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run Detection
    with st.spinner('Scanning Radar Image...'):
        results = model.predict(img_array, conf=conf_threshold)
        
        # Plotting the results
        res_plotted = results[0].plot()
        # Convert BGR (OpenCV) to RGB (Streamlit/PIL)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Calculate Data
        ship_count = len(results[0].boxes)
        latency = results[0].speed['inference']

    # 5. UI Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Detection Results")
        st.image(res_rgb, use_container_width=True)

    # 6. Metrics Dashboard
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Ships Detected", ship_count)
    m2.metric("Inference Time", f"{latency:.2f} ms")
    m3.metric("Model Architecture", "YOLOv8-Nano")
    
    st.success(f"Processing complete! Found {ship_count} vessels.")
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ✅ Load Model
model = YOLO("best.pt")  # Change to your model path

# ✅ Class Labels
CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}

# ✅ Custom Dark Theme CSS
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: #ffffff;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #e3e3e3;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 18px;
            text-align: center;
            color: #aaaaaa;
        }
        .stButton>button {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border: none !important;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #136c9b !important;
        }
        .stFileUploader {
            text-align: center !important;
        }
        .img-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
        }
        .result-box {
            background-color: #222831;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        .pneumonia {
            color: #ff4c4c !important;
        }
        .normal {
            color: #27ae60 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ UI Layout
st.markdown('<div class="title">Pneumonia Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a chest X-ray image to analyze for Pneumonia</div>', unsafe_allow_html=True)
st.markdown("---")

# ✅ File Uploader
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # ✅ Convert PIL Image to OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ✅ Resize Image
    img_resized = cv2.resize(image_np, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    # ✅ Ensure RGB format
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # ✅ Show progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(20)

    # ✅ Run YOLO Model
    results = model(img_resized)[0]
    progress_bar.progress(70)

    # ✅ Identify highest confidence prediction
    highest_conf_box = None
    max_confidence = 0
    detected_class = "Normal"

    for box in results.boxes.data:
        x1, y1, x2, y2, confidence, class_id = box
        class_id = int(class_id)

        if confidence > max_confidence:
            highest_conf_box = (int(x1), int(y1), int(x2), int(y2))
            max_confidence = confidence
            detected_class = CLASS_NAMES.get(class_id, "Unknown")

    progress_bar.progress(100)

    # ✅ Display Result
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)

        if detected_class == "Pneumonia" and highest_conf_box and max_confidence > 0.5:
            x1, y1, x2, y2 = highest_conf_box
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
            st.image(img_resized, caption=f"Pneumonia Detected - Confidence: {max_confidence:.2f}",
                     use_container_width=True)
            st.markdown(
                f'<div class="result-box pneumonia"> Pneumonia Detected (Confidence: {max_confidence:.2f})</div>',
                unsafe_allow_html=True)
        else:
            st.image(img_resized, caption="No Pneumonia Detected (Normal)", use_container_width=True)
            st.markdown('<div class="result-box normal"> No Pneumonia Detected (Normal)</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import logging
import os

# Suppress Streamlit context warning
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

# Streamlit config
st.set_page_config(page_title="Underwater Trash Detector", layout="wide")
st.title("🌊 Underwater Trash Detection App")

# Load YOLO model
@st.cache_resource
def load_model():
    model_path = r"C:/Users/anish/Downloads/yolo_trained_model/content/runs/detect/train/weights/best.pt"
    return YOLO(model_path)

with st.spinner("Loading model..."):
    model = load_model()
st.success("Model loaded successfully!")

# Image prediction with resizing
def make_prediction(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv)
    result_img = results[0].plot()

    # Resize output image to fixed width (e.g., 700 px)
    fixed_width = 700
    height, width = result_img.shape[:2]
    scaling_factor = fixed_width / width
    new_size = (fixed_width, int(height * scaling_factor))
    resized_img = cv2.resize(result_img, new_size)

    return resized_img

# Video processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        processed_frame = results[0].plot()
        out.write(processed_frame)

    cap.release()
    out.release()
    return temp_file.name

# File type selector
file_type = st.radio("Select file type", ["Image", "Video"])

# Image upload and detection
if file_type == "Image":
    upload = st.file_uploader("📤 Upload Underwater Image", type=["jpg", "jpeg", "png"])
    if upload:
        image = Image.open(upload)

        # Resize uploaded image before display
        fixed_width = 700
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        scaling_factor = fixed_width / width
        new_size = (fixed_width, int(height * scaling_factor))
        resized_input = cv2.resize(img_cv, new_size)
        resized_input_rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)

        st.image(resized_input_rgb, caption="📷 Uploaded Image", use_column_width=False)

        with st.spinner("🔍 Detecting..."):
            result_img = make_prediction(image)

        st.image(result_img, caption="✅ Detected Image", channels="BGR", use_column_width=False)
        st.success("Detection complete!")

# Video upload and detection
elif file_type == "Video":
    upload = st.file_uploader("📤 Upload Underwater Video", type=["mp4", "mov", "avi"])
    if upload:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(upload.read())
            video_path = temp_video.name

        with st.spinner("🔍 Processing video..."):
            output_path = process_video(video_path)

        st.video(output_path)
        st.success("🎥 Video detection complete!")

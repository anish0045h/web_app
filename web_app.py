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
st.title("🌊 Underwater Trash Detection")

# -----------------------------
# Load YOLO model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    # 🔴 FIX 1: Use a relative path. 
    # Ensure 'best.pt' is in the SAME folder as this python file on GitHub.
    model_path = "best.pt" 
    
    # If you have it in a subfolder named 'weights', use: "weights/best.pt"
    
    if not os.path.isfile(model_path):
        st.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        raise

with st.spinner("Loading model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# -----------------------------
# Image prediction
# -----------------------------
def make_prediction(img: Image.Image):
    """
    Takes a PIL image, runs YOLO, returns annotated BGR image resized for display.
    """
    # Convert PIL (RGB) -> OpenCV BGR
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    try:
        results = model.predict(img_cv, conf=0.40)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        raise

    annotated = results[0].plot()
    if annotated is None:
        st.error("No annotated image returned by model.plot()")
        raise RuntimeError("No annotated output")

    # Resize for display
    fixed_width = 700
    h, w = annotated.shape[:2]
    sf = fixed_width / w
    resized = cv2.resize(annotated, (fixed_width, int(h * sf)))

    # 'resized' is BGR
    return resized

# -----------------------------
# Video processing (save to file)
# -----------------------------
def process_video(video_path: str) -> str:
    """
    Processes an input video, runs YOLO on each frame,
    writes an annotated video to a temp file, and returns the path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open uploaded video")
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback FPS

    # Output temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()

    # Codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        st.error("Failed to open VideoWriter (codec/file issue)")
        raise RuntimeError("VideoWriter failed to open")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.predict(frame, conf=0.40)
        except Exception as e:
            st.error(f"Video prediction error: {e}")
            break

        annotated = results[0].plot()
        if annotated is None:
            continue

        if (annotated.shape[1], annotated.shape[0]) != (width, height):
            annotated = cv2.resize(annotated, (width, height))

        annotated_bgr = annotated.astype("uint8")
        out.write(annotated_bgr)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        st.error("No frames were written to the output video (frame_count = 0)")
        raise RuntimeError("No frames written")

    return temp_path

# -----------------------------
# Video streaming (live detection)
# -----------------------------
def stream_video(video_path: str):
    """
    Streams video with YOLO detections frame-by-frame to the UI.
    Includes optimization (skipping frames + resizing) to prevent lag.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open uploaded video")
        return

    stframe = st.empty()
    frame_count = 0  # Counter for skipping frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔴 FIX 2: Frame Skipping
        # Only process every 3rd frame (skip 2 frames)
        if frame_count % 3 != 0:
            continue

        # 🔴 FIX 3: Resize before processing
        # Reducing size to 640px width makes YOLO much faster on CPU
        frame = cv2.resize(frame, (640, 360))

        try:
            results = model.predict(frame, conf=0.40)
        except Exception as e:
            st.error(f"Video prediction error: {e}")
            break

        annotated = results[0].plot()
        if annotated is None:
            continue

        # Convert BGR -> RGB for Streamlit
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Update the placeholder
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# -----------------------------
# UI: File type selection
# -----------------------------
file_type = st.radio("Select file type", ["Image", "Video"])

# -----------------------------
# Image upload and detection
# -----------------------------
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

        st.image(resized_input_rgb, caption="📷 Uploaded Image", use_container_width=False)

        with st.spinner("🔍 Detecting..."):
            result_img = make_prediction(image)

        st.image(result_img, caption="✅ Detected Image", channels="BGR", use_container_width=False)
        st.success("Detection complete!")

# -----------------------------
# Video upload and detection
# -----------------------------
elif file_type == "Video":
    upload = st.file_uploader("📤 Upload Underwater Video", type=["mp4", "mov", "avi"])
    if upload:
        # Save uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(upload.read())
            video_path = temp_video.name

        mode = st.selectbox(
            "Choose how to run detection on the video:",
            ["Live detection (frame-by-frame)", "Generate processed video file"]
        )

        if mode == "Live detection (frame-by-frame)":
            st.info("🎥 Running live detection... (Optimized for performance)")
            stream_video(video_path)
            st.success("✅ Video processing complete!")

        else:  # Generate processed video file
            st.info("⏳ Processing full video... This may take a while.")
            with st.spinner("Processing video frames..."):
                output_path = process_video(video_path)

            st.video(output_path)
            st.success("🎥 Video detection complete!")

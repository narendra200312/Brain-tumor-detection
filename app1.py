import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

def load_model():
    model = YOLO("best.pt")  # Load your trained model
    return model

def detect_tumor(model, image):
    results = model.predict(image)
    detections = results[0].boxes.data  # Get bounding box data

    img = np.array(image)
    if len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2, conf, _ = box.tolist()
            label = f"Tumor ({conf:.2f})"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return Image.fromarray(img), True
    else:
        return image, False

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan, and the system will detect if a brain tumor is present.")

# Load YOLO model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload MRI scan (JPG, PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    
    if st.button("Detect Tumor"):
        result_image, detected = detect_tumor(model, image)
        
        with col2:
            st.image(result_image, caption="Detection Result", use_column_width=True)
        
        if detected:
            st.success("Tumor detected!")
        else:
            st.error("No tumor detected!")
            
st.sidebar.markdown("---")
st.sidebar.write("Created with 🤖 by Narendra, Jayesh & Nikitha")

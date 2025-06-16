import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load YOLOv11 model
MODEL_PATH = 'best.pt'  # Adjust if necessary
model = YOLO(MODEL_PATH)

# Streamlit setup
st.set_page_config(page_title="Brain Tumor Detection - Explainable AI", layout="wide")
st.title("🧠 Brain Tumor Detection using YOLOv11 + Explainability")

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

# Custom wrapper to extract feature maps before YOLO concat layers
class YoloFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def forward(self, x):
        return self.model(x)

class YOLODetectionTarget:
    def __call__(self, model_output):
        return model_output.sum()

def apply_yolo_gradcam(model, image_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resize image to model input size
    model_input_size = (640, 640)
    image_resized = cv2.resize(image_np, model_input_size)

    # Prepare input tensor
    input_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device).requires_grad_(True)

    # Feature model for Grad-CAM
    feature_model = YoloFeatureExtractor(model).to(device).eval()
    target_layer = model.model.model[9]  # Example: SPPF

    cam = GradCAM(model=feature_model, target_layers=[target_layer])
    targets = [YOLODetectionTarget()]

    # Generate Grad-CAM heatmap for resized image
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Resize Grad-CAM to original image size
    original_h, original_w = image_np.shape[:2]
    cam_resized = cv2.resize(grayscale_cam, (original_w, original_h))

    # Overlay Grad-CAM on original image
    visualization = show_cam_on_image(image_np.astype(np.float32) / 255.0, cam_resized, use_rgb=True)
    return visualization



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    st.subheader("Uploaded MRI Image vs Detection Result")
    col1, col2 = st.columns(2)

    # Run prediction
    results = model.predict(image, save=False, conf=0.2)[0]

    # Annotate image
    annotated_image = image_np.copy()
    heatmap = np.zeros((height, width), dtype=np.float32)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = int(box.cls[0])

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_image, f"Tumor {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf)

    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    overlay_image = cv2.addWeighted(annotated_image, 0.6, heatmap_colored, 0.4, 0)

    gradcam_image = apply_yolo_gradcam(model, image_np)

    col1, col2, col3 = st.columns([1, 1, 1])  # or [1, 1, 0.8] to emphasize control

    col1.image(image_np, caption="Original MRI Image", use_column_width=True)
    col2.image(overlay_image, caption="Detection + Confidence Heatmap", use_column_width=True)
    col3.image(gradcam_image, caption="Grad-CAM Explanation", use_column_width=True)  


    # Explanation
    st.subheader("🗞 Explanation of Predictions")

    if results.boxes:
        st.markdown("""
        ### What You're Seeing:
        - **Bounding boxes**: Detected tumor regions.
        - **Confidence score**: Model certainty for each box.
        - **Grad-CAM heatmap**: Shows **model's attention** — the parts of the image that influenced its decision most.

        ### Interpretation:
        - **Red areas** = highly influential regions.
        - If box aligns with red, model is focusing on meaningful tumor-like patterns.

        ⚠️ **Note**: This is an assistive AI tool. Final diagnosis must be made by a medical expert.
        """)

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            st.write(f"**Tumor {i+1}:** Location [({x1}, {y1}) - ({x2}, {y2})], Confidence: {conf:.2f}")
    else:
        st.markdown("""
        The model **did not detect any tumors** at the set confidence threshold (0.2).

        🔍 Tips:
        - Use higher-quality or clearer images.
        - Lower the confidence threshold in the code to increase sensitivity.

        Still, **absence of detection ≠ absence of tumor**. Always consult a radiologist.
        """)

st.sidebar.markdown("---")
st.sidebar.write("Created with 🤖 by Narendra, Jayesh & Nikitha")


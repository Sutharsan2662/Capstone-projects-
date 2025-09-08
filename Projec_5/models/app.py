import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import cv2
from ultralytics import YOLO

# ================================
# Page Config & Styling
# ================================
st.set_page_config(page_title="☀️ Solar Panel AI Diagnostics", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Load ResNet18 Classification Model
# ================================
@st.cache_resource
def load_classification_model():
    try:
        checkpoint = torch.load("best_resnet18_model.pth", map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            class_names = checkpoint.get("class_names", None)
            if class_names is None and "class_to_idx" in checkpoint:
                idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
                class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model, class_names, True
        else:
            class_names = ["Clean", "Dusty", "Bird-drop", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
            model.load_state_dict(checkpoint)
            model.eval()
            return model, class_names, True

    except FileNotFoundError:
        st.error("❌ Classification model file not found. Place `best_resnet18_model.pth` in the app directory.")
        return None, [], False
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None, [], False

# ================================
# Load YOLO Detection Model
# ================================
@st.cache_resource
def load_yolo_model():
    try:
        with open("data.yaml", "r") as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg["names"]

        model = YOLO("yolov8s.pt")  # make sure yolov8s.pt is present
        return model, class_names, True
    except FileNotFoundError:
        st.error("❌ YOLO files not found. Place `yolov8s.pt` and `data.yaml` in the app directory.")
        return None, [], False
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None, [], False

# ================================
# Prediction Function (Classification)
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_classification(model, image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs.numpy()

# ================================
# App Layout
# ================================
st.markdown('<div class="main-header">☀️ Solar Panel AI Diagnostics</div>', unsafe_allow_html=True)

# Sidebar: Choose Mode
with st.sidebar:
    st.header("🧠 Choose Mode")
    mode = st.radio("Select Task", ["Classification", "Object Detection"])

# ================================
# Classification Mode
# ================================
if mode == "Classification":
    st.header("📌 Classification Mode")

    with st.sidebar:
        st.subheader("⚙️ Model Status")
        clf_model, class_names, model_ready = load_classification_model()
        if model_ready:
            st.success("✅ Classification Model Loaded")
            st.write(f"**Classes Detected:** {len(class_names)}")
            for c in class_names:
                st.write(f"- {c}")
        else:
            st.stop()

        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        show_all_probs = st.checkbox("Show All Probabilities", value=True)

    uploaded_file = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])

    if uploaded_file and model_ready:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Could not read the uploaded image.")
            st.stop()

        st.image(image, caption="Uploaded Image", width=400)

        if st.button("🔍 Run Classification"):
            pred_idx, confidence, all_probs = predict_classification(clf_model, image)
            pred_class = class_names[pred_idx]

            st.markdown(f"""
            <div class="prediction-box">
                <h2>Prediction: {pred_class}</h2>
                <h3>Confidence: {confidence*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            if confidence < confidence_threshold:
                st.warning("⚠️ Low confidence detection. Please recheck manually.")

            if show_all_probs:
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame({
                    "Class": class_names,
                    "Probability": all_probs[0],
                    "Percentage": [f"{p*100:.1f}%" for p in all_probs[0]]
                }).sort_values("Probability", ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(prob_df["Class"], prob_df["Probability"])
                ax.set_xlim(0, 1)
                for i, (p, cls) in enumerate(zip(prob_df["Probability"], prob_df["Class"])):
                    ax.text(p + 0.01, i, f"{p*100:.1f}%", va="center")
                fig.tight_layout()
                st.pyplot(fig)

    # Training Performance
    st.markdown("---")
    st.subheader("📈 Training Performance")
    try:
        train_losses = np.load("train_losses.npy")
        val_losses = np.load("val_losses.npy")
        val_accs = np.load("val_accs.npy")

        epochs = range(1, len(train_losses) + 1)
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
            ax1.plot(epochs, val_losses, label="Val Loss", marker="s")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.legend()
            fig1.tight_layout()
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(epochs, val_accs, label="Val Accuracy", marker="^", color="green")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0, 1)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2)

    except FileNotFoundError:
        st.info("Training history files not found. Upload `train_losses.npy`, `val_losses.npy`, and `val_accs.npy`.")

# ================================
# Object Detection Mode
# ================================
elif mode == "Object Detection":
    st.header("📌 Object Detection Mode")

    with st.sidebar:
        st.subheader("⚙️ Model Status")
        yolo_model, yolo_class_names, yolo_ready = load_yolo_model()
        if yolo_ready:
            st.success("✅ YOLO Model Loaded")
            st.write(f"**Classes Detected:** {len(yolo_class_names)}")
            for c in yolo_class_names:
                st.write(f"- {c}")
        else:
            st.stop()

    uploaded_file = st.file_uploader("📤 Upload an image for detection", type=["jpg", "jpeg", "png"])

    if uploaded_file and yolo_ready:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Could not read the uploaded image.")
            st.stop()

        st.image(image, caption="Uploaded Image", width=400)

        if st.button("🔎 Run Detection"):
            results = yolo_model.predict(image, imgsz=640, conf=0.25)

            # Convert BGR → RGB for display
            res_bgr = results[0].plot()
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detection Results", width=600)

            boxes = results[0].boxes
            if boxes is not None:
                st.subheader("📊 Detected Faults")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- {yolo_class_names[cls_id]} ({conf*100:.1f}%)")

import streamlit as st
import sys
import os

# ================================
# Import Error Handling
# ================================
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime
    
    IMPORTS_SUCCESS = True
    ERROR_MESSAGE = None
    
except ImportError as e:
    IMPORTS_SUCCESS = False
    ERROR_MESSAGE = str(e)

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Solar Panel AI Diagnostics",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Error Handling Display
# ================================
if not IMPORTS_SUCCESS:
    st.markdown('<div class="main-header">☀️ Solar Panel AI Diagnostics</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="error-box">
        <h3>🚫 Import Error Detected</h3>
        <p><strong>Error:</strong> {ERROR_MESSAGE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔧 How to Fix This Issue")
    
    tab1, tab2, tab3 = st.tabs(["🔽 Quick Fix", "🔄 Environment Reset", "📋 Detailed Steps"])
    
    with tab1:
        st.markdown("### Option 1: Downgrade NumPy (Recommended)")
        st.code("pip install numpy<2", language="bash")
        st.markdown("### Option 2: Create New Environment")
        st.code("""
# Create new environment
conda create -n solar_env python=3.9
conda activate solar_env

# Install packages
pip install streamlit torch torchvision pillow matplotlib pandas numpy<2
        """, language="bash")
    
    with tab2:
        st.markdown("### Complete Environment Reset")
        st.code("""
# Uninstall conflicting packages
pip uninstall torch torchvision numpy -y

# Reinstall with compatible versions
pip install numpy==1.24.3
pip install torch==2.0.1 torchvision==0.15.2
pip install streamlit pillow matplotlib pandas
        """, language="bash")
    
    with tab3:
        st.markdown("### Detailed Troubleshooting Steps")
        st.markdown("""
        1. **Check your current versions:**
        ```bash
        pip list | grep -E "(torch|numpy)"
        ```
        
        2. **The issue:** PyTorch was compiled with NumPy 1.x but you have NumPy 2.x installed
        
        3. **Solutions (try in order):**
           - Downgrade NumPy: `pip install "numpy<2"`
           - Upgrade PyTorch: `pip install --upgrade torch torchvision`
           - Fresh install: Create new virtual environment
        
        4. **Verify installation:**
        ```python
        import torch
        import numpy as np
        print(f"PyTorch: {torch.__version__}")
        print(f"NumPy: {np.__version__}")
        ```
        """)
    
    st.stop()

# ================================
# Main Application (only runs if imports successful)
# ================================

# Initialize session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Model Loading
# ================================
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load("best_resnet18_model.pth", map_location=device)
        
        # Extract class mapping
        class_to_idx = checkpoint["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        # Load model
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        return model, class_names, True
    except FileNotFoundError:
        st.error("Model file 'best_resnet18_model.pth' not found. Please ensure the model file is in the same directory.")
        return None, [], False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, [], False

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown("## ⚙️ Model Status")
    
    with st.spinner("Loading AI model..."):
        model, class_names, model_loaded = load_model()
    
    if model_loaded:
        st.success("✅ Model Ready")
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Classes:** {len(class_names)}")
        
        # Show available classes
        st.markdown("**Detectable Faults:**")
        for class_name in class_names:
            st.markdown(f"• {class_name}")
            
    else:
        st.error("❌ Model Failed to Load")
        st.markdown("**Possible Issues:**")
        st.markdown("• Model file missing")
        st.markdown("• Incompatible model format")
        st.markdown("• Memory issues")
        st.stop()
    
    st.markdown("---")
    st.markdown("## 📊 Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    show_all_probs = st.checkbox("Show All Probabilities", value=True)

# ================================
# Image Transform & Prediction
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item(), probs.cpu().numpy()[0]

def get_fault_info(class_name):
    """Get detailed information about each fault type"""
    fault_descriptions = {
        'Bird-drop': {
            'severity': 'Medium',
            'description': 'Bird droppings can reduce panel efficiency by blocking sunlight and creating hot spots.',
            'recommendation': 'Clean panels regularly and consider installing bird deterrents or mesh barriers.',
            'urgency': 'Moderate - Clean within 1-2 weeks'
        },
        'Clean': {
            'severity': 'Good',
            'description': 'Panel appears to be in optimal condition with no visible defects or obstructions.',
            'recommendation': 'Continue regular maintenance schedule and periodic inspections.',
            'urgency': 'None - Maintain current schedule'
        },
        'Dusty': {
            'severity': 'Low',
            'description': 'Dust accumulation can reduce efficiency by 10-25% depending on thickness and coverage.',
            'recommendation': 'Schedule cleaning. In dusty environments, consider automated cleaning systems.',
            'urgency': 'Low - Clean within 1 month'
        },
        'Electrical-damage': {
            'severity': 'Critical',
            'description': 'Electrical damage poses serious safety risks and can cause complete system failure.',
            'recommendation': 'IMMEDIATE shutdown required. Contact certified solar technician immediately.',
            'urgency': 'URGENT - Address immediately'
        },
        'Physical-Damage': {
            'severity': 'High',
            'description': 'Physical damage can reduce efficiency and create safety hazards including fire risk.',
            'recommendation': 'Inspect extent of damage and replace affected panels. Check mounting integrity.',
            'urgency': 'High - Address within 1 week'
        },
        'Snow-Covered': {
            'severity': 'Medium',
            'description': 'Snow coverage completely prevents energy generation but typically resolves naturally.',
            'recommendation': 'Remove snow safely with proper tools or wait for natural melting.',
            'urgency': 'Weather dependent - Remove if safe'
        }
    }
    
    return fault_descriptions.get(class_name, {
        'severity': 'Unknown',
        'description': 'Classification uncertain. Manual inspection recommended.',
        'recommendation': 'Consult with qualified solar panel technician for proper assessment.',
        'urgency': 'Consult professional'
    })

# ================================
# Main App
# ================================
st.markdown('<div class="main-header">☀️ Solar Panel AI Diagnostics</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📸 Upload Solar Panel Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Image info
            st.markdown(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")

            # Prediction
            if st.button("🔍 Analyze Panel", type="primary", use_container_width=True):
                with st.spinner("Analyzing solar panel condition..."):
                    try:
                        pred_class, confidence, all_probs = predict(image)
                        st.session_state.prediction_count += 1
                        
                        # Get detailed fault information
                        fault_info = get_fault_info(pred_class)
                        confidence_pct = confidence * 100
                        
                        # Main prediction display
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🔍 Detection: {pred_class}</h2>
                            <h3>Confidence: {confidence_pct:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display recommendations based on severity
                        if fault_info['severity'] == 'Critical':
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>🚨 CRITICAL: {fault_info['severity']} Severity</h4>
                                <p><strong>Issue:</strong> {fault_info['description']}</p>
                                <p><strong>Action Required:</strong> {fault_info['recommendation']}</p>
                                <p><strong>Timeline:</strong> {fault_info['urgency']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fault_info['severity'] in ['High', 'Medium']:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>⚠️ {fault_info['severity']} Severity</h4>
                                <p><strong>Issue:</strong> {fault_info['description']}</p>
                                <p><strong>Recommendation:</strong> {fault_info['recommendation']}</p>
                                <p><strong>Timeline:</strong> {fault_info['urgency']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif fault_info['severity'] == 'Good':
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>✅ {fault_info['severity']} Condition</h4>
                                <p><strong>Status:</strong> {fault_info['description']}</p>
                                <p><strong>Recommendation:</strong> {fault_info['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="info-box">
                                <h4>ℹ️ {fault_info['severity']}</h4>
                                <p><strong>Status:</strong> {fault_info['description']}</p>
                                <p><strong>Recommendation:</strong> {fault_info['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Confidence warning
                        if confidence < confidence_threshold:
                            st.warning(f"⚠️ Low confidence detection ({confidence_pct:.1f}%). Consider manual inspection or retake image with better lighting/angle.")

                        # Show all probabilities
                        if show_all_probs:
                            st.markdown("### 📊 All Class Probabilities")
                            prob_df = pd.DataFrame({
                                'Class': class_names,
                                'Probability': all_probs,
                                'Percentage': [f"{p*100:.1f}%" for p in all_probs]
                            }).sort_values('Probability', ascending=False)
                            
                            # Display as table
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                            
                            # Create simple bar chart
                            fig, ax = plt.subplots(figsize=(8, 6))
                            bars = ax.barh(prob_df['Class'], prob_df['Probability'], 
                                          color=['#FF6B35' if cls == pred_class else '#4FACFE' for cls in prob_df['Class']])
                            ax.set_xlabel('Probability')
                            ax.set_title('Classification Probabilities')
                            ax.set_xlim(0, 1)
                            
                            # Add percentage labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                       f'{width*100:.1f}%', ha='left', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

with col2:
    st.markdown("### 📈 Analysis Stats")
    st.metric("Total Analyses", st.session_state.prediction_count)
    
    st.markdown("### ℹ️ System Info")
    st.markdown(f"**PyTorch:** {torch.__version__}")
    st.markdown(f"**NumPy:** {np.__version__}")
    st.markdown(f"**Device:** {device}")

# ================================
# Training Results (Optional)
# ================================
st.markdown("---")
st.subheader("📊 Model Training Performance")

try:
    val_losses = np.load("val_losses.npy")
    train_losses = np.load("train_losses.npy")
    val_accs = np.load("val_accs.npy")

    col1, col2 = st.columns(2)
    
    with col1:
        # Training vs Validation Loss
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', marker='o', label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', marker='s', label="Validation Loss", linewidth=2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        # Validation Accuracy
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(epochs, val_accs, 'g-', marker='^', linewidth=2, label="Validation Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy Over Time")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # Training summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Loss", f"{train_losses[-1]:.3f}")
    with col2:
        st.metric("Final Val Loss", f"{val_losses[-1]:.3f}")
    with col3:
        st.metric("Best Val Accuracy", f"{max(val_accs):.1%}")
    with col4:
        st.metric("Total Epochs", len(train_losses))

except FileNotFoundError:
    st.info("📁 Training history files not found. Upload val_losses.npy, train_losses.npy, and val_accs.npy to see training performance.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 AI-Powered Solar Panel Diagnostics | Built with Streamlit & PyTorch</p>
</div>
""", unsafe_allow_html=True)
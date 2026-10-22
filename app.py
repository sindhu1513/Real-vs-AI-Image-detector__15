"""AI vs Real Image Detector - Fixed Version"""
import sys
import streamlit as st

# === FIX PYTORCH + STREAMLIT COMPATIBILITY ISSUE ===
# This must be done before any torch imports
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# Disable Streamlit file watcher for torch
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# ==================================================

# Now import everything else
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNNClassifier
import base64
import io
from datetime import datetime

# PATHS
MODEL_PATH = "models/ai_real_classifier.pth"
UPLOAD_FOLDER = "uploads"

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stImage, .stImage > img {
        display: none !important;
    }
    
    .image-container {
        margin: 20px 0;
        padding: 15px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .perfect-image {
        width: auto;
        max-width: 500px;
        height: auto;
        max-height: 400px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .real-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 8px solid #28a745;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-left: 8px solid #ffc107;
    }
    
    .custom-progress {
        width: 100%;
        height: 20px;
        background: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = CNNClassifier(num_classes=2, input_size=32)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        class_names = checkpoint.get('class_names', ['real', 'ai'])
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, ['real', 'ai']

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_image(model, image_tensor):
    """Make prediction on the image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def pil_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    
    if image.format == 'PNG':
        image.save(buffered, format='PNG', optimize=False)
        mime = 'image/png'
    else:
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        image.save(buffered, format='JPEG', quality=95, optimize=True)
        mime = 'image/jpeg'
    
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:{mime};base64,{img_str}"

def display_image_perfect_size(image, caption=""):
    """Display image at perfect size"""
    img_data = pil_to_base64(image)
    
    html = f"""
    <div class="image-container">
        <img src="{img_data}" alt="{caption}" class="perfect-image">
        <div class="image-caption">{caption}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def main():
    st.title("🔍 AI vs Real Image Detector")
    st.markdown("Upload an image - perfect size display • No compression")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Status")
        if os.path.exists(MODEL_PATH):
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model not found")
            st.info("Run: `python train.py`")
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Open image
        original_image = Image.open(uploaded_file)
        
        # Save original
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        original_image.save(file_path)
        
        # Get image info
        img_width, img_height = original_image.size
        img_format = original_image.format or 'Unknown'
        file_size = len(uploaded_file.getvalue()) / 1024
        
        # Display image
        st.markdown("### 📸 Your Image")
        display_image_perfect_size(
            original_image, 
            f"{uploaded_file.name} • {img_width}x{img_height} • {file_size:.0f}KB"
        )
        
        # Image metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dimensions</div>
                <div class="metric-value">{img_width}×{img_height}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Format</div>
                <div class="metric-value">{img_format}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Size</div>
                <div class="metric-value">{file_size:.0f}KB</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis
        st.markdown("---")
        st.markdown("### 🔍 Analysis Results")
        
        with st.spinner("Analyzing..."):
            # Predict
            image_tensor = preprocess_image(original_image)
            prediction, confidence = predict_image(model, image_tensor)
            label = class_names[prediction]
            
            # Get probabilities
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Map probabilities
            if class_names[0].lower() == 'real':
                ai_prob = probabilities[1].item()
                real_prob = probabilities[0].item()
            else:
                ai_prob = probabilities[0].item()
                real_prob = probabilities[1].item()
            
            # Prediction card
            if label.lower() == 'real':
                st.markdown(f"""
                <div class="prediction-card real-card">
                    <h2 style="margin:0; color:#155724;">✅ REAL Image</h2>
                    <p style="font-size:1.3rem; margin:15px 0 0 0; color:#155724; font-weight:bold;">
                        {confidence*100:.1f}% Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card ai-card">
                    <h2 style="margin:0; color:#856404;">⚠️ AI-GENERATED</h2>
                    <p style="font-size:1.3rem; margin:15px 0 0 0; color:#856404; font-weight:bold;">
                        {confidence*100:.1f}% Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability bars
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.markdown("**🤖 AI-Generated**")
                progress_html = f"""
                <div class="custom-progress">
                    <div class="custom-progress-bar" style="width: {ai_prob*100}%;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                st.markdown(f"**{ai_prob:.1%}**")
            
            with col_p2:
                st.markdown("**📸 Real**")
                progress_html = f"""
                <div class="custom-progress">
                    <div class="custom-progress-bar" style="width: {real_prob*100}%;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                st.markdown(f"**{real_prob:.1%}**")
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_text = f"""AI vs REAL Detection Report
File: {uploaded_file.name}
Size: {img_width}x{img_height}
Prediction: {label.upper()}
Confidence: {confidence*100:.1f}%
AI: {ai_prob:.1%} | Real: {real_prob:.1%}"""
        
        st.download_button(
            label="📥 Download Report",
            data=result_text,
            file_name=f"detection_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding:40px; background:#f8f9fa; border-radius:16px; margin-top:30px;">
            <h3 style="color:#667eea;">🚀 Upload an image to get started</h3>
            <p style="color:#666;">Images are displayed at the perfect size - not too small, not too large</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
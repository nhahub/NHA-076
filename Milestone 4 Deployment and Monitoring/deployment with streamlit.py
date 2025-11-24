import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# ==================== Configuration ====================
st.set_page_config(
    page_title="Land Type Classifier",
    page_icon="🛰️",
    layout="wide"
)

# Model filename
MODEL_FILENAME = 'best_model.pth'

# Class names (same order as training)
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Model Loading ====================
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model"""
    
    # Check if model file exists
    if not os.path.exists(MODEL_FILENAME):
        return None, f"Model file '{MODEL_FILENAME}' not found in current directory"
    
    # Define model architecture (same as training)
    model = models.resnet50(pretrained=False)
    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(n_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(CLASSES))
    )
    
    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# ==================== Preprocessing ====================
def get_transform():
    """Same preprocessing as validation/test"""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy
    img_array = np.array(image)
    
    # Apply transformations
    transform = get_transform()
    transformed = transform(image=img_array)
    img_tensor = transformed['image']
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# ==================== Prediction ====================
def predict(model, image_tensor):
    """Make prediction"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

# ==================== UI ====================
def main():
    # Header
    st.title("🛰️ Land Type Classification")
    st.markdown("### Classify satellite images using Deep Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app classifies land types from **Sentinel-2 satellite images** using a 
        **ResNet50** model trained on the EuroSAT dataset.
        
        **Model Performance:**
        - Accuracy: **93.12%**
        - Classes: 10 land types
        """)
        
        st.markdown("---")
        st.header("📊 Classes")
        for i, cls in enumerate(CLASSES, 1):
            st.write(f"{i}. {cls}")
        
        st.markdown("---")
        st.info(f"🔧 **Model:** {MODEL_FILENAME}")
        st.info(f"💻 **Device:** {DEVICE}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image (JPG/PNG)", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=300)
            
            # Show image info
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                model, error = load_model()
            
            if model is None:
                st.error(f"❌ Error loading model: {error}")
                st.info(f"💡 Make sure '{MODEL_FILENAME}' is in the same directory as deployment.py")
                st.warning("📂 Current directory: " + os.getcwd())
                return
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    img_tensor = preprocess_image(image)
                    pred_idx, confidence, all_probs = predict(model, img_tensor)
                    predicted_class = CLASSES[pred_idx]
                    
                    # Display main prediction
                    st.success(f"### Predicted Class: **{predicted_class}**")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    st.markdown("---")
                    
                    # Show all probabilities
                    st.subheader("📊 All Class Probabilities")
                    
                    # Sort by probability
                    sorted_indices = np.argsort(all_probs)[::-1]
                    
                    for idx in sorted_indices:
                        prob = all_probs[idx]
                        class_name = CLASSES[idx]
                        
                        # Highlight predicted class
                        if idx == pred_idx:
                            st.markdown(f"**🎯 {class_name}**")
                        else:
                            st.write(class_name)
                        
                        st.progress(float(prob))
                        st.caption(f"{prob*100:.2f}%")
                        st.markdown("")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.info("👆 Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit • Model: ResNet50 • Dataset: EuroSAT</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== Run App ====================
if __name__ == "__main__":
    main()
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# --------- Load Models ---------
@st.cache_resource
def load_cifar10_model():
    return tf.keras.models.load_model("cnn_cifar10_optimized.h5")

@st.cache_resource
def load_mobilenet_model():
    # MobileNet expects 224x224 input, pretrained on ImageNet
    return tf.keras.applications.mobilenet.MobileNet(weights='imagenet')

cifar10_model = load_cifar10_model()
mobilenet_model = load_mobilenet_model()

# CIFAR-10 class names
cifar10_class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ImageNet labels for MobileNet
imagenet_labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(imagenet_labels_path) as f:
    imagenet_class_names = [line.strip() for line in f.readlines()]

# --------- Utility Functions ---------

def resize_with_padding(image, target_size):
    """Resize image preserving aspect ratio with padding to target_size."""
    return ImageOps.pad(image, target_size, method=Image.Resampling.LANCZOS, color=(0,0,0))

def preprocess_cifar10(image):
    """Resize and scale image for CIFAR-10 model."""
    img_resized = resize_with_padding(image, (32,32))
    img_array = np.array(img_resized) / 255.0
    return img_array.reshape(1, 32, 32, 3)

def preprocess_mobilenet(image):
    """Resize and preprocess for MobileNet."""
    img_resized = resize_with_padding(image, (224,224))
    img_array = np.array(img_resized)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array.reshape(1,224,224,3)

def predict_cifar10(image):
    img = preprocess_cifar10(image)
    preds = cifar10_model.predict(img)[0]
    return preds

def predict_mobilenet(image):
    img = preprocess_mobilenet(image)
    preds = mobilenet_model.predict(img)[0]
    return preds

def plot_confidence_bar(class_names, scores):
    fig, ax = plt.subplots(figsize=(8,3))
    bars = ax.barh(class_names, scores, color='skyblue')
    ax.set_xlim(0,1)
    ax.invert_yaxis()
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')
    plt.tight_layout()
    st.pyplot(fig)

# --------- Streamlit UI ---------

st.set_page_config(page_title="Advanced Image Classifier", layout="wide")

st.title("🧠 Advanced Image Classifier with Multiple Models & Augmentations")

st.markdown("""
Upload images for classification. Supports CIFAR-10 CNN model and MobileNet pretrained on ImageNet.

Features:
- Aspect-ratio preserving resize with padding
- Real-time augmentations (rotate, flip)
- Batch upload support
- Confidence bar chart visualization
- Confidence threshold warnings
- Responsive UI with side-by-side image display
""")

model_choice = st.selectbox("Choose Model", ["CIFAR-10 CNN", "MobileNet (ImageNet)"])

# Augmentation options
st.sidebar.header("Image Augmentations")
rotate_degree = st.sidebar.slider("Rotate (degrees)", -180, 180, 0, step=1)
flip_horizontal = st.sidebar.checkbox("Flip Horizontal")
flip_vertical = st.sidebar.checkbox("Flip Vertical")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                                         help="Warn if max confidence is below this value")

uploaded_files = st.file_uploader("Upload one or more Images (jpg, jpeg, png)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.write(f"### Image {idx+1}: {uploaded_file.name}")

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Apply augmentations
        if rotate_degree != 0:
            image = image.rotate(rotate_degree, expand=True)
        if flip_horizontal:
            image = ImageOps.mirror(image)
        if flip_vertical:
            image = ImageOps.flip(image)

        # Display original + resized images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Augmented Image", use_container_width=True)
        with col2:
            target_size = (32,32) if model_choice == "CIFAR-10 CNN" else (224,224)
            resized_image = resize_with_padding(image, target_size)
            st.image(resized_image, caption=f"Resized Input ({target_size[0]}x{target_size[1]})", use_container_width=True)

        # Predict
        with st.spinner("Predicting..."):
            if model_choice == "CIFAR-10 CNN":
                preds = predict_cifar10(image)
                class_names = cifar10_class_names
            else:
                preds = predict_mobilenet(image)
                class_names = imagenet_class_names

        predicted_idx = np.argmax(preds)
        predicted_class = class_names[predicted_idx]
        confidence = preds[predicted_idx]

        # Prediction and confidence
        st.subheader(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2f})")

        # Confidence warning
        if confidence < confidence_threshold:
            st.warning(f"⚠️ Confidence below threshold ({confidence_threshold:.2f}). Prediction may be unreliable.")

        # Show confidence bar chart (top 10 classes)
        top_indices = preds.argsort()[-10:][::-1]
        top_class_names = [class_names[i] for i in top_indices]
        top_scores = preds[top_indices]
        plot_confidence_bar(top_class_names, top_scores)

        st.markdown("---")
else:
    st.info("Upload images above to get predictions.")


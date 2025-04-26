
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Title
st.title("🧠 Deepfake Detection: CNN vs CapsNet")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Load models safely
@st.cache_resource
def load_models():
    cnn = load_model("cnn_model.h5")
    cap = load_model("capsnet_model.h5")
    return cnn, cap

try:
    cnn_model, capsnet_model = load_models()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Label classes
labels = ['Real', 'Fake']

# Preprocessing function
def preprocess(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # remove alpha if RGBA
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

# Prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼 Uploaded Image", use_column_width=False, width=250)

    try:
        image = Image.open(uploaded_file)
        processed = preprocess(image)

        cnn_pred = cnn_model.predict(processed)[0]
        cap_pred = capsnet_model.predict(processed)[0]

        cnn_result = labels[np.argmax(cnn_pred)]
        cap_result = labels[np.argmax(cap_pred)]

        st.markdown("### 🔍 Prediction Results:")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CNN Result")
            st.success(f"🧠 {cnn_result} ({cnn_pred[np.argmax(cnn_pred)]*100:.2f}%)")

        with col2:
            st.subheader("CapsNet Result")
            st.info(f"📦 {cap_result} ({cap_pred[np.argmax(cap_pred)]*100:.2f}%)")

    except Exception as e:
        st.error(f"Prediction error: {e}")

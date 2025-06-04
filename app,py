import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tempfile

st.set_page_config(page_title="🩻 Pneumonia Detection from X-Ray", layout="wide")
st.title("🩺 Pneumonia Detection using Chest X-Rays")

option = st.sidebar.radio("Choose input source:", ("Upload X-ray Image", "Camera"))

# Load model (placeholder: replace with your own .h5 model)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

def preprocess(image):
    image = image.convert('L')  # convert to grayscale
    image = image.resize((150, 150))  # model input size
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 150, 150, 1)
    return img_array

def predict(image):
    processed = preprocess(image)
    prediction = model.predict(processed)[0][0]
    if prediction > 0.5:
        return f"Prediction: 🚨 Pneumonia Detected (Confidence: {prediction:.2f})", "red"
    else:
        return f"Prediction: ✅ Normal (Confidence: {1 - prediction:.2f})", "green"

if option == "Upload X-ray Image":
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)
        result, color = predict(image)
        st.markdown(f"<h3 style='color:{color}'>{result}</h3>", unsafe_allow_html=True)

elif option == "Camera":
    st.warning("Camera access works only on local or full Render deployment.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    if run:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            output, color = predict(pil_img)
            cv2.putText(img, output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            FRAME_WINDOW.image(img, use_container_width=True)
        cap.release()
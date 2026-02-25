import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="ISL Translator", layout="centered")

st.title("🤟 Indian Sign Language to Text & Speech")
st.write("Show your hand sign using your camera")

# Initialize MediaPipe Hands correctly
mp_hands = mp.tasks.vision.HandLandmarker
mp_base_options = mp.tasks.BaseOptions
mp_vision_running_mode = mp.tasks.vision.RunningMode

# Load model file (MediaPipe requires model asset)
model_path = "hand_landmarker.task"

# Dummy prediction function
def predict_gesture():
    return "HELLO"

img_file = st.camera_input("📷 Show your hand sign")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    st.image(frame, caption="Captured Image")

    # For now simple detection placeholder
    gesture = predict_gesture()

    st.success(f"Detected Sign: {gesture}")

    # Browser speech (No API needed)
    speech_html = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{gesture}");
        window.speechSynthesis.speak(msg);
        </script>
    """
    st.markdown(speech_html, unsafe_allow_html=True)

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import base64

st.set_page_config(page_title="ISL Translator", layout="centered")

st.title("🤟 Indian Sign Language to Text & Speech")
st.write("Show a hand sign using your camera")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Dummy Gesture Prediction Function
# (Replace this with your trained model later)
def predict_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    # Simple rule example
    if thumb_tip.y < index_tip.y:
        return "HELLO"
    else:
        return "THANK YOU"

# Camera Input (Cloud Compatible)
img_file = st.camera_input("📷 Show your hand sign")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = predict_gesture(hand_landmarks.landmark)

            st.success(f"Detected Sign: {gesture}")

            # Convert text to speech (Browser based)
            speech_html = f"""
                <audio autoplay>
                <source src="https://api.voicerss.org/?key=YOUR_API_KEY&hl=en-us&src={gesture}" type="audio/mpeg">
                </audio>
            """
            st.markdown(speech_html, unsafe_allow_html=True)

    else:
        st.warning("No hand detected. Please try again.")

    st.image(frame, channels="BGR")

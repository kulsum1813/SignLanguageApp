import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="ISL Live Translator")

st.title("🤟 Live Indian Sign Language Translator")
st.write("Show your hand signs in front of the camera")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -------- Gesture Logic -------- #
def detect_gesture(hand_landmarks):

    landmarks = hand_landmarks.landmark

    fingers = []

    # Thumb
    if landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        if landmarks[tip].y < landmarks[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Gesture mapping
    if fingers == [0,1,0,0,0]:
        return "YOU"
    elif fingers == [0,1,1,0,0]:
        return "HELP"
    elif fingers == [1,1,1,1,1]:
        return "HELLO"
    elif fingers == [0,0,0,0,0]:
        return "STOP"
    elif fingers == [1,0,0,0,0]:
        return "GOOD"
    else:
        return ""

# -------- Video Processor -------- #
class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=1
        )
        self.last_word = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)

        word = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                word = detect_gesture(hand_landmarks)

                if word != "":
                    cv2.putText(img, word, (30,70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0,255,0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# WebRTC Camera
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="example",
    video_processor_factory=SignProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

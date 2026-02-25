import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Text to Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Simple rule-based gesture prediction
def predict_gesture(fingers):
    if fingers == [0,1,0,0,0]:
        return "YOU"
    elif fingers == [0,1,1,0,0]:
        return "HELP"
    elif fingers == [0,1,1,1,1]:
        return "HELLO"
    elif fingers == [0,0,0,0,0]:
        return "STOP"
    elif fingers == [1,1,1,1,1]:
        return "HI"
    else:
        return ""

st.title("🤟 Sign Language to Text & Speech Translator")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

last_word = ""

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    word = ""

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, c = frame.shape

            for id, lm in enumerate(handLms.landmark):
                lmList.append((int(lm.x * w), int(lm.y * h)))

            fingers = []

            # Thumb
            if lmList[4][0] > lmList[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            tips = [8, 12, 16, 20]
            for tip in tips:
                if lmList[tip][1] < lmList[tip - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            word = predict_gesture(fingers)

            if word != "" and word != last_word:
                speak(word)
                last_word = word

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Text: {word}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()

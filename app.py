# app.py
import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Live ISL Translator", layout="centered")
st.title("🤟 Live Indian Sign Language → Text → Speech")
st.markdown("Allow camera access when the browser asks. Speak output uses the browser's speech API.")

# MediaPipe setup (mp.solutions.hands)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------- Gesture detection (rule-based) ----------
def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: simple x comparison (works for typical camera mirror)
    try:
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    except:
        fingers.append(0)

    # Other fingers: tip y < pip y => finger up
    tips = [8, 12, 16, 20]
    for t in tips:
        try:
            fingers.append(1 if lm[t].y < lm[t - 2].y else 0)
        except:
            fingers.append(0)

    # Map gestures (tweak or extend as you record more data)
    if fingers == [0,1,0,0,0]:
        return "YOU"
    if fingers == [0,1,1,0,0]:
        return "HELP"
    if fingers == [1,1,1,1,1]:
        return "HELLO"
    if fingers == [0,0,0,0,0]:
        return "STOP"
    if fingers == [1,0,0,0,0]:
        return "GOOD"
    if fingers == [0,1,0,0,1]:
        return "THANKS"

    return ""

# ---------- Video processor ----------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.last_word = ""
        self._candidate = ""
        self._candidate_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)
        word = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                word = detect_gesture(hand_landmarks)

        # Debounce: candidate must persist for 3 frames
        if word == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = word
            self._candidate_count = 1

        if self._candidate_count >= 3 and self._candidate != self.last_word:
            self.last_word = self._candidate

        if self.last_word:
            cv2.putText(img, self.last_word, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Start WebRTC ----------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="isl-translator",
    video_processor_factory=SignProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# UI: poll detected word and call browser TTS
status_box = st.empty()
speak_box = st.empty()

if webrtc_ctx.video_processor:
    last_spoken = st.session_state.get("last_spoken_word", "")
    try:
        while webrtc_ctx.state.playing:
            proc = webrtc_ctx.video_processor
            current = getattr(proc, "last_word", "")
            if current and current != last_spoken:
                status_box.markdown(f"### Detected: **{current}**")
                # Browser TTS via injected JS
                js = f"""
                <script>
                (function() {{
                    const w = "{current}".replace(/"/g, '\\"');
                    var msg = new SpeechSynthesisUtterance(w);
                    msg.rate = 0.95;
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(msg);
                }})();
                </script>
                """
                speak_box.markdown(js, unsafe_allow_html=True)
                last_spoken = current
                st.session_state["last_spoken_word"] = last_spoken
            time.sleep(0.35)
    except Exception:
        pass
else:
    st.info("Click 'Start' in the video area to enable camera stream.")

"""
ISL Sign Language Translator - Professional Web Application
Real-time hand gesture recognition with speech output and W&B tracking
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from tensorflow import keras
import wandb
from datetime import datetime
import os
from gtts import gTTS
import tempfile
import base64

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ISL Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00A67E;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #00A67E 0%, #00875A 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        background: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== W&B INITIALIZATION ====================
@st.cache_resource
def initialize_wandb():
    try:
        wandb.login(key="YOUR_WANDB_API_KEY")
        run = wandb.init(
            project="isl-sign-language-translator",
            reinit=True
        )
        return run
    except:
        return None

wandb_run = initialize_wandb()

# ==================== ISL APPLICATION CLASS ====================
class ISLTranslator:

    def __init__(self):
        self.model = None
        self.labels = None

        # MediaPipe initialization - FIXED INDENTATION
        self.mp_hands = solutions.hands
        self.mp_draw = solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.total_predictions = 0
        self.confidence_scores = []
        self.prediction_history = []

        self.last_spoken = None
        self.speak_cooldown = 0

    @st.cache_resource
    def load_model(_self):
        try:
            model = keras.models.load_model("isl_model.h5", compile=False)

            if os.path.exists("isl_labels.npy"):
                labels = np.load("isl_labels.npy", allow_pickle=True).tolist()
            else:
                labels = [str(i) for i in range(1, 10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

            return model, labels
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None, None

    def text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file = fp.name

            with open(audio_file, "rb") as f:
                audio_bytes = f.read()

            audio_base64 = base64.b64encode(audio_bytes).decode()
            st.markdown(
                f"""
                <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True,
            )

            os.unlink(audio_file)
        except:
            pass

    def predict_gesture(self, features):
        features = features.reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)
        idx = np.argmax(prediction[0])
        confidence = float(prediction[0][idx])
        label = self.labels[idx]

        self.total_predictions += 1
        self.confidence_scores.append(confidence)
        self.prediction_history.append(
            {"label": label, "confidence": confidence, "time": datetime.now()}
        )

        if wandb_run:
            wandb.log({"prediction": label, "confidence": confidence})

        return label, confidence

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features)

            if len(features) == 63:
                return frame, *self.predict_gesture(features)

        return frame, None, 0.0

# ==================== MAIN APPLICATION ====================
def main():
    st.markdown('<div class="main-header">🤟 ISL Sign Language Translator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Indian Sign Language Recognition</div>', unsafe_allow_html=True)
    st.markdown("---")

    app = ISLTranslator()
    app.model, app.labels = app.load_model()

    if app.model is None:
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        start = st.checkbox("Start Camera")
        frame_window = st.image([])

        if start:
            cap = cv2.VideoCapture(0)

            while start:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame, pred, conf = app.process_frame(frame)

                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if pred and conf > 0.7:
                    st.markdown(
                        f'<div class="prediction-box">{pred}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="confidence-box">Confidence: {conf:.1%}</div>',
                        unsafe_allow_html=True
                    )

            cap.release()

    with col2:
        st.subheader("📊 Statistics")
        st.metric("Total Predictions", app.total_predictions)

# ==================== RUN ====================
if __name__ == "__main__":
    main()
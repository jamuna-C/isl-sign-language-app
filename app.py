import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from gtts import gTTS
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detector",
    page_icon="🤟",
    layout="wide"
)

# Title and description
st.title("🤟 ISL Sign Language Detector")
st.markdown("AI-Powered Hand Gesture Recognition with Voice Output 🔊")

# Initialize MediaPipe - CORRECTED
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('isl_model.h5')
        labels = np.load('isl_labels.npy')
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_model()

if model is None:
    st.error("⚠️ Model Loading Error")
    st.info("Please check that model files exist.")
    st.stop()

# Camera input
st.sidebar.header("Controls")
enable_camera = st.sidebar.checkbox("Enable Camera", value=False)

if enable_camera:
    img_file_buffer = st.camera_input("Capture a sign")
    
    if img_file_buffer is not None:
        # Convert to opencv format
        bytes_data = img_file_buffer.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        img_array = np.array(image)
        
        # Process with MediaPipe - CORRECTED
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        img_array,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract landmarks for prediction
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Make prediction
                    landmarks = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(landmarks, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Display results
                    st.image(img_array, caption="Processed Image", use_container_width=True)
                    st.success(f"Detected Sign: **{labels[predicted_class]}**")
                    st.info(f"Confidence: {confidence*100:.2f}%")
                    
                    # Text-to-speech
                    if st.button("🔊 Play Audio"):
                        tts = gTTS(text=str(labels[predicted_class]), lang='en')
                        tts.save("output.mp3")
                        st.audio("output.mp3")
            else:
                st.warning("No hand detected. Please try again.")
else:
    st.info("👈 Enable camera from the sidebar to start detection.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, MediaPipe, and TensorFlow")

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from gtts import gTTS
from PIL import Image
import io

# Initialize MediaPipe AFTER importing it
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Page config
st.set_page_config(
    page_title="ISL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Title
st.title("🤟 ISL Sign Language Recognition")
st.markdown("AI-Powered Hand Gesture Recognition with Voice Output 🔊")

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

if model is None or labels is None:
    st.error("Failed to load model or labels. Please check your files.")
    st.stop()

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
st.sidebar.info("Adjust the confidence threshold to control prediction sensitivity.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Camera Input")
    camera_input = st.camera_input("Capture a sign gesture")

with col2:
    st.subheader("🎯 Prediction")
    prediction_placeholder = st.empty()
    audio_placeholder = st.empty()

# Process image
if camera_input is not None:
    try:
        # Convert to PIL Image
        image = Image.open(camera_input)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Process with MediaPipe
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            
            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Pad or truncate to match model input (63 features for 21 landmarks * 3)
                landmarks = landmarks[:63] + [0] * max(0, 63 - len(landmarks))
                
                # Make prediction
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                if confidence >= confidence_threshold:
                    predicted_label = labels[predicted_class]
                    
                    # Display prediction
                    prediction_placeholder.success(
                        f"**Prediction:** {predicted_label}\n\n**Confidence:** {confidence:.2%}"
                    )
                    
                    # Generate audio
                    try:
                        tts = gTTS(text=str(predicted_label), lang='en')
                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        audio_placeholder.audio(audio_buffer, format='audio/mp3')
                    except Exception as e:
                        st.warning(f"Could not generate audio: {e}")
                    
                    # Draw landmarks on image
                    annotated_image = img_array.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                    
                    # Display annotated image
                    st.image(annotated_image, caption="Detected Hand Landmarks", use_container_width=True)
                else:
                    prediction_placeholder.warning(
                        f"Low confidence prediction ({confidence:.2%}). Please try again with a clearer gesture."
                    )
            else:
                prediction_placeholder.error("❌ No hand detected. Please ensure your hand is clearly visible in the frame.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("*Powered by TensorFlow, MediaPipe, and Streamlit*")

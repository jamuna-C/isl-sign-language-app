import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from PIL import Image
from gtts import gTTS
from io import BytesIO
import base64

# Page config
st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")

# Title
st.title("🤟 ISL Sign Language Recognition")
st.markdown("AI-Powered Hand Gesture Recognition with Voice Output 🔊")

# Initialize MediaPipe - CORRECTED
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
@st.cache_resource
def load_model():
    model = keras.models.load_model('isl_model.h5')
    labels = np.load('isl_labels.npy')
    return model, labels

try:
    model, labels = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
enable_voice = st.sidebar.checkbox("Enable Voice Output 🔊", value=True)

# Function to process hand landmarks
def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).reshape(1, -1)

# Function to generate voice
def generate_voice(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Camera Input")
    camera_input = st.camera_input("Capture a hand gesture")

with col2:
    st.subheader("🎯 Prediction Results")
    result_placeholder = st.empty()
    confidence_placeholder = st.empty()
    voice_placeholder = st.empty()

# Process image
if camera_input:
    # Convert to PIL Image
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            annotated_image = img_array.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            # Display annotated image
            col1.image(annotated_image, caption="Detected Hand", use_container_width=True)
            
            # Extract landmarks and predict
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = extract_landmarks(hand_landmarks)
            
            # Make prediction
            predictions = model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Display results
            if confidence >= confidence_threshold:
                predicted_label = labels[predicted_class]
                result_placeholder.success(f"**Prediction:** {predicted_label}")
                confidence_placeholder.info(f"**Confidence:** {confidence:.2%}")
                
                # Generate and play voice
                if enable_voice:
                    try:
                        audio_fp = generate_voice(predicted_label)
                        audio_bytes = audio_fp.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        </audio>
                        """
                        voice_placeholder.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Voice generation failed: {str(e)}")
            else:
                result_placeholder.warning(f"⚠️ Low confidence: {confidence:.2%}")
        else:
            col1.image(img_array, caption="Original Image", use_container_width=True)
            result_placeholder.error("❌ No hand detected! Please try again.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Click the **camera button** above to capture your hand gesture
    2. Make sure your hand is clearly visible and well-lit
    3. The AI will detect your hand and predict the ISL sign
    4. If voice is enabled, you'll hear the prediction
    5. Adjust the confidence threshold in the sidebar if needed
    
    **Tips for best results:**
    - Ensure good lighting
    - Keep your hand centered in the frame
    - Make clear, distinct gestures
    - Try different angles if detection fails
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, MediaPipe, and TensorFlow</p>
    </div>
    """,
    unsafe_allow_html=True
)

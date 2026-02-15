import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from PIL import Image
import io
from gtts import gTTS
import base64
import tempfile
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ISL Sign Language Detector",
    page_icon="🤟",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
    }
    
    .instructions-box {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .instructions-title {
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .instruction-item {
        color: #b0b0b0;
        font-size: 1.1rem;
        margin: 8px 0;
    }
    
    .webcam-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8a 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    
    .webcam-title {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .webcam-subtitle {
        color: #64b5f6;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .result-text {
        color: white;
        font-size: 4rem;
        font-weight: 900;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
    }
    
    .voice-indicator {
        background-color: #4caf50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        font-size: 1.2rem;
        animation: glow 1.5s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px #4caf50; }
        50% { box-shadow: 0 0 20px #4caf50, 0 0 30px #4caf50; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================
@st.cache_resource
def init_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        return mp_hands, mp_drawing, mp_drawing_styles, True
    except Exception as e:
        st.error(f"MediaPipe Error: {e}")
        return None, None, None, False

mp_hands, mp_drawing, mp_drawing_styles, MEDIAPIPE_OK = init_mediapipe()

# ============================================================================
# LOAD AI MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('isl_model.h5')
        labels = np.load('isl_labels.npy', allow_pickle=True)
        return model, labels, True
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, False

model, labels, MODEL_OK = load_model()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_landmarks(hand_landmarks):
    """Extract hand landmarks for AI"""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).reshape(1, -1)

def predict_gesture(landmarks, model, labels, threshold=0.70):
    """Predict ISL gesture"""
    try:
        predictions = model.predict(landmarks, verbose=0)
        confidence = np.max(predictions)
        class_idx = np.argmax(predictions)
        
        if confidence >= threshold:
            return labels[class_idx], confidence
        return None, confidence
    except:
        return None, 0.0

def text_to_speech(text):
    """Convert text to speech and return audio HTML"""
    try:
        # Create speech
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            
            # Read the audio file
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up temp file
            os.unlink(fp.name)
            
            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Create HTML audio player with autoplay
            audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """
            return audio_html
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None

def process_image(image, mp_hands, mp_drawing, mp_drawing_styles, model, labels):
    """Process uploaded image or webcam capture"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Convert back to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Predict
                landmarks = extract_landmarks(hand_landmarks)
                gesture, confidence = predict_gesture(landmarks, model, labels)
                
                return img_rgb, gesture, confidence, True
        
        return img_rgb, None, 0.0, False

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">📋 Instructions</div>
    <div class="instruction-item">1. Use camera or upload image</div>
    <div class="instruction-item">2. Ensure hand is clearly visible</div>
    <div class="instruction-item">3. Make ISL gesture</div>
    <div class="instruction-item">4. AI will detect & speak the sign! 🔊</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">📚 Supported Signs:</div>
    <div class="instruction-item">• <strong>Numbers:</strong> 1-9</div>
    <div class="instruction-item">• <strong>Alphabets:</strong> A-Z</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">🤖 AI Model Info</div>
    <div class="instruction-item">• Model: Deep Neural Network</div>
    <div class="instruction-item">• Signs: 35 ISL gestures</div>
    <div class="instruction-item">• Accuracy: ~95%</div>
    <div class="instruction-item">• Framework: TensorFlow/Keras</div>
    <div class="instruction-item">• Voice: Text-to-Speech 🔊</div>
</div>
""", unsafe_allow_html=True)

# Voice output toggle
enable_voice = st.sidebar.checkbox("🔊 Enable Voice Output", value=True, 
                                   help="Speak the detected sign")

# ============================================================================
# MAIN UI
# ============================================================================
st.markdown('<h1 style="text-align: center; color: #667eea;">🤟 ISL Sign Language Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; font-size: 1.2rem;">AI-Powered Hand Gesture Recognition with Voice Output 🔊</p>', unsafe_allow_html=True)

# Check if model loaded
if not MODEL_OK or not MEDIAPIPE_OK:
    st.error("⚠️ System not ready. Please ensure model files are present.")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📸 Camera Capture", "📁 Upload Image"])

# ============================================================================
# CAMERA CAPTURE TAB
# ============================================================================
with tab1:
    st.markdown("""
    <div class="webcam-container">
        <div class="webcam-title">📹 Live Webcam Detection</div>
        <div class="webcam-subtitle">
            Use the camera widget below to capture your hand gesture
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera input widget
    camera_photo = st.camera_input("📸 Take a photo of your hand gesture")
    
    if camera_photo is not None:
        # Open image
        image = Image.open(camera_photo)
        
        # Process image
        processed_img, gesture, confidence, hand_detected = process_image(
            image, mp_hands, mp_drawing, mp_drawing_styles, model, labels
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Captured Image")
            st.image(processed_img, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detection Result")
            
            if hand_detected and gesture:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-text">{gesture}</div>
                    <div style="color: white; font-size: 1.3rem; margin-top: 10px;">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Voice output
                if enable_voice:
                    st.markdown('<div class="voice-indicator">🔊 Speaking the detected sign...</div>', 
                              unsafe_allow_html=True)
                    speech_text = f"The detected sign is {gesture}"
                    audio_html = text_to_speech(speech_text)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                
            elif hand_detected:
                st.warning(f"👋 Hand detected but low confidence ({confidence:.1%})")
                st.info("Try:\n- Better lighting\n- Clearer gesture\n- Different angle")
            else:
                st.error("❌ No hand detected in image")
                st.info("Make sure your hand is clearly visible")

# ============================================================================
# UPLOAD IMAGE TAB
# ============================================================================
with tab2:
    st.markdown("""
    <div class="webcam-container">
        <div class="webcam-title">📁 Upload Image</div>
        <div class="webcam-subtitle">
            Upload an image of ISL hand gesture
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of an ISL hand gesture"
    )
    
    if uploaded_file is not None:
        # Open image
        image = Image.open(uploaded_file)
        
        # Process image
        processed_img, gesture, confidence, hand_detected = process_image(
            image, mp_hands, mp_drawing, mp_drawing_styles, model, labels
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(processed_img, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detection Result")
            
            if hand_detected and gesture:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-text">{gesture}</div>
                    <div style="color: white; font-size: 1.3rem; margin-top: 10px;">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Voice output
                if enable_voice:
                    st.markdown('<div class="voice-indicator">🔊 Speaking the detected sign...</div>', 
                              unsafe_allow_html=True)
                    speech_text = f"The detected sign is {gesture}"
                    audio_html = text_to_speech(speech_text)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                
            elif hand_detected:
                st.warning(f"👋 Hand detected but low confidence ({confidence:.1%})")
                st.info("Try a different image with:\n- Better lighting\n- Clearer gesture\n- Different angle")
            else:
                st.error("❌ No hand detected in image")
                st.info("Upload an image with a clearly visible hand gesture")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p style="font-size: 1.1rem;">
        <strong>🤖 Powered by Artificial Intelligence + Voice Output 🔊</strong>
    </p>
    <p>Built with ❤️ using <strong>Streamlit</strong> • <strong>MediaPipe</strong> • <strong>TensorFlow</strong> • <strong>gTTS</strong></p>
    <p style="font-size: 0.9rem; color: #999;">
        Deep Learning Model trained on 35 Indian Sign Language gestures with Text-to-Speech
    </p>
</div>
""", unsafe_allow_html=True)

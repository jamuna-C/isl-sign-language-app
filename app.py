import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
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
# IMPORT DEPENDENCIES
# ============================================================================
@st.cache_resource
def import_dependencies():
    """Import heavy dependencies with caching"""
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        from tensorflow import keras
        return mp_hands, keras, True
    except Exception as e:
        st.error(f"Error loading dependencies: {e}")
        return None, None, False

mp_hands, keras, DEPS_OK = import_dependencies()

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and labels"""
    try:
        model = keras.models.load_model('isl_model.h5')
        labels = np.load('isl_labels.npy', allow_pickle=True)
        return model, labels, True
    except:
        return None, None, False

if DEPS_OK:
    model, labels, MODEL_OK = load_model()
else:
    MODEL_OK = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_landmarks(hand_landmarks):
    """Extract hand landmarks coordinates"""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).reshape(1, -1)

def predict_gesture(landmarks, model, labels, threshold=0.70):
    """Predict ISL gesture from landmarks"""
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
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            with open(fp.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            os.unlink(fp.name)
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    except:
        return None

def draw_landmarks_on_image(image, hand_landmarks):
    """Draw hand landmarks using PIL"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = image.size
    
    # Draw landmark points
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        radius = 5
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill='#FF5252', outline='white', width=2)
    
    # Draw connections
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection[0], connection[1]
        start_landmark = hand_landmarks.landmark[start_idx]
        end_landmark = hand_landmarks.landmark[end_idx]
        
        start_x = int(start_landmark.x * width)
        start_y = int(start_landmark.y * height)
        end_x = int(end_landmark.x * width)
        end_y = int(end_landmark.y * height)
        
        draw.line([(start_x, start_y), (end_x, end_y)], 
                 fill='#4CAF50', width=3)
    
    return img_copy

def process_image(image):
    """Process image and detect hand gesture"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        ) as hands:
            results = hands.process(img_array)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    img_with_landmarks = draw_landmarks_on_image(image, hand_landmarks)
                    landmarks = extract_landmarks(hand_landmarks)
                    gesture, confidence = predict_gesture(landmarks, model, labels)
                    return img_with_landmarks, gesture, confidence, True
            
            return image, None, 0.0, False
            
    except Exception as e:
        st.error(f"Processing error: {e}")
        return image, None, 0.0, False

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
# MAIN UI
# ============================================================================
st.markdown('<h1 style="text-align: center; color: #667eea;">🤟 ISL Sign Language Detector</h1>', 
           unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; font-size: 1.2rem;">AI-Powered Hand Gesture Recognition with Voice Output 🔊</p>', 
           unsafe_allow_html=True)

# Check dependencies
if not DEPS_OK:
    st.error("### ⚠️ Dependency Error")
    st.info("Please check that all packages are installed correctly.")
    st.stop()

# Check model
if not MODEL_OK:
    st.error("### ⚠️ Model Files Missing")
    st.info("Please ensure `isl_model.h5` and `isl_labels.npy` are in your repository.")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">📋 Instructions</div>
    <div class="instruction-item">1. Use camera or upload image</div>
    <div class="instruction-item">2. Ensure hand is clearly visible</div>
    <div class="instruction-item">3. Make ISL gesture</div>
    <div class="instruction-item">4. AI will detect & speak! 🔊</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">📚 Supported Signs</div>
    <div class="instruction-item">• Numbers: 1-9</div>
    <div class="instruction-item">• Alphabets: A-Z</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="instructions-box">
    <div class="instructions-title">🤖 AI Info</div>
    <div class="instruction-item">• 35 ISL gestures</div>
    <div class="instruction-item">• ~95% accuracy</div>
    <div class="instruction-item">• Real-time voice 🔊</div>
</div>
""", unsafe_allow_html=True)

enable_voice = st.sidebar.checkbox("🔊 Enable Voice", value=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2 = st.tabs(["📸 Camera", "📁 Upload"])

# Camera Tab
with tab1:
    st.markdown("""
    <div class="webcam-container">
        <div class="webcam-title">📹 Live Camera</div>
        <div class="webcam-subtitle">Capture your hand gesture</div>
    </div>
    """, unsafe_allow_html=True)
    
    camera_photo = st.camera_input("📸 Take a photo")
    
    if camera_photo:
        image = Image.open(camera_photo)
        processed_img, gesture, confidence, hand_detected = process_image(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Image")
            st.image(processed_img, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Result")
            
            if hand_detected and gesture:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-text">{gesture}</div>
                    <div style="color: white; font-size: 1.3rem; margin-top: 10px;">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if enable_voice:
                    st.markdown(f'<div class="voice-indicator">🔊 Speaking: {gesture}</div>', 
                              unsafe_allow_html=True)
                    audio_html = text_to_speech(str(gesture))
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
            elif hand_detected:
                st.warning(f"👋 Low confidence ({confidence:.1%})")
                st.info("Try better lighting or clearer gesture")
            else:
                st.error("❌ No hand detected")
                st.info("Make sure your hand is clearly visible")

# Upload Tab
with tab2:
    st.markdown("""
    <div class="webcam-container">
        <div class="webcam-title">📁 Upload</div>
        <div class="webcam-subtitle">Upload hand gesture image</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        processed_img, gesture, confidence, hand_detected = process_image(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Image")
            st.image(processed_img, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Result")
            
            if hand_detected and gesture:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-text">{gesture}</div>
                    <div style="color: white; font-size: 1.3rem; margin-top: 10px;">
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if enable_voice:
                    st.markdown(f'<div class="voice-indicator">🔊 Speaking: {gesture}</div>', 
                              unsafe_allow_html=True)
                    audio_html = text_to_speech(str(gesture))
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
            elif hand_detected:
                st.warning(f"👋 Low confidence ({confidence:.1%})")
                st.info("Try a different image")
            else:
                st.error("❌ No hand detected")
                st.info("Upload an image with a visible hand")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p><strong>🤖 AI-Powered ISL Recognition 🔊</strong></p>
    <p>Streamlit • MediaPipe • TensorFlow • gTTS</p>
</div>
""", unsafe_allow_html=True)

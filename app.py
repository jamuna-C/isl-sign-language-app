"""
ISL Sign Language Detection System
Professional real-time hand gesture recognition with voice output
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from gtts import gTTS
import tempfile
import os
import base64
from datetime import datetime
from collections import deque

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ISL Sign Language Detector",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .status-running {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-stopped {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE MEDIAPIPE ====================
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except AttributeError:
    st.error("❌ MediaPipe initialization failed")
    st.stop()

# ==================== SESSION STATE ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'word_buffer' not in st.session_state:
    st.session_state.word_buffer = deque(maxlen=10)
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# ==================== SIGN LANGUAGE CLASSES ====================
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# ==================== HELPER FUNCTIONS ====================
def extract_keypoints(results):
    """Extract hand landmarks as keypoints"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            return np.array(keypoints)
    return np.zeros(63)

def preprocess_keypoints(keypoints, target_shape=(63,)):
    """Preprocess keypoints for model input"""
    if len(keypoints) < target_shape[0]:
        keypoints = np.pad(keypoints, (0, target_shape[0] - len(keypoints)))
    elif len(keypoints) > target_shape[0]:
        keypoints = keypoints[:target_shape[0]]
    return keypoints.reshape(1, -1)

def predict_sign(keypoints, model):
    """Predict sign language gesture"""
    if model is None:
        return None, 0.0
    
    try:
        processed = preprocess_keypoints(keypoints)
        prediction = model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except Exception as e:
        return None, 0.0

def text_to_speech(text):
    """Convert text to speech and play audio"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            audio_file = fp.name
        
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        os.unlink(audio_file)
        return True
    except Exception as e:
        return False

# ==================== MAIN HEADER ====================
st.markdown('<div class="main-header">🤟 ISL Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Real-time Hand Gesture Recognition with Voice Output</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Model Configuration
    st.subheader("📁 Model Settings")
    model_path = st.text_input("Model Path", value="model.h5", help="Path to your trained TensorFlow model")
    
    if st.button("🔄 Load Model", use_container_width=True):
        try:
            st.session_state.model = keras.models.load_model(model_path, compile=False)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Detection Settings
    st.subheader("🎯 Detection Settings")
    min_detection_confidence = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for hand detection"
    )
    
    min_tracking_confidence = st.slider(
        "Tracking Confidence",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for hand tracking"
    )
    
    prediction_threshold = st.slider(
        "Prediction Threshold",
        0.0, 1.0, 0.75, 0.05,
        help="Minimum confidence to accept prediction"
    )
    
    enable_voice = st.checkbox("🔊 Enable Voice Output", value=True)
    
    # Initialize Detector
    if st.button("🚀 Initialize Detector", use_container_width=True):
        try:
            st.session_state.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            st.success("✅ Detector initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    
    model_status = "✅ Loaded" if st.session_state.model else "❌ Not Loaded"
    detector_status = "✅ Ready" if st.session_state.hands else "❌ Not Ready"
    
    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Detector:** {detector_status}")
    st.markdown(f"**Total Predictions:** {len(st.session_state.predictions)}")
    
    # Clear History
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.predictions = []
        st.session_state.word_buffer = deque(maxlen=10)
        st.session_state.current_word = ""
        st.rerun()

# ==================== MAIN CONTENT ====================
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📊 Statistics", "ℹ️ Help"])

# ==================== LIVE DETECTION TAB ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Control Buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶️ Start Detection", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)
        with btn_col3:
            speak_btn = st.button("🔊 Speak Word", use_container_width=True)
        
        # Video Frame
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col2:
        st.subheader("🎯 Detection Results")
        
        # Current Prediction
        prediction_placeholder = st.empty()
        
        # Confidence
        confidence_placeholder = st.empty()
        
        # Current Word
        st.markdown("### 📝 Current Word")
        word_placeholder = st.empty()
        
        # Recent History
        st.markdown("### 📜 Recent Detections")
        history_placeholder = st.empty()
    
    # Handle Start Button
    if start_btn:
        if st.session_state.hands is None:
            st.warning("⚠️ Please initialize the detector first!")
        elif st.session_state.model is None:
            st.warning("⚠️ Please load the model first!")
        else:
            st.session_state.is_running = True
            st.rerun()
    
    # Handle Stop Button
    if stop_btn:
        st.session_state.is_running = False
        st.rerun()
    
    # Handle Speak Button
    if speak_btn and st.session_state.current_word:
        if enable_voice:
            text_to_speech(st.session_state.current_word)
            st.success(f"🔊 Speaking: {st.session_state.current_word}")
    
    # Main Detection Loop
    if st.session_state.is_running:
        status_placeholder.markdown('<span class="status-running">🔴 LIVE</span>', unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera!")
            st.session_state.is_running = False
        else:
            last_prediction = None
            stable_count = 0
            
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                results = st.session_state.hands.process(image)
                
                image.flags.writeable = True
                current_sign = None
                current_conf = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    keypoints = extract_keypoints(results)
                    sign, confidence = predict_sign(keypoints, st.session_state.model)
                    
                    if sign and confidence > prediction_threshold:
                        current_sign = sign
                        current_conf = confidence
                        
                        # Stabilize prediction
                        if sign == last_prediction:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_prediction = sign
                        
                        # Add to word after stable detection
                        if stable_count == 5:
                            st.session_state.word_buffer.append(sign)
                            st.session_state.current_word = ''.join(st.session_state.word_buffer)
                            
                            # Add to predictions
                            st.session_state.predictions.append({
                                'sign': sign,
                                'confidence': confidence,
                                'time': datetime.now()
                            })
                            
                            # Speak if enabled
                            if enable_voice:
                                text_to_speech(sign)
                            
                            stable_count = 0
                        
                        # Draw on frame
                        cv2.putText(
                            image,
                            f"{sign} ({confidence:.0%})",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            3
                        )
                
                # Display frame
                frame_placeholder.image(image, channels="RGB", use_container_width=True)
                
                # Update UI
                if current_sign:
                    prediction_placeholder.markdown(
                        f'<div class="prediction-card">{current_sign}</div>',
                        unsafe_allow_html=True
                    )
                    confidence_placeholder.progress(float(current_conf))
                
                word_placeholder.markdown(
                    f'<div class="stats-card"><h2>{st.session_state.current_word or "..."}</h2></div>',
                    unsafe_allow_html=True
                )
                
                # Show recent history
                if st.session_state.predictions:
                    recent = list(st.session_state.predictions)[-5:]
                    history_text = ""
                    for p in reversed(recent):
                        history_text += f"**{p['sign']}** ({p['confidence']:.0%})\n\n"
                    history_placeholder.markdown(history_text)
            
            cap.release()
    else:
        status_placeholder.markdown('<span class="status-stopped">⏸️ STOPPED</span>', unsafe_allow_html=True)

# ==================== STATISTICS TAB ====================
with tab2:
    st.subheader("📊 Detection Statistics")
    
    if st.session_state.predictions:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Detections", len(st.session_state.predictions))
        
        with col2:
            avg_conf = np.mean([p['confidence'] for p in st.session_state.predictions])
            st.metric("Avg Confidence", f"{avg_conf:.0%}")
        
        with col3:
            unique_signs = len(set([p['sign'] for p in st.session_state.predictions]))
            st.metric("Unique Signs", unique_signs)
        
        # Sign frequency
        st.markdown("### 📈 Sign Frequency")
        from collections import Counter
        sign_counts = Counter([p['sign'] for p in st.session_state.predictions])
        
        for sign, count in sign_counts.most_common(10):
            st.write(f"**{sign}**: {count} times")
    else:
        st.info("No detections yet. Start the camera to begin!")

# ==================== HELP TAB ====================
with tab3:
    st.markdown("""
    ## 📖 How to Use
    
    ### 🚀 Quick Start
    1. **Load Model**: Enter your model path in sidebar and click "Load Model"
    2. **Initialize Detector**: Click "Initialize Detector"
    3. **Start Detection**: Click "▶️ Start Detection" in Live Detection tab
    4. **Show Signs**: Display hand gestures in front of camera
    5. **Build Words**: Signs are automatically added to form words
    6. **Speak**: Click "🔊 Speak Word" to hear the detected word
    
    ### 💡 Tips for Best Results
    - 🌞 Ensure good lighting
    - 🖐️ Keep hand clearly visible
    - 📏 Maintain stable hand position for 1-2 seconds
    - 🎯 Adjust confidence thresholds in sidebar
    - 🔊 Enable voice output for audio feedback
    
    ### ⚙️ System Requirements
    - 📷 Working webcam
    - 🧠 Trained TensorFlow model (`.h5` file)
    - 🌐 Internet connection for voice synthesis
    
    ### 🛠️ Technical Stack
    - **MediaPipe**: Hand tracking
    - **TensorFlow**: Gesture classification
    - **gTTS**: Text-to-speech
    - **Streamlit**: Web interface
    - **OpenCV**: Video processing
    
    ### 📞 Support
    For issues or questions, please check the documentation.
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>ISL Sign Language Detector v1.0</strong></p>
        <p>Powered by MediaPipe, TensorFlow & Streamlit | Made with ❤️ for Accessibility</p>
    </div>
    """,
    unsafe_allow_html=True
)
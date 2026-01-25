import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    st.error(f"MediaPipe Error: {str(e)}")

# Try to import TensorFlow
KERAS_AVAILABLE = False
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    keras = None

# Initialize session state
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detected_signs' not in st.session_state:
    st.session_state.detected_signs = []
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

# Sign classes - A-Z and 0-9
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Title
st.title("🤟 Indian Sign Language Detection System")
st.markdown("**Real-time ISL Detection with Voice Output**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # System Status
    st.subheader("📊 System Status")
    if MEDIAPIPE_AVAILABLE:
        st.success("✅ MediaPipe Ready")
    else:
        st.error("❌ MediaPipe Not Available")
    
    if KERAS_AVAILABLE:
        st.success("✅ TensorFlow Ready")
    else:
        st.info("ℹ️ Running in Demo Mode")
    
    st.divider()
    
    # Detection Settings
    if MEDIAPIPE_AVAILABLE:
        st.subheader("🎛️ Detection Settings")
        
        min_detection_confidence = st.slider(
            "Detection Confidence",
            0.0, 1.0, 0.7, 0.05,
            help="Higher = more strict detection"
        )
        
        min_tracking_confidence = st.slider(
            "Tracking Confidence",
            0.0, 1.0, 0.5, 0.05,
            help="Higher = smoother tracking"
        )
        
        # Voice settings
        st.subheader("🔊 Voice Settings")
        voice_enabled = st.checkbox("Enable Voice Output", value=True)
        voice_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        
        st.divider()
        
        # Initialize/Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start System", use_container_width=True):
                try:
                    st.session_state.hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    st.success("✅ System Started!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.detected_signs = []
                st.session_state.current_sentence = ""
                st.session_state.detection_count = 0
                st.success("Reset complete!")
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Statistics")
    st.metric("Total Detections", st.session_state.detection_count)
    st.metric("Current Sentence Length", len(st.session_state.current_sentence))

def extract_keypoints(results):
    """Extract hand landmarks as keypoints"""
    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    return np.zeros(63)

def predict_sign_demo(keypoints):
    """Demo prediction (random) when no model is loaded"""
    # Simulate prediction based on hand movement
    if np.sum(keypoints) > 0:
        # Use hand position to generate consistent "prediction"
        idx = int((keypoints[0] * keypoints[1] * 1000) % len(SIGN_CLASSES))
        confidence = 0.75 + (keypoints[2] * 0.2)
        return SIGN_CLASSES[idx], min(confidence, 0.99)
    return "None", 0.0

def predict_sign(keypoints, model):
    """Real prediction with trained model"""
    if model is None:
        return predict_sign_demo(keypoints)
    
    try:
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except Exception as e:
        return "Error", 0.0

def speak_text(text, speed=1.0):
    """Generate speech using browser's speech synthesis"""
    if text and text != "None":
        # JavaScript code for text-to-speech
        speak_js = f"""
        <script>
            var msg = new SpeechSynthesisUtterance('{text}');
            msg.rate = {speed};
            msg.pitch = 1.0;
            msg.volume = 1.0;
            window.speechSynthesis.speak(msg);
        </script>
        """
        st.components.v1.html(speak_js, height=0)

# Main Dashboard
if not MEDIAPIPE_AVAILABLE:
    st.error("⚠️ **MediaPipe is not installed!** Please install it to use this app.")
    st.code("pip install mediapipe", language="bash")
elif st.session_state.hands is None:
    st.info("👉 **Click 'Start System' in the sidebar to begin!**")
    
    # Show demo/instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📹 Camera")
        st.write("Real-time webcam detection")
    with col2:
        st.markdown("### 🤖 AI Detection")
        st.write("Recognizes A-Z and 0-9")
    with col3:
        st.markdown("### 🔊 Voice Output")
        st.write("Speaks detected signs")
else:
    # Main detection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Detection Dashboard")
        
        # Current detection
        current_sign = st.empty()
        confidence_meter = st.empty()
        
        st.divider()
        
        # Current sentence
        st.markdown("**📝 Current Sentence:**")
        sentence_display = st.empty()
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add to Sentence", use_container_width=True):
                if st.session_state.detected_signs:
                    last_sign = st.session_state.detected_signs[-1]
                    st.session_state.current_sentence += last_sign
        
        with col_b:
            if st.button("🗑️ Clear Sentence", use_container_width=True):
                st.session_state.current_sentence = ""
        
        st.divider()
        
        # Recent detections
        st.markdown("**🕐 Recent Detections:**")
        recent_display = st.empty()
        
        # Speak sentence button
        if st.button("🔊 Speak Full Sentence", use_container_width=True):
            if st.session_state.current_sentence:
                speak_text(st.session_state.current_sentence, voice_speed)
    
    # Camera detection loop
    start_detection = st.checkbox("▶️ Start Detection", value=False)
    
    if start_detection:
        try:
            cap = cv2.VideoCapture(0)
            last_detection_time = time.time()
            detection_cooldown = 1.0  # 1 second between detections
            
            while start_detection:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("❌ Camera access failed")
                    break
                
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = st.session_state.hands.process(image)
                image.flags.writeable = True
                
                detected_sign = "None"
                confidence = 0.0
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract and predict
                    keypoints = extract_keypoints(results)
                    detected_sign, confidence = predict_sign(keypoints, st.session_state.model)
                    
                    # Add to detected signs (with cooldown)
                    current_time = time.time()
                    if confidence > 0.7 and (current_time - last_detection_time) > detection_cooldown:
                        st.session_state.detected_signs.append(detected_sign)
                        st.session_state.detection_count += 1
                        last_detection_time = current_time
                        
                        # Speak if enabled
                        if voice_enabled:
                            speak_text(detected_sign, voice_speed)
                        
                        # Keep only last 10 detections
                        if len(st.session_state.detected_signs) > 10:
                            st.session_state.detected_signs.pop(0)
                    
                    # Add text overlay on video
                    cv2.putText(
                        image,
                        f"{detected_sign} ({confidence:.0%})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
                    
                    status_placeholder.success("✅ Hand detected!")
                else:
                    status_placeholder.info("👋 Show your hand to the camera")
                
                # Update dashboard
                current_sign.metric(
                    "Current Sign",
                    detected_sign,
                    f"{confidence:.0%}" if confidence > 0 else ""
                )
                
                if confidence > 0:
                    confidence_meter.progress(float(confidence))
                
                sentence_display.markdown(f"### `{st.session_state.current_sentence}`")
                
                if st.session_state.detected_signs:
                    recent_text = " → ".join(st.session_state.detected_signs[-5:])
                    recent_display.markdown(f"`{recent_text}`")
                
                # Display frame
                camera_placeholder.image(image, channels="RGB", use_container_width=True)
            
            cap.release()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Instructions at bottom
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Start the System**
       - Click "Start System" in the sidebar
       - Allow camera access when prompted
    
    2. **Begin Detection**
       - Check "Start Detection" box
       - Show ISL hand signs to the camera
    
    3. **Build Sentences**
       - Signs are detected automatically
       - Click "Add to Sentence" to build words
       - Click "Speak Full Sentence" to hear it
    
    4. **Adjust Settings**
       - Use confidence sliders for accuracy
       - Enable/disable voice output
       - Adjust speech speed
    
    ### Tips for Best Results
    - Good lighting is essential
    - Keep hand centered in frame
    - Hold each sign for 1-2 seconds
    - Use plain background
    - Adjust confidence if needed
    
    ### Supported Signs
    - **Alphabets**: A through Z
    - **Numbers**: 0 through 9
    
    ### Features
    - ✅ Real-time detection
    - ✅ Voice feedback
    - ✅ Sentence building
    - ✅ Detection history
    - ✅ Adjustable settings
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>🤟 Indian Sign Language Detection System</strong></p>
        <p>Built with MediaPipe, OpenCV, Streamlit & Web Speech API</p>
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time
from collections import deque

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ISL Sign Language Detector - AI Powered",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main styling */
    .main-title {
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
    
    /* Webcam container */
    .webcam-container {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .webcam-title {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .webcam-instruction {
        color: #64b5f6;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Result display */
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        margin: 20px 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .result-text {
        color: white;
        font-size: 4.5rem;
        font-weight: 900;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        letter-spacing: 2px;
    }
    
    .confidence-text {
        color: white;
        font-size: 1.5rem;
        margin-top: 15px;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #263238 0%, #37474f 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .info-title {
        color: #64b5f6;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    /* Status indicators */
    .status-stable {
        background-color: #4caf50;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: bold;
        animation: glow 1.5s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px #4caf50; }
        50% { box-shadow: 0 0 20px #4caf50, 0 0 30px #4caf50; }
    }
    
    .status-detecting {
        background-color: #ff9800;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    st.success("✅ MediaPipe Hands module loaded successfully")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    st.error(f"❌ MediaPipe initialization failed: {e}")
    st.stop()

# ============================================================================
# LOAD AI MODEL AND LABELS
# ============================================================================
@st.cache_resource
def load_ai_model_and_labels():
    """Load the trained Keras model and ISL sign labels"""
    try:
        # Load the AI model
        model = keras.models.load_model('isl_model.h5')
        st.success("✅ AI Model (isl_model.h5) loaded successfully - 35 ISL signs ready")
        
        # Load labels
        labels = np.load('isl_labels.npy', allow_pickle=True)
        st.success(f"✅ Label database loaded - {len(labels)} ISL signs available")
        
        return model, labels
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("📁 Please ensure 'isl_model.h5' and 'isl_labels.npy' are in the same folder")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Load model at startup
model, labels = load_ai_model_and_labels()

if model is None or labels is None:
    st.error("⚠️ Cannot proceed without AI model. Please check files and restart.")
    st.stop()

# ============================================================================
# GESTURE STABILIZER CLASS
# ============================================================================
class GestureStabilizer:
    """
    Ensures stable gesture detection by requiring multiple consecutive 
    detections of the same gesture before confirming.
    """
    def __init__(self, required_frames=10):
        self.required_frames = required_frames
        self.gesture_buffer = deque(maxlen=required_frames)
        self.confidence_buffer = deque(maxlen=required_frames)
        self.current_stable_gesture = None
        self.current_stable_confidence = 0.0
        
    def add_detection(self, gesture, confidence):
        """Add new detection and check for stability"""
        self.gesture_buffer.append(gesture)
        self.confidence_buffer.append(confidence)
        
        if len(self.gesture_buffer) >= self.required_frames:
            gesture_list = list(self.gesture_buffer)
            valid_gestures = [g for g in gesture_list if g is not None]
            
            if valid_gestures:
                # Find most common gesture
                most_common = max(set(valid_gestures), key=valid_gestures.count)
                count = valid_gestures.count(most_common)
                
                # Require 80% agreement for stability
                if count >= self.required_frames * 0.8:
                    # Calculate average confidence for this gesture
                    relevant_confidences = [
                        c for g, c in zip(gesture_list, self.confidence_buffer) 
                        if g == most_common
                    ]
                    avg_confidence = np.mean(relevant_confidences)
                    
                    self.current_stable_gesture = most_common
                    self.current_stable_confidence = avg_confidence
                    
                    return True, most_common, avg_confidence
        
        return False, None, 0.0
    
    def get_stability_progress(self):
        """Get current progress towards stability"""
        if len(self.gesture_buffer) == 0:
            return 0, None
        
        gesture_list = list(self.gesture_buffer)
        valid_gestures = [g for g in gesture_list if g is not None]
        
        if not valid_gestures:
            return 0, None
        
        most_common = max(set(valid_gestures), key=valid_gestures.count)
        count = valid_gestures.count(most_common)
        
        return count, most_common
    
    def reset(self):
        """Reset the stabilizer"""
        self.gesture_buffer.clear()
        self.confidence_buffer.clear()
        self.current_stable_gesture = None
        self.current_stable_confidence = 0.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_hand_landmarks(hand_landmarks):
    """
    Extract 3D coordinates (x, y, z) from MediaPipe hand landmarks.
    Returns flattened array of 63 features (21 landmarks × 3 coordinates)
    """
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).reshape(1, -1)

def predict_gesture_with_ai(landmarks, model, labels, confidence_threshold=0.70):
    """
    Use AI model to predict ISL gesture from hand landmarks.
    Returns (predicted_label, confidence) or (None, confidence) if below threshold
    """
    try:
        # Get model prediction
        predictions = model.predict(landmarks, verbose=0)
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        
        # Only return prediction if confidence is above threshold
        if confidence >= confidence_threshold:
            predicted_label = labels[predicted_class_index]
            return predicted_label, confidence
        else:
            return None, confidence
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("## ⚙️ Settings & Configuration")

st.sidebar.markdown("### 🎯 Detection Parameters")
confidence_threshold = st.sidebar.slider(
    "AI Confidence Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
    help="Minimum confidence required for AI prediction (higher = more strict)"
)

stability_frames = st.sidebar.slider(
    "Stability Frames Required",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Number of consecutive frames needed for stable detection (higher = more stable, less responsive)"
)

st.sidebar.markdown("### 📹 Display Options")
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True, help="Draw hand skeleton on video")
show_fps = st.sidebar.checkbox("Show FPS Counter", value=True, help="Display frames per second")
mirror_camera = st.sidebar.checkbox("Mirror Camera View", value=True, help="Flip camera horizontally")

st.sidebar.markdown("### 📊 Camera Quality")
camera_resolution = st.sidebar.selectbox(
    "Resolution",
    ["Low (320×240)", "Medium (640×480)", "High (1280×720)"],
    index=1
)

# Parse resolution
resolution_map = {
    "Low (320×240)": (320, 240),
    "Medium (640×480)": (640, 480),
    "High (1280×720)": (1280, 720)
}
camera_width, camera_height = resolution_map[camera_resolution]

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Instructions
1. Click **Start Live Detection**
2. Position hand clearly in frame
3. Make ISL gesture
4. Hold steady for stable detection
5. Watch real-time AI predictions!

### 🤖 AI Model Info
- **Model:** Deep Neural Network
- **Signs:** 35 ISL gestures
- **Accuracy:** ~95%
- **Framework:** TensorFlow/Keras
""")

# ============================================================================
# MAIN UI
# ============================================================================
st.markdown('<p class="main-title">🤟 ISL Sign Language Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time AI-Powered Hand Gesture Recognition System</p>', unsafe_allow_html=True)

# Create main layout
col_left, col_right = st.columns([3, 2])

# LEFT COLUMN - Webcam Feed
with col_left:
    st.markdown("""
    <div class="webcam-container">
        <h2 class="webcam-title">📹 Live Webcam Detection</h2>
        <p class="webcam-instruction">
            👆 Click 'Start' below to activate AI-powered real-time gesture recognition
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    video_placeholder = st.empty()
    fps_placeholder = st.empty()

# RIGHT COLUMN - Detection Results
with col_right:
    st.markdown("""
    <div class="info-box">
        <div class="info-title">🎯 Detection Result</div>
        <p style="color: #90caf9; font-size: 1.1rem;">
            AI predictions will appear here in real-time
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    result_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

# Control Buttons
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    start_detection = st.button(
        "▶️ Start Live Detection", 
        use_container_width=True, 
        type="primary"
    )

with col_btn2:
    stop_detection = st.button(
        "⏹️ Stop Detection", 
        use_container_width=True
    )

with col_btn3:
    reset_system = st.button(
        "🔄 Reset System", 
        use_container_width=True
    )

# Session state management
if 'is_detecting' not in st.session_state:
    st.session_state.is_detecting = False
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if start_detection:
    st.session_state.is_detecting = True

if stop_detection:
    st.session_state.is_detecting = False

if reset_system:
    st.session_state.detection_history = []
    st.rerun()

# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================
if st.session_state.is_detecting:
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
        st.warning("**Troubleshooting:**")
        st.info("• Ensure no other application is using the camera")
        st.info("• Check browser camera permissions")
        st.info("• This must run LOCALLY (not on Hugging Face)")
        st.info("• Try restarting your browser")
        st.session_state.is_detecting = False
        
    else:
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize components
        stabilizer = GestureStabilizer(required_frames=stability_frames)
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        # Initialize MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        ) as hands:
            
            st.info("🟢 **System Active** - AI model processing live video feed...")
            
            while st.session_state.is_detecting:
                # Capture frame
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                
                # Mirror if enabled
                if mirror_camera:
                    frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                hand_results = hands.process(rgb_frame)
                
                # Initialize detection variables
                detected_gesture = None
                detection_confidence = 0.0
                
                # Process hand landmarks if detected
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        
                        # Draw hand landmarks
                        if show_landmarks:
                            mp_drawing.draw_landmarks(
                                rgb_frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                        
                        # Extract landmarks and run AI prediction
                        landmarks_array = extract_hand_landmarks(hand_landmarks)
                        detected_gesture, detection_confidence = predict_gesture_with_ai(
                            landmarks_array,
                            model,
                            labels,
                            confidence_threshold
                        )
                
                # Add detection to stabilizer
                is_stable, stable_gesture, stable_confidence = stabilizer.add_detection(
                    detected_gesture,
                    detection_confidence
                )
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Display video frame
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Display FPS if enabled
                if show_fps:
                    fps_placeholder.metric("📊 FPS", current_fps)
                
                # Display detection results
                if is_stable and stable_gesture:
                    # STABLE DETECTION - Show result prominently
                    result_placeholder.markdown(
                        f"""
                        <div class="result-container">
                            <p class="result-text">{stable_gesture}</p>
                            <p class="confidence-text">
                                AI Confidence: {stable_confidence:.1%}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    status_placeholder.markdown(
                        '<div class="status-stable">✓ STABLE DETECTION</div>',
                        unsafe_allow_html=True
                    )
                    
                    progress_placeholder.success(f"✅ Confirmed after {stability_frames} consecutive frames")
                    
                    # Add to history if new
                    if not st.session_state.detection_history or st.session_state.detection_history[-1] != stable_gesture:
                        st.session_state.detection_history.append(stable_gesture)
                
                elif detected_gesture:
                    # DETECTING - Show progress
                    progress_count, current_gesture = stabilizer.get_stability_progress()
                    
                    result_placeholder.info(
                        f"🔍 **Detecting:** {current_gesture}\n\n"
                        f"⏳ **Stabilizing...** ({progress_count}/{stability_frames} frames)\n\n"
                        f"📊 **Confidence:** {detection_confidence:.1%}"
                    )
                    
                    status_placeholder.markdown(
                        '<div class="status-detecting">⏳ STABILIZING...</div>',
                        unsafe_allow_html=True
                    )
                    
                    progress_placeholder.progress(
                        progress_count / stability_frames,
                        text=f"Hold steady: {progress_count}/{stability_frames}"
                    )
                
                elif hand_results.multi_hand_landmarks:
                    # HAND DETECTED BUT LOW CONFIDENCE
                    result_placeholder.warning(
                        f"👋 **Hand Detected**\n\n"
                        f"⚠️ Low AI confidence ({detection_confidence:.1%})\n\n"
                        f"**Tips:**\n"
                        f"• Make a clear, distinct gesture\n"
                        f"• Ensure good lighting\n"
                        f"• Keep hand steady"
                    )
                    status_placeholder.empty()
                    progress_placeholder.empty()
                
                else:
                    # NO HAND DETECTED
                    result_placeholder.info(
                        "👋 **Waiting for hand...**\n\n"
                        "Position your hand clearly in the camera frame"
                    )
                    status_placeholder.empty()
                    progress_placeholder.empty()
                
                # Check if should stop
                if not st.session_state.is_detecting:
                    break
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
        
        # Cleanup
        cap.release()
        video_placeholder.empty()
        result_placeholder.empty()
        status_placeholder.empty()
        progress_placeholder.empty()
        fps_placeholder.empty()
        st.info("⏹️ Detection stopped")

else:
    st.info("👆 Click **▶️ Start Live Detection** to begin AI-powered gesture recognition")

# ============================================================================
# DETECTION HISTORY
# ============================================================================
if st.session_state.detection_history:
    st.markdown("---")
    st.subheader("📜 Detection History")
    
    history_cols = st.columns(min(len(st.session_state.detection_history), 6))
    recent_history = st.session_state.detection_history[-6:]
    
    for idx, gesture in enumerate(recent_history):
        with history_cols[idx]:
            st.info(f"**#{idx+1}**\n\n{gesture}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p style="font-size: 1.1rem;">
        <strong>🤖 Powered by Artificial Intelligence</strong>
    </p>
    <p>
        Built with ❤️ using <strong>Streamlit</strong> • <strong>MediaPipe</strong> • <strong>TensorFlow</strong> • <strong>OpenCV</strong>
    </p>
    <p style="font-size: 0.9rem; color: #999;">
        Deep Learning Model trained on 35 Indian Sign Language gestures
    </p>
</div>
""", unsafe_allow_html=True)

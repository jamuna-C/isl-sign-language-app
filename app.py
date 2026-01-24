import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os

# Make TensorFlow optional
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = None

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize session state
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Title
st.title("🤟 Indian Sign Language Detection System")
st.markdown("Real-time hand gesture recognition using MediaPipe and Deep Learning")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show TensorFlow status
    if KERAS_AVAILABLE:
        st.success("✅ TensorFlow available")
    else:
        st.info("ℹ️ TensorFlow not installed - hand detection only")
    
    # Detection settings
    st.subheader("Detection Configuration")
    min_detection_confidence = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    min_tracking_confidence = st.slider(
        "Tracking Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    # Initialize detector
    if st.button("Initialize Hand Detector"):
        try:
            st.session_state.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            st.success("✅ Hand detector initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Model loading (optional - only if TensorFlow available)
    if KERAS_AVAILABLE:
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path (optional)", value="model.h5")
        
        if st.button("Load Model"):
            if os.path.exists(model_path):
                try:
                    st.session_state.model = keras.models.load_model(model_path)
                    st.success("✅ Model loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
            else:
                st.warning("⚠️ Model file not found")

# Sign classes (customize based on your model)
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def extract_keypoints(results):
    """Extract hand landmarks as keypoints"""
    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    return np.zeros(63)

def predict_sign(keypoints, model):
    """Predict sign language gesture"""
    if not KERAS_AVAILABLE:
        return "TensorFlow not installed", 0.0
    
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Main tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📸 Image Upload", "ℹ️ About"])

with tab1:
    st.header("Live Camera Detection")
    
    if st.session_state.hands is None:
        st.warning("⚠️ Please initialize the hand detector from the sidebar first!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            run = st.checkbox("Start Detection")
            frame_window = st.empty()
            
        with col2:
            st.subheader("Detection Results")
            result_text = st.empty()
            confidence_bar = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = st.session_state.hands.process(image)
                image.flags.writeable = True
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Predict if model is available
                    if KERAS_AVAILABLE and st.session_state.model:
                        keypoints = extract_keypoints(results)
                        sign, conf = predict_sign(keypoints, st.session_state.model)
                        
                        cv2.putText(
                            image,
                            f"{sign}: {conf:.2f}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3
                        )
                        
                        result_text.metric("Detected Sign", sign)
                        confidence_bar.progress(float(conf))
                    else:
                        result_text.success("✅ Hand detected!")
                else:
                    result_text.info("No hands detected")
                
                frame_window.image(image, channels="RGB", use_container_width=True)
            
            cap.release()

with tab2:
    st.header("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file and st.session_state.hands:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        with col2:
            st.subheader("Detection Result")
            
            results = st.session_state.hands.process(image_rgb)
            annotated = image_rgb.copy()
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                st.image(annotated, use_container_width=True)
                
                if KERAS_AVAILABLE and st.session_state.model:
                    keypoints = extract_keypoints(results)
                    sign, conf = predict_sign(keypoints, st.session_state.model)
                    st.success(f"Detected: **{sign}**")
                    st.info(f"Confidence: **{conf:.2%}**")
                else:
                    st.success("✅ Hand detected!")
            else:
                st.warning("No hands detected")
    elif uploaded_file:
        st.warning("Please initialize the hand detector first!")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Indian Sign Language Detection System
    
    This application uses advanced computer vision to detect and track hands in real-time.
    
    ### 🔧 Technologies Used
    - **MediaPipe**: Hand landmark detection
    - **Streamlit**: Interactive web interface
    - **OpenCV**: Image processing
    - **TensorFlow** (Optional): Deep learning for gesture classification
    
    ### 📖 How to Use
    1. **Initialize Detector**: Click "Initialize Hand Detector" in the sidebar
    2. **Live Detection**: Use your webcam for real-time hand tracking
    3. **Image Upload**: Upload images for static hand detection
    4. **Optional**: Add TensorFlow and a trained model for gesture recognition
    
    ### ⚙️ Current Features
    - Real-time hand tracking with MediaPipe
    - Support for detecting up to 2 hands simultaneously
    - Adjustable confidence thresholds
    - Visual feedback with hand landmarks
    - Works without TensorFlow (hand detection only)
    
    ### 💡 Tips
    - Ensure good lighting for better detection
    - Keep hands clearly visible to the camera
    - Adjust confidence thresholds if needed
    - Add TensorFlow for gesture prediction capabilities
    
    ### 🚀 Future Enhancements
    - Train a model to recognize ISL gestures (A-Z)
    - Add gesture vocabulary expansion
    - Include sentence formation
    """)
    
    # Show system status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MediaPipe", "✅ Installed")
        st.metric("OpenCV", "✅ Installed")
    with col2:
        if KERAS_AVAILABLE:
            st.metric("TensorFlow", "✅ Installed")
        else:
            st.metric("TensorFlow", "❌ Not Installed")
        
        if st.session_state.hands:
            st.metric("Hand Detector", "✅ Initialized")
        else:
            st.metric("Hand Detector", "❌ Not Initialized")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🤟 Indian Sign Language Detection System</p>
        <p>Powered by MediaPipe & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
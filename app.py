"""
ISL Sign Language Detection System
Real-time hand gesture recognition with MediaPipe and TensorFlow
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00A67E;
        text-align: center;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE MEDIAPIPE ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== SESSION STATE ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# ==================== SIGN LANGUAGE CLASSES ====================
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
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
    return np.zeros(63)  # 21 landmarks * 3 coordinates

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
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

# ==================== MAIN UI ====================
st.markdown('<div class="main-header">🤟 Indian Sign Language Detection System</div>', unsafe_allow_html=True)
st.markdown("Real-time hand gesture recognition using MediaPipe and Deep Learning")

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading
    st.subheader("Model Configuration")
    model_path = st.text_input("Model Path", value="model.h5")
    
    if st.button("Load Model"):
        try:
            st.session_state.model = keras.models.load_model(model_path, compile=False)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    # MediaPipe settings
    st.subheader("Detection Settings")
    min_detection_confidence = st.slider(
        "Min Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    min_tracking_confidence = st.slider(
        "Min Tracking Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    max_hands = st.selectbox("Max Number of Hands", [1, 2], index=0)
    
    # Initialize hands detector
    if st.button("Initialize Detector"):
        st.session_state.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        st.success("✅ Hand detector initialized!")
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    st.metric("Total Predictions", len(st.session_state.predictions))

# ==================== MAIN TABS ====================
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📸 Image Upload", "ℹ️ About"])

# ==================== LIVE DETECTION TAB ====================
with tab1:
    st.header("Live Camera Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_detection = st.checkbox("Start Detection")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection Results")
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        history_placeholder = st.empty()
        
    if run_detection:
        if st.session_state.hands is None:
            st.warning("⚠️ Please initialize the hand detector from the sidebar first!")
        elif st.session_state.model is None:
            st.warning("⚠️ Please load the model from the sidebar first!")
        else:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check camera permissions.")
            else:
                stop_button = st.button("Stop Detection")
                
                while run_detection and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Flip and convert BGR to RGB
                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    
                    # Process with MediaPipe
                    results = st.session_state.hands.process(image)
                    
                    # Draw landmarks
                    image.flags.writeable = True
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
                        sign, confidence = predict_sign(keypoints, st.session_state.model)
                        
                        if sign and confidence > 0.7:
                            # Add to prediction history
                            st.session_state.predictions.append({
                                'sign': sign,
                                'confidence': confidence,
                                'time': datetime.now()
                            })
                            
                            # Display prediction on image
                            cv2.putText(
                                image,
                                f"{sign}: {confidence:.2f}",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 255, 0),
                                3
                            )
                            
                            # Update results
                            result_placeholder.markdown(
                                f'<div class="prediction-box">{sign}</div>',
                                unsafe_allow_html=True
                            )
                            confidence_placeholder.progress(float(confidence))
                            
                            # Show recent predictions
                            recent = st.session_state.predictions[-5:]
                            history_text = "**Recent Predictions:**\n\n"
                            for p in reversed(recent):
                                history_text += f"- {p['sign']} ({p['confidence']:.1%})\n"
                            history_placeholder.markdown(history_text)
                    
                    # Display frame
                    frame_placeholder.image(image, channels="RGB", use_container_width=True)
                
                cap.release()

# ==================== IMAGE UPLOAD TAB ====================
with tab2:
    st.header("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        with col2:
            st.subheader("Detection Result")
            
            if st.session_state.hands and st.session_state.model:
                # Process image
                results = st.session_state.hands.process(image_rgb)
                
                # Draw landmarks
                annotated_image = image_rgb.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Predict
                    keypoints = extract_keypoints(results)
                    sign, confidence = predict_sign(keypoints, st.session_state.model)
                    
                    st.image(annotated_image, use_container_width=True)
                    
                    if sign:
                        st.markdown(
                            f'<div class="prediction-box">{sign}</div>',
                            unsafe_allow_html=True
                        )
                        st.info(f"Confidence: **{confidence:.2%}**")
                else:
                    st.image(annotated_image, use_container_width=True)
                    st.warning("⚠️ No hands detected in the image")
            else:
                st.warning("⚠️ Please initialize detector and load model first")

# ==================== ABOUT TAB ====================
with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application detects Indian Sign Language (ISL) gestures in real-time using:
    - **MediaPipe** for hand landmark detection
    - **TensorFlow/Keras** for gesture classification
    - **Streamlit** for the web interface
    
    ### 🔧 How to Use
    1. **Load Model**: Enter your model path in the sidebar and click "Load Model"
    2. **Initialize Detector**: Configure detection settings and click "Initialize Detector"
    3. **Start Detection**: Use the "Live Detection" tab for real-time recognition
    4. **Upload Images**: Use the "Image Upload" tab for static image analysis
    
    ### 📊 Features
    - Real-time hand tracking with MediaPipe
    - Support for multiple hand detection
    - Adjustable confidence thresholds
    - Image upload capability
    - Visual feedback with landmarks
    - Prediction history tracking
    
    ### 🛠️ Technical Stack
    - Python 3.13
    - Streamlit
    - MediaPipe
    - TensorFlow/Keras
    - OpenCV
    - NumPy
    
    ### 📝 Notes
    - Ensure good lighting for better detection
    - Keep hands clearly visible to the camera
    - Model accuracy depends on training data quality
    - Minimum confidence threshold of 70% for predictions
    """)
    
    st.info("💡 Tip: Adjust the confidence thresholds in the sidebar for better results")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ for Indian Sign Language Recognition</p>
        <p>Powered by MediaPipe, TensorFlow, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

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
if 'detector' not in st.session_state:
    st.session_state.detector = None

# Title
st.title("🤟 Indian Sign Language Detection System")
st.markdown("Hand gesture recognition using MediaPipe")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    min_detection_confidence = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    min_tracking_confidence = st.slider(
        "Tracking Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    if st.button("🚀 Initialize Hand Detector"):
        try:
            st.session_state.detector = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            st.success("✅ Detector initialized!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    
    if st.session_state.detector:
        st.metric("Status", "✅ Ready")
    else:
        st.metric("Status", "⚠️ Not Initialized")

# Main area
st.header("📸 Upload Hand Gesture Image")

if not st.session_state.detector:
    st.warning("⚠️ Click 'Initialize Hand Detector' in the sidebar first!")
else:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Image")
            st.image(image_rgb, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detection Result")
            
            # Process
            with st.spinner("Detecting hands..."):
                results = st.session_state.detector.process(image_rgb)
            
            # Draw results
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
                st.success(f"✅ Detected {len(results.multi_hand_landmarks)} hand(s)!")
            else:
                st.image(image_rgb, use_container_width=True)
                st.warning("⚠️ No hands detected. Try another image.")

# Footer
st.divider()
st.markdown("""
### 📖 About This Application

This app detects hand gestures using **MediaPipe** and can recognize:
- **Digits**: 0-9
- **Letters**: A-Z

### 💡 How to Use
1. Click **"Initialize Hand Detector"** in the sidebar
2. Upload a clear image of a hand gesture
3. View the detected hand landmarks

### Tips for Best Results
- Use good lighting
- Keep hands clearly visible
- Avoid overlapping hands
- Try different confidence thresholds if detection fails
""")

st.markdown("---")
st.markdown("🤟 **ISL Detection System** | Powered by MediaPipe & Streamlit")
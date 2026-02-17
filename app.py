import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from gtts import gTTS
from PIL import Image
import io

# Try to import MediaPipe, use fallback if not available
try:
    import mediapipe as mp
    from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
    from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
    MEDIAPIPE_AVAILABLE = True
except:
    try:
        from mediapipe.tasks.python import vision
        MEDIAPIPE_AVAILABLE = False
    except:
        MEDIAPIPE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="ISL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-size: 1.2rem;
    }
    .symbol-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 1rem 0;
    }
    .symbol-badge {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤟 ISL Sign Language Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Hand Gesture Recognition 🔊</div>', unsafe_allow_html=True)

# Load model and labels
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('isl_model.h5')
        labels = np.load('isl_labels.npy', allow_pickle=True)
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_model()

if model is None or labels is None:
    st.error("❌ Failed to load model or labels. Please check your files.")
    st.stop()

# Display supported symbols
st.sidebar.header("📚 Supported ISL Signs")
if labels is not None:
    st.sidebar.markdown(f"**Total Signs:** {len(labels)}")
    
    # Create a nice grid of all supported signs
    symbols_html = '<div class="symbol-grid">'
    for label in sorted(labels):
        symbols_html += f'<div class="symbol-badge">{label}</div>'
    symbols_html += '</div>'
    st.sidebar.markdown(symbols_html, unsafe_allow_html=True)

# Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)
enable_audio = st.sidebar.checkbox("Enable Voice Output", value=True)

st.sidebar.info("💡 **Tip:** Show clear hand gestures with good lighting!")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Camera Input")
    camera_input = st.camera_input("Show your ISL sign gesture")
    
    if camera_input is not None:
        # Display the captured image
        image = Image.open(camera_input)
        st.image(image, caption="Captured Image", use_container_width=True)

with col2:
    st.subheader("🎯 Detected Sign")
    prediction_container = st.container()

# Process image
if camera_input is not None:
    try:
        # Convert to PIL Image
        image = Image.open(camera_input)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        hand_detected = False
        landmarks = []
        
        # Try MediaPipe first with LOWER detection threshold for better detection
        if MEDIAPIPE_AVAILABLE:
            try:
                with Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.2,  # Lower threshold for better detection
                    min_tracking_confidence=0.2
                ) as hands:
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        hand_detected = True
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Extract landmarks EXACTLY as they come - RAW format
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
            except Exception as e:
                st.error(f"MediaPipe error: {e}")
                hand_detected = False
        
        if hand_detected and len(landmarks) > 0:
            # Ensure we have exactly 63 features (21 landmarks * 3 coordinates)
            if len(landmarks) < 63:
                landmarks.extend([0] * (63 - len(landmarks)))
            elif len(landmarks) > 63:
                landmarks = landmarks[:63]
            
            # Convert to numpy array with correct shape
            landmarks_array = np.array([landmarks], dtype=np.float32)
            
            # Make prediction
            prediction = model.predict(landmarks_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            with prediction_container:
                if confidence >= confidence_threshold:
                    predicted_label = str(labels[predicted_class])
                    
                    # Display BIG prediction
                    st.markdown(
                        f'<div class="prediction-box">{predicted_label}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display confidence
                    st.markdown(
                        f'<div class="confidence-box"><b>Confidence:</b> {confidence:.2%}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Generate audio
                    if enable_audio:
                        try:
                            tts = gTTS(text=predicted_label, lang='en', slow=False)
                            audio_buffer = io.BytesIO()
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.audio(audio_buffer, format='audio/mp3')
                        except Exception as e:
                            st.warning(f"⚠️ Audio generation failed: {e}")
                    
                    # Show top 5 predictions
                    st.markdown("**Top 5 Predictions:**")
                    top_indices = np.argsort(prediction[0])[-5:][::-1]
                    for idx in top_indices:
                        label = labels[idx]
                        conf = prediction[0][idx]
                        st.progress(float(conf), text=f"{label}: {conf:.2%}")
                    
                else:
                    st.warning(f"⚠️ Low confidence ({confidence:.2%}). Please try again!")
                    st.info("💡 Tips:\n- Ensure proper lighting\n- Position hand clearly\n- Hold gesture steady")
        else:
            with prediction_container:
                st.error("❌ No hand detected!")
                st.info("**Tips:**\n"
                       "- Show your hand clearly\n"
                       "- Good lighting is important\n"
                       "- Keep hand in frame\n"
                       "- Remove gloves")
        
        # ALWAYS show hand landmarks visualization below (separate from detection)
        st.markdown("---")
        if show_landmarks and MEDIAPIPE_AVAILABLE:
            try:
                with Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2
                ) as hands:
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        annotated_image = img_array.copy()
                        for hand_landmarks in results.multi_hand_landmarks:
                            draw_landmarks(
                                annotated_image,
                                hand_landmarks,
                                HAND_CONNECTIONS,
                                DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                                DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                            )
                        
                        st.subheader("🖐️ Hand Landmarks Visualization")
                        st.image(annotated_image, caption="MediaPipe Hand Detection - 21 Key Points", use_container_width=True)
                    else:
                        st.info("👋 Position your hand clearly in the frame to see landmarks visualization")
            except Exception as e:
                st.warning(f"Could not generate landmarks visualization: {e}")
    
    except Exception as e:
        st.error(f"❌ Error occurred during processing")
        st.exception(e)

# Instructions
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Quick Guide:
    
    1. **Position Your Hand:**
       - Show your ISL sign clearly
       - Keep hand centered in camera
       - Hold gesture steady
    
    2. **Lighting:**
       - Use good lighting
       - Avoid shadows
    
    3. **Capture:**
       - Click camera button
       - Wait for AI prediction
       - See your sign recognized!
    
    4. **Supported Signs:**
       - All signs shown in the sidebar
       - Numbers: 1-9
       - Letters: A-Z
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Powered by:</b> TensorFlow • MediaPipe • Streamlit</p>
    <p>🤟 Making ISL recognition easy and accessible</p>
</div>
""", unsafe_allow_html=True)

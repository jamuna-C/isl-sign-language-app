import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from gtts import gTTS
from PIL import Image
import io

# Import MediaPipe
try:
    import mediapipe as mp
    from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
    from mediapipe.python.solutions.drawing_utils import draw_landmarks
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    st.error("MediaPipe not available!")

# Page config
st.set_page_config(
    page_title="ISL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS
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
        model = keras.models.load_model('isl_model.h5', compile=False)
        labels = np.load('isl_labels.npy', allow_pickle=True).tolist()
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_model()

if model is None or labels is None:
    st.error("❌ Failed to load model or labels.")
    st.stop()

# Sidebar
st.sidebar.header("📚 Supported ISL Signs")
if labels:
    st.sidebar.markdown(f"**Total Signs:** {len(labels)}")
    symbols_html = '<div class="symbol-grid">'
    for label in sorted(labels):
        symbols_html += f'<div class="symbol-badge">{label}</div>'
    symbols_html += '</div>'
    st.sidebar.markdown(symbols_html, unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)
enable_audio = st.sidebar.checkbox("Enable Voice Output", value=True)
st.sidebar.info("💡 **Tip:** Show clear hand gestures with good lighting!")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Camera Input")
    camera_input = st.camera_input("Show your ISL sign gesture")
    
    if camera_input:
        image = Image.open(camera_input)
        st.image(image, caption="Captured Image", use_container_width=True)

with col2:
    st.subheader("🎯 Detected Sign")
    prediction_container = st.container()

# Process image
if camera_input:
    try:
        # Convert image
        image = Image.open(camera_input)
        img_array = np.array(image)
        image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        hand_detected = False
        features = []
        
        if MEDIAPIPE_AVAILABLE:
            try:
                # LOWER detection threshold for webcam images
                with Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.3,  # LOWER threshold
                    min_tracking_confidence=0.3
                ) as hands:
                    result = hands.process(rgb)
                    
                    if result.multi_hand_landmarks:
                        hand_detected = True
                        hand = result.multi_hand_landmarks[0]
                        
                        # Extract features
                        for lm in hand.landmark:
                            features.extend([lm.x, lm.y, lm.z])
                        
                        features = np.array(features)
                        st.success(f"✅ Hand detected! {len(features)} features extracted")
                    else:
                        st.warning("⚠️ No hand detected by MediaPipe. Try better lighting or clearer hand position.")
                        
            except Exception as e:
                st.error(f"MediaPipe error: {e}")
                hand_detected = False
        
        if hand_detected and len(features) == 63:
            # Predict
            pred = model.predict(features.reshape(1, -1), verbose=0)
            idx = np.argmax(pred[0])
            predicted_label = labels[idx]
            confidence = float(pred[0][idx])
            
            with prediction_container:
                if confidence >= confidence_threshold:
                    # Display prediction
                    st.markdown(
                        f'<div class="prediction-box">{predicted_label}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        f'<div class="confidence-box"><b>Confidence:</b> {confidence:.2%}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Audio
                    if enable_audio:
                        try:
                            tts = gTTS(text=predicted_label, lang='en', slow=False)
                            audio_buffer = io.BytesIO()
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.audio(audio_buffer, format='audio/mp3')
                        except:
                            pass
                    
                    # Top 5
                    st.markdown("**Top 5 Predictions:**")
                    top_indices = np.argsort(pred[0])[-5:][::-1]
                    for idx_top in top_indices:
                        label = labels[idx_top]
                        conf = pred[0][idx_top]
                        st.progress(float(conf), text=f"{label}: {conf:.2%}")
                else:
                    st.warning(f"⚠️ Low confidence ({confidence:.2%})")
                    st.info("💡 Tips:\n- Better lighting\n- Clear hand position\n- Hold steady")
        elif hand_detected and len(features) != 63:
            with prediction_container:
                st.error(f"❌ Invalid features: {len(features)} (expected 63)")
        else:
            with prediction_container:
                st.error("❌ No hand detected!")
                st.info("**Tips:**\n- Show hand clearly\n- Good lighting\n- Keep in frame\n- Remove gloves")
        
        # Visualization
        st.markdown("---")
        if show_landmarks and MEDIAPIPE_AVAILABLE:
            try:
                with Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                ) as hands:
                    result = hands.process(rgb)
                    
                    if result.multi_hand_landmarks:
                        annotated_image = image_bgr.copy()
                        hand = result.multi_hand_landmarks[0]
                        draw_landmarks(annotated_image, hand, HAND_CONNECTIONS)
                        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        
                        st.subheader("🖐️ Hand Landmarks Visualization")
                        st.image(annotated_rgb, caption="MediaPipe Hand Detection - 21 Key Points", use_container_width=True)
                    else:
                        st.info("👋 Position your hand clearly to see landmarks")
            except Exception as e:
                st.warning(f"Visualization error: {e}")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Quick Guide:
    
    1. **Position Your Hand:** Show ISL sign clearly, centered
    2. **Lighting:** Use good lighting, avoid shadows
    3. **Capture:** Click camera button, wait for prediction
    4. **Supported Signs:** All signs shown in sidebar
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Powered by:</b> TensorFlow • MediaPipe • Streamlit</p>
    <p>🤟 Making ISL recognition easy and accessible</p>
</div>
""", unsafe_allow_html=True)

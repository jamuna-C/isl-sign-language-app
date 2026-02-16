import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from gtts import gTTS
from PIL import Image
import io

# Import MediaPipe - CORRECT WAY that works on Streamlit Cloud
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

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
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .symbol-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(50px, 1fr));
        gap: 5px;
        margin: 1rem 0;
    }
    .symbol-badge {
        background: #e3f2fd;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤟 ISL Sign Language Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Hand Gesture Recognition with Voice Output 🔊</div>', unsafe_allow_html=True)

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
st.sidebar.header("📚 Supported ISL Symbols")
if labels is not None:
    st.sidebar.markdown(f"**Total Symbols:** {len(labels)}")
    
    # Create a grid of symbols
    symbols_html = '<div class="symbol-grid">'
    for label in labels:
        symbols_html += f'<div class="symbol-badge">{label}</div>'
    symbols_html += '</div>'
    st.sidebar.markdown(symbols_html, unsafe_allow_html=True)

# Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)
enable_audio = st.sidebar.checkbox("Enable Voice Output", value=True)

st.sidebar.info("💡 **Tip:** Ensure good lighting and clear hand visibility for best results!")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Camera Input")
    camera_input = st.camera_input("Capture your ISL sign gesture")
    
    if camera_input is not None:
        # Display the captured image
        image = Image.open(camera_input)
        st.image(image, caption="Captured Image", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")
    prediction_container = st.container()

# Text output section
st.subheader("📝 Detected Text Sequence")
if 'detected_sequence' not in st.session_state:
    st.session_state.detected_sequence = []

text_display = st.empty()
col_clear, col_speak = st.columns([1, 1])

with col_clear:
    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state.detected_sequence = []
        st.rerun()

with col_speak:
    speak_button = st.button("🔊 Speak Full Text", use_container_width=True)

# Display current sequence
if st.session_state.detected_sequence:
    text_display.markdown(
        f'<div class="prediction-box">{" ".join(st.session_state.detected_sequence)}</div>',
        unsafe_allow_html=True
    )
else:
    text_display.info("Detected signs will appear here...")

# Process image
if camera_input is not None:
    try:
        # Convert to PIL Image
        image = Image.open(camera_input)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Process with MediaPipe - using correct import
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Extract landmarks
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Ensure we have exactly 63 features (21 landmarks * 3 coordinates)
                if len(landmarks) < 63:
                    landmarks.extend([0] * (63 - len(landmarks)))
                elif len(landmarks) > 63:
                    landmarks = landmarks[:63]
                
                # Make prediction
                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                with prediction_container:
                    if confidence >= confidence_threshold:
                        predicted_label = str(labels[predicted_class])
                        
                        # Display prediction with styling
                        st.markdown(
                            f'<div class="prediction-box">{predicted_label}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Display confidence
                        st.markdown(
                            f'<div class="confidence-box"><b>Confidence:</b> {confidence:.2%}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Add to sequence
                        if st.button("➕ Add to Text Sequence"):
                            st.session_state.detected_sequence.append(predicted_label)
                            st.rerun()
                        
                        # Generate audio for single prediction
                        if enable_audio:
                            try:
                                tts = gTTS(text=predicted_label, lang='en', slow=False)
                                audio_buffer = io.BytesIO()
                                tts.write_to_fp(audio_buffer)
                                audio_buffer.seek(0)
                                st.audio(audio_buffer, format='audio/mp3')
                            except Exception as e:
                                st.warning(f"⚠️ Audio generation failed: {e}")
                        
                        # Show all predictions with confidence
                        st.markdown("**Top 5 Predictions:**")
                        top_indices = np.argsort(prediction[0])[-5:][::-1]
                        for idx in top_indices:
                            label = labels[idx]
                            conf = prediction[0][idx]
                            st.progress(float(conf), text=f"{label}: {conf:.2%}")
                        
                    else:
                        st.warning(f"⚠️ Low confidence ({confidence:.2%}). Please try again with a clearer gesture.")
                        st.info("💡 Tips:\n- Ensure proper lighting\n- Position hand clearly in frame\n- Hold gesture steady\n- Avoid background clutter")
                
                # Draw landmarks on image if enabled
                if show_landmarks:
                    annotated_image = img_array.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                    
                    st.subheader("🖐️ Hand Landmarks Detection")
                    st.image(annotated_image, caption="Detected Hand Landmarks", use_container_width=True)
            else:
                with prediction_container:
                    st.error("❌ No hand detected in the image!")
                    st.info("**Tips for better detection:**\n"
                           "- Position your hand clearly in the frame\n"
                           "- Ensure good lighting conditions\n"
                           "- Keep hand within the camera view\n"
                           "- Remove gloves or hand coverings")
    
    except Exception as e:
        st.error(f"❌ An error occurred during processing")
        st.exception(e)

# Speak full text sequence
if speak_button and st.session_state.detected_sequence:
    try:
        full_text = " ".join(st.session_state.detected_sequence)
        tts = gTTS(text=full_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format='audio/mp3')
    except Exception as e:
        st.error(f"Failed to generate speech: {e}")

# Instructions section
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Instructions for Best Results:
    
    1. **Position Your Hand:**
       - Keep your hand clearly visible in the frame
       - Center your hand in the camera view
       - Maintain a steady position
    
    2. **Lighting:**
       - Ensure adequate lighting on your hand
       - Avoid harsh shadows or backlighting
       - Use natural or bright indoor lighting
    
    3. **Background:**
       - Use a plain, contrasting background
       - Avoid cluttered or busy backgrounds
       - Keep background well-lit and uniform
    
    4. **Capture & Predict:**
       - Make your ISL sign gesture
       - Click the camera button to capture
       - Wait for the AI to process and predict
       - Add predicted sign to your text sequence
    
    5. **Build Sentences:**
       - Capture multiple signs one by one
       - Add each to your text sequence
       - Use "Speak Full Text" to hear the complete sentence
       - Clear text when starting a new sentence
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Powered by:</b> TensorFlow • MediaPipe • Streamlit</p>
    <p>🤟 Making communication accessible for everyone</p>
</div>
""", unsafe_allow_html=True)

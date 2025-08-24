import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

# Load model
MODEL = tf.keras.models.load_model("efficientnetv2s.h5")

# Load CSV
disease_info = pd.read_csv("disease_info.csv", encoding='cp1252')

# Classes
CLASSES = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy',
    'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust',
    'Corn(maize) Northern Leaf Blight', 'Corn(maize) healthy', 'Grape Black rot',
    'Grape Esca(Black Measles)', 'Grape Leaf blight(Isariopsis Leaf Spot)', 'Grape healthy',
    'Orange Haunglongbing(Citrus greening)', 'Peach Bacterial spot', 'Peach healthy',
    'Bell PepperBacterial_spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
    'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew',
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites (Two-spotted spider mite)', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# UI Setup
st.set_page_config(page_title="AgroVision", layout="wide", page_icon="🌿")

# Navigation Bar
menu = st.sidebar.radio("Navigation", ["Home", "Prediction", "About Us"])

if menu == "Home":
    # Custom HTML & CSS for dynamic styling
    st.markdown(
        """
        <style>
        .title-style {
            text-align: center;
            color: #2E8B57;
            font-weight: bold;
            font-size: 300%;
        }
        .subtitle-style {
            text-align: center;
            color: #4CAF50;
            font-weight: bold;
            font-size: 150%;
        }
        .list-style {
            color: #2E8B57;
            font-size: 120%;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title & Subtitle
    st.markdown("<div class='title-style'>🌿 Welcome to AgroVision 🌿</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-style'>Your Smart Assistant for Plant Health and Care</div>", unsafe_allow_html=True)

    # About Section
    st.markdown(
        """
        *AgroVision* is a cutting-edge smart assistant designed for farmers, gardeners, and enthusiasts. 
        Using advanced AI and image processing, this app makes *plant disease detection* and 
        *care recommendations* simple and effective. 🚀

        Whether you're managing a large farm or caring for your home garden, 
        *AgroVision* is here to guide you every step of the way. 🌾
        """
    )
    # Supported Plants Section
    st.markdown("### 🌱 *Plants We Can Diagnose*")
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between;">
            <ul style="flex: 1; list-style-type: none; padding-left: 1rem;">
                <li>🍎 Apple</li>
                <li>🍇 Grape</li>
                <li>🍅 Tomato</li>
                <li>🌽 Corn</li>
                <li>🥔 Potato</li>
                <li>🍑 Peach</li>
                <li>🫐 Blueberry</li>
            </ul>
            <ul style="flex: 1; list-style-type: none; padding-left: 1rem;">
                <li>🍒 Cherry</li>
                <li>🌶 Bell Pepper</li>
                <li>🍓 Strawberry</li>
                <li>❤ Raspberry</li>
                <li>🌱 Soybean</li>
                <li>🎃 Squash</li>
                <li>🍊 Orange</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Why Choose Section
    st.markdown("### 💡 *Why Choose AgroVision?*")
    st.markdown(
        """
        - *🌟 Quick and accurate disease detection* for a wide variety of crops.
        - *🧪 Step-by-step care recommendations* to keep your plants thriving.
        - *🌍 Empowering farmers and gardeners* worldwide with AI-powered tools.
        - *💚 Developed with love* to make crop care simple and effective for everyone.
        """
    )

elif menu == "Prediction":
    st.title("🧬 Plant Disease Prediction")
    uploaded_file = st.file_uploader("📷 Upload a Leaf Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Create prediction model with Softmax activation to interpret probabilities
        prob_model = tf.keras.Sequential([MODEL, tf.keras.layers.Softmax()])
        prediction = prob_model.predict(img_array)
        predicted_class = CLASSES[np.argmax(prediction[0])]

        # Display uploaded image and prediction result
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Leaf", use_container_width=True)
        with col2:
            st.success(f"🧬 Prediction: *{predicted_class}*")

        # Match predicted class to disease_info using a normalization function
        def normalize(text):
            return text.lower().replace(" ", "").replace(":", "").replace("_", "").replace("(", "").replace(")", "")

        norm_pred = normalize(predicted_class)
        match_row = disease_info[disease_info['disease_name'].apply(normalize).str.contains(norm_pred)]

        if not match_row.empty:
            disease_desc = match_row.iloc[0]['description']
            disease_steps = match_row.iloc[0]['Possible Steps']
            disease_img = match_row.iloc[0]['image_url']

            st.markdown("### 📝 Disease Description")
            st.image(disease_img, width=200)
            st.write(disease_desc)

            st.markdown("### 🧪 Possible Steps")
            st.write(disease_steps)
        else:
            st.warning("No disease description found.")

elif menu == "About Us":
    st.title("👨‍💻 About Us")
    st.markdown(
        """
        *AgroVision* aims to make plant care simple and effective for farmers, gardeners, and enthusiasts. 🚜

        Stay tuned for more updates as we plan to expand our app with additional features! 🌟

        ---
        ### 📬 Contact Us
        - 📧 Email: support@agrovision.app
        - 🌐 Website: [www.agrovision.app](http://www.agrovision.app)
        - 📱 Phone: +91-9999011111
        """
    )
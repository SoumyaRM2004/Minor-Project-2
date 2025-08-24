import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="AgroVision", layout="wide", page_icon="🌿")
st.markdown("""
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
""", unsafe_allow_html=True)

# -----------------------------
# CNN Model Architecture
# -----------------------------
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 14 * 14)
        x = self.dense_layers(x)
        return x

# -----------------------------
# Load Resources
# -----------------------------
@st.cache_resource
def load_model():
    model = CNN(39)
    model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_data():
    disease_info = pd.read_csv("disease_info.csv", encoding="cp1252")
    supplement_info = pd.read_csv("supplement_info.csv", encoding="cp1252")
    return disease_info, supplement_info

model = load_model()
disease_info, supplement_info = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio("Navigation", ["Home", "Prediction", "About Us"])

# -----------------------------
# Home Page
# -----------------------------
if menu == "Home":
    st.markdown("<div class='title-style'>🌿 Welcome to AgroVision 🌿</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-style'>Your Smart Assistant for Plant Health and Care</div>", unsafe_allow_html=True)

    st.markdown("""
        *AgroVision* is a cutting-edge smart assistant designed for farmers, gardeners, and enthusiasts. 
        Using advanced AI and image processing, this app makes *plant disease detection* and 
        *care recommendations* simple and effective. 🚀
    """)

    st.markdown("### 🌱 *Plants We Can Diagnose*")
    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("### 💡 *Why Choose AgroVision?*")
    st.markdown("""
        - *🌟 Quick and accurate disease detection* for a wide variety of crops.
        - *🧪 Step-by-step care recommendations* to keep your plants thriving.
        - *🌍 Empowering farmers and gardeners* worldwide with AI-powered tools.
        - *💚 Developed with love* to make crop care simple and effective for everyone.
    """)

# -----------------------------
# Prediction Page
# -----------------------------
elif menu == "Prediction":
    st.title("🌾 Upload Plant Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Leaf Image", width=300)  # Display uploaded image with smaller width

        if st.button("Predict"):
            image = image.resize((224, 224))
            image_tensor = TF.to_tensor(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()

            st.success(f"🔍 **Prediction**: {disease_info['disease_name'][predicted_class]}")
            st.markdown(f"💡 **Description**: {disease_info['description'][predicted_class]}")
            st.markdown(f"🛠️ **Precaution**: {disease_info['Possible Steps'][predicted_class]}")
            st.image(disease_info['image_url'][predicted_class], caption=f"Disease: {disease_info['disease_name'][predicted_class]}", width=300)  # Set smaller width

            supplement = supplement_info[supplement_info["disease_name"] == disease_info['disease_name'][predicted_class]]
            if not supplement.empty:
                st.markdown("🧪 **Suggested Supplement:**")
                st.markdown(f"- **Name**: {supplement['supplement name'].values[0]}")
                st.markdown(f"- **Description**: {supplement['description'].values[0]}")
                st.markdown(f"- [🛒 Buy Here]({supplement['buy link'].values[0]})")
                st.image(supplement['supplement image'].values[0], caption="Supplement Image", width=300)  # Set smaller width
            else:
                st.warning("No supplement info available for this disease.")

# -----------------------------
# About Us Page
# -----------------------------
elif menu == "About Us":
    st.title("👨‍💻 About Us")
    st.markdown("""
        *AgroVision* aims to make plant care simple and effective for farmers, gardeners, and enthusiasts. 🚜
        Stay tuned for more updates as we plan to expand our app with additional features! 🌟
        ---
        ### 📬 Contact Us
        - 📧 Email: support@agrovision.app
        - 🌐 Website: [www.agrovision.app](http://www.agrovision.app)
        - 📱 Phone: +91-9999011111
    """)
import streamlit as st
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection System")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

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
# Load Model and CSVs
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
    supplement_info.fillna('', inplace=True)
    return disease_info, supplement_info

model = load_model()
disease_info, supplement_info = load_data()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    image = image.convert("RGB")  # Fix: Ensure image has 3 channels (RGB)
    image = image.resize((224, 224))
    input_tensor = TF.to_tensor(image).unsqueeze(0)
    output = model(input_tensor)
    pred_index = torch.argmax(output).item()
    return pred_index

# -----------------------------
# Streamlit UI
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Submit Image", "Contact"])

if page == "Home":
    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909779.png", width=100)
    st.header("Welcome to Plant Disease Detection App")
    st.write("Upload a leaf image and detect the disease using AI!")

elif page == "Submit Image":
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True,width=250)

        if st.button("Predict"):
            pred = predict(image)
            st.success(f"**Disease Name:** {disease_info['disease_name'][pred]}")
            st.image(disease_info['image_url'][pred], caption='Reference Image',width=250)

            st.subheader("🧬 Description")
            st.write(disease_info['description'][pred])

            st.subheader("🛡️ Prevention")
            st.write(disease_info['Possible Steps'][pred])

            st.subheader("💊 Supplement Recommendation")

            supplement_img = supplement_info['supplement image'][pred]
            if supplement_img:
                try:
                    st.image(supplement_img, width=150)
                except Exception as e:
                    st.warning(f"Unable to display supplement image: {e}")
            else:
                st.info("No supplement image available.")

            supplement_name = supplement_info['supplement name'][pred]
            if supplement_name:
                st.write(f"**Name:** {supplement_name}")
            else:
                st.info("No supplement name provided.")

elif page == "Contact":
    st.header("📞 Contact Us")
    st.write("For any queries, contact us at: [your-email@example.com](mailto:your-email@example.com)")




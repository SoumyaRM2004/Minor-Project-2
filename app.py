from flask import Flask, render_template, request, session
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Load disease information with error handling using environment variable
try:
    disease_info = pd.read_csv(os.getenv("DISEASE_INFO_PATH", "disease_info.csv"), encoding="cp1252") 
except Exception as e:
    print(f"Error loading disease information: {e}")
    disease_info = pd.DataFrame()  # Fallback to an empty DataFrame

# Load supplement information with error handling using environment variable
try:
    supplement_info = pd.read_csv(os.getenv("SUPPLEMENT_INFO_PATH", "supplement_info.csv"), encoding="cp1252")
except Exception as e:
    print(f"Error loading supplement information: {e}")
    supplement_info = pd.DataFrame()  # Fallback to an empty DataFrame


# Load supplement information
supplement_info = pd.read_csv("supplement_info.csv", encoding="cp1252")

# CNN Model
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        return self.dense_layers(out)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(39)  
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.eval()

# Class Mapping
idx_to_classes = {index: name for index, name in enumerate(disease_info["disease_name"].tolist())}

# Prediction Function
def predict_disease(image):
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
        pred_index = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_index].item()
    return pred_index, confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file part"
    file = request.files['file']
    if file:
        image = Image.open(file).convert("RGB")
        try:
            pred_index, confidence = predict_disease(image)
        except Exception as e:
            return f"Error during prediction: {e}"

        predicted_class = idx_to_classes[pred_index]

        # Retrieve supplement information
        supplement_data = supplement_info[supplement_info['disease_name'] == predicted_class]

        # Store prediction in session
        if "history" not in session:
            session["history"] = []
        session["history"].append({"disease": predicted_class, "confidence": confidence})

        return render_template('result.html', predicted_class=predicted_class, confidence=confidence, disease_info=disease_info, supplement_data=supplement_data)

@app.route('/plants')
def plants():
    return render_template('plants.html', diseases=disease_info)

@app.route('/history')
def history():
    return render_template('history.html', history=session.get("history", []))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)

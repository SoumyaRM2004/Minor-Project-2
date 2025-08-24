
# 🌱 Plant Disease Detection & Management System  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?logo=streamlit&logoColor=white)  
![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-Data-Analysis-150458?logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-005C5C)  
![License](https://img.shields.io/badge/License-Academic-green)  

---

## 📖 Project Overview  

Plant diseases significantly impact global food production, reducing both **quality** and **yield**. Traditional disease detection methods—manual inspection or expert consultation—are **time-consuming, costly, and error-prone**.  

This project leverages **Convolutional Neural Networks (CNNs)** and a **Streamlit web application** to build an **automated plant disease detection system**.  

🔹 Farmers and researchers can upload an image of a plant leaf, and within seconds, the system:  
- Detects whether the leaf is **healthy or diseased**  
- Identifies the specific disease  
- Provides **treatment recommendations**  

This solution promotes **sustainable farming** by reducing unnecessary pesticide use and enabling **early intervention** to minimize crop losses.  

---

## 🎯 Purpose  

The primary aim is to build a **scalable, intelligent, and easy-to-use system** that:  
- Detects plant diseases from **leaf images** with **high accuracy**.  
- Provides **actionable treatment suggestions**.  
- Reduces crop loss and improves **farm productivity**.  
- Bridges the gap between **AI technology** and **traditional farming**.  

---

## 📌 Features  

- ✅ **Deep Learning-Based Disease Classification** (CNN, ANN, LSTM models)  
- ✅ **Advanced Preprocessing**: resizing, normalization, augmentation  
- ✅ **User-Friendly Web Interface** (Streamlit-powered)  
- ✅ **Real-Time Detection** from uploaded images  
- ✅ **Treatment Recommendations** for each prediction  
- ✅ **Scalability**: adaptable to multiple crops & regions  
- ✅ **Explainability** via visual results and confidence levels  

---

## 🗂️ File Structure  

```

├── 📂 data/                   # Training & testing datasets
├── 📓 Training.ipynb          # Model training Jupyter Notebook
├── 🐍 train.py                 # Streamlit-based web application
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # Project documentation

````

---

## ⚙️ System Requirements  

### 🔹 Hardware  
- CPU: Intel i5 / Ryzen 5 (i7/Ryzen 7 recommended)  
- RAM: 16 GB (32 GB recommended)  
- GPU: NVIDIA GTX 1660 Ti / RTX 3060+ (CUDA enabled)  
- Storage: 512 GB SSD  

### 🔹 Software  
- OS: Windows 10 / Ubuntu / macOS  
- Python 3.8+  
- TensorFlow / PyTorch  
- Streamlit  
- Pandas, NumPy, Matplotlib, scikit-learn  

---

## 📦 Installation  

1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
````

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app

```bash
streamlit run train.py
```

---

## 🚀 Usage

1. Upload a **leaf image** (JPG/PNG).
2. The system preprocesses the image (resize, normalize).
3. The **CNN model** predicts the disease type.
4. View **disease diagnosis + treatment suggestions**.

Example:

🌿 Uploaded: `Tomato_leaf.png`
🔍 Prediction: `Tomato___Early_Blight`
💊 Recommendation: "Apply fungicide spray with Chlorothalonil; rotate crops to prevent recurrence."

---

## 📊 Dataset

* Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* Classes: 12 (Corn, Potato, Tomato — Healthy & Diseased categories)
* Images: \~9,300 high-quality leaf samples

---

## 🔬 Model Development

* **CNN layers** for feature extraction
* **Data Augmentation** (rotation, zoom, contrast adjustments)
* Optimized with **Adam optimizer** & **cross-entropy loss**
* Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `F1-score`

---

## 📐 System Design

### 🔹 Data Flow Diagram (DFD)

*User → Upload Leaf → CNN Model → Prediction → Treatment Suggestions*

### 🔹 UML Use Case

* **Upload Image**
* **View Prediction**
* **Receive Recommendation**

---

## 📸 Screenshots

*(Add sample UI screenshots of your Streamlit app here)*

---

## 📈 Results

* Achieved **>92% classification accuracy** on test dataset.
* Demonstrated robust performance across **Corn, Potato, and Tomato** leaves.
* Real-time prediction speed: **<2 seconds per image**.

---

## 🔮 Future Scope

* 🌍 Expand to more crop species & regional diseases.
* 📱 Deploy as a **mobile app** for offline usage.
* 🌐 Integrate with **IoT smart farming systems**.
* 🤖 Add **real-time field monitoring** via drones & sensors.

---

## 📜 License

📌 This project is for **academic and non-commercial use only**.

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repo
2. Create a new branch (`feature-new`)
3. Commit your changes
4. Open a Pull Request

---

## 📬 Contact

👨‍💻 **Soumya Ranjan Mohapatra**
📧 [soumyasrm04@gmail.com](mailto:soumyasrm04@gmail.com)
🌐 [LinkedIn](https://www.linkedin.com/in/srmohapatra)
🐙 [GitHub](https://github.com/your-username)

---

💡 *Made with ❤️ to support smarter & sustainable farming!* 🌱

```

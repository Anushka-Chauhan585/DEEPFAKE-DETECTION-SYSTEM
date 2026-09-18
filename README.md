# 🧠 DEEPFAKE_DETECTION_MODEL

A full-fledged system for detecting deepfakes in images and videos using a custom-trained deep learning model powered by EfficientNetB0, wrapped in an interactive Streamlit interface.

---

## 📌 About

This project implements a deepfake detection pipeline from scratch. It includes:

- A training pipeline using EfficientNetB0 with fine-tuning.
- Real-time image and video classification using a Streamlit web app.
- Frame-by-frame video analysis for deepfake frame percentage.
- Robust handling of grayscale, RGB, and RGBA inputs.

The goal is to showcase applied machine learning, model deployment, and frontend integration for a practical AI application.

---

## 🗂 Repository Structure

`├── train/                     # Model architecture, training scripts` <br>
`├── test data/                 # Sample test videos and images` <br>
`├── streamlit/app.py           # Streamlit frontend code for inference` <br>
`├── streamlit/my_model.keras   # Saved trained model` <br>

---

### 📸 Output Section

<p align="center">
  <img src="demo/demo.png" width="30%"/>
  <img src="demo/screen.png" width="30%"/>
  <img src="demo/demo2.png" width="30%"/>
</p>



---

## 🚀 How It Works

1. **Training Phase**
   - EfficientNetB0 (imagenet weights) is used as a base model.
   - Custom dense layers are added for binary classification.
   - Trained on labeled real vs fake datasets.

2. **Streamlit App**
   - Users upload an image or video.
   - For images: Model instantly classifies as **Fake** or **Real**.
   - For videos: Each frame is predicted, and a percentage of fake frames is shown.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **OpenCV**
- **NumPy / PIL**

---

> ⚠️ This project is built for educational purposes and is not intended for production-grade deepfake detection.

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import time
import os

# Load model
model = tf.keras.models.load_model('my_model.keras')

# ---------- Video Prediction ----------
def img_pred(video_path, model, batch_size=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert the frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))

        if img.shape[-1] != 3:
            print(f"Warning: Frame does not have 3 channels, but has {img.shape[-1]} channels.")
            img = np.stack([img] * 3, axis=-1)

        img = img.reshape(1, 150, 150, 3)
        frames.append(img)

        if len(frames) >= batch_size:
            batch = np.vstack(frames)
            predictions = model.predict(batch)
            for p in predictions:
                p = np.argmax(p) if p.ndim == 1 else np.argmax(p, axis=1)[0]
                if p == 0:
                    fake_count += 1
            frames = []

    if frames:
        batch = np.vstack(frames)
        predictions = model.predict(batch)
        for p in predictions:
            p = np.argmax(p) if p.ndim == 1 else np.argmax(p, axis=1)[0]
            if p == 0:
                fake_count += 1

    cap.release()
    fake_percentage = (fake_count / frame_count) * 100 if frame_count > 0 else 0
    return fake_percentage, frame_count, fake_count

# ---------- Image Prediction ----------
def photo_pred(image, model):
    img = Image.open(image).resize((150, 150))
    img = np.array(img)

    print(f"Image shape: {img.shape}")

    if img.ndim == 3 and img.shape[-1] == 4:
        print("Image has 4 channels (RGBA), converting to RGB.")
        img = img[:, :, :3]

    elif img.ndim == 2:
        print("Image is grayscale, converting to RGB.")
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] != 3:
        raise ValueError(f"The image should have 3 color channels (RGB), but has {img.shape[-1]} channels.")

    img = img.reshape(1, 150, 150, 3)
    prediction = model.predict(img)
    p = np.argmax(prediction) if prediction.ndim == 1 else np.argmax(prediction, axis=1)[0]
    return "Fake" if p == 0 else "Real"

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(page_title="Deepfake Detection", layout="centered")
    st.markdown("<h1 style='text-align: center;'>Deepfake Detection</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    choice = st.radio("Select file type to upload", ["Video", "Image"], horizontal=True)

    if 'uploaded_video_path' not in st.session_state:
        st.session_state.uploaded_video_path = None

    if choice == "Video":
        st.subheader("Upload a Video for Deepfake Detection")
        video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

        if video_file:
            uploaded_video_path = "uploaded_video.mp4"

            if os.path.exists(uploaded_video_path):
                os.remove(uploaded_video_path)

            with open(uploaded_video_path, "wb") as f:
                f.write(video_file.getbuffer())

            st.session_state.uploaded_video_path = uploaded_video_path

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.video(video_file, start_time=0)

            st.markdown(f"**File:** {video_file.name}")

            with st.spinner("Analyzing video..."):
                time.sleep(1)
                fake_percentage, total_frames, fake_frames = img_pred(uploaded_video_path, model)

            st.markdown("### Prediction Result")
            result_color = "#ff4d4d" if fake_percentage > 50 else "#4CAF50"
            st.markdown(f"""
                <div style="padding: 1rem; border-radius: 10px; background-color: {result_color}; color: white;">
                    <h4 style="margin: 0;">{('Fake' if fake_percentage > 50 else 'Real')} Video</h4>
                    <p style="margin: 0;">Fake Frames: {fake_frames} / {total_frames} ({fake_percentage:.2f}%)</p>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("Show Details"):
                st.write(f"Total Frames: {total_frames}")
                st.write(f"Fake Detected: {fake_frames}")
                st.write(f"Fake Prediction Percentage: {fake_percentage:.2f}%")

    elif choice == "Image":
        st.subheader("Upload an Image for Deepfake Detection")
        image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if image_file:
            img = Image.open(image_file)

            # Show image in center using 1:2:1 layout
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img, caption='Uploaded Image', use_column_width=True)

            st.markdown(f"**File:** {image_file.name}")

            with st.spinner("Analyzing image..."):
                time.sleep(1)
                try:
                    prediction = photo_pred(image_file, model)
                except ValueError as e:
                    st.error(str(e))
                    return

            color = "#ff4d4d" if prediction == "Fake" else "#4CAF50"
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white;">
                <h4 style="margin: 0;">{prediction} Image</h4>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

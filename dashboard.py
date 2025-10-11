import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/object.pt"
    class_path = "model/class.h5"

    if not os.path.exists(yolo_path):
        st.error("❌ File YOLO tidak ditemukan di folder model/. Pastikan 'object.pt' sudah di-upload.")
        return None, None

    if not os.path.exists(class_path):
        st.error("❌ File model klasifikasi tidak ditemukan di folder model/. Pastikan 'class.h5' sudah di-upload.")
        return None, None

    # Load YOLO dan model klasifikasi
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(class_path, compile=False)
    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

if yolo_model is not None and classifier is not None:
    menu = st.sidebar.selectbox(
        "Pilih Mode:",
        ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
    )

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")  # pastikan RGB
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        if menu == "Deteksi Objek (YOLO)":
            # Deteksi objek menggunakan YOLO
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        elif menu == "Klasifikasi Gambar":
            # ==========================
            # Preprocessing Gambar
            # ==========================
            img_resized = img.resize((224, 224))  # sesuaikan ukuran input model
            img_array = np.array(img_resized).astype("float32")
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # normalisasi

            # ==========================
            # Prediksi
            # ==========================
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"### Hasil Prediksi: {class_index}")
            st.write("Probabilitas:", round(confidence * 100, 2), "%")

else:
    st.warning("⚠️ Model belum dimuat karena file model tidak ditemukan di folder `model/`.")

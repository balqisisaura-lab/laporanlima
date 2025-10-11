import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Load YOLO untuk deteksi objek
    yolo_model = YOLO("model/object.pt")
    # Load model klasifikasi (tanpa compile untuk mencegah error versi TF)
    classifier = tf.keras.models.load_model("model/class.h5", compile=False)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

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
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan input model
        img_array = np.array(img_resized).astype("float32")  # pastikan float32, bukan tuple
        img_array = np.expand_dims(img_array, axis=0)  # ubah ke bentuk (1,224,224,3)
        img_array = img_array / 255.0  # normalisasi

        st.write("Tipe data input:", type(img_array))
        st.write("Shape input:", img_array.shape)
        st.write("Dtype input:", img_array.dtype)

        # ==========================
        # Prediksi
        # ==========================
        prediction = classifier.predict(img_array)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", round(confidence * 100, 2), "%")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# Styling Keren (Tema Teknologi)
# ==============================
st.markdown("""
    <style>
        body {
            background-color: #0b0f19;
            color: #FFFFFF;
        }
        .title {
            text-align: center;
            font-size: 40px;
            color: #00FFFF;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7FFFD4;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00FFFF, #007BFF);
            color: white;
            border-radius: 12px;
            font-weight: bold;
            height: 3em;
            width: 100%;
            border: none;
            box-shadow: 0px 0px 20px #00FFFF;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #007BFF, #00FFFF);
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Header Dashboard
# ==============================
st.markdown("<h1 class='title'>🚀 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload gambar untuk diklasifikasi menggunakan model ResNet50</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# Fungsi Load Model
# ==============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model/object.pt", compile=False)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model TensorFlow: {e}")
        return None

model = load_model()

# ==============================
# Fungsi Prediksi
# ==============================
def predict_image(img, model):
    try:
        # 1️⃣ Ubah ukuran gambar ke 224x224
        img = img.resize((224, 224))
        img_array = np.array(img)

        # 2️⃣ Tambah dimensi batch
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        img_array = preprocess_input(img_array)

        # 3️⃣ Ambil fitur dari ResNet base (layer pertama)
        resnet_output = model.layers[0](img_array)
        pooled_features = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)

        # 4️⃣ Pastikan fitur berbentuk (1, 2048)
        pooled_features = np.expand_dims(pooled_features.numpy().flatten(), axis=0)

        # 5️⃣ Prediksi akhir
        prediction = model.predict(pooled_features)
        return prediction

    except Exception as e:
        st.error(f"⚠️ Error saat prediksi: {e}")
        return None

# ==============================
# Upload Gambar
# ==============================
uploaded_file = st.file_uploader("📂 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

    with col2:
        if st.button("🔍 Jalankan Prediksi"):
            with st.spinner("Model sedang menganalisis gambar... ⏳"):
                result = predict_image(img, model)

            if result is not None:
                st.success("✅ Prediksi Berhasil!")
                st.write("**Output Probabilitas:**")
                st.dataframe(result)
            else:
                st.error("❌ Prediksi gagal. Coba ulangi atau periksa model.")
else:
    st.info("⬆️ Silakan upload gambar untuk memulai prediksi.")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 AI Vision Dashboard — Powered by TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)

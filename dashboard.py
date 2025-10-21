import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ===================== CONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["YOLO Object Detection", "TensorFlow Rock–Paper–Scissors"]
)

# ==================== MODEL PYTORCH YOLO ====================
if model_choice == "YOLO Object Detection":
    st.header("🎯 Model YOLO - Object Detection")
    
    try:
        @st.cache_resource
        def load_yolo():
            return YOLO("model/object.pt")
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        st.success("✅ Model YOLO berhasil dimuat!")
        
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=["jpg", "jpeg", "png"],
            key="yolo"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Jalankan Deteksi", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    results = model(image)
                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        result_img = results[0].plot()
                        st.image(result_img, use_column_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Detail Deteksi")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.write(f"**{i}. {model.names[int(box.cls)]}**")
                            with col_b:
                                st.write(f"Confidence: **{box.conf[0]:.1%}**")
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file YOLO `.pt` ada di folder `model/`.")

# ==================== MODEL TENSORFLOW ROCK PAPER SCISSORS ====================
elif model_choice == "TensorFlow Rock–Paper–Scissors":
    st.header("🧠 Model TensorFlow - Rock Paper Scissors")
    
    try:
        @st.cache_resource
        def load_rps_model():
            return tf.keras.models.load_model("model/class.h5", compile=False)
        
        with st.spinner("Loading TensorFlow model..."):
            model = load_rps_model()
        
        st.success("✅ Model TensorFlow berhasil dimuat!")
        
        # Kelas model (urut sesuai urutan output layer kamu)
        class_names = ["Rock", "Paper", "Scissors"]
        
        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=["jpg", "jpeg", "png"],
            key="tf"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔮 Prediksi Gambar", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    # Preprocessing gambar
                    img_array = np.array(image.resize((224, 224))) / 255.0
                    
                    # Pastikan 3 channel (RGB)
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[-1] == 4:
                        img_array = img_array[..., :3]
                    
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    predictions = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_index]
                    confidence = predictions[0][predicted_index]
                    
                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", predicted_class)
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        with st.expander("📊 Detail Probabilitas Tiap Kelas"):
                            for i, prob in enumerate(predictions[0]):
                                st.progress(float(prob), text=f"{class_names[i]}: {prob:.4f}")
    
    except Exception as e:
        st.error(f"❌ Model tidak bisa dimuat: {str(e)}")
        st.info("""
        **💡 Tips:**
        1. Pastikan file model kamu `model_fixed.h5` ada di folder `model/`
        2. Pastikan model dilatih dengan 3 kelas: Rock, Paper, Scissors
        3. Kalau error karena versi TensorFlow, coba convert ulang model di lokal:
           ```python
           import tensorflow as tf
           model = tf.keras.models.load_model('model_fixed.h5')
           model.save('model_fixed_new.h5')
           ```
        """)

st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")

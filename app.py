import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Klasifikasi Kelapa",
    page_icon="🌴",
    layout="centered"
)

# =========================
# CSS (UI MODERN + ICON SUPPORT)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #eef2f7, #ffffff);
}

/* TITLE */
h1 {
    color: #2c3e50 !important;
    text-align: center;
}

/* TEXT */
p, label {
    color: #2c3e50 !important;
    font-weight: 500;
}

/* RADIO BUTTON STYLE */
div[role="radiogroup"] {
    display: flex;
    gap: 10px;
}

div[role="radiogroup"] label {
    background: white;
    padding: 12px 20px;
    border-radius: 12px;
    border: 2px solid #ddd;
    cursor: pointer;
    transition: 0.3s;
}

/* HOVER */
div[role="radiogroup"] label:hover {
    border: 2px solid #27ae60;
}

/* FILE UPLOADER */
.stFileUploader {
    background: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #ddd;
}

/* CAMERA */
[data-testid="stCameraInput"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #ddd;
}

/* RESULT BOX */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.saved_model.load('model/saved_model_kelapa')
    return model.signatures["serving_default"]

infer = load_model()

# =========================
# HEADER
# =========================
st.markdown("<h1>🌴 Klasifikasi Bibit Kelapa Pandan</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
    <span style='color:#7f8c8d;'>Sistem AI untuk menentukan kualitas bibit kelapa</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# PILIH INPUT (DENGAN ICON)
# =========================
option = st.radio(
    "📌 Pilih sumber gambar:",
    ["📂 Upload File", "📷 Gunakan Kamera"],
    horizontal=True
)

st.info("Silakan upload gambar atau gunakan kamera untuk analisis.")

img = None

# =========================
# INPUT
# =========================
if option == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)

elif option == "📷 Gunakan Kamera":
    camera_image = st.camera_input("Ambil gambar")
    if camera_image:
        img = Image.open(camera_image)

# =========================
# PROSES
# =========================
if img is not None:

    col1, col2 = st.columns(2)

    # tampil gambar
    with col1:
        st.image(img, caption="Gambar Input", use_column_width=True)

    # preprocessing (FIX FLOAT32)
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # loading
    with st.spinner("⏳ Sistem sedang menganalisis..."):
        time.sleep(1)
        pred = infer(tf.constant(img_array, dtype=tf.float32))
        score = float(list(pred.values())[0][0][0])

    # =========================
    # HASIL
    # =========================
    with col2:
        st.subheader("Hasil Analisis")

        if score > 0.5:
            st.markdown(
                f"<div class='result-box' style='background:#d4efdf; color:#1e8449;'>🟢 Bibit Unggul<br><b>{score*100:.2f}%</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box' style='background:#f5b7b1; color:#922b21;'>🔴 Bibit Tidak Unggul<br><b>{score*100:.2f}%</b></div>",
                unsafe_allow_html=True
            )

    st.divider()

    # =========================
    # GRAFIK
    # =========================
    st.subheader("📊 Grafik Confidence")

    labels = ['Tidak Unggul', 'Unggul']
    values = [1-score, score]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim([0,1])
    ax.set_ylabel("Probabilitas")

    st.pyplot(fig)
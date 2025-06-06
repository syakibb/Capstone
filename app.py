import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="EcoSort AI - Pilah Sampah Cerdas",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Kustom ---
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("File 'style.css' tidak ditemukan. Beberapa elemen mungkin tidak tampil optimal.")

# --- Inisialisasi Session State ---
if 'total_points' not in st.session_state:
    st.session_state.total_points = 0
if 'classification_log' not in st.session_state:
    st.session_state.classification_log = []
if 'last_processed_id' not in st.session_state:
    st.session_state.last_processed_id = None
if 'image_to_show' not in st.session_state:
    st.session_state.image_to_show = None
if 'result_to_show' not in st.session_state:
    st.session_state.result_to_show = None

# --- Muat Model dan Label Encoder ---
@st.cache_resource
def load_components():
    model = load_model('waste_classifier_fast_cnn_final.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

try:
    model, le = load_components()
except FileNotFoundError:
    st.error("File model atau label encoder tidak ditemukan. Pastikan ada di direktori yang sama.")
    st.stop()

# --- Fungsi-fungsi Bantuan ---
def preprocess_image(image_pil):
    img = image_pil.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def show_educational_info(class_name_encoded):
    # Menampilkan info edukasi seperti screenshot
    if class_name_encoded == "O":
        st.warning(
            """
            🌿 **Sampah Organik Terdeteksi!**
            Jenis sampah ini berasal dari sisa makhluk hidup dan mudah terurai alami.
            Contoh: Sisa makanan, daun, ranting, sayuran, buah.

            **💡 Tips Penanganan:**
            - Olah menjadi kompos untuk pupuk tanaman.
            - Beberapa jenis dapat dijadikan pakan ternak.
            - Pastikan tidak tercampur dengan sampah anorganik.
            """
        )
    elif class_name_encoded == "R":
        st.info(
            """
            🔩 **Sampah Anorganik Terdeteksi!**
            Jenis sampah ini umumnya sulit terurai alami, banyak yang dapat didaur ulang.
            Contoh: Botol plastik, kertas, kaleng, kaca, kemasan.

            **💡 Tips Penanganan:**
            - Pilah sesuai jenis material (plastik, kertas, logam, dll.).
            - Bersihkan dan keringkan sebelum disetor ke bank sampah.
            - Upayakan untuk mengurangi penggunaan produk sekali pakai.
            """
        )

# --- Tata Letak Aplikasi ---
st.markdown("<h1 style='text-align: center; color: #38761D;'>🌿 EcoSort AI: Pilah Sampah Jadi Mudah! ♻️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #66BB6A;'>Identifikasi Cerdas Sampah Organik & Anorganik</h3>", unsafe_allow_html=True)
st.markdown("<p class='main-description' style='text-align: center;'>Unggah atau ambil foto sampah Anda. AI akan langsung mengklasifikasi, memberi poin, dan edukasi!</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("🌟 Poin Anda 🌟")
    st.markdown(f"<p class='sidebar-text'>Total Poin: <strong>{st.session_state.total_points}</strong></p>", unsafe_allow_html=True)
    st.caption("Dapatkan 10 poin untuk setiap gambar baru!")
    st.header("📜 Log Klasifikasi (Sesi Ini)")
    if st.session_state.classification_log:
        for i, log_item in enumerate(reversed(st.session_state.classification_log[-5:])):
            st.markdown(f"<div class='log-entry'>{i+1}. {log_item['name']}: {log_item['class']} ({log_item['conf']:.2f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='sidebar-text'>Belum ada klasifikasi.</p>", unsafe_allow_html=True)
    if st.button("🔄 Reset Sesi"):
        keys_to_delete = [k for k in st.session_state.keys() if k not in ['model', 'le']]
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()
    st.markdown("---")
    st.subheader("Tentang Platform Ini")
    st.markdown("""<div class="sidebar-info-custom"><p>Platform cerdas ini dirancang untuk memudahkan masyarakat dalam mengidentifikasi sampah organik dan anorganik melalui teknologi machine learning. Dapatkan juga panduan edukatif untuk penanganan sampah yang lebih bijak.</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='sidebar-text' style='text-align: center;'>Lingkungan Bersih, Hidup Lebih Sehat!</p>", unsafe_allow_html=True)

# --- Kolom Utama & Input ---
col1, col2 = st.columns([2, 3])
image_buffer = None
with col1:
    st.subheader("🖼️ Input Gambar Sampah")
    input_method = st.radio("Pilih metode input:", ("Upload Gambar", "Gunakan Kamera"), horizontal=True)
    if input_method == "Upload Gambar":
        image_buffer = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    else:
        image_buffer = st.camera_input("Ambil foto sampah")

# --- Logika Pemrosesan Terpusat ---
if image_buffer is not None:
    # Buat ID unik untuk setiap input
    if hasattr(image_buffer, 'id'): # Input dari kamera memiliki atribut 'id'
        current_file_id = image_buffer.id
    else: # Input dari uploader tidak punya 'id', kita buat dari nama dan ukuran
        current_file_id = f"{image_buffer.name}-{image_buffer.size}"
    
    # Proses hanya jika ID file ini baru
    if current_file_id != st.session_state.last_processed_id:
        with st.spinner('AI sedang menganalisis... 🤔'):
            image = Image.open(image_buffer)
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)

            class_idx = np.argmax(predictions[0])
            class_name_encoded = le.classes_[class_idx]
            confidence = predictions[0][class_idx] * 100
        
        # Tambah poin dan log
        st.session_state.total_points += 10
        st.balloons()
        
        file_name = image_buffer.name if hasattr(image_buffer, 'name') else f"capture_{current_file_id}.jpg"
        display_class_name = "Organik" if class_name_encoded == "O" else "Anorganik"
        
        st.session_state.classification_log.append({
            "name": file_name,
            "class": display_class_name,
            "conf": confidence,
        })
        
        # Simpan hasil dan gambar untuk ditampilkan
        st.session_state.image_to_show = image
        st.session_state.result_to_show = {
            "name": file_name,
            "class_encoded": class_name_encoded,
            "display_class": display_class_name,
            "confidence": confidence
        }
        
        # Tandai file ini sudah diproses dan panggil rerun SATU KALI
        st.session_state.last_processed_id = current_file_id
        st.rerun()

# --- Tampilan Hasil ---
with col2:
    st.subheader("🔍 Hasil Klasifikasi & Info")
    
    if st.session_state.image_to_show:
        display_image = st.session_state.image_to_show.copy()
        display_image.thumbnail((450, 350)) # Batasi ukuran gambar
        
        st.markdown("<div class='image-container-centered'>", unsafe_allow_html=True)
        st.image(display_image, caption=f"Gambar: {st.session_state.result_to_show['name']}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.result_to_show:
        result = st.session_state.result_to_show
        
        # Box 1: Hasil deteksi
        if result['display_class'] == "Organik":
            st.success(f"Terdeteksi: {result['display_class']} (Akurasi: {result['confidence']:.2f}%)")
        else:
            st.error(f"Terdeteksi: {result['display_class']} (Akurasi: {result['confidence']:.2f}%)")
        
        # Box 2: Info Edukasi
        show_educational_info(result["class_encoded"])
    else:
        st.info("Hasil klasifikasi akan muncul di sini.")

st.markdown("---")
st.caption("Dibuat untuk membantu pengelolaan sampah yang lebih baik.")

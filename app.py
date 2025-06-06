import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
# import cv2

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
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'current_file_processed' not in st.session_state:
    st.session_state.current_file_processed = False
if 'uploader_key_counter' not in st.session_state: # Untuk mereset file uploader
    st.session_state.uploader_key_counter = 0


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
    st.error("File model atau label encoder tidak ditemukan. Pastikan 'waste_classifier_fast_cnn_final.h5' dan 'label_encoder.pkl' ada di direktori yang sama dengan app.py.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

def preprocess_image(image_pil):
    img = image_pil.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def show_educational_info_and_add_points(class_name_encoded, image_name_for_point_logic):
    points_earned = 0
    add_points_for_this_image = True

    # Cek apakah gambar dengan nama ini sudah pernah mendapatkan poin di log
    for log_entry in st.session_state.classification_log:
        if log_entry['name'] == image_name_for_point_logic and log_entry['points'] > 0:
            add_points_for_this_image = False
            break
    
    # Jika ini adalah entri log pertama, pasti tambahkan poin
    if not st.session_state.classification_log:
        add_points_for_this_image = True

    if class_name_encoded == "O":
        st.success(
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
        if add_points_for_this_image: points_earned = 10
    elif class_name_encoded == "R":
        st.warning(
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
        if add_points_for_this_image: points_earned = 10
    else:
        st.write("Tidak ada informasi edukasi untuk kategori ini.")

    if points_earned > 0:
        st.session_state.total_points += points_earned
        st.balloons()
    return points_earned

# --- Tata Letak Aplikasi ---

st.markdown("<h1 style='text-align: center; color: #38761D;'>🌿 EcoSort AI: Pilah Sampah Jadi Mudah! ♻️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #66BB6A;'>Identifikasi Cerdas Sampah Organik & Anorganik</h3>", unsafe_allow_html=True)
st.markdown("<p class='main-description' style='text-align: center;'>Unggah gambar sampah Anda, biarkan AI mengklasifikasi, dapatkan poin, dan pelajari cara penanganan yang tepat!</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("🌟 Poin Anda 🌟")
    st.markdown(f"<p class='sidebar-text'>Total Poin: <strong>{st.session_state.total_points}</strong></p>", unsafe_allow_html=True)
    st.caption("Dapatkan 10 poin untuk setiap klasifikasi gambar baru!")

    st.header("📜 Log Klasifikasi (Sesi Ini)")
    if st.session_state.classification_log:
        for i, log_item in enumerate(reversed(st.session_state.classification_log[-5:])):
            st.markdown(f"<div class='log-entry'>{i+1}. {log_item['name']}: {log_item['class']} ({log_item['conf']:.2f}%) - Poin: {log_item['points']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='sidebar-text'>Belum ada klasifikasi.</p>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Poin & Log Sesi Ini"):
        st.session_state.total_points = 0
        st.session_state.classification_log = []
        st.session_state.last_uploaded_filename = None
        st.session_state.current_file_processed = False
        st.session_state.uploader_key_counter += 1 # Ubah kunci uploader untuk meresetnya
        st.rerun()

    st.markdown("---")
    st.subheader("Tentang Platform Ini")
    st.markdown(
        """
        <div class="sidebar-info-custom">
        <p>Platform cerdas ini dirancang untuk memudahkan masyarakat dalam mengidentifikasi sampah organik dan anorganik melalui teknologi machine learning. Dapatkan juga panduan edukatif untuk penanganan sampah yang lebih bijak.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("<p class='sidebar-text' style='text-align: center;'>Lingkungan Bersih, Hidup Lebih Sehat!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🖼️ Input Gambar Sampah")
    input_method = st.radio("Pilih metode input:", ("Upload Gambar", "Gunakan Kamera"), horizontal=True, key="input_method_radio")

    image_to_process_for_model = None # Gambar PIL asli untuk model
    image_for_display = None # Gambar PIL yang mungkin di-resize untuk tampilan
    
    uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"

    if input_method == "Upload Gambar":
        uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"], key=uploader_key)
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_uploaded_filename or not st.session_state.current_file_processed:
                image_to_process_for_model = Image.open(uploaded_file)
                image_for_display = image_to_process_for_model.copy() # Buat salinan untuk display
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.session_state.current_file_processed = False
            else: # File sama dan sudah diproses, hanya siapkan untuk display ulang
                image_for_display = Image.open(uploaded_file) # Buka lagi hanya untuk display
    else: # Gunakan Kamera
        img_file_buffer = st.camera_input("Ambil foto sampah")
        if img_file_buffer is not None:
            # Beri nama unik untuk webcam capture untuk logika poin
            webcam_filename = f"webcam_capture_{len(st.session_state.classification_log)}.jpg"
            if webcam_filename != st.session_state.last_uploaded_filename or not st.session_state.current_file_processed:
                image_to_process_for_model = Image.open(img_file_buffer)
                image_for_display = image_to_process_for_model.copy()
                st.session_state.last_uploaded_filename = webcam_filename
                st.session_state.current_file_processed = False
            else:
                image_for_display = Image.open(img_file_buffer)


with col2:
    st.subheader("🔍 Hasil Klasifikasi & Info")
    if image_for_display is not None:
        # --- PENYESUAIAN UKURAN GAMBAR UNTUK TAMPILAN ---
        display_image_resized = image_for_display.copy()
        MAX_DISPLAY_HEIGHT = 350  # Tentukan tinggi maksimal gambar tampilan dalam piksel
        MAX_DISPLAY_WIDTH = 500   # Tentukan lebar maksimal gambar tampilan dalam piksel

        # Pertahankan rasio aspek sambil memastikan tidak melebihi max_height dan max_width
        img_width, img_height = display_image_resized.size
        ratio = min(MAX_DISPLAY_WIDTH / img_width, MAX_DISPLAY_HEIGHT / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Gunakan Image.Resampling.LANCZOS untuk PIL versi baru, Image.LANCZOS untuk yang lama
        try:
            display_image_resized = display_image_resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError: # Fallback untuk PIL versi lama
            display_image_resized = display_image_resized.resize((new_width, new_height), Image.LANCZOS)
        
        st.image(display_image_resized, caption=f'Gambar: {st.session_state.last_uploaded_filename}')
        # --- AKHIR PENYESUAIAN UKURAN GAMBAR ---

        if not st.session_state.current_file_processed and image_to_process_for_model is not None:
            with st.spinner('AI sedang menganalisis gambar... 🤔'):
                processed_img = preprocess_image(image_to_process_for_model)
                predictions = model.predict(processed_img)
                class_idx = np.argmax(predictions[0])
                class_name_encoded = le.classes_[class_idx]
                confidence = predictions[0][class_idx] * 100

            display_class_name = "Organik" if class_name_encoded == "O" else "Anorganik"
            
            if display_class_name == "Organik":
                st.markdown(f"<div class='stAlert stSuccess'>Terdeteksi: <strong>{display_class_name}</strong> (Akurasi: {confidence:.2f}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='stAlert stWarning'>Terdeteksi: <strong>{display_class_name}</strong> (Akurasi: {confidence:.2f}%)</div>", unsafe_allow_html=True)
            
            points_earned_this_round = show_educational_info_and_add_points(class_name_encoded, st.session_state.last_uploaded_filename)

            st.session_state.classification_log.append({
                "name": st.session_state.last_uploaded_filename,
                "class": display_class_name,
                "conf": confidence,
                "points": points_earned_this_round
            })
            st.session_state.current_file_processed = True
            st.rerun() # Rerun untuk update tampilan sidebar & mencegah re-proses

        elif st.session_state.current_file_processed and st.session_state.classification_log:
             # Tampilkan hasil terakhir jika file sama dan sudah diproses
            last_log = None
            for log in reversed(st.session_state.classification_log):
                if log['name'] == st.session_state.last_uploaded_filename:
                    last_log = log
                    break
            
            if last_log:
                display_class_name = last_log['class']
                confidence = last_log['conf']
                class_name_encoded_for_edu = "O" if display_class_name == "Organik" else "R"

                if display_class_name == "Organik":
                    st.markdown(f"<div class='stAlert stSuccess'>Terdeteksi: <strong>{display_class_name}</strong> (Akurasi: {confidence:.2f}%)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='stAlert stWarning'>Terdeteksi: <strong>{display_class_name}</strong> (Akurasi: {confidence:.2f}%)</div>", unsafe_allow_html=True)
                show_educational_info_and_add_points(class_name_encoded_for_edu, st.session_state.last_uploaded_filename) # Poin tidak akan bertambah
            else:
                st.info("Hasil klasifikasi akan muncul di sini.")
    else:
        st.info("Silakan upload gambar atau gunakan kamera untuk memulai klasifikasi.")

st.markdown("---")
st.caption("Dibuat untuk membantu pengelolaan sampah yang lebih baik.")
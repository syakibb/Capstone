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
# Pastikan path file CSS sudah benar
try:
    with open("static/style.css") as f:
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
# Menambahkan state untuk notifikasi agar tidak hilang setelah rerun
if 'notification' not in st.session_state:
    st.session_state.notification = None
if 'confidence_warning' not in st.session_state:
    st.session_state.confidence_warning = None


# --- Muat Model dan Label Encoder ---
@st.cache_resource
def load_components():
    model = load_model('model/waste_classifier_fast_cnn_final.h5')
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

try:
    model, le = load_components()
except Exception as e:
    st.error(f"Error saat memuat model: {e}. Pastikan file model dan label encoder ada.")
    st.stop()

# --- Fungsi-fungsi Bantuan ---
def preprocess_image(image_pil):
    img = image_pil.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Tata Letak Aplikasi ---
st.markdown("<h1 class='main-title' style='text-align: center; color: #38761D;'>🌿 EcoSort AI: Pilah Sampah Jadi Mudah! ♻️</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title' style='text-align: center; color: #66BB6A;'>Identifikasi Sampah Organik & Anorganik dengan Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("<p class='main-description' style='text-align: center;'>Unggah atau ambil foto sampah Anda. AI akan langsung mengklasifikasi, memberi poin, dan edukasi!</p>", unsafe_allow_html=True)

# --- Tempat untuk Menampilkan Notifikasi setelah Rerun ---
if st.session_state.notification:
    msg = st.session_state.notification
    st.success(f"Kamu mendapatkan {msg['points']} poin karena mengklasifikasikan sampah {msg['class']}! 🌟")
    st.session_state.notification = None # Hapus setelah ditampilkan

if st.session_state.confidence_warning:
    st.warning(st.session_state.confidence_warning)
    st.session_state.confidence_warning = None # Hapus setelah ditampilkan

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌟 Poin & Level Anda 🌟")

    points = st.session_state.total_points
    if points < 50:
        level, level_number, progress, next_level_points = "🌱 Pemula Hijau", 1, points / 50, 50
    elif points < 150:
        level, level_number, progress, next_level_points = "🌿 Penjaga Alam", 2, (points - 50) / 100, 150
    elif points < 300:
        level, level_number, progress, next_level_points = "♻️ Pahlawan Daur Ulang", 3, (points - 150) / 150, 300
    else:
        level, level_number, progress, next_level_points = "🌍 Legenda Bumi", 4, 1.0, "Max"

    st.markdown(f"<p class='sidebar-text'><strong>Level:</strong> {level}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='sidebar-text'><strong>Total Poin:</strong> {points}</p>", unsafe_allow_html=True)
    st.progress(progress)

    if level_number < 4:
        st.caption(f"{next_level_points - points} poin lagi untuk naik level!")
    else:
        st.caption("🎉 Anda telah mencapai level tertinggi!")
    
    st.header("📜 Log Klasifikasi")
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
    st.markdown("""<div class="sidebar-info-custom"><p>Platform cerdas ini dirancang untuk memudahkan masyarakat dalam memilah sampah organik dan anorganik melalui teknologi machine learning.</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='sidebar-text' style='text-align: center;'>Lingkungan Bersih, Hidup Lebih Sehat!</p>", unsafe_allow_html=True)


# --- KONTEN UTAMA ---
col1, col2 = st.columns([2, 3], gap="large")

# --- KOLOM KIRI (INPUT) ---
with col1:
    # Kita tambahkan div wrapper di sini agar strukturnya konsisten
    st.markdown("<div class='input-container-wrapper'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("📸 Unggah Gambar Sampah")
        input_method = st.radio("Pilih metode input:", ("Upload Gambar", "Gunakan Kamera"), horizontal=True, label_visibility="collapsed")
        
        image_buffer = None
        if input_method == "Upload Gambar":
            image_buffer = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            image_buffer = st.camera_input("Ambil foto sampah", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Logika Pemrosesan Terpusat ---
if image_buffer is not None:
    current_file_id = image_buffer.id if hasattr(image_buffer, 'id') else f"{image_buffer.name}-{image_buffer.size}"
    
    if current_file_id != st.session_state.last_processed_id:
        with st.spinner('AI sedang menganalisis... 🤔'):
            image = Image.open(image_buffer)
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)

            class_idx = np.argmax(predictions[0])
            class_name_encoded = le.classes_[class_idx]
            confidence = predictions[0][class_idx] * 100
        
        display_class_name = "Organik" if class_name_encoded == "O" else "Anorganik"
        
        # --- LOGIKA POIN BARU ---
        points_to_add = 8 if class_name_encoded == "O" else 10 # Organik=8, Anorganik=10
        st.session_state.total_points += points_to_add
        st.balloons()
        
        # --- PERSIAPKAN NOTIFIKASI UNTUK DITAMPILKAN SETELAH RERUN ---
        st.session_state.notification = {
            "points": points_to_add,
            "class": display_class_name.lower()
        }

        # --- PERSIAPKAN PERINGATAN AKURASI JIKA PERLU ---
        if confidence < 70:
            st.session_state.confidence_warning = f"⚠️ Akurasi di bawah 70% ({confidence:.1f}%). Gambar mungkin kurang jelas atau tidak fokus. Coba lagi dengan gambar yang lebih baik."
        
        file_name = image_buffer.name if hasattr(image_buffer, 'name') else f"capture_{current_file_id}.jpg"
        
        st.session_state.classification_log.append({
            "name": file_name, "class": display_class_name, "conf": confidence,
        })
        
        st.session_state.image_to_show = image
        st.session_state.result_to_show = {
            "name": file_name,
            "class_encoded": class_name_encoded,
            "display_class": display_class_name,
            "confidence": confidence
        }
        
        st.session_state.last_processed_id = current_file_id
        st.rerun()

# --- KOLOM KANAN (HASIL) ---
with col2:
    if not st.session_state.result_to_show:
        # Tampilan placeholder juga kita bungkus dengan div agar sejajar
        st.markdown("<div class='result-container-placeholder'>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("📊 Hasil Klasifikasi & Info")
            st.info("Hasil klasifikasi akan muncul di sini.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        result = st.session_state.result_to_show

        # Tentukan kelas CSS untuk kontainer hasil
        if result['class_encoded'] == "O":
            result_container_class = "result-container-organic"
        else:
            result_container_class = "result-container-anorganic"

        # Bungkus seluruh kontainer hasil dengan div yang memiliki kelas dinamis
        st.markdown(f"<div class='{result_container_class}'>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("📊 Hasil Klasifikasi & Info")

            display_image = st.session_state.image_to_show.copy()

            # Tentukan kelas untuk border gambar dinamis
            if result['class_encoded'] == "O":
                image_container_class = "organic-container"
            else:
                image_container_class = "anorganic-container"

            st.markdown(f"<div class='{image_container_class}'>", unsafe_allow_html=True)
            st.image(display_image, caption=f"Gambar: {result['name']}", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Header hasil
            logo = "🍂" if result['class_encoded'] == "O" else "♻️"
            st.markdown(f"<h4 class='result-header'>{logo} Terdeteksi: <strong>{result['display_class']}</strong> (Akurasi: {result['confidence']:.2f}%)</h4>", unsafe_allow_html=True)

            # Expander untuk edukasi
            with st.expander("💡 Lihat Tips Penanganan"):
                if result["class_encoded"] == "O":
                    st.markdown("""
                    Jenis sampah ini berasal dari sisa makhluk hidup dan mudah terurai alami.
                    - **Contoh**: Sisa makanan, daun, ranting, sayuran, buah.
                    - **Aksi**: Olah menjadi kompos untuk pupuk tanaman atau beberapa jenis dapat dijadikan pakan ternak.
                    """)
                else:
                    st.markdown("""
                    Jenis sampah ini umumnya sulit terurai alami, namun banyak yang dapat didaur ulang.
                    - **Contoh**: Botol plastik, kertas, kaleng, kaca, kemasan.
                    - **Aksi**: Pilah sesuai jenis material, bersihkan, dan keringkan sebelum disetor ke bank sampah.
                    """)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Bagian Edukasi Tambahan ---
st.markdown("---")
st.subheader("⚠️ Dampak Membuang Sampah Sembarangan")

# Fungsi bantuan untuk membuat kartu info agar kode tidak berulang
def info_card(icon, title, text, card_class):
    st.markdown(f"""
    <div class="info-card {card_class}">
        <div class="card-icon">{icon}</div>
        <div class="card-content">
            <h5>{title}</h5>
            <p>{text}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Membuat TABS
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Lingkungan", "❤️ Kesehatan", "👥 Sosial & Ekonomi", "💡 Solusi"])

with tab1: # Konten Tab Lingkungan
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        info_card("💧", "Pencemaran Air", "Sampah yang dibuang ke sungai, danau, atau laut menyebabkan pencemaran air. Zat berbahaya dari sampah meresap ke dalam air tanah dan mencemari sumber air minum.", "card-problem")
        info_card("💨", "Pencemaran Udara", "Pembakaran sampah menghasilkan gas beracun dan partikel berbahaya yang mencemari udara. Gas rumah kaca dari sampah organik yang membusuk berkontribusi pada perubahan iklim.", "card-problem")
    with col2:
        info_card("🐠", "Kerusakan Ekosistem", "Sampah plastik membahayakan kehidupan laut dan satwa liar. Hewan dapat terperangkap atau memakan sampah plastik yang menyebabkan kematian dan gangguan ekosistem.", "card-problem")
        info_card("🌊", "Banjir", "Sampah yang menyumbat saluran air dan drainase menyebabkan banjir saat hujan deras. Banjir dapat merusak infrastruktur dan menyebabkan kerugian ekonomi yang besar.", "card-problem")

    st.markdown("""
    <div class="fact-box fact-problem">
        <b>Fakta:</b> Setiap tahun, sekitar 8 juta ton sampah plastik berakhir di lautan. Jika tidak ada perubahan, pada tahun 2050 diperkirakan akan ada lebih banyak plastik daripada ikan di laut (berdasarkan berat).
    </div>
    """, unsafe_allow_html=True)

with tab2: # Konten Tab Kesehatan
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        info_card("🦟", "Penyakit Menular", "Sampah menjadi tempat berkembang biak nyamuk, tikus, dan serangga pembawa penyakit seperti demam berdarah, malaria, leptospirosis, dan diare.", "card-problem")
        info_card("🚰", "Kontaminasi Air Minum", "Zat berbahaya dari sampah dapat meresap ke dalam tanah dan mencemari sumber air minum, menyebabkan penyakit seperti diare, kolera, dan keracunan.", "card-problem")
    with col2:
        info_card("😮‍💨", "Gangguan Pernapasan", "Gas beracun dari pembakaran sampah dan dekomposisi sampah dapat menyebabkan gangguan pernapasan, iritasi mata, dan memperburuk kondisi asma.", "card-problem")
        info_card("⏳", "Dampak Jangka Panjang", "Paparan jangka panjang terhadap bahan kimia berbahaya dari sampah dapat meningkatkan risiko kanker, gangguan hormonal, dan masalah kesehatan kronis lainnya.", "card-problem")

    st.markdown("""
    <div class="fact-box fact-problem">
        <b>Fakta:</b> Menurut WHO, sekitar 2 miliar orang di dunia tidak memiliki akses ke pengelolaan sampah yang memadai, yang berkontribusi pada 4 juta kematian per tahun akibat penyakit yang berhubungan dengan sanitasi buruk.
    </div>
    """, unsafe_allow_html=True)

with tab3: # Konten Tab Sosial & Ekonomi
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        info_card("💸", "Biaya Pembersihan", "Pemerintah harus mengeluarkan dana besar untuk membersihkan sampah yang dibuang sembarangan. Dana ini seharusnya bisa dialokasikan untuk pendidikan, kesehatan, atau infrastruktur.", "card-problem")
        info_card("🏞️", "Dampak pada Pariwisata", "Destinasi wisata yang kotor dan dipenuhi sampah kehilangan daya tarik. Hal ini berdampak pada pendapatan dari sektor pariwisata dan mata pencaharian masyarakat lokal.", "card-problem")
    with col2:
        info_card("📉", "Penurunan Nilai Properti", "Area yang dipenuhi sampah mengalami penurunan nilai properti. Rumah dan bisnis di sekitar tempat pembuangan sampah ilegal bernilai lebih rendah di pasar properti.", "card-problem")
        info_card("⚖️", "Ketidakadilan Sosial", "Tempat pembuangan sampah ilegal sering berada di dekat komunitas berpenghasilan rendah, menciptakan ketidakadilan lingkungan. Masyarakat ini menanggung beban kesehatan dan sosial yang tidak proporsional.", "card-problem")

    st.markdown("""
    <div class="fact-box fact-problem">
        <b>Fakta:</b> Menurut Bank Dunia, biaya ekonomi dari pengelolaan sampah yang buruk di negara berkembang bisa mencapai 5-10% dari PDB lokal, termasuk biaya kesehatan, produktivitas yang hilang, dan kerusakan lingkungan.
    </div>
    """, unsafe_allow_html=True)

with tab4: # Konten Tab Solusi
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        info_card("♻️", "Prinsip 3R", "Terapkan prinsip Reduce (kurangi), Reuse (gunakan kembali), dan Recycle (daur ulang) dalam kehidupan sehari-hari. Kurangi penggunaan plastik sekali pakai dan pilih produk ramah lingkungan.", "card-solution")
        info_card("📚", "Edukasi Masyarakat", "Tingkatkan kesadaran masyarakat tentang dampak membuang sampah sembarangan melalui kampanye, pendidikan di sekolah, dan media sosial.", "card-solution")
    with col2:
        info_card("🗑️", "Pemilahan Sampah", "Pisahkan sampah organik dan anorganik di rumah. Sampah organik dapat dikomposkan, sementara sampah anorganik dapat didaur ulang atau dibuang dengan benar.", "card-solution")
        info_card("📜", "Kebijakan dan Penegakan", "Dukung kebijakan pengelolaan sampah yang baik dan penegakan hukum terhadap pembuangan sampah sembarangan. Laporkan pelanggaran ke pihak berwenang.", "card-solution")

    st.markdown("""
    <div class="fact-box fact-solution">
        <b>Aksi Nyata:</b> Mulailah dari hal kecil seperti membawa tas belanja sendiri, botol minum isi ulang, dan menolak sedotan plastik. Bergabunglah dengan komunitas peduli lingkungan untuk kegiatan bersih-bersih lingkungan secara berkala.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# Membuat Kontainer CTA (Call to Action)
st.markdown("""
<div class="cta-container">
    <h2>Jadilah Bagian dari Solusi, Bukan Masalah!</h2>
    <p>Mulai dari diri sendiri untuk memilah sampah dan membuangnya pada tempatnya. Bersama-sama kita bisa menciptakan lingkungan yang lebih bersih dan sehat.</p>
    <a href="#" class="cta-button">Sebarkan Website Ini</a>
</div>
""", unsafe_allow_html=True)


# Membuat Footer
st.markdown("""
<div class="footer-container">
    <div class="footer-left">
        © 2025 EcoSort AI - Klasifikasi Sampah Organik dan Anorganik - Dibuat oleh tim CC25-CR438		
    </div>
    <div class="footer-right">
        <a href="https://github.com/syakibb/ecosort-ai" target="_blank" title="Lihat kode di GitHub">❓</a>
    </div>
</div>
""", unsafe_allow_html=True)

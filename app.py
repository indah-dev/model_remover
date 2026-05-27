import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import io
import base64
from pdf2image import convert_from_bytes
from huggingface_hub import hf_hub_download  # Import baru untuk Hugging Face

# ==============================================================================
# 1. KONFIGURASI DASHBOARD
# ==============================================================================
st.set_page_config(
    page_title="Sistem Restorasi Dokumen KCV",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. INJEKSI CSS REVISI (LATAR BELAKANG & VISIBILITAS TEKS)
# ==============================================================================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Pastikan huruf besar/kecil nama file "bg-masthead.jpg" sama persis dengan yang ada di GitHub
bg_image_base64 = get_base64_of_bin_file("bg-masthead.jpg")

if bg_image_base64:
    bg_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(30, 35, 35, 0.50), rgba(30, 35, 35, 0.50)), url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = """
    .stApp {
        background-color: #2D3231;
    }
    """

st.markdown(f"""
    <style>
    {bg_css}
    
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Tipografi */
    h1, h2, h3, h4, p, span, li {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #FFFFFF;
    }}

    /* Glassmorphism Card */
    .glass-card {{
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}

    /* Tombol Stylish */
    .stDownloadButton > button, .stButton > button {{
        background-color: #1D809F;
        color: white !important;
        border-radius: 30px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stDownloadButton > button:hover, .stButton > button:hover {{
        background-color: #155d74;
        transform: scale(1.03);
    }}

    /* Kotak Upload */
    .stFileUploader > div > div {{
        background-color: rgba(29, 128, 159, 0.15);
        border: 2px dashed #1D809F;
        border-radius: 15px;
        padding: 25px;
    }}

    /* Watermark Style */
    .watermark {{
        text-align: center;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 60px;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER UTAMA APLIKASI
# ==============================================================================
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 45px; text-shadow: 2px 2px 4px rgba(0,0,0,0.6);'>Document Shadow Removal</h1>", unsafe_allow_html=True)

# ==============================================================================
# 3. BAGIAN INTRODUKSI APLIKASI
# ==============================================================================
st.markdown("""
<div class='glass-card'>
    <h3 style='margin-top:0; color:#00F2FE;'>💡 Tentang Aplikasi</h3>
    <p style='text-align: justify; line-height: 1.6; font-size: 1.1rem;'>
        Pernahkah Anda memfoto dokumen penting (seperti Kartu Keluarga, nota, atau halaman buku) menggunakan HP, tetapi hasilnya gelap sebelah karena tertutup <b>bayangan tangan atau bayangan HP Anda sendiri</b>? 
    </p>
    <p style='text-align: justify; line-height: 1.6; font-size: 1.1rem;'>
        Aplikasi ini hadir sebagai solusi instan untuk masalah tersebut! Cukup masukkan foto dokumen yang berbayang, dan kecerdasan buatan (AI) di dalam sistem ini akan langsung mendeteksi lalu <b>menyapu bersih bayangan gelap tersebut</b> agar kertas dokumen Anda kembali putih, bersih, dan rapi secara merata.
    </p>
    <h4 style='color: #00F2FE; margin-top: 20px;'>✨ Keunggulan Utama:</h4>
    <ul style='line-height: 1.8; font-size: 1.05rem; padding-left: 20px;'>
        <li><b>Warna Asli Tidak Luntur:</b> Aplikasi pemindai biasa umumnya memaksa dokumen menjadi hitam-putih kaku yang merusak gambar asli. Sistem ini berbeda; warna-warna penting seperti <b>stempel resmi biru/merah, tanda tangan pulpen, logo, maupun grafik berwarna</b> akan tetap dijaga agar tetap asli dan tajam.</li>
        <li><b>Tulisan Tetap Jelas:</b> Proses pembersihan fokus hanya pada bayangan yang mengganggu, sehingga huruf atau teks di bawah bayangan tidak akan ikut terhapus atau kabur.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. METRIK & MANAJEMEN MODEL
# ==============================================================================
def psnr_metric(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    return 20 * tf.math.log(1.0 / (tf.sqrt(mse) + 1e-7)) / tf.math.log(10.0)

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

@st.cache_resource
def load_unet_model():
    # Mengunduh model langsung dari Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="indah-dev/model-restorasi-kcv", 
        filename="model_unet_rgb_final.keras"
    )
    
    # Memuat model yang sudah diunduh beserta custom objects
    return keras.models.load_model(
        model_path, 
        custom_objects={'psnr_metric': psnr_metric, 'ssim_metric': ssim_metric}
    )

# ==============================================================================
# 5. SIDEBAR KONTROL
# ==============================================================================
with st.sidebar:
    st.markdown("<h2>⚙️ Parameter Sistem</h2>", unsafe_allow_html=True)
    st.write("---")
    try:
        model = load_unet_model()
        st.success("🤖 Status AI: Siap Digunakan")
    except Exception as e:
        st.error(f"🔴 Status AI: Gagal Dimuat. Detail: {e}")
        st.stop()
        
    st.markdown("<div class='glass-card' style='padding: 15px;'>", unsafe_allow_html=True)
    max_dimension = st.sidebar.slider("Dimensi Maksimal Resize", min_value=400, max_value=1200, value=800, step=100)
    padding_multiple = st.sidebar.select_slider("Faktor Bantalan (Padding)", options=[16, 32, 64], value=32)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Eksperimen Konsentrasi KCV")

# ==============================================================================
# 6. LOGIKA PREPROCESSING
# ==============================================================================
def pad_to_multiple(img_array, multiple=32):
    h, w = img_array.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return padded_img, h, w

def smart_resize_for_ai(image_array, max_dim=800):
    h, w = image_array.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_array

# ==============================================================================
# 7. RUANG KERJA UTAMA (WORKSPACE UPLOAD)
# ==============================================================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📥 Tarik & Lepas Dokumen Anda di Sini (Mendukung Gambar Tunggal maupun File PDF Multi-Halaman)", 
    type=["jpg", "jpeg", "png", "pdf"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    filename = uploaded_file.name
    pages = []
    
    if filename.lower().endswith('.pdf'):
        with st.spinner("🔄 Mengekstrak halaman PDF secara berurutan..."):
            pdf_bytes = uploaded_file.read()
            # JALUR POPPLER WINDOWS SUDAH DIHAPUS AGAR SESUAI DENGAN SERVER CLOUD LINUX
            pages = convert_from_bytes(pdf_bytes, dpi=150)
    else:
        image_raw = Image.open(uploaded_file).convert('RGB')
        pages = [image_raw]

    enhanced_pages = []

    for i, page in enumerate(pages):
        st.markdown(f"<h3 style='color: #00F2FE;'>📄 Memproses Halaman {i+1} dari {len(pages)}</h3>", unsafe_allow_html=True)
        
        img_arr = np.array(page, dtype=np.float32) / 255.0
        
        with st.spinner(f"Membasmi bayangan pada halaman {i+1}... Mohon tunggu sebentar..."):
            img_arr_resized = smart_resize_for_ai(img_arr, max_dim=max_dimension)
            padded_img, orig_h, orig_w = pad_to_multiple(img_arr_resized, multiple=padding_multiple)
            
            img_batch = np.expand_dims(padded_img, axis=0)
            pred = model.predict(img_batch, verbose=0)[0]
            
            pred_cropped = pred[:orig_h, :orig_w, :]
            pred_uint8 = (np.clip(pred_cropped, 0, 1) * 255).astype(np.uint8)
            cleaned_page_image = Image.fromarray(pred_uint8)
            enhanced_pages.append(cleaned_page_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-weight: bold; margin-top:0;'>Foto Dokumen Asli (Kotor/Berbayang)</p>", unsafe_allow_html=True)
            st.image(img_arr_resized, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-weight: bold; margin-top:0;'>Hasil Bersih Bebas Bayangan</p>", unsafe_allow_html=True)
            st.image(cleaned_page_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # PANEL EXPORT AKHIR
    st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Proses Selesai! Berkas Siap Diunduh</h3>", unsafe_allow_html=True)
    
    export_buffer = io.BytesIO()
    if filename.lower().endswith('.pdf'):
        enhanced_pages[0].save(export_buffer, format="PDF", save_all=True, append_images=enhanced_pages[1:])
        download_name = f"RESTORED_{filename}"
        mime_type = "application/pdf"
    else:
        enhanced_pages[0].save(export_buffer, format="JPEG")
        download_name = f"RESTORED_{filename.split('.')[0]}.jpg"
        mime_type = "image/jpeg"
        
    st.download_button(
        label=f"💾 UNDUH BERKAS HASIL RESTORASI",
        data=export_buffer.getvalue(),
        file_name=download_name,
        mime=mime_type
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 8. WATERMARK HAK CIPTA PENULIS
# ==============================================================================
st.markdown("<div class='watermark'>created by indah lestari</div>", unsafe_allow_html=True)

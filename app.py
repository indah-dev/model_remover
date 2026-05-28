import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import io
import base64
import gc
from pdf2image import convert_from_bytes
from huggingface_hub import hf_hub_download

# ==============================================================================
# 1. KONFIGURASI DASHBOARD
# ==============================================================================
st.set_page_config(
    page_title="Sistem Restorasi Dokumen KCV",
    page_icon="✨",
    layout="wide"
)

# ==============================================================================
# 2. INJEKSI CSS (TAMPILAN DIPERTAHANKAN + KOTAK UPLOAD DARK MODE)
# ==============================================================================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

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

    /* Kotak Upload Dark Mode */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: #1F2229 !important;
        border: 1px solid #3E4351 !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderDropzone"] div, 
    [data-testid="stFileUploaderDropzone"] span, 
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {{
        color: #E2E8F0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: #0F1115 !important;
        color: #FFFFFF !important;
        border: 1px solid #3E4351 !important;
        border-radius: 6px !important;
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
    model_path = hf_hub_download(
        repo_id="dalgiuyu/model-restorasi-kcv", 
        filename="model_unet_rgb_final.keras"
    )
    return keras.models.load_model(
        model_path, 
        custom_objects={'psnr_metric': psnr_metric, 'ssim_metric': ssim_metric}
    )

# ==============================================================================
# 5. PARAMETER DEFAULT (BERJALAN DI LATAR BELAKANG)
# ==============================================================================
max_dimension = 800
padding_multiple = 32

try:
    model = load_unet_model()
except Exception as e:
    st.error(f"🔴 AI Gagal Dimuat. Pastikan Repositori Hugging Face diset 'Public' dan namanya benar! Detail: {e}")
    st.stop()

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
    "📥 Tarik & Lepas Dokumen Anda di Sini (Maksimal 200 MB)", 
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=False
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Pengecekan ukuran file
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 200:
        st.error(f"❌ Ukuran file terlalu besar ({file_size_mb:.1f} MB). Batas maksimal adalah 200 MB.")
        st.stop()

    filename = uploaded_file.name
    pages = []
    
    if filename.lower().endswith('.pdf'):
        with st.spinner("🔄 Mengekstrak halaman PDF... (Tunggu sebentar jika file tebal)"):
            pdf_bytes = uploaded_file.read()
            # Turunkan DPI agar memori server tidak penuh
            pages = convert_from_bytes(pdf_bytes, dpi=120)
    else:
        image_raw = Image.open(uploaded_file).convert('RGB')
        pages = [image_raw]

    enhanced_pages = []
    total_pages = len(pages)
    
    st.markdown(f"<p style='color: #00F2FE; font-weight: bold;'>Memproses {total_pages} halaman dokumen...</p>", unsafe_allow_html=True)
    progress_bar = st.progress(0)

    for i, page in enumerate(pages):
        status_text = st.empty()
        status_text.markdown(f"**Membasmi bayangan pada halaman {i+1} dari {total_pages}...**")
        
        img_arr = np.array(page, dtype=np.float32) / 255.0
        
        img_arr_resized = smart_resize_for_ai(img_arr, max_dim=max_dimension)
        padded_img, orig_h, orig_w = pad_to_multiple(img_arr_resized, multiple=padding_multiple)
        
        img_batch = np.expand_dims(padded_img, axis=0)
        pred = model.predict(img_batch, verbose=0)[0]
        
        pred_cropped = pred[:orig_h, :orig_w, :]
        pred_uint8 = (np.clip(pred_cropped, 0, 1) * 255).astype(np.uint8)
        cleaned_page_image = Image.fromarray(pred_uint8)
        enhanced_pages.append(cleaned_page_image)
        
        # --- TRIK 1: LIMITASI RENDER UI BROWSER ---
        # Hanya tampilkan preview perbandingan untuk Halaman Pertama (index 0) saja
        if i == 0:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-weight: bold; margin-top:0;'>Foto Asli (Preview Halaman 1)</p>", unsafe_allow_html=True)
                st.image(img_arr_resized, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-weight: bold; margin-top:0;'>Hasil Bersih (Preview Halaman 1)</p>", unsafe_allow_html=True)
                st.image(cleaned_page_image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Update Progress Bar
        progress_bar.progress((i + 1) / total_pages)
        status_text.empty()
        
        # --- TRIK 2: GARBAGE COLLECTION ---
        # Bersihkan memori RAM secara paksa setiap selesai 1 halaman
        del page, img_arr, img_arr_resized, padded_img, img_batch, pred, pred_cropped, pred_uint8
        gc.collect()

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
        label=f"💾 UNDUH BERKAS HASIL RESTORASI ({total_pages} Halaman)",
        data=export_buffer.getvalue(),
        file_name=download_name,
        mime=mime_type
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 8. WATERMARK HAK CIPTA PENULIS
# ==============================================================================
st.markdown("<div class='watermark'>created by indah lestari</div>", unsafe_allow_html=True)

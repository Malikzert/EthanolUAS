import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# Setup paths relatif
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logreg_model.pkl")
EXAMPLE_PATH = os.path.join(BASE_DIR, "example.txt")  # file contoh input

# --------------------------------------
# Load model + scaler
# --------------------------------------
@st.cache_data
def load_model(path):
    model_package = joblib.load(path)
    return model_package["scaler"], model_package["logreg"]

scaler, logreg = load_model(MODEL_PATH)

# --------------------------------------
# Judul aplikasi
# --------------------------------------
st.title("Prediksi Level Etanol")
st.markdown("""
Masukkan 1751 nilai sinyal etanol, pisahkan dengan koma (,).  
Contoh: `-0.9357,-0.9360,-0.9361,...`
""")

# ======================================================
# TAMBAHAN: Pilih metode input (manual / file)
# ======================================================
st.subheader("Metode Input Data")
input_method = st.radio(
    "Pilih cara memasukkan data:",
    ("Input Manual", "Upload File")
)

input_list = None  # variabel bersama

# --------------------------------------
# Input Manual (ASLI – TIDAK DIUBAH)
# --------------------------------------
if input_method == "Input Manual":
    user_input = st.text_area("Masukkan nilai (1751 fitur)")

# --------------------------------------
# Input File (TAMBAHAN)
# --------------------------------------
elif input_method == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload file .txt atau .csv (1751 fitur)",
        type=["txt", "csv"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".txt"):
                content = uploaded_file.read().decode("utf-8")
                input_list = [float(x.strip()) for x in content.split(",")]

            elif uploaded_file.name.endswith(".csv"):
                data = np.loadtxt(uploaded_file, delimiter=",")
                input_list = data.flatten().tolist()

            st.success(f"File berhasil dibaca ({len(input_list)} fitur)")

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            input_list = None

# --------------------------------------
# Tombol Prediksi (LOGIKA ASLI + PENYESUAIAN MINIMAL)
# --------------------------------------
if st.button("Prediksi"):
    try:
        # Ambil input dari manual jika dipilih
        if input_method == "Input Manual":
            input_list = [float(x.strip()) for x in user_input.split(",")]

        # Validasi input
        if input_list is None:
            st.error("Data input belum tersedia.")
        elif len(input_list) != 1751:
            st.error(f"Input harus 1751 nilai, sekarang: {len(input_list)}")
        else:
            # Konversi ke array 2D
            input_array = np.array(input_list).reshape(1, -1)

            # Scaling
            input_scaled = scaler.transform(input_array)

            # Prediksi kelas dan probabilitas
            pred = logreg.predict(input_scaled)
            pred_proba = logreg.predict_proba(input_scaled)[0]

            # Mapping prediksi ke kadar alkohol
            def pred_to_alcohol(pred):
                if pred == 1:
                    return "35%"
                elif pred == 2:
                    return "38%"
                elif pred == 3:
                    return "40%"
                elif pred == 4:
                    return "45%"
                else:
                    return "Unknown"

            kadar_alcohol = pred_to_alcohol(pred[0])
            st.success(
                f"Hasil prediksi level etanol: {pred[0]} "
                f"(kadar alkohol: {kadar_alcohol})"
            )

            # --------------------------------------
            # Statistik input (ASLI)
            # --------------------------------------
            st.subheader("Statistik Input")
            st.write("Rata-rata:", np.mean(input_list))
            st.write("Median:", np.median(input_list))
            st.write("Standar deviasi:", np.std(input_list))
            st.write("Min:", np.min(input_list))
            st.write("Max:", np.max(input_list))

            # Distribusi nilai
            st.subheader("Distribusi Nilai Sinyal")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(input_list, bins=50, kde=True, ax=ax)
            ax.set_xlabel("Nilai Sinyal")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

            # Probabilitas prediksi
            st.subheader("Probabilitas Prediksi per Kelas")
            classes = logreg.classes_
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=classes, y=pred_proba, ax=ax2)
            ax2.set_xlabel("Kelas")
            ax2.set_ylabel("Probabilitas")
            st.pyplot(fig2)

            # Analisis kelas dominan
            max_class_idx = np.argmax(pred_proba)
            st.info(
                f"Model cenderung memprediksi kelas `{classes[max_class_idx]}` "
                f"karena probabilitas tertinggi "
                f"({pred_proba[max_class_idx]*100:.2f}%)."
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# --------------------------------------
# Contoh input otomatis dari example.txt (ASLI)
# --------------------------------------
if st.checkbox("Isi contoh otomatis"):
    try:
        with open(EXAMPLE_PATH, "r") as f:
            example_content = f.read().strip()
        st.text_area("Contoh input", value=example_content, height=150)
    except Exception as e:
        st.error(f"Gagal membaca file example.txt: {e}")

import streamlit as st
import pandas as pd
import joblib
import sqlite3
import datetime

# --- PENGATURAN DATABASE (TETAP SAMA) ---
DB_FILE = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sleep_quality INTEGER,
            headaches INTEGER,
            academic_performance INTEGER,
            study_load INTEGER,
            extracurricular INTEGER,
            prediction_result TEXT,
            prediction_probability REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(inputs, result, probability):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO predictions (
            timestamp, sleep_quality, headaches, academic_performance, 
            study_load, extracurricular, prediction_result, prediction_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now,
        int(inputs['Sleep_Quality']),
        int(inputs['Headaches_per_week']),
        int(inputs['Academic_Performance']),
        int(inputs['Study_Load']),
        int(inputs['Extracurricular_Activities']),
        result,
        float(probability)
    ))
    conn.commit()
    conn.close()

def load_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        df_history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()
        return df_history
    except Exception as e:
        return pd.DataFrame()

# --- AKHIR DARI PENGATURAN DATABASE ---

# Muat model
MODEL_FILENAME = 'model_stres_dataset_baru.joblib'
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"File model ('{MODEL_FILENAME}') tidak ditemukan.")
    st.stop()

# Inisialisasi database
init_db()

st.title('Analisis Faktor Penyebab Stres Mahasiswa')
st.subheader('Prediksi Risiko Stres: Tinggi vs. Rendah/Normal')
st.write("Aplikasi ini menggunakan model Random Forest dengan akurasi 94.23%")

# --- INPUT SIDEBAR (TETAP SAMA) ---
st.sidebar.header('Masukkan Data Anda:')
st.sidebar.write("Silakan nilai dari 1 (Sangat Rendah) hingga 5 (Sangat Tinggi)")

def user_input_features():
    sleep = st.sidebar.slider('Kualitas Tidur (Sleep_Quality)', 1, 5, 3)
    headaches = st.sidebar.slider('Sakit Kepala per Minggu (Headaches_per_week)', 1, 5, 2)
    performance = st.sidebar.slider('Performa Akademik (Academic_Performance)', 1, 5, 3)
    load = st.sidebar.slider('Beban Belajar (Study_Load)', 1, 5, 3)
    extra = st.sidebar.slider('Aktivitas Ekstrakurikuler', 1, 5, 2)

    data = {
        'Sleep_Quality': sleep,
        'Headaches_per_week': headaches,
        'Academic_Performance': performance,
        'Study_Load': load,
        'Extracurricular_Activities': extra
    }
    
    column_order = [
        'Sleep_Quality', 'Headaches_per_week', 'Academic_Performance', 
        'Study_Load', 'Extracurricular_Activities'
    ]
    features = pd.DataFrame(data, index=[0])
    return features

# Ambil nilai slider saat ini
input_df = user_input_features()

# Tampilkan nilai yang dipilih (tanpa prediksi)
st.subheader('Data Input (Pilihan Anda Saat Ini):')
st.write(input_df)

# --- PERUBAHAN BESAR: TAMBAHKAN TOMBOL SUBMIT ---
st.sidebar.write("---") # Garis pemisah

# Tombol ini akan menjalankan prediksi DAN penyimpanan
if st.sidebar.button('Prediksi & Simpan ke Riwayat'):
    
    # --- SEMUA LOGIKA PREDIKSI & SIMPAN PINDAH KE DALAM SINI ---
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Hasil Prediksi:')
        
        input_data_dict = input_df.iloc[0].to_dict()

        if prediction[0] == 1:
            result_text = "STRES TINGGI"
            proba_display = prediction_proba[0][1]
            st.error(f'**Diprediksi: {result_text}** (Keyakinan: {proba_display*100:.2f}%)')
        else:
            result_text = "STRES RENDAH/NORMAL"
            proba_display = prediction_proba[0][0]
            st.success(f'**Diprediksi: {result_text}** (Keyakinan: {proba_display*100:.2f}%)')

        # --- SIMPAN KE DATABASE (HANYA SETELAH DITEKAN) ---
        save_prediction(input_data_dict, result_text, proba_display)
        st.success("Prediksi ini telah disimpan ke Riwayat.")

        st.subheader('Rincian Keyakinan Model:')
        proba_df = pd.DataFrame(prediction_proba, columns=['Stres Rendah (0)', 'Stres Tinggi (1)'], index=['Probabilitas'])
        st.dataframe(proba_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

else:
    # Pesan default jika tombol belum ditekan
    st.info("Atur slider di sebelah kiri dan klik 'Prediksi & Simpan ke Riwayat' untuk melihat hasil.")


# --- TAMPILKAN RIWAYAT PREDIKSI (TETAP SAMA) ---
st.subheader('Riwayat Prediksi')
st.write("Menampilkan 10 prediksi terakhir yang disimpan di database:")
history_df = load_history()
st.dataframe(history_df.head(10))

if st.button('Muat Ulang Riwayat'):
    st.rerun()
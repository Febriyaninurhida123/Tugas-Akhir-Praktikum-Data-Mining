# 🧠 Analisis & Prediksi Risiko Stroke dengan Machine Learning
LINK APLIKASI : (BELLOW)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediksi-risiko-stroke.streamlit.app/)

Proyek ini adalah implementasi aplikasi web interaktif untuk tugas akhir mata kuliah Data Mining. Aplikasi ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi stroke dan memprediksi risiko stroke pada pasien menggunakan algoritma machine learning, khususnya **K-Nearest Neighbors (KNN)** dan **Naive Bayes**.

Aplikasi ini dibangun menggunakan Python dengan library Streamlit untuk antarmuka pengguna yang modern dan responsif.

## 📊 Tampilan Aplikasi

Berikut adalah tampilan utama dari aplikasi web yang telah dibangun:
![tampilan-laman](https://github.com/user-attachments/assets/1802fb78-e13b-49fa-ab2b-4a87212551f9)

## ✨ Fitur Utama

- **Prediksi Interaktif**: Pengguna dapat memasukkan data pasien melalui form input untuk mendapatkan prediksi risiko stroke secara real-time.
- **Perbandingan Model**: Memungkinkan pengguna memilih antara dua algoritma klasifikasi (KNN dan Naive Bayes) untuk melakukan prediksi.
- **Visualisasi Kinerja**: Menampilkan metrik evaluasi seperti akurasi dan *confusion matrix* untuk kedua model secara berdampingan.
- **Analisis Faktor Risiko**: Menyajikan tabel berisi faktor-faktor paling berpengaruh yang diidentifikasi dari dataset training, memberikan wawasan tambahan tentang pemicu stroke.
- **Antarmuka Modern**: Didesain dengan tata letak satu halaman yang bersih dan intuitif untuk pengalaman pengguna yang lebih baik.

## ⚙️ Alur Kerja Data Mining

Proyek ini mengikuti alur kerja standar dalam data mining, mulai dari pra-pemrosesan data hingga evaluasi model.

1.  **Pemuatan Data**: Menggunakan dataset stroke yang berisi informasi demografis dan medis pasien.
2.  **Pra-pemrosesan Data**:
    - Menghapus kolom yang tidak relevan (`id`).
    - Mengisi nilai yang hilang (*missing values*) pada kolom `bmi` dengan nilai rata-rata (mean imputation).
    - Menghapus data anomali (pasien dengan gender 'Other').
3.  **Encoding & Scaling**:
    - **Label Encoding**: Mengubah fitur-fitur kategorikal (seperti `gender`, `work_type`, `smoking_status`) menjadi representasi numerik.
    - **Standard Scaling**: Menyamakan skala semua fitur numerik agar tidak ada fitur yang mendominasi, langkah ini sangat penting untuk model KNN.
4.  **Pelatihan Model**:
    - Memisahkan data menjadi data latih (80%) dan data uji (20%).
    - Melatih dua model klasifikasi: **K-Nearest Neighbors (KNN)** dan **Gaussian Naive Bayes**.
5.  **Evaluasi**: Mengukur kinerja model pada data uji menggunakan metrik **akurasi** dan memvisualisasikan hasilnya dengan **confusion matrix**.
6.  **Deployment**: Model yang telah dilatih, beserta scaler dan encoder, disimpan dalam file `.pkl` untuk digunakan oleh aplikasi Streamlit.

## 🛠️ Teknologi yang Digunakan

- **Bahasa**: Python 3
- **Framework Aplikasi Web**: Streamlit
- **Analisis Data**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualisasi Data**: Matplotlib, Seaborn, Plotly

## 🚀 Cara Menjalankan Proyek Secara Lokal

Proses pengembangan aplikasi ini dibagi menjadi dua tahap utama: membuat script untuk pelatihan model dan membuat script untuk aplikasi antarmuka pengguna (UI).

### Tahap 1: Script Pelatihan Model (`train_model.py`)

Langkah pertama adalah membuat script terpisah untuk proses training. Tujuannya adalah agar model tidak dilatih ulang setiap kali ada prediksi baru, sehingga aplikasi menjadi lebih efisien. Script ini bertanggung jawab untuk memuat dataset, melakukan pra-pemrosesan data, melatih model, dan menyimpan semua artefak yang dibutuhkan (model, scaler, encoder) ke dalam file `.pkl`.

Berikut adalah kode lengkap untuk `train_model.py`:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import os

# Membuat folder 'models' jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Memuat Dataset
df = pd.read_csv('Stroke_Dataset_Trainingcsv.csv')

# 2. Pra-pemrosesan Data
# Menghapus kolom 'id' karena tidak relevan
df = df.drop('id', axis=1)

# Mengatasi nilai yang hilang (missing values) di kolom 'bmi' dengan nilai rata-rata
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Menghapus baris dengan gender 'Other' karena hanya ada satu dan bisa menjadi noise
df = df[df['gender'] != 'Other']

# 3. Encoding Fitur Kategorikal
encoders = {}
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Menyimpan encoders
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# 4. Memisahkan Fitur (X) dan Target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# 5. Scaling Fitur Numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menyimpan scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 6. Memisahkan Data Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Melatih dan Menyimpan Model K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
with open('models/model_knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

# 8. Melatih dan Menyimpan Model Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
with open('models/model_nb.pkl', 'wb') as f:
    pickle.dump(nb, f)

# 9. Menyimpan akurasi
accuracies = {'knn': accuracy_knn, 'naive_bayes': accuracy_nb}
with open('models/accuracies.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

print("Training selesai! Artefak telah disimpan di folder 'models'.")
````

### Tahap 2: Script Aplikasi Streamlit (`app.py`)

Script ini adalah inti dari aplikasi. Fungsinya adalah memuat model yang sudah dilatih dari file `.pkl`, membuat antarmuka pengguna (form input), menerima input dari pengguna, melakukan prediksi berdasarkan input tersebut, dan menampilkan hasilnya secara visual, lengkap dengan metrik evaluasi model.

Berikut adalah kode untuk `app.py`:

```python
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# =================================================================================
# KONFIGURASI HALAMAN & GAYA CSS
# =================================================================================

st.set_page_config(
    page_title="Analisis & Prediksi Risiko Stroke",
    page_icon="🧠",
    layout="wide"
)

# Kustomisasi CSS untuk tampilan yang lebih modern
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Anda bisa membuat file style.css atau langsung inject di sini
st.markdown("""
<style>
    .st-emotion-cache-18ni7ap {
        padding-top: 2rem;
    }
    .st-emotion-cache-z5fcl4 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    .stButton>button {
        border-radius: 20px;
        border: 2px solid #007bff;
        color: #007bff;
        background-color: transparent;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #0056b3;
        color: white;
        background-color: #0056b3;
    }
    .st-emotion-cache-10trblm {
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================
# PEMUATAN DATA & MODEL (DENGAN CACHE)
# =================================================================================

@st.cache_resource
def load_resources():
    """Memuat semua resource (model, scaler, encoders, data) dari file."""
    try:
        with open('models/model_knn.pkl', 'rb') as f: model_knn = pickle.load(f)
        with open('models/model_nb.pkl', 'rb') as f: model_nb = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('models/encoders.pkl', 'rb') as f: encoders = pickle.load(f)
        with open('models/accuracies.pkl', 'rb') as f: accuracies = pickle.load(f)
        
        train_data = pd.read_csv('Stroke_Dataset_Trainingcsv.csv')
        test_data = pd.read_csv('Stroke_dataset_Testing.csv')
        
        return model_knn, model_nb, scaler, encoders, accuracies, train_data, test_data
    except FileNotFoundError:
        return None

resources = load_resources()
if resources is None:
    st.error("⚠️ File model/data tidak ditemukan. Pastikan Anda telah menjalankan script `train_model.py` dan file dataset ada di direktori yang sama.", icon="🚨")
    st.stop()
model_knn, model_nb, scaler, encoders, accuracies, train_data, test_data = resources


# =================================================================================
# ANALISIS FAKTOR RISIKO (FUNGSI BARU)
# =================================================================================

@st.cache_data
def analyze_risk_factors(_df):
    """Menganalisis dan menghitung pengaruh faktor risiko dari data training."""
    stroke_patients = _df[_df['stroke'] == 1]
    
    factors = {
        'Hipertensi': f"{stroke_patients['hypertension'].mean() * 100:.1f}%",
        'Penyakit Jantung': f"{stroke_patients['heart_disease'].mean() * 100:.1f}%",
        'Rata-rata Usia': f"{stroke_patients['age'].mean():.1f} tahun",
        'Status Menikah': f"{(stroke_patients['ever_married'] == 'Yes').mean() * 100:.1f}%",
        'Level Glukosa > 125 mg/dL (Diabetes)': f"{(stroke_patients['avg_glucose_level'] > 125).mean() * 100:.1f}%",
        'BMI > 25 (Overweight/Obesitas)': f"{(stroke_patients['bmi'] > 25).mean() * 100:.1f}%",
        'Perokok atau Mantan Perokok': f"{stroke_patients['smoking_status'].isin(['smokes', 'formerly smoked']).mean() * 100:.1f}%"
    }
    
    factor_df = pd.DataFrame(list(factors.items()), columns=['Faktor Risiko', 'Persentase/Nilai pada Pasien Stroke'])
    return factor_df


# =================================================================================
# TAMPILAN UTAMA
# =================================================================================

# --- HEADER ---
st.title("🧠 Analisis & Prediksi Risiko Stroke")
st.write("""
Selamat datang di aplikasi analisis risiko stroke. Aplikasi ini menggunakan model Machine Learning untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan faktor-faktor kesehatan dan gaya hidup. 
Di bawah ini, Anda dapat melihat evaluasi kinerja model, faktor risiko utama yang diidentifikasi dari data, lalu memasukkan data Anda sendiri untuk mendapatkan prediksi.
""")
st.divider()

# --- EVALUASI MODEL & FAKTOR RISIKO ---
col1, col2 = st.columns([0.55, 0.45], gap="large")

with col1:
    st.subheader("📊 Evaluasi & Kinerja Model")
    st.write("Perbandingan akurasi dan *confusion matrix* dari dua model yang digunakan pada data uji.")
    
    # Pre-processing data uji
    test_data_clean = test_data.drop('id', axis=1).copy()
    test_data_clean['bmi'].fillna(test_data_clean['bmi'].mean(), inplace=True)
    test_data_clean = test_data_clean[test_data_clean['gender'] != 'Other']
    X_test = test_data_clean.drop('stroke', axis=1)
    y_test = test_data_clean['stroke']
    for col in encoders:
        X_test[col] = encoders[col].transform(X_test[col])
    X_test_scaled = scaler.transform(X_test)

    # Prediksi untuk confusion matrix
    y_pred_knn = model_knn.predict(X_test_scaled)
    y_pred_nb = model_nb.predict(X_test_scaled)
    
    # Tampilkan Akurasi
    acc_col1, acc_col2 = st.columns(2)
    with acc_col1:
        st.metric(label="Akurasi KNN", value=f"{accuracies['knn']:.2%}")
    with acc_col2:
        st.metric(label="Akurasi Naive Bayes", value=f"{accuracies['naive_bayes']:.2%}")

    # Tampilkan Confusion Matrix
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[0], xticklabels=['Rendah', 'Tinggi'], yticklabels=['Rendah', 'Tinggi'])
    axes[0].set_title("Confusion Matrix KNN")
    axes[0].set_ylabel('Aktual')
    axes[0].set_xlabel('Prediksi')
    
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=['Rendah', 'Tinggi'], yticklabels=['Rendah', 'Tinggi'])
    axes[1].set_title("Confusion Matrix Naive Bayes")
    axes[1].set_ylabel('Aktual')
    axes[1].set_xlabel('Prediksi')
    
    st.pyplot(fig)

with col2:
    st.subheader("❗ Faktor Risiko Utama")
    st.write("Analisis dari data training menunjukkan seberapa umum faktor-faktor ini pada pasien yang **terkena stroke**.")
    risk_factors_df = analyze_risk_factors(train_data)
    st.dataframe(risk_factors_df, use_container_width=True, hide_index=True)


st.divider()

# --- FORM INPUT DATA PASIEN ---
st.subheader("📝 Input Data Pasien")
st.write("Isi formulir di bawah ini dengan data pasien, lalu klik tombol prediksi.")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ('Laki-laki', 'Perempuan'), index=0)
        age = st.slider("Usia", 1, 100, 50)
    with col2:
        hypertension = st.selectbox("Memiliki Hipertensi?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Memiliki Penyakit Jantung?", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with col3:
        avg_glucose_level = st.slider("Rata-rata Level Glukosa", 50.0, 300.0, 100.0)
        bmi = st.slider("Indeks Massa Tubuh (BMI)", 10.0, 60.0, 25.0)

    col4, col5 = st.columns(2)
    with col4:
        ever_married = st.selectbox("Status Pernikahan", ('Sudah Menikah', 'Belum Menikah'))
        work_type = st.selectbox("Jenis Pekerjaan", ('Swasta', 'Wiraswasta', 'Pemerintahan', 'Anak-anak', 'Tidak Pernah Bekerja'))
    with col5:
        Residence_type = st.selectbox("Tipe Tempat Tinggal", ('Perkotaan', 'Pedesaan'))
        smoking_status = st.selectbox("Status Merokok", ('Dulu merokok', 'Tidak pernah merokok', 'Merokok', 'Tidak diketahui'))

    st.divider()
    
    # Pilihan Algoritma & Tombol Prediksi
    metode = st.radio("Pilih Algoritma Prediksi", ("K-Nearest Neighbors (KNN)", "Naive Bayes"), horizontal=True)
    
    # Tombol di tengah
    _, col_btn, _ = st.columns([1, 0.4, 1])
    with col_btn:
        predict_button = st.button("🚀 Prediksi Risiko Stroke", use_container_width=True, type="primary")


# --- HASIL PREDIKSI ---
if predict_button:
    # Mapping dan pre-processing input
    gender_map = {'Laki-laki': 'Male', 'Perempuan': 'Female'}
    ever_married_map = {'Belum Menikah': 'No', 'Sudah Menikah': 'Yes'}
    work_type_map = {'Pemerintahan': 'Govt_job', 'Tidak Pernah Bekerja': 'Never_worked', 'Wiraswasta': 'Self-employed', 'Swasta': 'Private', 'Anak-anak': 'children'}
    residence_type_map = {'Pedesaan': 'Rural', 'Perkotaan': 'Urban'}
    smoking_status_map = {'Dulu merokok': 'formerly smoked', 'Tidak pernah merokok': 'never smoked', 'Merokok': 'smokes', 'Tidak diketahui': 'Unknown'}
    
    data = {'gender': gender_map[gender], 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 'ever_married': ever_married_map[ever_married], 
            'work_type': work_type_map[work_type], 'Residence_type': residence_type_map[Residence_type], 'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 
            'smoking_status': smoking_status_map[smoking_status]}
    input_df = pd.DataFrame(data, index=[0])

    input_df_encoded = input_df.copy()
    for col in encoders:
        le = encoders[col]
        input_df_encoded[col] = input_df_encoded[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    input_scaled = scaler.transform(input_df_encoded)

    # Prediksi
    if metode == "K-Nearest Neighbors (KNN)":
        prediction = model_knn.predict(input_scaled)
        prediction_proba = model_knn.predict_proba(input_scaled)
    else:
        prediction = model_nb.predict(input_scaled)
        prediction_proba = model_nb.predict_proba(input_scaled)

    st.divider()
    st.subheader("✅ Hasil Prediksi Anda")

    with st.container(border=True):
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            if prediction[0] == 1:
                st.error("### Hasil: Berisiko Tinggi Terkena Stroke", icon="🚨")
                st.write(f"Berdasarkan model **{metode}**, pasien memiliki kemungkinan **tinggi** untuk mengalami stroke. Sangat disarankan untuk segera berkonsultasi dengan tenaga medis profesional untuk pemeriksaan dan penanganan lebih lanjut.")
            else:
                st.success("### Hasil: Berisiko Rendah Terkena Stroke", icon="✅")
                st.write(f"Berdasarkan model **{metode}**, pasien memiliki kemungkinan **rendah** untuk mengalami stroke. Tetap jaga pola hidup sehat untuk meminimalisir risiko di masa depan.")
            
            st.metric(label="Tingkat Keyakinan (Probabilitas Stroke)", value=f"{prediction_proba[0][1]:.2%}")
            st.warning("Disclaimer: Hasil ini adalah estimasi berdasarkan model dan tidak menggantikan diagnosis medis profesional.", icon="⚠️")

        with col2:
            prob = prediction_proba[0][1]
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Probabilitas Risiko (%)", 'font': {'size': 18}},
                gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                         'bar': {'color': "red" if prob > 0.5 else "green"},
                         'steps' : [{'range': [0, 50], 'color': "lightgreen"}, {'range': [50, 100], 'color': "lightcoral"}]}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), font=dict(color="darkblue", family="Arial"))
            st.plotly_chart(fig, use_container_width=True)
```

## 🚀 Cara Menjalankan Aplikasi

Berikut adalah panduan untuk menjalankan aplikasi ini dari file sumber yang telah disediakan.

#### Prasyarat

Pastikan Anda memiliki file-file berikut dalam satu direktori:

  - `train_model.py`
  - `app.py`
  - `Stroke_Dataset_Trainingcsv.csv`
  - `Stroke_dataset_Testing.csv`

#### Langkah 1: Instalasi Library

Buka terminal atau command prompt, lalu install semua library yang dibutuhkan dengan perintah:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn plotly
```

#### Langkah 2: Latih Model

Jalankan script `train_model.py` satu kali untuk menghasilkan folder `models` yang berisi file-file `.pkl`.

```bash
python train_model.py
```

#### Langkah 3: Jalankan Aplikasi Streamlit

Setelah folder `models` berhasil dibuat, jalankan aplikasi utama:

```bash
streamlit run app.py
```

Browser akan otomatis terbuka dan menampilkan aplikasi yang siap digunakan.

## 📁 Struktur Proyek

  - **`models/`**: Folder berisi semua file hasil pelatihan model.
      - `model_knn.pkl`
      - `model_nb.pkl`
      - `scaler.pkl`
      - `encoders.pkl`
      - `accuracies.pkl`
  - **`app.py`**: Script utama untuk menjalankan aplikasi Streamlit.
  - **`train_model.py`**: Script untuk melatih model dari dataset.
  - **`requirements.txt`**: Daftar library Python yang dibutuhkan proyek.
  - **`Stroke_Dataset_Trainingcsv.csv`**: Dataset untuk melatih model.
  - **`Stroke_dataset_Testing.csv`**: Dataset untuk menguji model.
  - **`README.md`**: File laporan proyek ini.

-----

Dibuat oleh: **Febriyani Nurhida**

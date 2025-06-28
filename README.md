# 🧠 Analisis & Prediksi Risiko Stroke dengan Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_APLIKASI_STREAMLIT_ANDA)

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

# Fungsi untuk memuat semua resource
@st.cache_resource
def load_resources():
    with open('models/model_knn.pkl', 'rb') as f:
        model_knn = pickle.load(f)
    with open('models/model_nb.pkl', 'rb') as f:
        model_nb = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/accuracies.pkl', 'rb') as f:
        accuracies = pickle.load(f)
    test_data = pd.read_csv('Stroke_dataset_Testing.csv')
    return model_knn, model_nb, scaler, encoders, accuracies, test_data

# Memuat semua resource
model_knn, model_nb, scaler, encoders, accuracies, test_data = load_resources()

st.title("🧠 Prediksi Risiko Stroke")
# ... (Sisa kode app.py seperti yang Anda berikan) ...
# Kode lengkap app.py tidak disertakan di sini untuk keringkasan,
# namun merujuk pada file app.py dalam repositori ini.
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

```
```

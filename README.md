# 🧠 Analisis & Prediksi Risiko Stroke dengan Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_APLIKASI_STREAMLIT_ANDA)

Proyek ini adalah implementasi aplikasi web interaktif untuk tugas akhir mata kuliah Data Mining. Aplikasi ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi stroke dan memprediksi risiko stroke pada pasien menggunakan algoritma machine learning, khususnya **K-Nearest Neighbors (KNN)** dan **Naive Bayes**.

Aplikasi ini dibangun menggunakan Python dengan library Streamlit untuk antarmuka pengguna yang modern dan responsif.

## 📊 Tampilan Aplikasi

Berikut adalah tampilan utama dari aplikasi web yang telah dibangun:

![Tampilan Aplikasi](image_250a1e.jpg)

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

Untuk menjalankan aplikasi ini di komputermu, ikuti langkah-langkah berikut:

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/Febriyaninurhida123/Final-Task-Data-Mining.git](https://github.com/Febriyaninurhida123/Final-Task-Data-Mining.git)
    cd Final-Task-Data-Mining
    ```

2.  **Install Dependensi**
    Pastikan kamu memiliki file `requirements.txt` di dalam folder. Lalu, jalankan perintah berikut:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Latih Model (Hanya sekali)**
    Jalankan script ini untuk melakukan pra-pemrosesan data dan menghasilkan file-file model (`.pkl`) di dalam folder `models`.
    ```bash
    python train_model.py
    ```

4.  **Jalankan Aplikasi Streamlit**
    Setelah model berhasil dibuat, jalankan aplikasi web dengan perintah berikut:
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan terbuka secara otomatis di browsermu.

## 📁 Struktur Proyek
---
.

├── models/
│   ├── model_knn.pkl
│   ├── model_nb.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── accuracies.pkl
│
├── app.py                      # Script utama aplikasi Streamlit
├── train_model.py              # Script untuk melatih model
├── requirements.txt            # Daftar dependensi Python
├── Stroke_Dataset_Trainingcsv.csv   # Dataset untuk training
├── Stroke_dataset_Testing.csv  # Dataset untuk testing
└── README.md                   # File ini

---
Dibuat oleh: **Febriyani Nurhida**

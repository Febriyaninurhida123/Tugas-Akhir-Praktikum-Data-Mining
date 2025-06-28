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
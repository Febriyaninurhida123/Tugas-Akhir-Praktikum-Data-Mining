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
# Mengubah fitur kategori menjadi angka agar bisa diproses model
encoders = {}
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Menyimpan encoders untuk digunakan di aplikasi Streamlit
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# 4. Memisahkan Fitur (X) dan Target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# 5. Scaling Fitur Numerik
# Penting untuk KNN agar skala fitur tidak mendominasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menyimpan scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 6. Memisahkan Data Training dan Testing
# Menggunakan data yang sudah di-scale
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Melatih dan Menyimpan Model K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluasi akurasi KNN
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Akurasi KNN: {accuracy_knn:.4f}")

# Menyimpan model KNN
with open('models/model_knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

# 8. Melatih dan Menyimpan Model Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Evaluasi akurasi Naive Bayes
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Akurasi Naive Bayes: {accuracy_nb:.4f}")

# Menyimpan model Naive Bayes
with open('models/model_nb.pkl', 'wb') as f:
    pickle.dump(nb, f)

# 9. Menyimpan akurasi untuk ditampilkan di aplikasi
accuracies = {'knn': accuracy_knn, 'naive_bayes': accuracy_nb}
with open('models/accuracies.pkl', 'wb') as f:
    pickle.dump(accuracies, f)

print("\nTraining selesai! Model, encoders, scaler, dan akurasi telah disimpan di folder 'models'.")
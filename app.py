import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Judul Aplikasi
st.title("Prediksi ikan")

# Input Data Baru
st.sidebar.header("Masukkan Data Baru:")
panjang_tubuh = st.sidebar.number_input("Panjang Tubuh ", min_value=0.0, value=39.1, step=0.1)
lebar_tubuh = st.sidebar.number_input("Lebar Tubuh ", min_value=0.0, value=18.7, step=0.1)
panjang_sirip = st.sidebar.number_input("Panjang Sirip ", min_value=0, value=181, step=1)
lebar_sirip = st.sidebar.number_input("Lebar Sirip ", min_value=0, value=3750, step=1)

# Masukkan data ke DataFrame
data_baru = pd.DataFrame({
    'panjang_tubuh': [panjang_tubuh],
    'lebar_tubuh': [lebar_tubuh],
    'panjang_sirip': [panjang_sirip],
    'lebar_sirip': [lebar_sirip],
})

st.subheader("Data Baru:")
st.write(data_baru)

# Tombol Prediksi
if st.button("Prediksi"):
    # Muat label encoder untuk kolom kategorikal
    label_encoders = load('label_encoder.pkl')  # Pastikan file ini tersedia

    # Muat model yang telah disimpan
    model_rf = load('rf_model.joblib')

    # Lakukan prediksi dengan data baru
    y_pred_baru = model_rf.predict(data_baru)

    # Kembalikan hasil prediksi ke bentuk kategorikal (species)
    spesies_encoder = label_encoders['spesies']  # Pastikan ada encoder untuk 'spesies'
    y_pred_categorical = spesies_encoder.inverse_transform(y_pred_baru)
    
    # Buat DataFrame untuk hasil prediksi
    hasil_prediksi = data_baru.copy()
    hasil_prediksi['Hasil Prediksi (spesies)'] = y_pred_categorical

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(hasil_prediksi)
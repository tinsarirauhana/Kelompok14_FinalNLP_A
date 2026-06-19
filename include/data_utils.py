"""
Modul ini menyediakan fungsi utilitas untuk manipulasi dataset dalam alur kerja machine learning.
Fungsi utama mencakup pembagian dataset (dataset splitting) menjadi subset training, validation,
dan testing menggunakan metode stratified sampling untuk menjaga distribusi label tetap konsisten.
Selain itu, modul ini menyediakan fungsi untuk menghitung bobot kelas (class weight) secara otomatis
guna menangani masalah ketidakseimbangan data (class imbalance) pada fungsi loss CrossEntropyLoss.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pandas import DataFrame

def split_dataset(
    df: DataFrame, 
    label: str, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    random_start: int = 42
) -> tuple:
    # Tahap 1: Memisahkan subset test dari keseluruhan dataset berdasarkan test_ratio yang ditentukan
    df_trainval, df_test = train_test_split(
        df,
        test_size = test_ratio,
        stratify = df[label],  # Stratify memastikan persentase label pada subset test sama dengan dataset awal
        random_state = random_start
    )
    
    # Menghitung rasio validation yang disesuaikan setelah dataset dikurangi oleh subset test
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    # Tahap 2: Memisahkan subset trainval menjadi subset train dan val secara independen
    df_train, df_val = train_test_split(
        df_trainval,
        test_size = val_ratio_adjusted,
        stratify = df_trainval[label],  # Stratify menjaga konsistensi distribusi label pada subset train dan val
        random_state = random_start
    )
    
    return df_train, df_val, df_test

def compute_weights(df: DataFrame, label: str) -> dict:
    # Mengambil seluruh nilai label dari DataFrame dan mencari daftar kelas unik
    labels = df[label].values
    classes = np.unique(labels)
    
    # Menghitung nilai bobot menggunakan formula seimbang (balanced) dari scikit-learn
    weights = compute_class_weight(
        class_weight = "balanced",
        classes = classes,
        y = labels
    )
    
    # Menggabungkan daftar nama kelas dan nilai bobot menjadi sebuah dictionary Python
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    
    return weight_dict
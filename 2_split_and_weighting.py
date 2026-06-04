"""
STEP 2 - Stratified Split dan Class Weighting
Membagi dataset menjadi train/validation/test dengan proporsi stratified,
dan menghitung bobot kelas untuk menangani class imbalance.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

INPUT_PATH       = "data/preprocessed_dataset.csv"
OUTPUT_TRAIN     = "data/train.csv"
OUTPUT_VAL       = "data/val.csv"
OUTPUT_TEST      = "data/test.csv"
OUTPUT_WEIGHTS   = "data/class_weights.json"

RATIO_TRAIN = 0.70
RATIO_VAL   = 0.20
RATIO_TEST  = 0.10

RANDOM_STATE = 42


def split_dataset(df):
    """
    Stratified split: 70% train, 20% val, 10% test.
    Stratified berarti proporsi setiap label dipertahankan di setiap subset.
    """
    # Tahap 1: pisahkan test (10%) dari sisa data (90%)
    df_trainval, df_test = train_test_split(
        df,
        test_size=RATIO_TEST,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # Tahap 2: dari 90% sisa, pisahkan val (~22.2% dari sisa = 20% dari total)
    val_ratio_adjusted = RATIO_VAL / (RATIO_TRAIN + RATIO_VAL)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adjusted,
        stratify=df_trainval["label"],
        random_state=RANDOM_STATE
    )

    return df_train, df_val, df_test


def compute_weights(df_train):
    """
    Menghitung class weight menggunakan sklearn compute_class_weight
    dengan parameter 'balanced' untuk menangani ketidakseimbangan kelas.
    """
    labels = df_train["label"].values
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    return weight_dict


def print_split_info(df_train, df_val, df_test, total):
    print("Hasil pembagian dataset:")
    print(f"  Train : {len(df_train)} baris ({len(df_train)/total*100:.1f}%)")
    print(f"  Val   : {len(df_val)} baris ({len(df_val)/total*100:.1f}%)")
    print(f"  Test  : {len(df_test)} baris ({len(df_test)/total*100:.1f}%)")
    print()

    for nama, subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = subset["label"].value_counts()
        print(f"Distribusi label - {nama}:")
        for label, count in dist.items():
            print(f"  {label:<12}: {count} ({count/len(subset)*100:.1f}%)")
        print()


def run_split():
    print("Memuat dataset hasil preprocessing...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total data: {len(df)} baris")
    print()

    df_train, df_val, df_test = split_dataset(df)

    # Simpan split
    df_train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8")
    df_val.to_csv(OUTPUT_VAL,   index=False, encoding="utf-8")
    df_test.to_csv(OUTPUT_TEST,  index=False, encoding="utf-8")

    print_split_info(df_train, df_val, df_test, total=len(df))

    # Hitung dan simpan class weights
    weight_dict = compute_weights(df_train)

    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(weight_dict, f, indent=2)

    print("Class weights (untuk CrossEntropyLoss):")
    for label, w in weight_dict.items():
        print(f"  {label:<12}: {w:.4f}")
    print()
    print("Catatan: Bobot lebih tinggi berarti kelas tersebut lebih sedikit datanya")
    print("         dan akan diberi perhatian lebih saat pelatihan model.")
    print()
    print(f"File disimpan:")
    print(f"  {OUTPUT_TRAIN}")
    print(f"  {OUTPUT_VAL}")
    print(f"  {OUTPUT_TEST}")
    print(f"  {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    run_split()

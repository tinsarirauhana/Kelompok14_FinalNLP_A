"""
STEP 1 - Preprocessing Teks
Melakukan pembersihan dan normalisasi teks SMS sebelum digunakan untuk pelatihan model.
"""

import pandas as pd
import re
import os

INPUT_PATH  = "data/raw_dataset.csv"
OUTPUT_PATH = "data/preprocessed_dataset.csv"

COL_TEXT  = "Teks"
COL_LABEL = "label"

LABEL_MAP = {
    "normal"   : "normal",
    "promo"    : "promo",
    "penipuan" : "penipuan",
    "ham"      : "normal",
    "spam"     : "promo",
}


# -----------------------------------------------------------------------
# Fungsi pembersihan teks
# -----------------------------------------------------------------------

def remove_url(text):
    """Menghapus URL (http, https, www)."""
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_extra_symbols(text):
    """Menghapus karakter berulang dan simbol yang tidak bermakna."""
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text

def normalize_whitespace(text):
    """Menghapus spasi ganda dan whitespace di awal/akhir."""
    return re.sub(r"\s+", " ", text).strip()

def normalize_case(text):
    """Mengubah teks menjadi huruf kecil semua."""
    return text.lower()

def preprocess_text(text):
    """Pipeline preprocessing lengkap."""
    if not isinstance(text, str):
        return ""
    text = remove_url(text)
    text = remove_extra_symbols(text)
    text = normalize_whitespace(text)
    text = normalize_case(text)
    return text

def normalize_label(label):
    """Menyeragamkan label ke skema 3 kelas."""
    if not isinstance(label, str):
        return None
    label = label.strip().lower()
    return LABEL_MAP.get(label, None)


# -----------------------------------------------------------------------
# Pipeline utama
# -----------------------------------------------------------------------

def run_preprocessing():
    print("Memuat dataset mentah...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total data awal: {len(df)} baris")

    print(f"Kolom tersedia: {list(df.columns)}")

    if COL_TEXT not in df.columns or COL_LABEL not in df.columns:
        print(f"Kolom '{COL_TEXT}' atau '{COL_LABEL}' tidak ditemukan.")
        print("Sesuaikan variabel COL_TEXT dan COL_LABEL di bagian atas skrip ini.")
        return

    df = df[[COL_TEXT, COL_LABEL]].dropna()
    print(f"Setelah hapus nilai kosong: {len(df)} baris")

    df["label_clean"] = df[COL_LABEL].apply(normalize_label)
    tidak_dikenal = df["label_clean"].isna().sum()
    if tidak_dikenal > 0:
        print(f"Peringatan: {tidak_dikenal} baris memiliki label tidak dikenal dan akan dihapus.")
        print("Label unik yang ditemukan:", df[COL_LABEL].unique())
    df = df.dropna(subset=["label_clean"])

    print("Melakukan preprocessing teks...")
    df["text_clean"] = df[COL_TEXT].apply(preprocess_text)

    df = df[df["text_clean"].str.len() > 0]

    df_output = df[["text_clean", "label_clean"]].rename(
        columns={"text_clean": "text", "label_clean": "label"}
    )

    df_output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print()
    print("Preprocessing selesai.")
    print(f"Total data bersih  : {len(df_output)} baris")
    print(f"Distribusi label   :")
    print(df_output["label"].value_counts().to_string())
    print()
    print("Contoh hasil preprocessing (3 baris):")
    for i, row in df_output.head(3).iterrows():
        print(f"  [{row['label']}] {row['text'][:80]}")
    print()
    print(f"File disimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    run_preprocessing()
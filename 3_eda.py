"""
STEP 3 - Exploratory Data Analysis (EDA)
Analisis distribusi data untuk memahami karakteristik dataset
sebelum digunakan dalam pelatihan model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.use("Agg")

INPUT_PATH  = "data/preprocessed_dataset.csv"
OUTPUT_DIR  = "output"

# Warna konsisten per label
LABEL_COLORS = {
    "normal"   : "#4C72B0",
    "promo"    : "#DD8452",
    "penipuan" : "#C44E52",
}


# -----------------------------------------------------------------------
# Fungsi analisis
# -----------------------------------------------------------------------

def analyze_label_distribution(df):
    """Distribusi jumlah data per label."""
    dist = df["label"].value_counts().reset_index()
    dist.columns = ["label", "jumlah"]
    dist["persentase"] = (dist["jumlah"] / len(df) * 100).round(2)
    return dist


def analyze_text_length(df):
    """Statistik panjang teks per label."""
    df = df.copy()
    df["panjang_karakter"] = df["text"].str.len()
    df["jumlah_kata"]      = df["text"].str.split().str.len()

    stats = df.groupby("label")[["panjang_karakter", "jumlah_kata"]].agg(
        ["min", "max", "mean", "median"]
    ).round(1)

    return df, stats


# -----------------------------------------------------------------------
# Fungsi visualisasi
# -----------------------------------------------------------------------

def plot_label_distribution(dist, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribusi Label Dataset SMS Spam Indonesia", fontsize=13, fontweight="bold")

    colors = [LABEL_COLORS.get(l, "#999999") for l in dist["label"]]

    # Bar chart
    axes[0].bar(dist["label"], dist["jumlah"], color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Jumlah Data per Kelas")
    axes[0].set_xlabel("Kelas")
    axes[0].set_ylabel("Jumlah")
    for i, (_, row) in enumerate(dist.iterrows()):
        axes[0].text(i, row["jumlah"] + 5, str(row["jumlah"]), ha="center", fontsize=10)

    # Pie chart
    axes[1].pie(
        dist["jumlah"],
        labels=dist["label"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2}
    )
    axes[1].set_title("Proporsi Kelas")

    plt.tight_layout()
    path = os.path.join(output_dir, "1_distribusi_label.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik disimpan: {path}")


def plot_text_length(df_with_length, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribusi Panjang Teks per Kelas", fontsize=13, fontweight="bold")

    for label, group in df_with_length.groupby("label"):
        color = LABEL_COLORS.get(label, "#999999")
        axes[0].hist(group["panjang_karakter"], bins=40, alpha=0.6, label=label, color=color)
        axes[1].hist(group["jumlah_kata"],      bins=30, alpha=0.6, label=label, color=color)

    axes[0].set_title("Distribusi Panjang Karakter")
    axes[0].set_xlabel("Jumlah Karakter")
    axes[0].set_ylabel("Frekuensi")
    axes[0].legend()

    axes[1].set_title("Distribusi Jumlah Kata")
    axes[1].set_xlabel("Jumlah Kata")
    axes[1].set_ylabel("Frekuensi")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "2_distribusi_panjang_teks.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik disimpan: {path}")


def plot_boxplot_length(df_with_length, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Boxplot Panjang Teks per Kelas", fontsize=13, fontweight="bold")

    palette = {k: v for k, v in LABEL_COLORS.items()}

    sns.boxplot(data=df_with_length, x="label", y="panjang_karakter",
                hue="label", palette=palette, legend=False,
                ax=axes[0], linewidth=1.2)
    axes[0].set_title("Panjang Karakter")
    axes[0].set_xlabel("Kelas")
    axes[0].set_ylabel("Jumlah Karakter")

    sns.boxplot(data=df_with_length, x="label", y="jumlah_kata",
                hue="label", palette=palette, legend=False,
                ax=axes[1], linewidth=1.2)
    axes[1].set_title("Jumlah Kata")
    axes[1].set_xlabel("Kelas")
    axes[1].set_ylabel("Jumlah Kata")

    plt.tight_layout()
    path = os.path.join(output_dir, "3_boxplot_panjang_teks.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafik disimpan: {path}")


# -----------------------------------------------------------------------
# Pipeline utama
# -----------------------------------------------------------------------

def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Memuat dataset hasil preprocessing...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total data: {len(df)} baris")
    print()

    # 1. Distribusi label
    dist = analyze_label_distribution(df)
    print("Distribusi label:")
    print(dist.to_string(index=False))
    print()

    # 2. Panjang teks
    df_len, stats_len = analyze_text_length(df)
    print("Statistik panjang teks per label:")
    print(stats_len.to_string())
    print()

    # 3. Cek duplikasi
    duplikat = df.duplicated(subset=["text"]).sum()
    print(f"Jumlah teks duplikat: {duplikat}")
    print()

    # 4. Cek teks terlalu pendek (kurang dari 5 karakter)
    terlalu_pendek = (df["text"].str.len() < 5).sum()
    print(f"Teks sangat pendek (<5 karakter): {terlalu_pendek}")
    print()

    # 5. Visualisasi
    print("Membuat visualisasi...")
    plot_label_distribution(dist, OUTPUT_DIR)
    plot_text_length(df_len, OUTPUT_DIR)
    plot_boxplot_length(df_len, OUTPUT_DIR)

    print()
    print("EDA selesai. Seluruh grafik disimpan di folder:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
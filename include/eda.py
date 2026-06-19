"""
Modul ini menyediakan kelas EDA (Exploratory Data Analysis) untuk melakukan analisis statistik
dan visualisasi data teks pada dataset SMS spam. Fungsi utama dalam kelas ini mencakup kalkulasi
distribusi label, analisis deskriptif panjang teks (berdasarkan jumlah karakter dan kata), serta
pembuatan grafik visualisasi seperti bar chart, pie chart, histogram distribusi, dan boxplot
untuk mengidentifikasi karakteristik serta sebaran data pada masing-masing kelas target.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

class EDA:
    def __init__(self, color_map: dict[str, str]) -> None:
        # Menyimpan konfigurasi warna global untuk visualisasi grafik tiap kelas target
        self.color_map = color_map
    
    def analyze_label_distribution(self, df: DataFrame) -> DataFrame:
        # Menghitung total kemunculan setiap sampel label unik di dalam dataset
        dist = df["label"].value_counts().reset_index()
        dist.columns = ["label", "jumlah"]
        # Menghitung nilai persentase proporsi masing-masing label dengan presisi dua angka belakang koma
        dist["persentase"] = (dist["jumlah"] / len(df) * 100).round(2)
        
        return dist
    
    def analyze_text_length(self, df: DataFrame) -> tuple:
        temp_df = df.copy()
        # Menghitung panjang total karakter string teks
        temp_df["panjang_karakter"] = df["text"].str.len()
        # Menghitung jumlah kata berdasarkan pemisahan karakter spasi
        temp_df["jumlah_kata"] = df["text"].str.split().str.len()
        
        # Melakukan kalkulasi statistik deskriptif agregat kelompok berdasarkan label kelas
        stats = temp_df.groupby("label")[["panjang_karakter", "jumlah_kata"]].agg(
            ["min", "max", "mean", "median"]
        ).round(1)

        return temp_df, stats
    
    def plot_label_distribution(
        self, 
        df: DataFrame, 
        figsize: tuple[int, int] = (7, 4)
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        fig.suptitle("Distribusi Label Dataset SMS Spam Indonesia", fontsize = 13, fontweight = "bold")

        # Mengambil urutan warna grafik yang sesuai dengan nama label dari dictionary color_map
        colors = [self.color_map.get(label, "#999999") for label in df["label"]]
        
        # Merancang visualisasi grafik batang (bar chart) untuk frekuensi absolut data
        axes[0].bar(df["label"], df["jumlah"], color = colors, edgecolor = "white", linewidth = 0.8)
        axes[0].set_title("Jumlah Data per Kelas")
        axes[0].set_xlabel("Kelas")
        axes[0].set_ylabel("Jumlah")
        
        # Menampilkan teks angka frekuensi di atas masing-masing bar grafik batang
        for index, (_, row) in enumerate(df.iterrows()):
            axes[0].text(index, row["jumlah"] + 5, str(row["jumlah"]), ha = "center", fontsize = 10)
        
        # Merancang visualisasi grafik lingkaran (pie chart) untuk persentase proporsi data   
        axes[1].pie(
            df["jumlah"],
            labels = df["label"],
            autopct = "%1.1f%%",
            colors = colors,
            startangle = 140,
            wedgeprops = {
                "edgecolor": "white", 
                "linewidth": 1.2
            }
        )
        axes[1].set_title("Proporsi Kelas")
        
        plt.tight_layout()
        
    def plot_text_length(
        self, 
        df: DataFrame, 
        figsize: tuple[int, int] = (8, 4)
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        fig.suptitle("Distribusi Panjang Teks per Kelas", fontsize = 13, fontweight = "bold")

        # Melakukan looping untuk membuat visualisasi histogram pada tiap subset kelompok data label
        for label, group in df.groupby("label"):
            color = self.color_map.get(label, "#999999")
            axes[0].hist(group["panjang_karakter"], bins = 40, alpha = 0.6, label = label, color = color)
            axes[1].hist(group["jumlah_kata"], bins = 30, alpha = 0.6, label = label, color = color)

        # Mengatur elemen judul, teks label, dan legend pada subplot histogram karakter
        axes[0].set_title("Distribusi Panjang Karakter")
        axes[0].set_xlabel("Jumlah Karakter")
        axes[0].set_ylabel("Frekuensi")
        axes[0].legend()

        # Mengatur elemen judul, teks label, dan legend pada subplot histogram kata
        axes[1].set_title("Distribusi Jumlah Kata")
        axes[1].set_xlabel("Jumlah Kata")
        axes[1].set_ylabel("Frekuensi")
        axes[1].legend()

        plt.tight_layout()
        
    def plot_boxplot_length(
        self, 
        df: DataFrame, 
        figsize: tuple[int, int] = (8, 4)
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        fig.suptitle("Boxplot Panjang Teks per Kelas", fontsize = 13, fontweight = "bold")

        palette = {key: value for key, value in self.color_map.items()}

        # Membuat grafik boxplot untuk memetakan variasi dan outlier panjang karakter teks
        sns.boxplot(
            data = df, x = "label", y = "panjang_karakter",
            hue = "label", palette = palette, legend = False,
            ax = axes[0], linewidth = 1.2
        )
        axes[0].set_title("Panjang Karakter")
        axes[0].set_xlabel("Kelas")
        axes[0].set_ylabel("Jumlah Karakter")

        # Membuat grafik boxplot untuk memetakan variasi dan outlier kuantitas kata teks
        sns.boxplot(
            data = df, x = "label", y = "jumlah_kata",
            hue = "label", palette = palette, legend = False,
            ax = axes[1], linewidth = 1.2
        )
        axes[1].set_title("Jumlah Kata")
        axes[1].set_xlabel("Kelas")
        axes[1].set_ylabel("Jumlah Kata")

        plt.tight_layout()
# Bagian 1 - Data & Preprocessing
## Tinsari Rauhana (2308107010038)
## Proyek Akhir NLP - Kelompok 14

---

## Struktur File

```
nlp_project/
|-- 1_preprocessing.py         # Pembersihan dan normalisasi teks
|-- 3_split_and_weighting.py   # Stratified split + class weighting
|-- 4_eda.py                   # Exploratory Data Analysis
|
|-- data/
|   |-- raw_dataset.csv        # Dataset mentah (hasil step 1)
|   |-- preprocessed_dataset.csv  # Dataset bersih (hasil step 2)
|   |-- train.csv              # Data latih 70% (hasil step 3)
|   |-- val.csv                # Data validasi 20% (hasil step 3)
|   |-- test.csv               # Data uji 10% (hasil step 3)
|   |-- class_weights.json     # Bobot kelas untuk CrossEntropyLoss (hasil step 3)
|
|-- output/
    |-- 1_distribusi_label.png       # Grafik distribusi kelas
    |-- 2_distribusi_panjang_teks.png
    |-- 3_boxplot_panjang_teks.png
```

---

## Cara Menjalankan (Urutan)

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan satu per satu sesuai urutan:
```bash
python 1_collect_dataset.py
python 2_preprocessing.py
python 3_split_and_weighting.py
python 4_eda.py
```

---

## Penjelasan Singkat Tiap Langkah

### Step 1 - Preprocessing Teks
Pipeline pembersihan teks yang mencakup:
- Penghapusan URL
- Penghapusan simbol berlebihan dan karakter berulang
- Normalisasi whitespace (hapus spasi ganda)
- Normalisasi huruf menjadi lowercase
- Normalisasi label ke skema 3 kelas yang seragam

### Step 2 - Stratified Split dan Class Weighting
Pembagian data secara stratified:
- 70% data training
- 20% data validasi
- 10% data pengujian

Distribusi label dipertahankan di setiap subset menggunakan stratify dari sklearn.
Class weight dihitung dengan compute_class_weight(balanced) untuk menangani
ketidakseimbangan kelas pada fungsi loss CrossEntropyLoss.

### Step 3 - Exploratory Data Analysis (EDA)
Analisis dan visualisasi dataset yang mencakup:
- Distribusi jumlah data per kelas (bar chart dan pie chart)
- Distribusi panjang teks per kelas (histogram)
- Boxplot panjang teks per kelas
- Pengecekan duplikasi dan teks terlalu pendek

---

## Output untuk Laporan
Grafik-grafik dari step 3 dapat langsung dimasukkan ke bab Metodologi atau
bab Hasil pada laporan akhir untuk menjelaskan karakteristik dataset.

File train.csv, val.csv, test.csv, dan class_weights.json akan digunakan
pada tahap fine-tuning model.

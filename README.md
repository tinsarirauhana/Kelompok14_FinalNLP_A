# Klasifikasi Teks SMS Spam Indonesia Menggunakan IndoBERT dan XLM-RoBERTa

Proyek ini berfokus pada implementasi dan komparasi performa dua arsitektur berbasis Transformer, yaitu IndoBERT dan XLM-RoBERTa, untuk menyelesaikan tugas klasifikasi teks multi-kelas pada korpus SMS Bahasa Indonesia ke dalam tiga kategori: normal, promo, dan penipuan. Alur kerja yang dibangun di dalam proyek ini mencakup seluruh siklus pengembangan model pembelajaran mendalam, mulai dari tahapan preprocessing teks, analisis karakteristik data, fine-tuning model dengan skema checkpointing otomatis, evaluasi performa komprehensif menggunakan matriks klasifikasi, hingga ekspor grafik komputasi model ke format biner ONNX untuk kebutuhan deployment di lingkungan produksi.

## Prerequisites

Sebelum menjalankan proyek ini, pastikan sistem Anda telah memenuhi beberapa persyaratan perangkat lunak berikut:

* **Python**: Versi 3.10 atau versi yang lebih baru.
* **Package Manager**: `uv` (package manager Python yang cepat sebagai pengganti `pip`).
* **Akselerasi Hardware**: Driver CUDA (opsional, sangat direkomendasikan jika ingin melakukan proses training menggunakan GPU).

## Langkah Menjalankan Program

Ikuti instruksi di bawah ini secara berurutan untuk memasang dependensi dan mengeksekusi program:

### 1. Clone Repository

Unduh source code proyek dari repository GitHub menggunakan perintah berikut:

```bash
git clone https://github.com/tinsarirauhana/Kelompok14_FinalNLP_A.git
cd Kelompok14_FinalNLP_A
```

### 2. Persiapan Environment dan Install Dependency

Gunakan tool `uv` untuk membuat virtual environment baru serta menginstal seluruh library yang dibutuhkan secara otomatis berdasarkan file lock yang tersedia:

```bash
# Membuat virtual environment berbasis .venv
uv venv

# Mengaktifkan virtual environment (Linux/macOS)
source .venv/bin/activate

# Mengaktifkan virtual environment (Windows)
.venv\Scripts\activate

# Sinkronisasi dan instalasi seluruh dependensi terdaftar
uv sync
```

### 3. Eksekusi Program Utama

Buka text editor atau IDE pilihan Anda (seperti VS Code atau Jupyter Lab), pastikan kernel notebook telah diarahkan ke virtual environment `.venv` yang baru dibuat, lalu buka dan jalankan file utama:

```text
main.ipynb
```

Eksekusi sel di dalam notebook tersebut secara berurutan dari atas hingga akhir untuk melihat seluruh proses eksperimen.

## Struktur Directory

Berikut adalah representasi pohon struktur directory dari proyek ini beserta penjelasan fungsi untuk masing-masing komponen:

```text
.
|-- README.md                     # Dokumentasi utama panduan proyek
|-- data                          # Directory penyimpanan file data teks dan konfigurasi bobot
|   |-- class_weights.json        # File konfigurasi bobot kelas untuk menangani data imbalance
|   |-- dataset.csv               # Raw dataset asli teks SMS Indonesia
|   |-- preprocessed.csv          # Dataset hasil tahapan pembersihan dan normalisasi teks
|   |-- test.csv                  # Subset dataset khusus untuk tahapan pengujian akhir (testing)
|   |-- train.csv                 # Subset dataset utama untuk tahapan pelatihan model (training)
|   `-- val.csv                   # Subset dataset untuk tahapan validasi selama looping pelatihan
|-- include                       # Directory berisi modul eksternal dan script pembantu (helper)
|   |-- data_utils.py             # Fungsi utilitas untuk memuat data dan kalkulasi class weights
|   |-- dataset.py                # Implementasi class custom SMSDataset berbasis PyTorch Dataset
|   |-- eda.py                    # Fungsi visualisasi distribusi kelas dan visualisasi panjang teks
|   |-- helper.py                 # Kelas TrainHelper berisi method loop training, evaluasi, dan ekspor ONNX
|   `-- preprocessor.py           # Modul pembersihan teks, case folding, dan penghapusan karakter khusus
|-- main.ipynb                    # File Jupyter Notebook utama tempat eksekusi alur kerja eksperimen
|-- model                         # Directory lokal tempat penyimpanan output model dan tokenizer
|   |-- indobert                  # Folder khusus untuk checkpoint dan biner model IndoBERT
|   |   |-- config.json           # Konfigurasi parameter arsitektur model IndoBERT
|   |   |-- model.onnx            # Grafik komputasi model IndoBERT dalam format biner ONNX
|   |   |-- model.safetensors     # File bobot parameter terlatih (fine-tuned weights) IndoBERT
|   |   |-- tokenizer.json        # Kosakata dan struktur tokenizer biner IndoBERT
|   |   `-- tokenizer_config.json # Konfigurasi hyperparameter tokenizer IndoBERT
|   `-- xlm_roberta               # Folder khusus untuk checkpoint dan biner model XLM-RoBERTa
|       |-- config.json           # Konfigurasi parameter arsitektur model XLM-RoBERTa
|       |-- model.onnx            # Grafik komputasi model XLM-RoBERTa dalam format biner ONNX
|       |-- model.safetensors     # File bobot parameter terlatih (fine-tuned weights) XLM-RoBERTa
|       |-- tokenizer.json        # Kosakata dan struktur tokenizer biner XLM-RoBERTa
|       `-- tokenizer_config.json # Konfigurasi hyperparameter tokenizer XLM-RoBERTa
|-- pyproject.toml                # File konfigurasi utama proyek dan dependensi metadata untuk uv
|-- requirements.txt              # Daftar standar library dependensi Python yang dibutuhkan
`-- uv.lock                       # File pengunci versi dependensi spesifik untuk konsistensi environment

6 directories, 26 files
```

## Ringkasan Tahapan Proyek

Alur kerja eksperimen di dalam file `main.ipynb` dibagi secara terstruktur ke dalam 5 bagian utama sebagai berikut:

### Bagian 1: Preprocessing Teks

Tahap awal berfokus pada pembersihan dokumen mentah dari text noise yang dapat menurunkan akurasi model. Proses ini memanfaatkan modul `preprocessor.py` untuk melakukan konversi teks menjadi huruf kecil (*case folding*), penghapusan karakter khusus seperti tanda baca, angka, dan emoji, eliminasi whitespace berlebih, serta pembersihan tautan URL atau nomor telepon tertentu agar menghasilkan korpus teks yang bersih dan seragam.

### Bagian 2: Split Dataset dan Perhitungan Class Weight

Pada tahap ini, data yang telah bersih dipisah (*split*) menjadi tiga bagian secara proporsional, yaitu data training, data validasi, dan data testing. Mengingat distribusi jumlah sampel antar kategori tidak seimbang (*imbalanced data*), dilakukan perhitungan statistik bobot kelas menggunakan metode kebalikan frekuensi sampel. Nilai bobot ini kemudian disimpan ke dalam file `class_weights.json` untuk disuntikkan ke fungsi loss criterion saat proses pelatihan.

### Bagian 3: Exploratory Data Analysis (EDA)

Tahap EDA diimplementasikan menggunakan fungsi di dalam modul `eda.py` untuk memahami karakteristik intrinsik data secara visual. Proses analisis mencakup pembuatan grafik batang jumlah data per kelas dan diagram lingkaran proporsional kelas untuk memeriksa tingkat ketimpangan data. Selain itu, dibuat pula grafik histogram dan boxplot distribusi panjang karakter serta jumlah kata per kelas untuk melihat pola penulisan teks pada kategori normal, promo, dan penipuan.

### Bagian 4: Model Training dan Fine-Tuning IndoBERT

Tahap ini berfokus pada proses fine-tuning model pre-trained `indobenchmark/indobert-base-p1` pada korpus SMS lokal menggunakan class pembantu `TrainHelper`. Model dilatih menggunakan optimizer AdamW dan linear learning rate scheduler selama 3 epoch. Proses validasi loss dipantau di setiap akhir epoch untuk memicu skema checkpointing otomatis jika ditemukan model terbaik. Setelah selesai, model dikonversi ke format ONNX dan kinerjanya diuji menggunakan data testing untuk menampilkan classification report serta visualisasi confusion matrix.

### Bab 5: Model Training dan Fine-Tuning XLM-RoBERTa

Sebagai langkah komparasi, tahap akhir ini menerapkan prosedur fine-tuning yang sama menggunakan arsitektur cross-lingual `FacebookAI/xlm-roberta-base`. Pipeline input data dibangun ulang menggunakan tokenizer khusus berbasis SentencePiece bawaan XLM-RoBERTa. Setelah melalui siklus pelatihan 3 epoch dengan pencatatan nilai performa melalui history dictionary dan pengamanan file bobot terbaik, model optimal XLM-RoBERTa diekspor ke format biner ONNX menggunakan konfigurasi *dynamic axes*. Evaluasi performa kemudian ditampilkan secara komprehensif melalui metrik klasifikasi dan grafik visualisasi.
"""
Modul ini menyediakan kelas utilitas untuk melakukan tahapan preprocessing data teks pada dataset.
Fungsi utama dari modul ini mencakup pembersihan komponen string teks (noise removal) dari link atau URL,
penghapusan karakter simbol yang tidak relevan, serta reduksi pengulangan karakter yang berlebihan.
Selain itu, kelas ini juga menangani standardisasi format teks melalui normalisasi spasi, pembersihan
karakter kosong, normalisasi huruf kecil (case folding), serta pemetaan nilai label ke dalam skema
kategori target yang valid untuk kebutuhan pelatihan model Transformer.
"""

import re

class DataPreprocessor:
    def _remove_url(self, text: str) -> str:
        # Menghapus link atau URL berbasis http/https dan awalan www dari dalam teks
        return re.sub(r"http\S+|www\.\S+", "", text)

    def _remove_extra_symbols(self, text: str) -> str:
        # Mengganti karakter non-alfanumerik dengan spasi, kecuali tanda baca dasar (.,!?)
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        
        # Mereduksi pengulangan karakter yang sama lebih dari dua kali menjadi dua kali saja (misal: "banyakkk" menjadi "banyakk")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        
        return text

    def _normalize_whitespace(self, text: str) -> str:
        # Mengubah spasi berlebih atau spasi ganda di dalam teks menjadi satu spasi tunggal dan menghapus spasi di awal/akhir teks
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_case(self, text: str) -> str:
        # Mengubah format seluruh huruf menjadi huruf kecil (lowercase) untuk konsistensi tokenisasi
        return text.lower()
    
    def _normalize_label(self, label_map: dict[str, str], label: str) -> (str | None):
        # Validasi preventif untuk memastikan input label memiliki tipe data string
        if not isinstance(label, str):
            return None
        
        # Menghapus spasi di ujung label dan mengubahnya menjadi huruf kecil
        label = label.strip().lower()
        
        # Mengembalikan nilai label hasil pemetaan dari dictionary label_map, atau None jika tidak terdaftar
        return label_map.get(label, None)

    def preprocess_text(self, text: str) -> str:
        # Mengantisipasi nilai kosong atau tipe data non-string agar tidak menyebabkan error saat proses pembersihan
        if not isinstance(text, str):
            return ""
        
        # Menjalankan seluruh tahapan pembersihan teks secara berurutan (pipeline)
        text = self._remove_url(text)
        text = self._remove_extra_symbols(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_case(text)
        
        return text
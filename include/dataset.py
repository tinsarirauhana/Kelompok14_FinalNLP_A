"""
Modul ini menyediakan kelas kustom SMSDataset berbasis PyTorch Dataset untuk memetakan data teks.
Fungsi utama dari kelas ini adalah melakukan enkapsulasi terhadap DataFrame teks dan label,
menjalankan proses tokenisasi menggunakan objek tokenizer Hugging Face, menerapkan pengaturan
panjang maksimal (padding dan truncation), serta mengembalikan representasi tensor input_ids,
attention_mask, dan label tensor yang siap dikonsumsi oleh DataLoader untuk pelatihan model Transformer.
"""

import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from typing import Any

class SMSDataset(Dataset):
    def __init__(
        self, 
        label_id: dict[str, int], 
        df: DataFrame, 
        tokenizer: Any, 
        max_len: int
    ) -> None:
        self.label_id = label_id
        self.texts = df["text"].tolist()
        # Mengubah nilai label string menjadi integer index berdasarkan dictionary label_id
        self.labels = df["label"].map(label_id).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Melakukan tokenisasi teks dinamis berdasarkan index data
        encoding = self.tokenizer(
            self.texts[index],
            max_length = self.max_len,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        
        # Mengembalikan dictionary berisi tensor yang dimensinya disesuaikan menggunakan squeeze()
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[index], dtype = torch.long),
        }
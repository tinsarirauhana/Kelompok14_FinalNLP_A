"""
Modul ini menyediakan kelas pembantu TrainHelper untuk mengotomatisasi alur kerja (workflow)
pelatihan, evaluasi, dan pengujian model arsitektur IndoBERT atau XLM-RoBERTa. Fungsi utama mencakup manajemen
looping epoch training (termasuk kalkulasi loss dan backpropagation), looping evaluasi validasi,
prediksi data pengujian, serta pembuatan visualisasi performa berupa grafik confusion matrix
dan kurva perkembangan loss beserta akurasi model selama proses fine-tuning berlangsung.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Any

class TrainHelper:
    def __init__(self, label_id_map: dict[int, str]) -> None:
        self.label_id_map = label_id_map
    
    def train_epoch(
        self,
        model: Any, 
        loader: Any, 
        optimizer: Any, 
        scheduler: Any, 
        criterion: Any, 
        device: Any
    ) -> tuple[Any | float, Any | float]:
        # Mengubah model ke dalam mode pelatihan (aktifkan dropout dan batch normalization)
        model.train()
        total_loss, total_correct = 0, 0

        # Iterasi mengambil batch data yang dialirkan oleh DataLoader
        for batch in loader:
            # Mengirimkan tensor input teks dan label ke memory hardware target (CPU atau GPU CUDA)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Mengosongkan akumulasi gradien sebelum proses kalkulasi baru
            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            loss = criterion(outputs.logits, labels)
            
            # Melakukan backpropagation untuk menghitung gradien error
            loss.backward()

            # Menerapkan gradient clipping untuk mencegah masalah exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            
            # Memperbarui bobot parameter model dan learning rate penjadwal
            optimizer.step()
            scheduler.step()

            # Akumulasi nilai loss dan jumlah prediksi yang benar dalam satu batch
            total_loss += loss.item()
            total_correct += (outputs.logits.argmax(dim = 1) == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / len(loader.dataset)
        
        return avg_loss, accuracy


    def eval_epoch(
        self,
        model: Any, 
        loader: Any, 
        criterion: Any, 
        device: Any
    ) -> tuple[Any | float, Any | float]:
        # Mengubah model ke dalam mode evaluasi (menonaktifkan dropout)
        model.eval()
        total_loss, total_correct = 0, 0

        # Mematikan kalkulasi gradien untuk menghemat memori GPU
        with torch.no_grad():
            for batch in loader:
                # Alokasi batch data evaluasi ke hardware target
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Forward pass data validasi tanpa memperbarui bobot model
                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                loss = criterion(outputs.logits, labels)

                # Akumulasi loss dan total prediksi benar untuk data validasi
                total_loss += loss.item()
                total_correct += (outputs.logits.argmax(dim = 1) == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / len(loader.dataset)
        
        return avg_loss, accuracy

    def evaluate_test(
        self,
        model: Any, 
        loader: Any, 
        device: Any
    ) -> tuple[list, list]:
        # Mengubah model ke mode evaluasi untuk pengujian akhir
        model.eval()
        all_preds, all_labels = [], []

        # Mematikan pelacakan gradien selama inferensi data pengujian
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Mendapatkan output prediksi logits dari test dataset
                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                
                # Mengambil index kelas dengan nilai probabilitas logits tertinggi
                preds = outputs.logits.argmax(dim = 1)

                # Memindahkan tensor kembali ke CPU dan mengonversinya ke array NumPy
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_preds

    def plot_confusion_matrix(
        self,
        labels: Any,
        preds: Any,
        name: str,
        figsize: tuple[int, int] = (7, 4)
    ) -> None:
        # Menghitung nilai confusion matrix berdasarkan label aktual dan hasil prediksi
        cm = confusion_matrix(labels, preds)
        label_names = [self.label_id_map[index] for index in range(len(self.label_id_map))]

        # Menginisialisasi ukuran dan komponen visual heatmap grafik via Seaborn
        plt.figure(figsize = figsize)
        sns.heatmap(
            cm, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = label_names, yticklabels = label_names
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Label Sebenarnya")
        plt.xlabel("Label Prediksi")
        plt.tight_layout()

    def plot_training_curve(
        self, 
        history: Any, 
        name: str,
        figsize: tuple[int, int] = (8, 4)
    ) -> None:
        # Membuat range sumbu X berdasarkan total jumlah epoch pelatihan yang dilalui
        epochs = range(1, len(history["train_loss"]) + 1)


        # Inisialisasi struktur subplot untuk menampilkan dua grafik berdampingan
        fig, axes = plt.subplots(1, 2, figsize = figsize)
        fig.suptitle(f"Training Curve - {name}", fontsize = 13, fontweight = "bold")

        # Menggambar kurva perbandingan loss training dan validasi
        axes[0].plot(epochs, history["train_loss"], label = "Train")
        axes[0].plot(epochs, history["val_loss"], label = "Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Menggambar kurva perbandingan akurasi training dan validasi
        axes[1].plot(epochs, history["train_acc"], label = "Train")
        axes[1].plot(epochs, history["val_acc"], label = "Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        plt.tight_layout()
        
    def export_to_onnx(
        self,
        model: Any,
        tokenizer: Any,
        export_path: str,
        max_len: int,
        device: Any
    ) -> None:
        # Memastikan model berada dalam mode evaluasi untuk menonaktifkan lapisan dropout
        model.eval()

        # Membuat teks dummy untuk memicu pembuatan representasi bentuk struktur grafik input model
        dummy_text = ("Sistem deteksi otomatis sedang melakukan verifikasi terhadap pesan teks ini " +
            "untuk pengujian performa arsitektur model.")
        inputs = tokenizer(
            dummy_text,
            max_length = max_len,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        # Mengirimkan komponen data dummy input ke memory hardware target (CPU atau GPU CUDA)
        dummy_input_ids = inputs["input_ids"].to(device)
        dummy_attention_mask = inputs["attention_mask"].to(device)
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)

        # Mengeksekusi proses konversi grafik komputasi PyTorch ke format biner standar ONNX
        torch.onnx.export(
            model,
            dummy_inputs,
            export_path,
            export_params = True,
            opset_version = 14,
            do_constant_folding = True,
            input_names = ["input_ids", "attention_mask"],
            output_names = ["logits"],
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"}
            }
        )
        
        print(f"Model berhasil dikonversi ke format ONNX dan disimpan di: {export_path}")
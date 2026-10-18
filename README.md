# Thesis Galvanic Skin Response LMS

Repo terpadu untuk penelitian klasifikasi beban kognitif pada aktivitas LMS
berbasis sinyal Galvanic Skin Response (GSR). Isinya mencakup firmware pembaca
sensor, dataset responden anonim, praproses dan representasi data, visualisasi
hasil evaluasi, serta prediksi interaktif terhadap CSV baru.

Data direkam menggunakan Grove GSR Sensor dan dikonversi oleh ESP32-S3.

## Isi repo

- `firmware/gsr_reader/gsr_reader.ino` — pembacaan GSR 12-bit pada pin A0,
  konversi tampilan ke 10-bit, dan keluaran serial 115200 baud.
- `data pc */` — rekaman GSR mentah dengan ID responden anonim `r_N`.
- `nasa_tlx_results/` — skor NASA-TLX yang sudah memakai ID anonim yang sama.
- `dataset_fitur/` — fitur tabular hasil ekstraksi sinyal.
- `app_streamlit/showcase.py` — napak tilas data mentah, praproses, pelabelan,
  ekstraksi fitur, empat representasi model, dan evaluasi.
- `app_streamlit/app.py` — unggah CSV baru dan prediksi SVM, GCN, GLCT-CNN,
  serta AttnBiLSTM.
- `streamlit_app.py` — pintu masuk tunggal untuk kedua halaman di atas.

## Menjalankan Streamlit

Gunakan Python 3.11 atau versi yang kompatibel dengan PyTorch, lalu jalankan:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Halaman **Proses Data sampai Evaluasi** menampilkan seluruh alur penelitian.
Halaman **Prediksi CSV Baru** menerima berkas dengan kolom wajib:

- `TimeStamp`
- `gsrRaw`
- `Activity_Code` (`1` watching, `2` reading, `3` discussion, `4` quiz)

Artefak `nodes_train.npz`, `windows_train.npz`, dan `eval_cache.npz` sudah
disertakan agar dashboard dapat langsung dipakai tanpa melatih ulang semuanya.

## Firmware Arduino

Buka `firmware/gsr_reader/gsr_reader.ino` di Arduino IDE, pilih board yang
mendukung `analogReadResolution(12)`, unggah program, lalu buka Serial Monitor
pada 115200 baud.

## Privasi data

Nama responden tidak disimpan dalam dataset, tabel fitur, tabel NASA-TLX, atau
cache aplikasi. Seluruh komponen menggunakan ID anonim `r_N` (dan `pilot_N`
untuk data pilot yang hanya tersimpan sebagai metadata evaluasi).

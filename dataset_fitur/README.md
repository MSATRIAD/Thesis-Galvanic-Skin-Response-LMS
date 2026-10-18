# dataset_fitur — Fitur GSR ter-ekstraksi (data tabular)

Tiap baris = 1 window 30 detik (300 sampel @10Hz), sudah dibersihkan + z-score per subjek.

## Kolom metadata (sama di semua file)
- `window_id` : indeks window
- `subject`   : ID responden anonim (`r_N`)
- `activity`  : watching/reading/discussion/quiz
- `label`     : Low/High (dari WWL, ambang tertil extreme)
- `wwl`       : nilai NASA-TLX WWL kontinu 0-100 (sebelum dibinerkan)

## File
- `features_basic.csv`    : 12 fitur tangan (set ringkas)
- `features_extended.csv` : 37 fitur tangan (tonik, fasik/SCR, statistik, turunan, spektral, Hjorth)
- `features_glct.csv`     : 18 fitur statistik dari transform GLCT

Total 1480 window | 20 subjek | Low=834 High=646.
Sumber pipeline: experiments/shared/ (data.py, eda_preprocessing.py, feature_sets.py).

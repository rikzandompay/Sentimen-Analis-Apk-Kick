# Dashboard Analisis Sentimen Ulasan Kick (Naïve Bayes vs Linear SVM)

Dashboard web satu halaman yang menampilkan hasil penelitian analisis sentimen ulasan aplikasi Kick di
Google Play Store (Multinomial Naïve Bayes vs Linear SVM), lalu memungkinkan penguji mencoba kedua model
secara langsung. Spesifikasinya ada di `PRD.md`.

Model dan angka yang dipakai sekarang berasal dari run Colab notebook
`analisis_sentimen_kick_nb_svm_v5b_colab.ipynb` (preprocessing **v5**, `metrics.json["versi"]["preprocessing"]`).

Semua angka di halaman dibaca dari `artifacts/metrics.json`. Tidak ada angka hasil eksperimen yang
ditulis di kode antarmuka.

## Isi folder

```text
app.py                  server FastAPI: /api/metrics, /api/predict, /api/health, dan halaman static/
preprocessing.py        preprocessing v5, identik dengan Bagian 5 notebook v5 (112 entri slang, huruf berulang,
                        negasi Inggris, stopword Indonesia dan Inggris, negasi dipertahankan)
artifacts/              tfidf.joblib, nb.joblib, svm.joblib, metrics.json dari run Colab notebook v5b
build_artifacts.py      LAMA (alur notebook v4): jangan dijalankan, lihat "Membuat ulang artifacts/"
verifikasi.py           LAMA: membandingkan build v4 lokal dengan run Colab v4
pembanding_colab/       LAMA: artifacts.zip dari run Colab v4, hanya untuk verifikasi.py
datakick_raw.csv        data mentah scraping untuk build v4 (kolom content, score, at); TIDAK disajikan ke web
static/                 index.html, style.css, app.js, tema.js (tema terang/gelap)
static/fonts/           Source Serif 4 dan Source Sans 3 (SIL Open Font License), disajikan sendiri tanpa CDN
DESIGN.md               arah desain (editorial jurnal), palet, tipografi, dan alasan setiap keputusan
requirements.txt        versi pustaka dikunci
```

`datakick_preprocessed_revisi.csv` di folder ini adalah keluaran build v4. Keluaran run v5b yang sepasang
dengan `artifacts/` ada di folder `Documents/Jurnal Sentimen Analisis/` bersama `artifacts.zip`-nya.

## Prasyarat

- Python 3.13 (run Colab memakai 3.13.16; verifikasi lokal memakai 3.13.5)
- Koneksi internet saat pertama kali jalan, untuk mengunduh data NLTK (`punkt`, `punkt_tab`, `stopwords`).
  Unduhan ini otomatis.

## Instalasi

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Atau dengan uv:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
```

Versi `scikit-learn==1.6.1` jangan diubah. Berkas `.joblib` hanya terbaca dengan versi yang sama
dengan saat pelatihan, dan `/api/health` melaporkan apakah versinya cocok.

## Membuat ulang `artifacts/`

> **Jangan jalankan `build_artifacts.py` untuk model sekarang.** Skrip itu salinan notebook **v4**
> (preprocessing lama, data `datakick_raw.csv` 12 Oktober). Menjalankannya akan **menimpa** model v5 di
> `artifacts/` dengan model v4. `verifikasi.py` juga masih membandingkan dengan run Colab v4, jadi untuk
> artifacts v5 hasilnya pasti "berbeda".
>
> Untuk memperbarui model v5, jalankan notebook v5b di Colab, lalu ganti isi `artifacts/` dengan
> `artifacts.zip` hasilnya, dan `preprocessing.py` dengan versi yang sepasang dengan notebook itu.
> Setelah itu periksa `/api/health` (versi scikit-learn harus cocok).

Bagian di bawah menjelaskan alur v4 lama, untuk riwayat:

```bash
.venv/bin/python build_artifacts.py     # sekitar 1 menit (nested CV paling lama)
.venv/bin/python verifikasi.py          # kode keluar 0 = sama dengan run Colab
```

`build_artifacts.py` menjalankan alur notebook tanpa mengubah preprocessing, split, TF-IDF, ROS,
penalaan, model, nested CV, McNemar, ablasi, maupun `random_state`. Perbedaannya dengan notebook hanya:

- `SUMBER_DATA = "csv"`, `FILE_DATA_MENTAH = "datakick_raw.csv"`, `TANGGAL_AMBIL_CSV = "2026-10-17"`
  (`BATAS_ULASAN` tetap `"2026-10-12 23:59:59"`)
- baris khusus Colab dibuang (`!pip`, `!zip`, `google.colab`, `google_play_scraper`, `display`, `plt.show`);
  tabel hasil dicetak dengan `print`

`verifikasi.py` melakukan tiga pemeriksaan:

1. Setiap angka di `artifacts/metrics.json` dibandingkan dengan `pembanding_colab/metrics.json` sampai
   4 desimal. Ada dua teks yang memang berbeda: `versi.python` dan `data.tanggal_scraping`. Colab
   mencatat `date.today()` dalam UTC (16 Oktober), sedangkan skrip memakai tanggal WIB (17 Oktober).
2. Kosakata dan IDF TF-IDF, parameter Naive Bayes, serta koefisien SVM dibandingkan dengan model `.joblib` dari Colab.
3. Keluaran `preprocessing.py` dibandingkan dengan kolom `text_clean` hasil notebook untuk semua ulasan.

Jika notebook dijalankan ulang di Colab dengan data baru, ganti isi `pembanding_colab/` dengan
`artifacts.zip` yang baru dan `datakick_raw.csv` dengan CSV yang baru, lalu jalankan kedua perintah di atas lagi.

## API key

`/api/predict` punya dua jalur akses:

- **Klien luar** (curl, `/docs`, Postman, aplikasi lain) wajib membawa header `X-API-Key`.
- **Halaman dashboard** tidak perlu mengetik kunci. Saat halaman dibuka, server memberi cookie sesi
  `sesi_dashboard` yang ditandatangani HMAC, bersifat HttpOnly (tidak bisa dibaca JavaScript),
  `SameSite=Strict`, dan berlaku 12 jam. Cookie palsu atau kedaluwarsa ditolak; muat ulang halaman memberi
  sesi baru.

`/api/metrics` dan `/api/health` tetap terbuka karena hanya berisi hasil paper dan status server.
Kuncinya dibaca dari variabel lingkungan `KICK_API_KEY`, atau dari file `.env` di folder proyek:

```text
KICK_API_KEY=isi-dengan-kunci-acak
```

Buat kunci baru dengan `.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(24))"`. File `.env`
tidak ikut disajikan ke web dan sudah masuk `.gitignore`; jangan dibagikan. Server menolak menyala jika
kuncinya belum diatur. Di hosting (Hugging Face Spaces atau Render), isi `KICK_API_KEY` sebagai secret.

- `/docs`: klik **Authorize**, isi kuncinya, lalu coba `/api/predict`.
- Terminal: tambahkan `-H "X-API-Key: <kunci>"` pada perintah curl.

Tanpa kunci atau dengan kunci salah, server menjawab **401** "API key salah atau belum diisi".

## Menjalankan server

```bash
.venv/bin/uvicorn app:app --port 8000
```

Setelah itu buka <http://127.0.0.1:8000>. TF-IDF, kedua model, metrics, dan stemmer dimuat sekali saat
server menyala. Halaman dan API disajikan dari server yang sama, jadi CORS tidak perlu diatur.

### API

| Endpoint | Metode | Isi |
| --- | --- | --- |
| `/api/metrics` | GET | seluruh isi `artifacts/metrics.json` |
| `/api/predict` | POST | `{"text": "...", "model": "nb" atau "svm"}`, menghasilkan label, skor, dan tahap preprocessing |
| `/api/health` | GET | status server, model yang termuat, dan kecocokan versi scikit-learn |

```bash
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <kunci>" \
  -d '{"text": "Aplikasi lemot sering buffer dan logout sendiri", "model": "svm"}'
```

Jawaban mengikuti kontrak bagian 10 PRD (`label`, `confidence`, `confidence_kind`, `probabilities`,
`preprocessing` berisi `original`, `cleaned`, `slang_normalized`, `after_stopword`, `stemmed`,
`replacements`, `negations_kept`). Ada dua tambahan: `model` dan `vocabulary`
(`{"known", "total"}`, yaitu berapa kata hasil preprocessing yang ada di kosakata TF-IDF).

- `confidence_kind = "probability"` untuk NB (`predict_proba`).
- `confidence_kind = "logistic_margin"` untuk SVM: 1 / (1 + e^(−d)) dari `decision_function`.
  Ini perkiraan keyakinan, bukan probabilitas terkalibrasi.
- Label selalu sama dengan `model.predict`. Urutan kelas model adalah `['Negatif', 'Positif']`.

Kode status:

| Kode | Kapan |
| --- | --- |
| 400 | teks kosong, lebih dari 500 karakter, model tidak dikenal, atau body tidak valid |
| 401 | header `X-API-Key` salah, atau tidak ada kunci maupun cookie sesi yang sah |
| 413 | body lebih dari 16 KB |
| 422 | teks kosong setelah preprocessing (hanya emoji, angka, tautan, atau kata umum) |
| 429 | lebih dari 60 prediksi per menit dari satu alamat IP |
| 500 | galat server |

## Keamanan dan privasi

- Teks ulasan tidak disimpan dan tidak dicatat. Log akses uvicorn hanya mencatat path, bukan isi permintaan.
- Semua teks ditampilkan lewat `textContent`, tidak pernah `innerHTML`. Server juga mengirim header
  Content-Security-Policy yang hanya mengizinkan skrip dari server sendiri.
- `datakick_raw.csv` dan `artifacts/` berada di luar `static/`, sehingga tidak bisa diunduh lewat web.

## Deploy (Hugging Face Spaces atau Render)

- Sertakan folder `artifacts/` dan `preprocessing.py` apa adanya, lalu pakai `requirements.txt` ini.
- Di balik proxy, jalankan `uvicorn app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"`.
  Tanpa opsi ini, batas laju menganggap semua pengunjung berasal dari satu IP (proxy).
- Hosting gratis "tidur" jika lama tidak dipakai. Buka halaman beberapa menit sebelum presentasi; halaman
  akan menampilkan pesan "Server sedang menyala" selama server bangun.

## Catatan sebelum sidang

- Preset positif ("Streaming lancar jernih mantap tidak ada iklan") **diprediksi Negatif** oleh kedua
  model v5 (NB 56,0%, SVM 58,8%; pada model v4: NB 65,5%, SVM 57,6%). Kata `jernih` tidak ada di kosakata,
  sedangkan `tidak` (paling kuat), `lancar`, dan `iklan` condong ke ulasan negatif. Preset negatif diprediksi benar oleh
  kedua model (NB 84,4%, SVM 62,1%). Sesuai PRD, yang diganti adalah kalimat presetnya (konstanta `PRESET`
  di `static/app.js`), bukan modelnya.
- Judul di header memakai judul notebook. Sesuaikan dengan judul final paper di `static/index.html`.

# -*- coding: utf-8 -*-
"""Membuat ulang artifacts/ (tfidf.joblib, nb.joblib, svm.joblib, metrics.json) secara lokal.

Isi skrip ini adalah alur notebook analisis_sentimen_kick_nb_svm_v4_colab.ipynb apa adanya.
Penyesuaian terhadap notebook hanya:
- SUMBER_DATA = "csv", FILE_DATA_MENTAH = "datakick_raw.csv", TANGGAL_AMBIL_CSV = "2026-10-17"
- baris khusus Colab dibuang: !pip, !zip, google.colab, google_play_scraper, display, plt.show
  (display tabel hasil diganti print agar tetap terbaca di terminal)
Preprocessing, split, TF-IDF, ROS, tuning, model, nested CV, McNemar, ablasi, dan random_state tidak diubah.

Jalankan:  .venv/bin/python build_artifacts.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # path relatif di bawah dihitung dari folder proyek

import matplotlib
matplotlib.use("Agg")   # tanpa jendela grafik (plt.show dibuang)

# ## 2. Import Library

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from scipy.stats import binomtest

# Download resource NLTK (hanya perlu dijalankan sekali di awal)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Catat versi pustaka untuk bagian Metode Penelitian di paper
import sys, sklearn, imblearn, importlib.metadata as md
print("Python              :", sys.version.split()[0])
print("scikit-learn        :", sklearn.__version__)
print("imbalanced-learn    :", imblearn.__version__)
print("NLTK                :", nltk.__version__)
print("Sastrawi            :", md.version("Sastrawi"))

# ## 3. Data Ulasan

# Pilih sumber data:
#   "scraping" -> hanya di notebook Colab (google_play_scraper tidak dipakai di skrip lokal)
#   "csv"      -> baca file data mentah yang sudah tersimpan (hasil dapat direproduksi persis)
SUMBER_DATA = "csv"
FILE_DATA_MENTAH = "datakick_raw.csv"

# Ulasan setelah tanggal ini dibuang agar data mendekati scraping 12 Oktober 2026.
# Isi None jika tidak ingin memotong. Catat di paper: "ulasan hingga <tanggal>, diambil pada <tanggal ambil>".
BATAS_ULASAN = "2026-10-12 23:59:59"
# Isi hanya jika SUMBER_DATA = "csv" (tanggal CSV itu diambil, format YYYY-MM-DD)
TANGGAL_AMBIL_CSV = "2026-10-17"

if SUMBER_DATA == "scraping":
    raise SystemExit("Scraping hanya dijalankan di notebook Colab; skrip ini membaca FILE_DATA_MENTAH.")
else:
    df = pd.read_csv(FILE_DATA_MENTAH)[['content', 'score', 'at']]
    TANGGAL_AMBIL = TANGGAL_AMBIL_CSV
    print(f"Data dibaca dari '{FILE_DATA_MENTAH}'")

# Pemotongan per tanggal ulasan (lihat BATAS_ULASAN di atas)
df['at'] = pd.to_datetime(df['at'])
print(f"Rentang tanggal ulasan: {df['at'].min()} s.d. {df['at'].max()}")
if BATAS_ULASAN:
    n_sebelum_potong = len(df)
    df = df[df['at'] <= BATAS_ULASAN]
    print(f"Ulasan setelah {BATAS_ULASAN} dibuang: {n_sebelum_potong - len(df)} baris")
df = df[['content', 'score']].copy()

# Membersihkan data yang tidak memiliki teks ulasan (hanya rating tanpa komentar)
df = df.dropna(subset=['content'])
df = df[df['content'].str.strip() != '']

print(f"Total data setelah scraping dan filter awal: {len(df)} baris")
n_berteks = len(df)   # dipakai untuk metrics.json

# ### 3.1 Menghapus Data Duplikat

jumlah_sebelum = len(df)
df = df.drop_duplicates(subset='content').reset_index(drop=True)
jumlah_sesudah = len(df)
n_dedup = len(df)   # dipakai untuk metrics.json

print(f"Jumlah data sebelum penghapusan duplikat : {jumlah_sebelum}")
print(f"Jumlah data setelah penghapusan duplikat  : {jumlah_sesudah}")
print(f"Jumlah duplikat yang dihapus              : {jumlah_sebelum - jumlah_sesudah}")

# ## 4. Pelabelan Sentimen

my_df = df.copy()   # data analisis = hasil scraping + penghapusan duplikat (Bagian 3), bukan CSV hasil run lama

def pelabelan(score):
    if score < 3:
        return 'Negatif'
    elif score > 3:
        return 'Positif'
    else:
        return None  # skor 3 (netral) dikeluarkan dari analisis biner

my_df['Label'] = my_df['score'].apply(pelabelan)

print("Distribusi Sentimen (sebelum penghapusan data netral):")
print(my_df['Label'].value_counts(dropna=False))

dist_label = {('Netral' if pd.isna(k) else str(k)): int(v) for k, v in my_df['Label'].value_counts(dropna=False).items()}   # untuk metrics.json
my_df = my_df.dropna(subset=['Label']).reset_index(drop=True)

print("\nJumlah data setelah data netral (skor=3) dikeluarkan:", len(my_df))

# ## 5. Text Preprocessing

def remove_emoji(text):
    if not isinstance(text, str):
        return text
    # Menghapus emoji dan karakter non-ASCII lainnya
    return text.encode('ascii', 'ignore').decode('ascii')

SLANG = {
    # negasi
    'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak','enggak':'tidak','engga':'tidak','gk':'tidak',
    'tdk':'tidak','tak':'tidak','bkn':'bukan','blm':'belum','belom':'belum','jgn':'jangan','gausah':'tidak usah',
    'gabisa':'tidak bisa','gabsa':'tidak bisa','gbs':'tidak bisa','gaada':'tidak ada','gada':'tidak ada',
    # penghubung / kata fungsi
    'yg':'yang','dgn':'dengan','dg':'dengan','utk':'untuk','untk':'untuk','krn':'karena','karna':'karena','tp':'tapi',
    'tpi':'tapi','jd':'jadi','sm':'sama','dr':'dari','dlm':'dalam','pd':'pada','trs':'terus','lg':'lagi','lgi':'lagi',
    'sdh':'sudah','udh':'sudah','udah':'sudah','emg':'memang','emang':'memang','kyk':'seperti','kayak':'seperti',
    'gmn':'bagaimana','gimana':'bagaimana','knp':'kenapa','org':'orang','aja':'saja','aj':'saja','doang':'saja',
    'bs':'bisa','bgt':'sangat','banget':'sangat','sgt':'sangat','sy':'saya','gw':'saya','gue':'saya','gua':'saya','lu':'kamu',
    # istilah domain aplikasi streaming
    'apk':'aplikasi','app':'aplikasi','apps':'aplikasi','aplikasinya':'aplikasi','hp':'ponsel',
    'lemot':'lambat','lag':'lambat','ngelag':'lambat','lagging':'lambat','lelet':'lambat','loadingnya':'loading',
    'buffering':'buffer','ngebuffer':'buffer','eror':'error','erorr':'error','ngebug':'bug','bugs':'bug',
    'nntn':'nonton','streamingnya':'streaming','stream':'streaming',
    # ekspresi sentimen
    'bgs':'bagus','mantul':'mantap','mantab':'mantap','kren':'keren','jlk':'jelek','parah':'buruk',
    'thx':'terima kasih','makasih':'terima kasih','terimakasih':'terima kasih',
}

def cleaning_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return word_tokenize(text)

def slang_normalization(tokens):
    hasil = []
    for word in tokens:
        # .split() agar hasil normalisasi multi-kata (mis. 'gabisa' -> 'tidak bisa') dipecah
        # menjadi token terpisah dan diperlakukan konsisten oleh tahap stopword removal
        hasil.extend(SLANG.get(word, word).split())
    return hasil

list_stopwords = set(stopwords.words('indonesian'))
list_stopwords.update(['nya', 'yg', 'ga', 'gak', 'dgn', 'utk', 'aja'])

# Kata negasi DIPERTAHANKAN: stopword NLTK Indonesia memuat kata-kata ini, padahal ia membalik polaritas
# sentimen (mis. "tidak bagus" akan menjadi "bagus" jika 'tidak' dibuang).
KATA_NEGASI = {'tidak', 'tak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang'}
list_stopwords -= KATA_NEGASI

def stopword_removal(tokens):
    return [word for word in tokens if word not in list_stopwords]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    text = remove_emoji(text)
    text = cleaning_text(text)
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = slang_normalization(tokens)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)

print("Memulai Preprocessing Teks... (bisa beberapa menit karena proses stemming)")
my_df['text_clean'] = my_df['content'].apply(preprocess_text)

# Menyimpan data hasil preprocessing ke file CSV
my_df.to_csv("datakick_preprocessed_revisi.csv", index=False)
print("Data hasil preprocessing telah disimpan ke 'datakick_preprocessed_revisi.csv'")

# ### 5.1 Memfilter Ulasan yang Menjadi Kosong Setelah Cleaning

jumlah_sebelum = len(my_df)
my_df = my_df[my_df['text_clean'].str.strip() != ''].reset_index(drop=True)
jumlah_sesudah = len(my_df)

print(f"Jumlah data sebelum filter teks kosong : {jumlah_sebelum}")
print(f"Jumlah data setelah filter teks kosong  : {jumlah_sesudah}")
print(f"Jumlah baris yang dihapus                : {jumlah_sebelum - jumlah_sesudah}")

# Menghitung jumlah masing-masing kelas sentimen setelah pembersihan
kelas_counts = my_df['Label'].value_counts()

plt.figure(figsize=(6, 4.5))
sns.set_style("whitegrid")

# Membuat bar chart
ax = sns.barplot(
    x=kelas_counts.index,
    y=kelas_counts.values,
    hue=kelas_counts.index,
    palette=['#4C72B0', '#C44E52'],
    legend=False
)

# Menambahkan anotasi angka jumlah data di atas batang
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)} ulasan',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='black',
                xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribusi Kelas Sentimen (Positif vs Negatif)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Kelas Sentimen', fontsize=10)
plt.ylabel('Jumlah Ulasan', fontsize=10)
plt.ylim(0, max(kelas_counts.values) * 1.15) # Memberi ruang untuk anotasi
plt.tight_layout()

# ## 6. Split Data & TF-IDF (Perbaikan Data Leakage)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    my_df['text_clean'], my_df['Label'],
    test_size=0.3, random_state=42, stratify=my_df['Label']
)

tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train_text)   # fit HANYA pada data train
X_test = tfidf.transform(X_test_text)         # transform saja pada data test (tanpa fit ulang)

print(f"Jumlah data train : {X_train.shape[0]} baris, {X_train.shape[1]} fitur")
print(f"Jumlah data test  : {X_test.shape[0]} baris, {X_test.shape[1]} fitur")

# ## 7. Penanganan Ketidakseimbangan Kelas (Oversampling)

# Satu konfigurasi oversampling untuk seluruh eksperimen (single split, tuning, dan K-Fold CV)
SAMPLING_STRATEGY = 0.85   # Menggunakan tipe float (0.85), bukan string ('0.85')

print("Distribusi kelas pada data train sebelum resampling:")
print(y_train.value_counts())

ros = RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY, random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("\nDistribusi kelas setelah ROS:")
print(y_train_resampled.value_counts())

# ## 8. Pelatihan Model Naive Bayes
# ### 8.1 Penentuan Parameter `alpha` (Hyperparameter Tuning pada Data Train)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('ros', RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY, random_state=42)),
    ('nb', MultinomialNB())
])

param_grid = {'nb__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]}

# Kriteria penalaan = akurasi (sebanding dengan penelitian pembanding); Macro-F1 tetap dilaporkan di setiap evaluasi.
# Jika ingin menala berdasarkan Macro-F1, ganti scoring='f1_macro' di sini dan di nested CV -> SEMUA angka paper berubah.
grid_search = GridSearchCV(grid_pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
grid_search.fit(X_train_text, y_train)   # HANYA data train (TF-IDF & ROS dikerjakan di dalam pipeline)

best_alpha = grid_search.best_params_['nb__alpha']
print("Alpha terbaik (data train)  :", best_alpha)
print(f"Akurasi CV pada data train  : {grid_search.best_score_:.2%}")

# ### 8.2 Pelatihan Model Final

nb_model = MultinomialNB(alpha=best_alpha)
nb_model.fit(X_train_resampled, y_train_resampled)

y_pred = nb_model.predict(X_test)  # evaluasi tetap menggunakan data test ASLI (tidak di-resample)

print("Model berhasil dilatih.")

# ## 9. Evaluasi Model (Single Split)

akurasi = accuracy_score(y_test, y_pred)
f1_nb = f1_score(y_test, y_pred, average='macro')
print(f"Akurasi Model  : {akurasi:.2%}")
print(f"Macro-F1 Model : {f1_nb:.2%}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=nb_model.classes_)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix - Naive Bayes')
plt.tight_layout()

# ## 10. Validasi Tambahan: Nested Stratified K-Fold Cross-Validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # fold luar

nested_nb = GridSearchCV(grid_pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy')

cv_nb = cross_validate(
    nested_nb, my_df['text_clean'], my_df['Label'],
    cv=skf, scoring=['accuracy', 'f1_macro'], return_estimator=True
)

cv_scores = cv_nb['test_accuracy']
cv_f1_nb = cv_nb['test_f1_macro']

print("Akurasi tiap fold   :", np.round(cv_scores, 4))
print(f"Rata-rata akurasi   : {cv_scores.mean():.2%}")
print(f"Standar deviasi     : {cv_scores.std():.2%}")
print(f"Rata-rata macro-F1  : {cv_f1_nb.mean():.2%}")
print("Macro-F1 tiap fold  :", np.round(cv_f1_nb, 4))
print(f"Std. dev. macro-F1  : {cv_f1_nb.std():.2%}")
print("Alpha terbaik tiap fold:", [e.best_params_['nb__alpha'] for e in cv_nb['estimator']])

# ## 11. Pemodelan dengan Support Vector Machine (SVM)
# ### 11.1 Penentuan Parameter `C` (Hyperparameter Tuning pada Data Train)

grid_pipeline_svm = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('ros', RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY, random_state=42)),
    ('svm', LinearSVC(random_state=42))
])

param_grid_svm = {'svm__C': [0.01, 0.1, 1.0, 10.0]}

grid_search_svm = GridSearchCV(grid_pipeline_svm, param_grid=param_grid_svm, cv=inner_cv, scoring='accuracy')
grid_search_svm.fit(X_train_text, y_train)   # HANYA data train

best_C = grid_search_svm.best_params_['svm__C']
print("C terbaik untuk SVM (data train) :", best_C)
print(f"Akurasi CV pada data train       : {grid_search_svm.best_score_:.2%}")

# ### 11.2 Pelatihan dan Evaluasi Model SVM (Single Split)

# Inisialisasi dan latih model SVM pada data train yang sudah di-resample (C terbaik dari Bagian 11.1)
svm_model = LinearSVC(C=best_C, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Lakukan prediksi pada data test
y_pred_svm = svm_model.predict(X_test)
print("Model SVM berhasil dilatih dan dievaluasi pada data test.")

akurasi_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
print(f"Akurasi Model SVM  : {akurasi_svm:.2%}")
print(f"Macro-F1 Model SVM : {f1_svm:.2%}\n")

print("Classification Report SVM:")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix - SVM')
plt.tight_layout()

# ### 11.3 Validasi Tambahan SVM: Nested Stratified K-Fold Cross-Validation

nested_svm = GridSearchCV(grid_pipeline_svm, param_grid=param_grid_svm, cv=inner_cv, scoring='accuracy')

cv_svm = cross_validate(
    nested_svm, my_df['text_clean'], my_df['Label'],
    cv=skf, scoring=['accuracy', 'f1_macro'], return_estimator=True
)

cv_scores_svm = cv_svm['test_accuracy']
cv_f1_svm = cv_svm['test_f1_macro']

print("Akurasi tiap fold (SVM)  :", np.round(cv_scores_svm, 4))
print(f"Rata-rata akurasi (SVM)   : {cv_scores_svm.mean():.2%}")
print(f"Standar deviasi (SVM)     : {cv_scores_svm.std():.2%}")
print(f"Rata-rata macro-F1 (SVM)  : {cv_f1_svm.mean():.2%}")
print("Macro-F1 tiap fold (SVM) :", np.round(cv_f1_svm, 4))
print(f"Std. dev. macro-F1 (SVM)  : {cv_f1_svm.std():.2%}")
print("C terbaik tiap fold:", [e.best_params_['svm__C'] for e in cv_svm['estimator']])

# ## 12. Perbandingan Kinerja Model (Naïve Bayes vs SVM)
# ### 12.1 Tabel dan Grafik Perbandingan

df_compare = pd.DataFrame({
    'Model': ['Naïve Bayes', 'Naïve Bayes', 'SVM (Linear)', 'SVM (Linear)'],
    'Metode': ['Single Split', 'Nested K-Fold CV (Mean)', 'Single Split', 'Nested K-Fold CV (Mean)'],
    'Akurasi': [akurasi, cv_scores.mean(), akurasi_svm, cv_scores_svm.mean()],
    'Macro-F1': [f1_nb, cv_f1_nb.mean(), f1_svm, cv_f1_svm.mean()]
})
print(df_compare.round(4).to_string(index=False))

# Membuat plot
plt.figure(figsize=(8, 5.5))
sns.set_style("whitegrid")

ax = sns.barplot(
    x='Model',
    y='Akurasi',
    hue='Metode',
    data=df_compare,
    palette='Set2'
)

# Menampilkan label nilai akurasi di atas masing-masing bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.2%}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black',
                    xytext=(0, 5),
                    textcoords='offset points')

plt.title('Perbandingan Akurasi Model: Naïve Bayes vs SVM', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Algoritma Model', fontsize=11)
plt.ylabel('Akurasi', fontsize=11)
plt.ylim(0, 1.1)                       # sumbu-y dimulai dari 0 (tidak dipotong)
plt.yticks(np.arange(0, 1.01, 0.2))
plt.legend(title='Metode Evaluasi', loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.tight_layout()

# ### 12.2 Uji Signifikansi NB vs SVM (McNemar Exact Test)

nb_benar  = (y_pred == y_test.values)
svm_benar = (y_pred_svm == y_test.values)

b = int(np.sum(nb_benar & ~svm_benar))    # NB benar, SVM salah
c = int(np.sum(~nb_benar & svm_benar))    # NB salah, SVM benar

print(f"NB benar & SVM salah : {b}")
print(f"NB salah & SVM benar : {c}")

if b + c == 0:
    print("Kedua model memberi prediksi identik pada data test (uji tidak dapat dihitung).")
else:
    p_value = binomtest(min(b, c), n=b + c, p=0.5).pvalue
    print(f"p-value (McNemar exact): {p_value:.4f}")
    print("Perbedaan SIGNIFIKAN pada alpha=0.05" if p_value < 0.05 else "Perbedaan TIDAK signifikan pada alpha=0.05")

# ### 12.3 Pengaruh Oversampling (Ablation)

def ringkas_cv(estimator):
    hasil = cross_validate(estimator, my_df['text_clean'], my_df['Label'], cv=skf,
                           scoring=['accuracy', 'f1_macro'])
    return hasil['test_accuracy'].mean(), hasil['test_f1_macro'].mean()

def buat_pipeline(klasifier, dengan_ros):
    langkah = [('tfidf', TfidfVectorizer(max_features=5000))]
    if dengan_ros:
        langkah.append(('ros', RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY, random_state=42)))
    langkah.append(('clf', klasifier))
    return ImbPipeline(langkah)

baris = []
for nama, klf in [('Naïve Bayes', MultinomialNB(alpha=best_alpha)),
                  ('SVM (Linear)', LinearSVC(C=best_C, random_state=42))]:
    for dengan_ros in [False, True]:
        acc, f1 = ringkas_cv(buat_pipeline(klf, dengan_ros))
        baris.append({'Model': nama,
                      'Oversampling': 'Dengan ROS' if dengan_ros else 'Tanpa ROS',
                      'Akurasi (CV)': acc,
                      'Macro-F1 (CV)': f1})

df_ablation = pd.DataFrame(baris)
print(df_ablation.round(4).to_string(index=False))

# ## 14. Ekspor Artefak untuk Dashboard

import json, os, joblib, sys, sklearn, imblearn
os.makedirs("artifacts", exist_ok=True)

joblib.dump(tfidf, "artifacts/tfidf.joblib")
joblib.dump(nb_model, "artifacts/nb.joblib")
joblib.dump(svm_model, "artifacts/svm.joblib")

def cm_dict(m):  # baris = aktual, kolom = prediksi, urutan Negatif, Positif
    return {"tn": int(m[0][0]), "fp": int(m[0][1]), "fn": int(m[1][0]), "tp": int(m[1][1])}

metrics = {
  "versi": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__},
  "data": {"tanggal_scraping": TANGGAL_AMBIL, "batas_ulasan": BATAS_ULASAN,
           "berteks": n_berteks, "setelah_dedup": n_dedup,
           "distribusi_label": dist_label, "akhir": len(my_df),
           "distribusi_akhir": {k: int(v) for k, v in my_df["Label"].value_counts().items()},
           "latih": {k: int(v) for k, v in y_train.value_counts().items()},
           "uji": {k: int(v) for k, v in y_test.value_counts().items()},
           "latih_setelah_ros": {k: int(v) for k, v in y_train_resampled.value_counts().items()},
           "jumlah_fitur": int(X_train.shape[1])},
  "model": {
    "nb":  {"alpha": best_alpha,
            "single_split": {"akurasi": akurasi, "macro_f1": f1_nb, "cm": cm_dict(cm)},
            "nested_cv": {"akurasi_mean": cv_scores.mean(), "akurasi_std": cv_scores.std(),
                          "macro_f1_mean": cv_f1_nb.mean(), "macro_f1_std": cv_f1_nb.std()}},
    "svm": {"C": best_C,
            "single_split": {"akurasi": akurasi_svm, "macro_f1": f1_svm, "cm": cm_dict(cm_svm)},
            "nested_cv": {"akurasi_mean": cv_scores_svm.mean(), "akurasi_std": cv_scores_svm.std(),
                          "macro_f1_mean": cv_f1_svm.mean(), "macro_f1_std": cv_f1_svm.std()}}},
  "mcnemar": {"b": b, "c": c, "p_value": p_value, "alpha": 0.05, "signifikan": p_value < 0.05},
  "ablasi": df_ablation.to_dict(orient="records"),
}
with open("artifacts/metrics.json", "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("metrics.json tersimpan di artifacts/ (bersama tfidf.joblib, nb.joblib, svm.joblib).")

# -*- coding: utf-8 -*-
"""Verifikasi hasil build lokal terhadap hasil run notebook di Google Colab.

1. artifacts/metrics.json vs pembanding_colab/metrics.json: semua angka harus sama sampai 4 desimal.
2. Model lokal vs model Colab (jika pembanding_colab/*.joblib ada): vocabulary dan IDF TF-IDF,
   parameter Naive Bayes dan Linear SVM.
3. preprocessing.py (dipakai server) vs kolom text_clean hasil notebook (datakick_preprocessed_revisi.csv).

Keluar dengan kode 1 jika ada yang berbeda.  Jalankan:  .venv/bin/python verifikasi.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

AKAR = Path(__file__).parent
LOKAL = AKAR / "artifacts"
COLAB = AKAR / "pembanding_colab"

# Teks yang memang berbeda karena lingkungan/konfigurasi, bukan karena hasil eksperimen
BEDA_DIHARAPKAN = {
    "versi.python": "versi patch Python lokal berbeda dari Colab",
    "data.tanggal_scraping": "diisi dari TANGGAL_AMBIL_CSV (tanggal WIB), Colab mencatat date.today() dalam UTC",
}


def ratakan(obj, awalan=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from ratakan(v, f"{awalan}.{k}" if awalan else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from ratakan(v, f"{awalan}[{i}]")
    else:
        yield awalan, obj


def angka(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def bandingkan_metrics():
    lokal = dict(ratakan(json.loads((LOKAL / "metrics.json").read_text(encoding="utf-8"))))
    colab = dict(ratakan(json.loads((COLAB / "metrics.json").read_text(encoding="utf-8"))))
    beda, info, n_angka = [], [], 0
    for kunci in sorted(set(lokal) | set(colab)):
        if kunci not in lokal or kunci not in colab:
            beda.append(f"{kunci}: hanya ada di {'lokal' if kunci in lokal else 'Colab'}")
            continue
        a, b = lokal[kunci], colab[kunci]
        if angka(a) and angka(b):
            n_angka += 1
            if round(a, 4) != round(b, 4):
                beda.append(f"{kunci}: lokal {a!r} vs Colab {b!r}")
        elif a != b:
            if kunci in BEDA_DIHARAPKAN:
                info.append(f"{kunci}: lokal {a!r} vs Colab {b!r} ({BEDA_DIHARAPKAN[kunci]})")
            else:
                beda.append(f"{kunci}: lokal {a!r} vs Colab {b!r}")
    print(f"[1] metrics.json: {n_angka} angka dibandingkan sampai 4 desimal")
    for baris in info:
        print("    catatan :", baris)
    for baris in beda:
        print("    BEDA    :", baris)
    print("    hasil   :", "SAMA" if not beda else f"{len(beda)} PERBEDAAN")
    return not beda


def bandingkan_model():
    if not all((COLAB / f).exists() for f in ("tfidf.joblib", "nb.joblib", "svm.joblib")):
        print("[2] model: pembanding_colab/*.joblib tidak ada, dilewati")
        return True
    L = {n: joblib.load(LOKAL / f"{n}.joblib") for n in ("tfidf", "nb", "svm")}
    C = {n: joblib.load(COLAB / f"{n}.joblib") for n in ("tfidf", "nb", "svm")}
    cek = {
        "tfidf.vocabulary_": L["tfidf"].vocabulary_ == C["tfidf"].vocabulary_,
        "nb.classes_ / svm.classes_": list(L["nb"].classes_) == list(C["nb"].classes_) == list(L["svm"].classes_)
                                      == list(C["svm"].classes_) == ["Negatif", "Positif"],
    }
    selisih = {
        "tfidf.idf_": np.abs(L["tfidf"].idf_ - C["tfidf"].idf_).max(),
        "nb.feature_log_prob_": np.abs(L["nb"].feature_log_prob_ - C["nb"].feature_log_prob_).max(),
        "nb.class_log_prior_": np.abs(L["nb"].class_log_prior_ - C["nb"].class_log_prior_).max(),
        "svm.coef_": np.abs(L["svm"].coef_ - C["svm"].coef_).max(),
        "svm.intercept_": np.abs(L["svm"].intercept_ - C["svm"].intercept_).max(),
    }
    print("[2] model lokal vs model Colab")
    for nama, ok in cek.items():
        print(f"    {nama:28s}: {'sama' if ok else 'BEDA'}")
    for nama, s in selisih.items():
        print(f"    {nama:28s}: selisih maksimum {s:.2e}")
    return all(cek.values()) and all(s < 1e-6 for s in selisih.values())


def bandingkan_preprocessing():
    from preprocessing import SLANG, preprocess_text
    data = pd.read_csv(AKAR / "datakick_preprocessed_revisi.csv", keep_default_na=False)
    hasil = data["content"].map(preprocess_text)
    beda = data[hasil != data["text_clean"]]
    print(f"[3] preprocessing.py vs notebook: {len(data)} ulasan, {len(SLANG)} entri slang, {len(beda)} berbeda")
    for _, r in beda.head(5).iterrows():
        print(f"    BEDA: {r['content'][:60]!r}")
    return beda.empty


if __name__ == "__main__":
    ok_metrics = bandingkan_metrics()
    ok_model = bandingkan_model()
    ok_pre = bandingkan_preprocessing()
    sys.exit(0 if (ok_metrics and ok_model and ok_pre) else 1)

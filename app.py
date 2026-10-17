# -*- coding: utf-8 -*-
"""Server dashboard analisis sentimen ulasan Kick (Naive Bayes vs Linear SVM).

TF-IDF, kedua model, dan metrics.json dimuat sekali saat server menyala dari artifacts/.
Halaman disajikan dari static/. /api/predict menerima header X-API-Key (KICK_API_KEY di .env) untuk
klien luar, atau cookie sesi yang diberikan server saat halaman dashboard dibuka.

Jalankan:  .venv/bin/uvicorn app:app --port 8000
"""
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import sklearn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from preprocessing import preprocess_steps

ARTEFAK = Path(__file__).parent / "artifacts"
tfidf = joblib.load(ARTEFAK / "tfidf.joblib")
MODELS = {"nb": joblib.load(ARTEFAK / "nb.joblib"), "svm": joblib.load(ARTEFAK / "svm.joblib")}
METRICS = json.loads((ARTEFAK / "metrics.json").read_text(encoding="utf-8"))
ANALYZER = tfidf.build_analyzer()

MAKS_KARAKTER = 500
MAKS_BODY_BYTE = 16_384      # permintaan prediksi lebih besar dari ini ditolak sebelum dibaca
BATAS_LAJU = 60              # permintaan prediksi per alamat IP per menit
_riwayat_ip = defaultdict(deque)

preprocess_steps("aplikasi bagus")   # memuat data tokenizer NLTK sekarang, bukan saat permintaan pertama


def _baca_api_key():
    """API key dari variabel lingkungan KICK_API_KEY, atau dari baris KICK_API_KEY=... di file .env."""
    kunci = os.environ.get("KICK_API_KEY", "").strip()
    berkas = Path(__file__).parent / ".env"
    if not kunci and berkas.exists():
        for baris in berkas.read_text(encoding="utf-8").splitlines():
            nama, _, nilai = baris.partition("=")
            if nama.strip() == "KICK_API_KEY":
                kunci = nilai.strip().strip("\"'")
    if not kunci:
        raise RuntimeError("KICK_API_KEY belum diatur. Isi file .env dengan KICK_API_KEY=<kunci>, lihat README.")
    return kunci


API_KEY = _baca_api_key()
_header_api_key = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="API key untuk /api/predict (nilai KICK_API_KEY di server)")


# Halaman dashboard tidak perlu mengetik API key: saat halaman dibuka, server memberi cookie sesi
# bertanda tangan HMAC (HttpOnly, tidak bisa dibaca JavaScript). Klien luar (curl, /docs, aplikasi lain)
# tetap wajib mengirim header X-API-Key.
COOKIE_SESI = "sesi_dashboard"
UMUR_SESI = 12 * 3600                       # detik
_RAHASIA_SESI = secrets.token_bytes(32)     # baru setiap server menyala; sesi lama cukup dengan muat ulang halaman


def _buat_sesi():
    waktu = str(int(time.time()))
    return f"{waktu}.{hmac.new(_RAHASIA_SESI, waktu.encode(), hashlib.sha256).hexdigest()}"


def _sesi_sah(nilai):
    waktu, _, tanda = (nilai or "").partition(".")
    if not waktu.isdigit() or time.time() - int(waktu) > UMUR_SESI:
        return False
    return secrets.compare_digest(tanda, hmac.new(_RAHASIA_SESI, waktu.encode(), hashlib.sha256).hexdigest())


def wajib_api_key(request: Request, kunci: str | None = Depends(_header_api_key)):
    # compare_digest: lama perbandingan tidak bocorkan berapa karakter yang sudah benar
    if kunci and secrets.compare_digest(kunci, API_KEY):
        return
    if not kunci and _sesi_sah(request.cookies.get(COOKIE_SESI)):
        return
    raise HTTPException(401, "API key salah atau belum diisi.", headers={"WWW-Authenticate": "API-Key"})

app = FastAPI(title="Dashboard Sentimen Kick", docs_url="/docs", redoc_url=None)


class Permintaan(BaseModel):
    text: str
    model: str  # "nb" atau "svm"


@app.exception_handler(RequestValidationError)
async def body_tidak_valid(request, exc):
    # 422 dicadangkan untuk "teks kosong setelah preprocessing" (PRD bagian 10)
    return JSONResponse(status_code=400, content={"detail": "Format permintaan tidak valid: perlu text dan model."})


@app.middleware("http")
async def batas_dan_header(request: Request, call_next):
    if request.url.path == "/api/predict":
        panjang = request.headers.get("content-length")
        if panjang and panjang.isdigit() and int(panjang) > MAKS_BODY_BYTE:
            return JSONResponse(status_code=413, content={"detail": "Permintaan terlalu besar."})
    respons = await call_next(request)
    # 304: halaman dari cache browser saat dimuat ulang tetap perlu sesi baru
    if request.url.path in ("/", "/index.html") and respons.status_code in (200, 304):
        respons.set_cookie(COOKIE_SESI, _buat_sesi(), max_age=UMUR_SESI, path="/api", httponly=True,
                           samesite="strict", secure=request.url.scheme == "https")
    respons.headers["X-Content-Type-Options"] = "nosniff"
    respons.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith("/docs"):
        respons.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
    return respons


def _cek_laju(ip):
    sekarang = time.monotonic()
    jejak = _riwayat_ip[ip]
    while jejak and sekarang - jejak[0] > 60:
        jejak.popleft()
    if len(jejak) >= BATAS_LAJU:
        raise HTTPException(429, "Terlalu banyak permintaan. Tunggu sebentar lalu coba lagi.")
    jejak.append(sekarang)


def _logistik(d):
    # bentuk stabil dari 1 / (1 + e^-d) untuk d besar positif maupun negatif
    if d >= 0:
        return 1.0 / (1.0 + math.exp(-d))
    e = math.exp(d)
    return e / (1.0 + e)


@app.get("/api/health")
def health():
    versi_latih = METRICS["versi"]["scikit-learn"]
    return {
        "status": "ok",
        "models": sorted(MODELS),
        "scikit_learn": sklearn.__version__,
        "scikit_learn_saat_pelatihan": versi_latih,
        "versi_cocok": sklearn.__version__ == versi_latih,
    }


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict", dependencies=[Depends(wajib_api_key)])
def predict(r: Permintaan, request: Request):
    _cek_laju(request.client.host if request.client else "-")
    if r.model not in MODELS:
        raise HTTPException(400, "Model harus nb atau svm.")
    if not r.text.strip():
        raise HTTPException(400, "Teks ulasan kosong.")
    if len(r.text) > MAKS_KARAKTER:
        raise HTTPException(400, f"Teks lebih dari {MAKS_KARAKTER} karakter.")

    steps = preprocess_steps(r.text)
    if not steps["stemmed"].strip():
        raise HTTPException(422, "Teks kosong setelah preprocessing (hanya emoji, angka, tautan, atau kata umum).")

    X = tfidf.transform([steps["stemmed"]])
    m = MODELS[r.model]
    label = str(m.predict(X)[0])   # label = prediksi model apa adanya; urutan kelas ['Negatif', 'Positif']
    if r.model == "nb":
        probs = {str(k): float(v) for k, v in zip(m.classes_, m.predict_proba(X)[0])}
        kind = "probability"
    else:
        pos = _logistik(float(m.decision_function(X)[0]))   # skor margin > 0 berarti 'Positif'
        probs = {"Positif": pos, "Negatif": 1.0 - pos}
        kind = "logistic_margin"

    istilah = ANALYZER(steps["stemmed"])
    return {
        "model": r.model,
        "label": label,
        "confidence": probs[label],
        "confidence_kind": kind,
        "probabilities": {"Positif": probs["Positif"], "Negatif": probs["Negatif"]},
        "preprocessing": steps,
        # tambahan di luar kontrak PRD: berapa kata hasil preprocessing yang dikenal TF-IDF
        "vocabulary": {"known": sum(t in tfidf.vocabulary_ for t in istilah), "total": len(istilah)},
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")  # harus paling akhir

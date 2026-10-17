# -*- coding: utf-8 -*-
"""Preprocessing teks ulasan Kick, versi v5.

Bagian "INTI" di bawah adalah salinan persis kode Bagian 5 notebook
analisis_sentimen_kick_nb_svm_v5_colab.ipynb. Jangan mengubahnya di satu tempat saja:
model di artifacts/ dilatih dengan kode ini, jadi web dan notebook harus identik.
"""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def _pastikan_nltk():
    for path, nama in [("tokenizers/punkt_tab", "punkt_tab"),
                       ("tokenizers/punkt", "punkt"),
                       ("corpora/stopwords", "stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(nama, quiet=True)


_pastikan_nltk()

# ===================== INTI (identik dengan notebook v5, Bagian 5) =====================
def remove_emoji(text):
    if not isinstance(text, str):
        return text
    # Menghapus emoji dan karakter non-ASCII lainnya
    return text.encode('ascii', 'ignore').decode('ascii')

SLANG = {
    # negasi
    'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak','enggak':'tidak','engga':'tidak','gk':'tidak',
    'tdk':'tidak','tak':'tidak','bkn':'bukan','blm':'belum','belom':'belum','blom':'belum','jgn':'jangan','gausah':'tidak usah',
    'gabisa':'tidak bisa','gabsa':'tidak bisa','gbs':'tidak bisa','gaada':'tidak ada','gada':'tidak ada',
    'kaga':'tidak','kagak':'tidak','gajelas':'tidak jelas',
    # negasi bahasa Inggris (bentuk tanpa apostrof)
    'dont':'not','cant':'not','cannot':'not','doesnt':'not','didnt':'not','isnt':'not','wont':'not',
    # penghubung / kata fungsi
    'yg':'yang','dgn':'dengan','dg':'dengan','utk':'untuk','untk':'untuk','krn':'karena','karna':'karena','tp':'tapi',
    'tpi':'tapi','jd':'jadi','sm':'sama','dr':'dari','dlm':'dalam','pd':'pada','trs':'terus','lg':'lagi','lgi':'lagi',
    'sdh':'sudah','udh':'sudah','udah':'sudah','dah':'sudah','emg':'memang','emang':'memang','kyk':'seperti','kayak':'seperti',
    'gmn':'bagaimana','gimana':'bagaimana','knp':'kenapa','org':'orang','aja':'saja','aj':'saja','doang':'saja',
    'bs':'bisa','bgt':'sangat','banget':'sangat','sgt':'sangat','sy':'saya','gw':'saya','gue':'saya','gua':'saya','lu':'kamu',
    'gini':'begini','gitu':'begitu','kalo':'kalau','klo':'kalau','cuman':'cuma','pake':'pakai','mulu':'terus',
    # istilah domain aplikasi streaming
    'apk':'aplikasi','app':'aplikasi','apps':'aplikasi','aplikasinya':'aplikasi','hp':'ponsel',
    'lemot':'lambat','lag':'lambat','ngelag':'lambat','lagging':'lambat','lelet':'lambat','loadingnya':'loading',
    'buffering':'buffer','ngebuffer':'buffer','eror':'error','erorr':'error','ngebug':'bug','bugs':'bug',
    'nntn':'tonton','nonton':'tonton','liat':'lihat','streamingnya':'streaming','stream':'streaming',
    'mendownload':'download','didownload':'download','ngedownload':'download',
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

def normalisasi_huruf_berulang(text):
    # 'bagusss' -> 'bagus', 'kecewaaa' -> 'kecewa'; huruf ganda seperti 'maaf' tidak berubah
    return re.sub(r'(.)\1{2,}', r'\1', text)

def normalisasi_negasi_inggris(text):
    # "don't" menjadi "don t" setelah cleaning; dikembalikan ke "not" agar negasi tidak hilang
    return re.sub(r'\b(?:can|don|doesn|didn|isn|aren|wasn|weren|won|couldn|shouldn|wouldn|haven|hasn)\s+t\b', 'not', text)

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

# Partikel dan sapaan informal yang tidak membawa sentimen
PARTIKEL_INFORMAL = {'ya', 'yah', 'sih', 'nih', 'ni', 'loh', 'lho', 'dong', 'deh', 'kok', 'si', 'ku', 'mu',
                     'kak', 'bang', 'mas', 'gan', 'bro', 'sis', 'min'}
list_stopwords |= PARTIKEL_INFORMAL

# Banyak ulasan berbahasa Inggris: stopword Inggris ikut dibuang, kecuali kata negasinya
KATA_NEGASI_INGGRIS = {'not', 'no', 'nor'}
list_stopwords |= set(stopwords.words('english'))

# Kata negasi DIPERTAHANKAN: stopword NLTK Indonesia memuat kata-kata ini, padahal ia membalik polaritas
# sentimen (mis. "tidak bagus" akan menjadi "bagus" jika 'tidak' dibuang).
KATA_NEGASI = {'tidak', 'tak', 'bukan', 'belum', 'jangan', 'tanpa', 'kurang'}
# Kata bermuatan makna yang ada di stopword NLTK Indonesia juga dipertahankan
# (mis. "tidak bisa login" akan menjadi "tidak login" jika 'bisa' dibuang).
KATA_BERMAKNA = {'bisa', 'jelas', 'baik', 'lama', 'sering', 'masalah', 'benar', 'lebih', 'terlalu'}
list_stopwords -= KATA_NEGASI | KATA_NEGASI_INGGRIS | KATA_BERMAKNA

def stopword_removal(tokens):
    # token satu huruf (sisa singkatan seperti 's', 't') ikut dibuang
    return [word for word in tokens if word not in list_stopwords and len(word) > 1]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

@lru_cache(maxsize=None)
def stem_kata(word):
    hasil = stemmer.stem(word)
    # Sufiks -nya pada kata serapan yang tidak dikenali kamus Sastrawi (mis. 'servernya' -> 'server')
    if hasil == word and word.endswith('nya') and len(word) > 5:
        hasil = SLANG.get(word[:-3], word[:-3])
    return hasil

def stemming(tokens):
    return [stem_kata(word) for word in tokens]

def preprocess_text(text):
    text = remove_emoji(text)
    text = cleaning_text(text)
    text = case_folding(text)
    text = normalisasi_huruf_berulang(text)
    text = normalisasi_negasi_inggris(text)
    tokens = tokenizing(text)
    tokens = slang_normalization(tokens)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)

# =================== AKHIR INTI ===================


def preprocess_steps(text):
    """Hasil tiap tahap, sesuai kontrak /api/predict (PRD bagian 10)."""
    original = "" if text is None else str(text)
    t = case_folding(cleaning_text(remove_emoji(original)))
    cleaned = normalisasi_negasi_inggris(normalisasi_huruf_berulang(t))
    tokens = tokenizing(cleaned)
    replacements, seen = [], set()
    for w in tokens:
        if w in SLANG and SLANG[w] != w and w not in seen:
            replacements.append({"from": w, "to": SLANG[w]})
            seen.add(w)
    slang = slang_normalization(tokens)
    after_stop = stopword_removal(slang)
    stemmed = stemming(after_stop)
    negations = []
    for w in after_stop:
        if w in KATA_NEGASI | KATA_NEGASI_INGGRIS and w not in negations:
            negations.append(w)
    return {
        "original": original,
        "cleaned": cleaned,
        "slang_normalized": " ".join(slang),
        "after_stopword": " ".join(after_stop),
        "stemmed": " ".join(stemmed),
        "replacements": replacements,
        "negations_kept": negations,
    }

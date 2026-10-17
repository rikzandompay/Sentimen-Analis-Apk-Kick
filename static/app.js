"use strict";

// Semua angka hasil eksperimen dibaca dari /api/metrics (artifacts/metrics.json); tidak ada yang ditulis di sini.
// Teks dari pengguna dan dari server selalu dimasukkan lewat textContent / text node, tidak pernah innerHTML.

const NAMA_MODEL = { nb: "Multinomial NB", svm: "Linear SVM" };
const KUNCI_ABLASI = { "Naïve Bayes": "nb", "SVM (Linear)": "svm" };
const KELAS = [["Positif", "pos"], ["Negatif", "neg"]];
const PRESET = {
  positif: "Streaming mantap lancar banget",
  negatif: "Aplikasi lemot sering buffer dan logout sendiri",
};
const MAKS_KARAKTER = 500;
const AMBANG_RAGU = 0.6;
const LANGKAH_CM = 6;   // ramp confusion matrix: kelas .cm-t0 sampai .cm-t6 (warna per tema di style.css)
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
const namaModel = (k) => NAMA_MODEL[k] ?? k;

/* ---------- Format angka dan tanggal (id-ID) ---------- */

const angka = (x, d = 0) => x.toLocaleString("id-ID", { minimumFractionDigits: d, maximumFractionDigits: d });
const persen = (x, d = 2) => `${angka(x * 100, d)}%`;
const parameter = (x) => x.toLocaleString("id-ID", { maximumFractionDigits: 6 });
const nilaiP = (p) => (p < 0.0001 ? "< 0,0001" : angka(p, 4));
const jumlah = (obj) => Object.values(obj).reduce((a, b) => a + b, 0);
const bulat2 = (x) => Math.round(x * 10000);   // pembanding pada presisi yang ditampilkan (2 desimal persen)

function tanggal(s) {
  const [y, m, d] = String(s).slice(0, 10).split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d));
  if (Number.isNaN(t.getTime())) return String(s);
  return new Intl.DateTimeFormat("id-ID", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }).format(t);
}

/* ---------- Pembuat elemen ---------- */

function el(tag, attrs = {}, ...anak) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  for (const a of anak) if (a != null) e.append(a);
  return e;
}

function svg(tag, attrs = {}, ...anak) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const a of anak) if (a != null) e.append(a);
  return e;
}

const teksSvg = (attrs, isi) => svg("text", attrs, isi);

/* ---------- Tema terang / gelap ---------- */

const tombolTema = $("tombol-tema");
const mediaGelap = window.matchMedia("(prefers-color-scheme: dark)");

function temaAktif() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t === "dark" || t === "light") return t;
  return mediaGelap.matches ? "dark" : "light";
}

function perbaruiTombolTema() {
  tombolTema.textContent = temaAktif() === "dark" ? "Mode terang" : "Mode gelap";
}

tombolTema.addEventListener("click", () => {
  const baru = temaAktif() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", baru);
  try { localStorage.setItem("tema", baru); } catch { /* penyimpanan diblokir: tema tetap berlaku untuk sesi ini */ }
  perbaruiTombolTema();
});
mediaGelap.addEventListener("change", perbaruiTombolTema);
perbaruiTombolTema();

/* ---------- Tooltip (hover dan fokus keyboard) ---------- */

const tooltip = $("tooltip");

function tampilTip(target) {
  const teks = target.getAttribute("data-tip");
  if (!teks) return;
  tooltip.textContent = teks;
  tooltip.hidden = false;
  const r = target.getBoundingClientRect();
  const t = tooltip.getBoundingClientRect();
  const x = Math.max(8, Math.min(r.left + r.width / 2 - t.width / 2, window.innerWidth - t.width - 8));
  let y = r.top - t.height - 8;
  if (y < 8) y = r.bottom + 8;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

const sembunyiTip = () => { tooltip.hidden = true; };
const targetTip = (ev) => (ev.target instanceof Element ? ev.target.closest("[data-tip]") : null);

document.addEventListener("mouseover", (ev) => { const t = targetTip(ev); if (t) tampilTip(t); });
document.addEventListener("mouseout", (ev) => {
  const t = targetTip(ev);
  if (t && !(ev.relatedTarget instanceof Node && t.contains(ev.relatedTarget))) sembunyiTip();
});
document.addEventListener("focusin", (ev) => { const t = targetTip(ev); if (t) tampilTip(t); });
document.addEventListener("focusout", sembunyiTip);
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") sembunyiTip(); });
window.addEventListener("scroll", sembunyiTip, { passive: true });

/* ---------- Banner status server ---------- */

let ulangBanner = null;

function banner(teks, ulang = null) {
  $("banner").hidden = !teks;
  $("banner-teks").textContent = teks || "";
  ulangBanner = ulang;
  $("banner-ulang").hidden = !ulang;
}

$("banner-ulang").addEventListener("click", () => { if (ulangBanner) ulangBanner(); });

async function ambilJSON(url, batasMs) {
  const res = await fetch(url, { signal: AbortSignal.timeout(batasMs) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ---------- Header dan ringkasan KPI (PRD bagian 5) ---------- */

function modelTerbaik(m) {
  let terbaik = null;
  for (const [k, v] of Object.entries(m.model)) {
    if (!terbaik || v.nested_cv.akurasi_mean > m.model[terbaik].nested_cv.akurasi_mean) terbaik = k;
  }
  return terbaik;
}

function statusMcNemar(mc, ringkas = false) {
  const alfa = angka(mc.alpha, 2);
  const teks = ringkas
    ? `McNemar p = ${nilaiP(mc.p_value)}, ${mc.signifikan ? "signifikan" : "tidak signifikan"} pada α = ${alfa}`
    : `${mc.signifikan ? "Signifikan" : "Tidak signifikan"} pada α = ${alfa}`;
  return el("span", { class: `status ${mc.signifikan ? "signifikan" : "tidak-signifikan"}` },
    el("span", { class: "status-ikon", "aria-hidden": "true" }, mc.signifikan ? "✓" : "="), teks);
}

function isiKpi(m, terbaik) {
  const d = m.data;
  $("kpi-berteks").textContent = angka(d.berteks);
  $("kpi-bersih").textContent = `ulasan berteks, ${angka(d.akhir)} bersih untuk analisis`;
  // batas_ulasan (dan tanggal_scraping) bisa null: bagian yang kosong tidak ditampilkan
  const bagianTanggal = [];
  if (d.batas_ulasan) bagianTanggal.push(`ulasan hingga ${tanggal(d.batas_ulasan)}`);
  if (d.tanggal_scraping) bagianTanggal.push(`diambil ${tanggal(d.tanggal_scraping)}`);
  const ketTanggal = bagianTanggal.join(", ");
  $("kpi-tanggal").textContent = ketTanggal && ketTanggal[0].toUpperCase() + ketTanggal.slice(1);
  $("kpi-tanggal").hidden = !ketTanggal;
  const tip = `${angka(d.berteks)} adalah ulasan berteks yang berhasil diambil saat scraping`
    + (d.batas_ulasan ? " (setelah dipotong pada batas tanggal ulasan)" : "")
    + `, bukan seluruh ulasan aplikasi di Play Store. Setelah duplikat, ulasan netral, dan teks kosong dibuang, tersisa ${angka(d.akhir)} ulasan.`;
  $("info-dataset").setAttribute("data-tip", tip);
  $("info-dataset").setAttribute("aria-label", tip);

  const latih = jumlah(d.latih);
  const uji = jumlah(d.uji);
  const pLatih = Math.round((latih / (latih + uji)) * 100);
  $("kpi-split").textContent = `${angka(latih)} / ${angka(uji)}`;
  $("kpi-rasio").textContent = `Rasio ${pLatih}:${100 - pLatih}, stratified`;

  const kunci = Object.keys(m.model);
  $("kpi-algoritma").textContent = `${kunci.length} Model`;
  $("kpi-algoritma-nama").textContent = kunci.map(namaModel).join(" dan ");

  $("kpi-terbaik-nilai").textContent = persen(m.model[terbaik].nested_cv.akurasi_mean);
  $("kpi-terbaik-model").textContent = namaModel(terbaik);
  $("kpi-terbaik-mcnemar").replaceChildren(statusMcNemar(m.mcnemar, true));
}

/* ---------- Bagian 1: distribusi kelas dan resampling (PRD bagian 6) ---------- */

function langkahRapi(x) {
  const pangkat = 10 ** Math.floor(Math.log10(x));
  const sisa = x / pangkat;
  return (sisa <= 1 ? 1 : sisa <= 2 ? 2 : sisa <= 5 ? 5 : 10) * pangkat;
}

function kolomBulat(x, yAtas, lebar, tinggi, r) {
  const rr = Math.min(r, tinggi, lebar / 2);
  const yBawah = yAtas + tinggi;
  return `M${x},${yBawah} V${yAtas + rr} Q${x},${yAtas} ${x + rr},${yAtas} H${x + lebar - rr} `
    + `Q${x + lebar},${yAtas} ${x + lebar},${yAtas + rr} V${yBawah} Z`;
}

function grafikDistribusi(dist, total) {
  const W = 300, H = 260, kiri = 44, kanan = 8, atas = 50, bawah = 32;
  const tinggiPlot = H - atas - bawah;
  const maks = Math.max(...KELAS.map(([k]) => dist[k] ?? 0));
  const langkah = langkahRapi(maks / 4);
  const puncak = Math.ceil(maks / langkah) * langkah;
  const y = (v) => atas + tinggiPlot - (v / puncak) * tinggiPlot;

  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Grafik batang distribusi kelas sentimen" });
  for (let v = 0; v <= puncak; v += langkah) {
    s.append(svg("line", { class: v === 0 ? "dasar" : "grid", x1: kiri, x2: W - kanan, y1: y(v), y2: y(v) }));
    s.append(teksSvg({ class: "tick", x: kiri - 8, y: y(v) + 4, "text-anchor": "end" }, angka(v)));
  }
  const pita = (W - kiri - kanan) / KELAS.length;
  KELAS.forEach(([k, cls], i) => {
    const v = dist[k] ?? 0;
    const cx = kiri + pita * (i + 0.5);
    const tip = `${k}: ${angka(v)} ulasan (${persen(v / total, 1)} dari ${angka(total)})`;
    s.append(svg("g", { class: "kolom", tabindex: 0, role: "img", "data-tip": tip, "aria-label": tip },
      svg("rect", { class: "area-sentuh", x: cx - pita / 2 + 10, y: atas - 46, width: pita - 20, height: tinggiPlot + 46, rx: 4 }),
      svg("path", { class: `batang isi-${cls}`, d: kolomBulat(cx - 12, y(v), 24, y(0) - y(v), 4) }),
      teksSvg({ class: "nilai", x: cx, y: y(v) - 24, "text-anchor": "middle" }, angka(v)),
      teksSvg({ class: "nilai-persen", x: cx, y: y(v) - 8, "text-anchor": "middle" }, persen(v / total, 1)),
      teksSvg({ class: "label-x", x: cx, y: H - 8, "text-anchor": "middle" }, k)));
  });
  return s;
}

function baris(dt, dd, kelas = null) {
  const label = kelas ? [el("span", { class: `swatch swatch-${kelas}`, "aria-hidden": "true" }), dt] : [dt];
  return el("div", {}, el("dt", {}, ...label), el("dd", {}, dd));
}

function isiDistribusi(m) {
  const d = m.data;
  const dist = d.distribusi_akhir;
  const total = jumlah(dist);
  $("cap-gambar-1").textContent = `Distribusi kelas sentimen setelah pembersihan (n = ${angka(total)}).`;
  $("grafik-distribusi").replaceChildren(grafikDistribusi(dist, total));
  const daftar = [
    baris("Positif", angka(dist.Positif ?? 0), "pos"),
    baris("Negatif", angka(dist.Negatif ?? 0), "neg"),
    baris("Total", angka(total)),
  ];
  if (dist.Positif && dist.Negatif) daftar.push(baris("Rasio pos : neg", `${angka(dist.Positif / dist.Negatif, 2)} : 1`));
  if (d.distribusi_label?.Netral != null) daftar.push(baris("Netral dikeluarkan", angka(d.distribusi_label.Netral)));
  $("ringkas-distribusi").replaceChildren(...daftar);
}

function isiResampling(m) {
  const sebelum = m.data.latih;
  const sesudah = m.data.latih_setelah_ros;
  const nilaiSesudah = Object.values(sesudah);
  const rasio = Math.min(...nilaiSesudah) / Math.max(...nilaiSesudah);
  $("cap-gambar-2").textContent = `Data latih sebelum dan sesudah RandomOverSampler (rasio minoritas : mayoritas ${angka(rasio, 2)}).`;
  $("cap-gambar-2-uji").textContent = ` (${angka(m.data.uji.Positif ?? 0)} positif, ${angka(m.data.uji.Negatif ?? 0)} negatif)`;

  const maks = Math.max(jumlah(sebelum), jumlah(sesudah));
  const barisTabel = [["Sebelum ROS", sebelum, null], ["Sesudah ROS", sesudah, sebelum]].map(([nama, data, asal]) => {
    const total = jumlah(data);
    const batang = el("div", { class: "ros-batang" });
    batang.style.width = `${(total / maks) * 100}%`;
    const sel = [];
    for (const [k, cls] of KELAS) {
      const v = data[k] ?? 0;
      const tambah = asal ? v - (asal[k] ?? 0) : 0;
      let tip = `${k}, ${nama.toLowerCase()}: ${angka(v)}`;
      if (tambah > 0) tip += ` (${angka(asal[k])} asli + ${angka(tambah)} salinan acak)`;
      const seg = el("div", { class: `ros-segmen ${cls}`, tabindex: 0, role: "img", "data-tip": tip, "aria-label": tip });
      seg.style.flex = `${v} 0 0`;
      batang.append(seg);
      sel.push(el("td", {}, angka(v), tambah > 0 ? el("span", { class: "ros-tambah" }, `+${angka(tambah)} salinan`) : null));
    }
    return el("tr", {}, el("th", { scope: "row" }, nama), ...sel, el("td", {}, angka(total)), el("td", {}, batang));
  });
  document.querySelector("#tabel-ros tbody").replaceChildren(...barisTabel);
}

/* ---------- Bagian 2: benchmark dan confusion matrix (PRD bagian 7) ---------- */

function hyperparameter(x) {
  if ("alpha" in x) return `α = ${parameter(x.alpha)}`;
  if ("C" in x) return `C = ${parameter(x.C)}`;
  return "–";
}

function teksMcNemar(mc) {
  return mc.signifikan
    ? `selisih signifikan pada α = ${angka(mc.alpha, 2)} (McNemar p = ${nilaiP(mc.p_value)})`
    : `selisih tidak signifikan pada α = ${angka(mc.alpha, 2)} (McNemar p = ${nilaiP(mc.p_value)})`;
}

function isiBenchmark(m, terbaik) {
  const nUji = jumlah(m.data.uji);
  $("kepala-single").textContent = `Single split (${angka(nUji)} data uji)`;
  $("cap-tabel-1").textContent = `Akurasi dan Macro-F1 pada single split dan nested 5-fold CV; pada nested CV, penalaan hyperparameter diulang di setiap fold luar.`;
  const kunci = Object.keys(m.model);
  const kolom = [
    { ambil: (x) => x.single_split.akurasi },
    { ambil: (x) => x.single_split.macro_f1 },
    { ambil: (x) => x.nested_cv.akurasi_mean, sd: (x) => x.nested_cv.akurasi_std },
    { ambil: (x) => x.nested_cv.macro_f1_mean },
  ];
  const tertinggi = kolom.map((k) => Math.max(...kunci.map((n) => bulat2(k.ambil(m.model[n])))));
  const tip = `Akurasi nested 5-fold CV tertinggi; ${teksMcNemar(m.mcnemar)}.`;
  const barisTabel = kunci.map((n) => {
    const x = m.model[n];
    const nama = el("span", { class: "nama-model" }, namaModel(n));
    if (n === terbaik) nama.append(el("span", { class: "lencana", tabindex: 0, "data-tip": tip, "aria-label": `BEST MODEL: ${tip}` }, "BEST MODEL"));
    const tr = el("tr", { class: n === terbaik ? "terbaik" : "" },
      el("th", { scope: "row" }, nama),
      el("td", { class: "kiri" }, hyperparameter(x)));
    kolom.forEach((k, i) => {
      const v = k.ambil(x);
      const td = el("td", { class: bulat2(v) === tertinggi[i] ? "tebal" : "" }, persen(v));
      if (k.sd) td.append(" ", el("span", { class: "sd" }, `± ${angka(k.sd(x) * 100, 2)}`));
      tr.append(td);
    });
    return tr;
  });
  document.querySelector("#tabel-benchmark tbody").replaceChildren(...barisTabel);

  const lain = kunci.filter((n) => n !== terbaik).map(namaModel).join(" dan ");
  $("catatan-tabel-1").textContent = `BEST MODEL: akurasi nested CV tertinggi; terhadap ${lain}, ${teksMcNemar(m.mcnemar)}. `
    + "SD memakai pembagi N (np.std, ddof=0). Penalaan hyperparameter memakai akurasi, hanya pada data latih.";
}

const ARTI_CM = {
  tn: ["TN", "aktual Negatif, diprediksi Negatif"],
  fp: ["FP", "aktual Negatif, diprediksi Positif"],
  fn: ["FN", "aktual Positif, diprediksi Negatif"],
  tp: ["TP", "aktual Positif, diprediksi Positif"],
};

function isiConfusionMatrix(m) {
  const kunci = Object.keys(m.model);
  $("cap-gambar-3").textContent = `Confusion matrix kedua model pada ${angka(jumlah(m.data.uji))} data uji.`;
  const maks = Math.max(...kunci.flatMap((n) => Object.values(m.model[n].single_split.cm)));
  const sel = (n, cm, kode) => {
    const v = cm[kode];
    const [singkat, arti] = ARTI_CM[kode];
    const tip = `${namaModel(n)} · ${singkat}: ${angka(v)} ulasan (${arti})`;
    return el("div", { class: `cm-sel cm-t${Math.round((v / maks) * LANGKAH_CM)}`, role: "cell", tabindex: 0, "data-tip": tip, "aria-label": tip },
      el("span", { class: "cm-angka" }, angka(v)), el("span", { class: "cm-kode" }, singkat));
  };
  const kepala = (teks) => el("div", { class: "cm-kepala", role: "columnheader" }, teks);
  const kepalaBaris = (teks) => el("div", { class: "cm-kepala baris-kepala", role: "rowheader" }, teks);

  $("grid-cm").replaceChildren(...kunci.map((n) => {
    const cm = m.model[n].single_split.cm;
    const benar = cm.tn + cm.tp;
    const total = cm.tn + cm.fp + cm.fn + cm.tp;
    return el("div", {},
      el("p", { class: "cm-judul" }, namaModel(n)),
      el("div", { class: "cm", role: "table", "aria-label": `Confusion matrix ${namaModel(n)}` },
        el("div", { class: "cm-row", role: "row" }, kepala(""), kepala("Prediksi Negatif"), kepala("Prediksi Positif")),
        el("div", { class: "cm-row", role: "row" }, kepalaBaris("Aktual Negatif"), sel(n, cm, "tn"), sel(n, cm, "fp")),
        el("div", { class: "cm-row", role: "row" }, kepalaBaris("Aktual Positif"), sel(n, cm, "fn"), sel(n, cm, "tp"))),
      el("p", { class: "cm-benar" }, `Benar ${angka(benar)} dari ${angka(total)} (${persen(benar / total)})`));
  }));
}

/* ---------- Bagian 3: ablasi dan McNemar ---------- */

function isiAblasi(m) {
  const kelompok = new Map();
  for (const r of m.ablasi) {
    if (!kelompok.has(r.Model)) kelompok.set(r.Model, {});
    kelompok.get(r.Model)[r.Oversampling] = r;
  }
  const barisTabel = [];
  for (const [nama, k] of kelompok) {
    const label = namaModel(KUNCI_ABLASI[nama] ?? nama);
    const tanpa = k["Tanpa ROS"];
    const dengan = k["Dengan ROS"];
    const ada = [tanpa, dengan].filter(Boolean);
    const span = ada.length + (tanpa && dengan ? 1 : 0);
    ada.forEach((r, i) => {
      const tr = el("tr", { class: i === 0 ? "awal-grup" : "" });
      if (i === 0) tr.append(el("th", { scope: "rowgroup", rowspan: span }, label));
      tr.append(el("td", { class: "kiri" }, r.Oversampling),
        el("td", {}, persen(r["Akurasi (CV)"])), el("td", {}, persen(r["Macro-F1 (CV)"])));
      barisTabel.push(tr);
    });
    if (tanpa && dengan) {
      const sel = (selisih) => el("td", { class: selisih >= 0 ? "naik" : "turun" },
        `${selisih >= 0 ? "▲ +" : "▼ −"}${angka(Math.abs(selisih * 100), 2)} pp`);
      barisTabel.push(el("tr", { class: "selisih" }, el("td", { class: "kiri" }, "Selisih"),
        sel(dengan["Akurasi (CV)"] - tanpa["Akurasi (CV)"]), sel(dengan["Macro-F1 (CV)"] - tanpa["Macro-F1 (CV)"])));
    }
  }
  document.querySelector("#tabel-ablasi tbody").replaceChildren(...barisTabel);
  const hp = Object.values(m.model).map(hyperparameter).join(" dan ");
  $("ket-ablasi").textContent = `Selisih = dengan ROS − tanpa ROS, dalam poin persentase (pp). Memakai hyperparameter hasil penalaan `
    + `pada data latih (${hp}) dan fold yang sama untuk kedua kondisi. Estimasi tanpa kebocoran ada di Tabel 1 (nested CV).`;
}

function isiMcNemar(m) {
  const mc = m.mcnemar;
  const nUji = jumlah(m.data.uji);
  $("ket-mcnemar").textContent = `Prediksi NB dan SVM dibandingkan pada ${angka(nUji)} data uji yang sama (single split).`;
  $("mcnemar-p").textContent = nilaiP(mc.p_value);
  $("mcnemar-status").replaceChildren(statusMcNemar(mc));
  $("mcnemar-b").textContent = angka(mc.b);
  $("mcnemar-c").textContent = angka(mc.c);
  $("mcnemar-tafsir").textContent = mc.signifikan
    ? `Selisih akurasi kedua model signifikan pada α = ${angka(mc.alpha, 2)} untuk data uji ini.`
    : `Selisih akurasi NB dan SVM belum cukup besar untuk dinyatakan berbeda. Data uji hanya ${angka(nUji)} ulasan, sehingga daya uji terbatas.`;
}

function isiKolofon(m) {
  const v = m.versi;
  $("kolofon").textContent = `Semua angka dibaca dari artifacts/metrics.json · Python ${v.python} · scikit-learn ${v["scikit-learn"]} · `
    + `imbalanced-learn ${v["imbalanced-learn"]}${v.preprocessing ? ` · preprocessing ${v.preprocessing}` : ""} · `
    + "Huruf: Source Serif 4 dan Source Sans 3 (SIL Open Font License)";
}

function render(m) {
  const terbaik = modelTerbaik(m);
  isiKpi(m, terbaik);
  isiDistribusi(m);
  isiResampling(m);
  isiBenchmark(m, terbaik);
  isiConfusionMatrix(m);
  isiAblasi(m);
  isiMcNemar(m);
  isiKolofon(m);
}

async function muatMetrics() {
  const lambat = setTimeout(() => banner("Server sedang menyala… hosting gratis bisa butuh beberapa puluh detik."), 1500);
  try {
    const m = await ambilJSON("/api/metrics", 90000);
    clearTimeout(lambat);
    banner(null);
    try {
      render(m);
    } catch (e) {
      banner(`Isi metrics.json tidak sesuai struktur yang diharapkan (${e.message}).`);
      return;
    }
    cekVersi();
  } catch {
    clearTimeout(lambat);
    banner("Data hasil penelitian gagal dimuat. Pastikan server berjalan.", muatMetrics);
  }
}

async function cekVersi() {
  try {
    const h = await ambilJSON("/api/health", 10000);
    if (!h.versi_cocok) {
      banner(`Peringatan: scikit-learn di server (${h.scikit_learn}) berbeda dari saat pelatihan (${h.scikit_learn_saat_pelatihan}); hasil prediksi bisa berbeda dari paper.`);
    }
  } catch { /* status kesehatan hanya pelengkap */ }
}

/* ---------- Bagian 4: playground (PRD bagian 8) ---------- */

const form = $("form-prediksi");
const teks = $("teks-ulasan");
const penghitung = $("penghitung");
const tPrediksi = $("tombol-prediksi");
const tBandingkan = $("tombol-bandingkan");
const wadahHasil = $("hasil");
const kosongAwal = $("hasil-kosong");
let sibuk = false;
let aksiTerakhir = null;
let tampilTunggal = false;   // hasil satu model sedang ditampilkan (untuk ganti model otomatis)

const modelTerpilih = () => form.querySelector("input[name=model]:checked").value;

function perbaruiForm() {
  const n = teks.value.length;
  penghitung.textContent = `${angka(n)} / ${angka(MAKS_KARAKTER)}`;
  penghitung.classList.toggle("dekat", n >= MAKS_KARAKTER * 0.9 && n < MAKS_KARAKTER);
  penghitung.classList.toggle("penuh", n >= MAKS_KARAKTER);
  const kosong = !teks.value.trim();
  tPrediksi.disabled = kosong || sibuk;
  tBandingkan.disabled = kosong || sibuk;
}

teks.addEventListener("input", perbaruiForm);
teks.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey) && !tPrediksi.disabled) {
    ev.preventDefault();
    form.requestSubmit();
  }
});

for (const tombol of document.querySelectorAll("[data-preset]")) {
  tombol.addEventListener("click", () => {
    teks.value = PRESET[tombol.dataset.preset];
    perbaruiForm();
    teks.focus();
  });
}

$("tombol-bersihkan").addEventListener("click", () => {
  teks.value = "";
  aksiTerakhir = null;
  tampilTunggal = false;
  perbaruiForm();
  wadahHasil.replaceChildren(kosongAwal);
  teks.focus();
});

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  jalankan([modelTerpilih()]);
});
tBandingkan.addEventListener("click", () => jalankan(["nb", "svm"]));
for (const pilihan of form.querySelectorAll("input[name=model]")) {
  pilihan.addEventListener("change", () => { if (tampilTunggal && teks.value.trim()) jalankan([modelTerpilih()]); });
}

class GalatPrediksi extends Error {
  constructor(jenis, detail = "") {
    super(jenis);
    this.jenis = jenis;
    this.detail = typeof detail === "string" ? detail : "";
  }
}

// Halaman tidak mengirim API key: server memberi cookie sesi (HttpOnly) saat halaman dibuka,
// dan fetch ke origin yang sama membawanya otomatis. API key hanya untuk klien luar.
async function prediksi(text, model) {
  let res;
  try {
    res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
      signal: AbortSignal.timeout(60000),
    });
  } catch {
    throw new GalatPrediksi("server");
  }
  if (res.ok) return res.json();
  let detail = "";
  try { detail = (await res.json()).detail; } catch { /* badan respons bukan JSON */ }
  if (res.status === 401) throw new GalatPrediksi("kunci", detail);
  if (res.status === 422) throw new GalatPrediksi("kosong", detail);
  if (res.status === 429) throw new GalatPrediksi("laju", detail);
  if (res.status === 400 || res.status === 413) throw new GalatPrediksi("masukan", detail);
  throw new GalatPrediksi("server", detail);
}

async function jalankan(daftarModel) {
  const isi = teks.value;
  if (!isi.trim() || sibuk) return;
  aksiTerakhir = daftarModel;
  sibuk = true;
  perbaruiForm();
  tPrediksi.textContent = "Memproses…";
  wadahHasil.setAttribute("aria-busy", "true");
  const tunda = setTimeout(() => wadahHasil.replaceChildren(el("p", { class: "memuat" }, "Memproses ulasan…")), 300);
  try {
    const jawaban = await Promise.all(daftarModel.map((m) => prediksi(isi, m)));
    clearTimeout(tunda);
    tampilHasil(jawaban);
    tampilTunggal = jawaban.length === 1;
  } catch (e) {
    clearTimeout(tunda);
    tampilTunggal = false;
    tampilGalat(e instanceof GalatPrediksi ? e : new GalatPrediksi("server"));
  } finally {
    sibuk = false;
    tPrediksi.textContent = "Prediksi";
    wadahHasil.removeAttribute("aria-busy");
    perbaruiForm();
  }
}

function tampilGalat(e) {
  const pesan = {
    kunci: ["Sesi halaman sudah berakhir", "Server menolak permintaan (HTTP 401). Muat ulang halaman untuk mendapatkan sesi baru."],
    kosong: ["Tulis ulasan dengan kata-kata", e.detail || "Teks hanya berisi emoji, angka, atau tautan."],
    laju: ["Terlalu banyak permintaan", "Tunggu sebentar lalu coba lagi."],
    masukan: ["Ulasan tidak dapat diproses", e.detail || "Periksa kembali isi ulasan."],
    server: ["Server tidak dapat dihubungi", "Server mungkin sedang menyala (hosting gratis bisa butuh beberapa puluh detik). Coba lagi sebentar lagi."],
  }[e.jenis] || ["Terjadi galat", e.detail];
  const kotak = el("div", { class: "galat", role: "alert" },
    el("p", { class: "galat-judul" }, pesan[0]),
    el("p", { class: "galat-detail" }, pesan[1]));
  if (e.jenis === "server" || e.jenis === "laju") {
    const ulang = el("button", { type: "button", class: "tombol kecil" }, "Coba lagi");
    ulang.addEventListener("click", () => { if (aksiTerakhir) jalankan(aksiTerakhir); });
    kotak.append(ulang);
  }
  wadahHasil.replaceChildren(kotak);
}

/* ---------- Hasil prediksi (PRD bagian 9) ---------- */

function peringatan(teksPeringatan) {
  return el("p", { class: "peringatan" }, el("span", { class: "peringatan-ikon", "aria-hidden": "true" }, "!"), teksPeringatan);
}

function blokModel(j) {
  const positif = j.label === "Positif";
  const blok = el("section", { class: "hasil-model", "aria-label": `Hasil ${namaModel(j.model)}` });
  blok.append(el("p", { class: "hasil-model-nama" }, namaModel(j.model)));

  const keyakinan = el("div", { class: "keyakinan" },
    el("span", { class: "keyakinan-nilai" }, persen(j.confidence, 1)),
    el("span", { class: "keyakinan-jenis" },
      j.confidence_kind === "probability" ? "peluang kelas (predict_proba)" : "perkiraan keyakinan, bukan probabilitas terkalibrasi"));
  if (j.confidence_kind === "logistic_margin") {
    const p = j.probabilities.Positif;
    const d = Math.log(p / (1 - p));
    if (Number.isFinite(d)) keyakinan.append(el("span", { class: "keyakinan-margin" }, `fungsi logistik dari skor margin d = ${angka(d, 2)}`));
  }
  blok.append(el("div", { class: "hasil-atas" },
    el("span", { class: `polaritas ${positif ? "pos" : "neg"}` }, positif ? "POSITIF" : "NEGATIF"), keyakinan));

  const prob = el("div", { class: "prob", role: "list", "aria-label": "Skor per kelas" });
  for (const [k, cls] of KELAS) {
    const v = j.probabilities[k];
    const isi = el("div", { class: `prob-isi ${cls}` });
    isi.style.width = `${v * 100}%`;
    prob.append(el("div", { class: "prob-baris", role: "listitem" },
      el("span", { class: "prob-label" }, k), el("div", { class: "prob-jalur" }, isi), el("span", { class: "prob-nilai" }, persen(v, 1))));
  }
  blok.append(prob);

  if (j.confidence < AMBANG_RAGU) blok.append(peringatan("Model kurang yakin untuk ulasan ini"));
  if (j.vocabulary && j.vocabulary.known === 0) {
    blok.append(peringatan("Tidak ada kata yang dikenal model; hasil hanya mengikuti kecenderungan bawaan model."));
  }
  return blok;
}

function teksTahap(s, negasi = new Set()) {
  const wadah = el("div", { class: "tahap-isi" });
  const token = s.split(" ").filter(Boolean);
  if (!token.length) {
    wadah.append(el("span", { class: "kosong-teks" }, "(kosong)"));
    return wadah;
  }
  token.forEach((t, i) => {
    if (i) wadah.append(" ");
    wadah.append(negasi.has(t) ? el("mark", { class: "negasi" }, t) : t);
  });
  return wadah;
}

function barisChip(chip, kosong = null) {
  if (!chip.length) return kosong ? el("div", { class: "chip-baris" }, el("span", { class: "kosong-teks" }, kosong)) : null;
  return el("div", { class: "chip-baris" }, ...chip);
}

const tahap = (nama, isi, ket = null) => el("li", {}, el("span", { class: "tahap-no", "aria-hidden": "true" }),
  el("span", { class: "tahap-nama" }, nama, ket ? el("span", { class: "tahap-ket" }, ket) : null), isi);
const kotakTeks = (label, isi) => el("div", { class: "teks-kotak" },
  el("p", { class: "teks-kotak-label" }, label), el("p", { class: "teks-kotak-isi" }, isi));

function inspektur(j) {
  const p = j.preprocessing;
  const negasi = new Set(p.negations_kept);
  const slang = teksTahap(p.slang_normalized);
  slang.append(barisChip(p.replacements.map((r) => el("span", { class: "chip" }, `${r.from} → ${r.to}`)), "tidak ada kata slang"));
  const stop = teksTahap(p.after_stopword, negasi);
  const chipNegasi = barisChip(p.negations_kept.map((w) => el("span", { class: "chip" }, `${w}: dipertahankan`)));
  if (chipNegasi) stop.append(chipNegasi);

  const wadah = el("section", { class: "inspektur", "aria-label": "Inspektur teks" },
    el("h4", {}, "Inspektur teks"),
    el("div", { class: "inspektur-dua" }, kotakTeks("Teks asli", p.original), kotakTeks("Teks akhir (masuk ke TF-IDF)", p.stemmed)),
    el("ol", { class: "tahap" },
      tahap("Pembersihan dan normalisasi", teksTahap(p.cleaned), "huruf kecil, huruf berulang, negasi Inggris"),
      tahap("Normalisasi slang", slang),
      tahap("Hapus stopword", stop, "Indonesia dan Inggris; negasi dipertahankan"),
      tahap("Stemming", teksTahap(p.stemmed, negasi))));
  if (j.vocabulary) {
    wadah.append(el("p", { class: "kosakata" },
      `${angka(j.vocabulary.known)} dari ${angka(j.vocabulary.total)} kata hasil preprocessing dikenal kosakata TF-IDF.`));
  }
  return wadah;
}

function tampilHasil(jawaban) {
  const grid = el("div", { class: `hasil-model-grid${jawaban.length > 1 ? " dua" : ""}` }, ...jawaban.map(blokModel));
  wadahHasil.replaceChildren(grid, inspektur(jawaban[0]));
}

perbaruiForm();
muatMetrics();

# DESIGN.md: Dashboard Sentimen Kick

Arah dipilih pemilik proyek pada 17 Oktober 2026: **editorial jurnal**, tema **terang dengan tombol gelap**.

## Arah

- **Identitas:** pendamping paper penelitian. Halaman terasa seperti jurnal ilmiah yang dicetak rapi, bukan aplikasi SaaS.
- **Kepribadian:** tenang, teliti, berwibawa. Angka hasil penelitian adalah bintangnya.
- **Pembaca:** dosen pembimbing dan penguji (membaca hasil), penulis (presentasi di proyektor).
- **Dial:** ENERGY 2 / RHYTHM 2 / MOTION 1

## Palet

Token ada di `static/style.css` (`:root` untuk terang, blok `data-theme="dark"` untuk gelap).

| Peran | Terang | Gelap |
| --- | --- | --- |
| Kertas (latar) | `#f8f5ee` | `#171614` |
| Permukaan zona interaktif | `#fffdf8` | `#201e1b` |
| Tinta / tinta-2 / tinta-3 | `#1b1a17` / `#4b4740` / `#66615a` | `#ede8dc` / `#c7c0b1` / `#a39c8d` |
| Aksen (biru tinta) | `#24407e` | `#9db8ec` |
| Positif / Negatif | `#007400` / `#eb5c5b` | `#1d8a2f` / `#e4605e` |
| Ramp confusion matrix | sepia, terang ke gelap | sepia, gelap ke terang |

Semua pasangan teks lolos WCAG AA (diperiksa dengan `contrast-check.py`). Pasangan hijau/merah lolos validator
palet dataviz: tema terang CVD ΔE 8,5; tema gelap 6,6, yang sah hanya dengan encoding kedua, dan setiap
warna kelas memang selalu didampingi label teks.

## Tipografi

- **Source Serif 4** (sumbu optical size) untuk judul, angka besar, dan p-value.
- **Source Sans 3** untuk teks, label, dan angka tabel (`tabular-nums`).
- Monospace sistem hanya untuk menampilkan token hasil preprocessing apa adanya.
- Font disimpan di `static/fonts/` (SIL OFL), tidak memakai CDN.

## Komposisi

- Urutan baris mengikuti PRD bagian 13: header, KPI, distribusi dan resampling, tabel benchmark, ablasi dan McNemar, playground dan hasil.
- Bagian 1 sampai 3 adalah "halaman cetak": isi langsung di atas kertas, dipisah garis dan ruang kosong, tanpa kartu.
- Bagian 4 (Coba model) satu-satunya zona dengan panel permukaan, karena hanya bagian ini yang interaktif.

## Motif identitas

Keterangan bernomor ala jurnal (**Tabel 1.**, **Gambar 2.**) dan aturan tabel booktabs (garis tebal di atas
dan bawah, garis tipis di bawah kepala tabel, tanpa garis vertikal).

## Keputusan dan alasannya

| Keputusan | Alasan satu baris |
| --- | --- |
| Kertas putih hangat, bukan putih murni | Membawa nuansa halaman jurnal cetak dan lebih nyaman dibaca lama |
| Aksen biru tinta hanya pada BEST MODEL dan tombol Prediksi | Dua momen kunci halaman: kesimpulan dan aksi; di tempat lain aksen akan kehilangan arti |
| Hijau dan merah hanya untuk Positif dan Negatif | Ketentuan PRD; warna kelas tidak boleh bermakna lain |
| Serif untuk judul dan angka besar | Bahasa visual paper ilmiah; optical size menjaga judul besar tetap tajam |
| Sans satu keluarga (Source) untuk teks | Serasi dengan serifnya dan memiliki angka tabular untuk tabel |
| Tanpa kartu di bagian 1 sampai 3 | Hasil penelitian dibaca seperti paper; kartu akan membuatnya terasa seperti template dashboard |
| Panel permukaan hanya di bagian 4 | Menandai satu-satunya area yang bisa diisi dan diklik |
| Tabel booktabs | Konvensi tabel jurnal; angka lebih mudah dibandingkan tanpa garis vertikal |
| Ramp sepia untuk confusion matrix | Gradasi tinta ala gambar cetak, tidak mengambil hue aksen atau hue kelas |
| Ruang kosong besar antarbagian, rapat di dalam bagian | Ruang kosong menjadi struktur yang memisahkan bab, bukan sisa ruang |
| Radius 2 sampai 6px | Editorial terasa tegas; radius kecil membedakan kontrol dari isi |
| Gerak hanya hover dan fokus (MOTION 1) | Penguji membaca angka; animasi tidak menambah informasi |
| Tema terang default, tombol gelap tersedia | Proyektor membuat tema gelap pudar; mode gelap untuk membaca di layar |

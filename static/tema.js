// Dimuat sinkron di <head>: menerapkan tema yang pernah dipilih sebelum halaman digambar, agar tidak berkedip.
// Tanpa pilihan tersimpan, tema mengikuti pengaturan sistem (prefers-color-scheme di style.css).
(function () {
  try {
    var tema = localStorage.getItem("tema");
    if (tema === "light" || tema === "dark") document.documentElement.setAttribute("data-theme", tema);
  } catch (e) { /* penyimpanan diblokir: pakai tema sistem */ }
})();

Gunakan dataset berikut untuk menilai kemampuan deteksi dan ketepatan area yang ditandai. Pilih citra beserta mask acuannya agar hasil dapat dihitung, bukan hanya dibandingkan secara visual.

#### CoMoFoD

Acuan untuk menguji salinan yang mengalami perubahan geometri, kompresi JPEG, noise, atau blur. Tersedia 200 set citra 512×512 dan 60 set citra 3000×2000.

[Situs resmi](http://vcl.fer.hr/comofod/) · [Unduh melalui Kaggle](https://www.kaggle.com/datasets/tusharchauhan1898/comofod)

**Catatan pengujian:** nama berkas dengan `_O` menandai citra asli, `_F` citra manipulasi, dan `_B` mask biner. Gunakan citra asli untuk memeriksa alarm palsu dan citra manipulasi untuk mengukur deteksi.

#### CASIA 2.0

Sumber foto manipulasi yang lebih beragam. Untuk eksperimen CekTimpa, pilih sampel copy-move dan pastikan mask acuan tersedia serta sesuai dengan citra yang diuji.

[Unduh melalui Kaggle](https://www.kaggle.com/datasets/santoshanand10203040/casiav2)

**Catatan pengujian:** pisahkan manipulasi copy-move dari jenis manipulasi lain sebelum menghitung hasil benchmark.

#### COVERAGE

Digunakan untuk memeriksa apakah sistem dapat membedakan area salinan dari objek asli yang memang tampak mirip. Dataset ini penting untuk mengevaluasi alarm palsu pada pola dan objek berulang.

[Unduh melalui Kaggle](https://www.kaggle.com/datasets/ma7moud3tef/coverage) · [Informasi dari penyusun](https://github.com/wenbihan/coverage)

**Catatan pengujian:** evaluasi pasangan citra asli dan manipulasi. Jika mask sumber dan target disediakan terpisah, gabungkan keduanya untuk dibandingkan dengan keluaran CekTimpa.

---

#### Citra sumber untuk eksperimen sendiri

[USC-SIPI](https://sipi.usc.edu/database/), [Unsplash](https://unsplash.com/), dan [Pexels](https://www.pexels.com/) dapat dipakai sebagai bahan citra. Citra sumber ini belum otomatis menjadi dataset benchmark dengan ground truth. Simpan mask saat membuat manipulasi agar area sumber dan salinan diketahui.

#### Demo bawaan aplikasi

Demo sintetis menyediakan contoh positif, contoh negatif, dan mask acuan untuk memeriksa alur aplikasi. Keberhasilan pada demo belum mewakili kemampuan pada foto nyata.

**Sebelum evaluasi:** samakan ukuran dan orientasi citra dengan mask, lalu pisahkan data berdasarkan foto sumber agar variasi foto yang sama tidak tersebar antara validasi dan pengujian.

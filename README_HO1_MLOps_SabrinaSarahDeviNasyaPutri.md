# EDA \& Feature Engineering: Crime Data Chicago

Laporan hasil hands-on task EDA, Feature Engineering, dan Pseudo-Labeling data kriminalitas Chicago.



\---

## 1\. Penjelasan Singkat Dataset

Dataset yang digunakan adalah **Chicago Crimes Dataset** (City of Chicago Data Portal), dibatasi pada rentang **5 tahun terakhir (2021–2026)** untuk menjaga relevansi terhadap kondisi terkini dan mengantisipasi *concept drift* — penurunan performa model akibat perubahan distribusi data seiring waktu (Uchida \& Yoshida, 2023). Pemilihan rentang ini juga sejalan dengan pendekatan Jean \& Roy (2026) yang membangun model prediksi kriminalitas Chicago menggunakan data multi-tahun serupa.

**Proses cleaning yang dilakukan:**

1. Konversi kolom `Date` ke tipe `datetime` agar informasi jam, hari, dan bulan dapat diekstraksi.
2. Menghapus baris tanpa informasi tanggal maupun koordinat (`Latitude`/`Longitude`).
3. Memfilter koordinat di luar bounding box Kota Chicago untuk menghilangkan titik lokasi yang tidak valid.
4. Membentuk fitur waktu dasar: `hour`, `dow` (day of week), dan `month`.

**Hasil cleaning:**

|Tahap|Jumlah|
|-|-|
|Shape awal|1.002.744 baris, 22 kolom|
|Dibuang (tanggal/koordinat kosong)|8.091 baris|
|Dibuang (di luar bounding box)|2 baris|
|**Shape final**|**994.651 baris (\~99,2% data tersisa)**|



\---

## 2\. Justifikasi Keputusan Desain

### a. Severity Scoring

Karena dataset tidak memiliki kolom target "tingkat risiko", skor keparahan (`severity\_score`) dibangun sebagai **pseudo-label berbasis domain knowledge**, bukan hasil training model maupun skor resmi kepolisian. Skema penyusunannya:

* **Base severity per `Primary Type`** mengacu pada hierarki NIBRS/FBI (*Crimes Against Persons > Crimes Against Property > Crimes Against Society*) serta klasifikasi *felony/misdemeanor*, dibagi menjadi 5 tier (0–100), misalnya `HOMICIDE`/`CRIM SEXUAL ASSAULT` di tier tertinggi (75–100) dan `LIQUOR LAW VIOLATION`/`GAMBLING` di tier terendah (0–9). Kategori yang tidak terdaftar eksplisit diberi skor default 25 (netral-rendah) agar pipeline tetap berjalan.
* **Severity modifier** dari kata kunci pada `Description` (mis. *AGGRAVATED*, *ARMED: HANDGUN*, *DOMESTIC*) — sehingga pendekatan ini bersifat *scoring engine*, bukan tabulasi manual per kombinasi, dan otomatis mencakup seluruh kombinasi `Primary Type` × `Description` yang benar-benar muncul di data.
* Kalibrasi tabel skor menggunakan contoh ilustratif pada instruksi hands-on sebagai *anchor point*, sehingga konsisten secara internal (mis. `ROBBERY` bersenjata api selalu lebih berat dari `ROBBERY` tanpa senjata).

### b. Relevansi Waktu dan Lokasi (Space-Time Decay)

Kontribusi satu kejadian terhadap risiko suatu area **tidak dianggap konstan** — melainkan meluruh berdasarkan waktu dan jarak:

* **Temporal decay:** *exponential decay* terhadap `days\_since` (selisih hari dari tanggal referensi/tanggal terakhir di dataset), dengan `half\_life = 180 hari (\~6 bulan)`. Rumus: `w\_temporal = exp(-λ·Δt)`, `λ = ln(2)/half\_life`. Kejadian lama tetap dipertahankan sebagai konteks historis tapi bobotnya mengecil, sejalan dengan Han et al. (2023) yang menerapkan siklus temporal 6 bulan pada prediksi hotspot kriminal.
* **Spatial decay:** *Gaussian ring-weight* pada grid H3 (heksagonal), `w\_spatial(k) = exp(-k²/2σ²)` dengan `σ = 1`, dibatasi hingga ring ke-2 (\~1–1,5 km pada resolusi H3 8). H3 dipilih dibanding grid kotak biasa karena keenam tetangga ring-1 pada heksagon **berjarak seragam** dari pusat sel, sehingga tidak ada bias arah diagonal seperti pada grid kotak (di mana tetangga diagonal \~41% lebih jauh dari tetangga aksial).
* **Normalisasi ke 0–100:** menggunakan `log1p` sebelum min-max scaling (bukan min-max langsung), karena distribusi `risk\_score\_raw` sangat *right-skewed*. `log1p` memampatkan nilai ekstrem tanpa mendesak seluruh sel "sedang" menumpuk ke 0, sekaligus tetap mempertahankan jarak relatif antar nilai (berbeda dari percentile rank yang hanya mempertahankan urutan).

### c. Representasi Fitur

* **Fitur temporal — cyclical encoding (sin-cos):** karena waktu bersifat siklikal (pukul 23.00 berdekatan dengan 00.00; Desember berdekatan dengan Januari), representasi linear biasa akan salah menginterpretasikan jarak antarwaktu. Diterapkan pada `hour`, `dow`, dan `month` menggunakan formula `x\_sin = sin(2π·t/T)`, `x\_cos = cos(2π·t/T)`. Fitur biner `is\_weekend` ditambahkan karena komposisi jenis kejahatan berbeda antara weekday dan weekend.
* **Fitur spasial — dual representasi grid:**

  * *Grid kotak* (pembulatan `Latitude`/`Longitude` 2 desimal, ±1 km per sel) dipakai untuk fitur agregat historis (`grid\_crime\_count`, `grid\_arrest\_rate`, `grid\_theft\_share`, `grid\_narcotics\_share`), dipilih setelah membandingkan trade-off ukuran sel 0,1° (terlalu kasar, 16 sel), 0,001° (terlalu sparse, 36.055 sel), dan 0,01° (kompromi terbaik, 721 sel).
  * *Grid heksagonal H3 (resolusi 8)* dipakai khusus untuk perhitungan spatial decay, karena sifat jarak-seragamnya. Resolusi 8 dipilih setelah membandingkan res 7 (152 sel, terlalu kasar), res 8 (kompromi terbaik), dan res 9 (4.920 sel, terlalu sparse).
* Kolom penyusun label (`severity\_score`, `days\_since`, `temporal\_weight`, `Primary Type`, `Description`) sengaja tetap disertakan di dataset akhir, bukan hanya `risk\_score`, agar pipeline tetap **auditable**.



\---

## 3\. Insight Singkat dari EDA

* **Jam kejadian paling berpengaruh:** frekuensi tertinggi tercatat pukul 00.00 (diduga *midnight bucket* — waktu tidak diketahui pasti sehingga dicatat administratif jam 00.00), dengan puncak sekunder pada 12.00–17.00 dan titik terendah 04.00–06.00.
* **Interaksi hari × jam lebih informatif** daripada masing-masing secara terpisah — konsentrasi kejadian tertinggi terjadi siang-sore di hampir semua hari, dengan lonjakan pada Jumat dan Sabtu malam.
* **Pola musiman nyata:** kejadian meningkat mulai Juli, memuncak Agustus–Oktober, lalu menurun November–Desember.
* **`THEFT` paling sering terjadi**, diikuti `BATTERY` dan `CRIMINAL DAMAGE`, tetapi frekuensi tinggi **tidak berarti tingkat risiko tinggi** — `NARCOTICS` punya arrest rate \~96% sedangkan `MOTOR VEHICLE THEFT` hanya \~3%.
* **Setiap jenis kejahatan punya "sidik jari" waktu dan lokasi sendiri**: `NARCOTICS` terkonsentrasi sore-malam dan area jalan/gang, `BATTERY`/`ASSAULT` meningkat malam hari, `MOTOR VEHICLE THEFT` di area jalan/parkir, `OFFENSE INVOLVING CHILDREN` di area permukiman.
* **Distribusi spasial tidak merata** — hotspot jelas terlihat di bagian tengah, selatan, dan barat kota, dan pola hotspot ini **berbeda antar jenis kejahatan** (`NARCOTICS` lebih terkonsentrasi vs `THEFT` lebih menyebar).
* **Peta `risk\_score` akhir sejalan tapi tidak identik** dengan peta hotspot mentah — beberapa sel dengan jumlah kejadian tinggi ternyata risk score-nya moderat (didominasi kejahatan ringan/lama), sebaliknya ada sel dengan kejadian sedikit tapi risk score tinggi (kejadian berat \& baru) — membuktikan `risk\_score` membawa informasi lebih kaya dibanding `crime\_count` biasa.



\---

## 4\. Refleksi Singkat

**Kendala:**

* Representasi koordinat mentah (`Latitude`/`Longitude`) terlalu presisi sehingga hampir setiap kejadian punya lokasi unik, menyulitkan identifikasi pola spasial antarwilayah.
* Grid kotak biasa memiliki bias geometris — jarak diagonal antar sel tetangga \~41% lebih jauh dibanding jarak aksial — yang berisiko menghasilkan bobot spatial decay yang bias arah.
* Distribusi `risk\_score\_raw` hasil space-time decay sangat *right-skewed*, sehingga normalisasi min-max langsung membuat mayoritas sel menumpuk di nilai rendah dan sulit dibedakan.
* Tidak tersedia ground truth "tingkat risiko" pada dataset, sehingga label harus dibangun sendiri secara bertanggung jawab tanpa terkesan sembarangan atau tidak dapat dipertanggungjawabkan.

**Solusi:**

* Melakukan agregasi spasial melalui **grid aggregation** (pembulatan koordinat) untuk grid kotak, dan mengadopsi **H3 hexagonal grid** khusus untuk perhitungan proximity/spatial decay agar bobot antar-tetangga seragam ke segala arah.
* Menentukan ukuran sel/resolusi grid melalui **perbandingan trade-off eksplisit** (jumlah sel vs jumlah sel dengan observasi sangat sedikit) alih-alih memilih secara sembarangan.
* Menerapkan transformasi **`log1p` sebelum min-max scaling** untuk mengatasi skewness pada normalisasi `risk\_score`, dengan histogram sebagai validasi visual bahwa distribusi hasil menjadi lebih tersebar dan informatif.
* Menyusun **severity scoring engine** yang dikalibrasi terhadap contoh ilustratif pada instruksi hands-on sebagai anchor point, serta mengikuti kerangka acuan yang dapat dipertanggungjawabkan (hierarki NIBRS/FBI), sekaligus tetap menyertakan kolom penyusun label pada dataset akhir agar seluruh proses **auditable** dan dapat ditelusuri ulang.

\---



## Rangkuman

|Tahap|Ringkasan|
|-|-|
|Cleaning|\~1 juta baris mentah → 994.651 kejadian valid (99,2%)|
|EDA|10 visualisasi; pola kuat pada jam, hari, bulan, dan lokasi; didominasi `THEFT` \& `BATTERY`|
|Feature Engineering|Cyclical encoding (sin-cos) untuk waktu; grid kotak 0,01° + H3 res 8 untuk lokasi; fitur agregat historis per area|
|Pseudo-Labeling|`risk\_score` = severity scoring → temporal decay (exponential, half-life 180 hari) → spatial decay (Gaussian ring-weight H3) → normalisasi `log1p` + min-max (0–100)|
|Output|`features\_labels.csv` / `features\_labels.parquet`, lengkap dengan kolom penyusun label untuk keperluan Hands-On 2|




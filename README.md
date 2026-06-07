<div align="center">

# 🌤️ WeatherAI

**Aplikasi Prediksi Cuaca Cerdas Berbasis AI**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/version-3.0.0-brightgreen?style=flat-square" alt="version"/>
  <img src="https://img.shields.io/badge/status-active-success?style=flat-square" alt="status"/>
  <img src="https://img.shields.io/badge/language-Indonesia-red?style=flat-square" alt="language"/>
</p>

> WeatherAI adalah aplikasi web full-stack yang menggabungkan data cuaca real-time, machine learning, computer vision, dan asisten AI berbasis Gemini untuk memberikan pengalaman prakiraan cuaca yang cerdas, personal, dan interaktif.

---

</div>

## 📋 Daftar Isi

- [✨ Fitur Unggulan](#-fitur-unggulan)
- [🏗️ Arsitektur Sistem](#️-arsitektur-sistem)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Struktur Proyek](#-struktur-proyek)
- [⚙️ Instalasi & Setup](#️-instalasi--setup)
- [🔑 Konfigurasi Environment](#-konfigurasi-environment)
- [🚀 Menjalankan Aplikasi](#-menjalankan-aplikasi)
- [📡 API Endpoints](#-api-endpoints)
- [🤖 Modul AI & Machine Learning](#-modul-ai--machine-learning)
- [🌐 Halaman & Tampilan](#-halaman--tampilan)
- [📊 Database](#-database)
- [🤝 Kontribusi](#-kontribusi)
- [📄 Lisensi](#-lisensi)

---

## ✨ Fitur Unggulan

### 🌍 Data Cuaca Real-Time
- Suhu, kelembaban, tekanan udara, kecepatan angin langsung dari **Open-Meteo API**
- Prakiraan **6 hari ke depan** dengan data lengkap
- UV Index dan curah hujan harian
- **Kualitas udara** (AQI, PM2.5, PM10) via Open-Meteo Air Quality API
- Waktu lokal otomatis berdasarkan koordinat GPS dengan dukungan **zona waktu Indonesia** (WIB, WITA, WIT)

### 🤖 Asisten AI — Ashley
- Chatbot cuaca cerdas berbasis **Google Gemini 2.5 Flash**
- Sistem **rotasi multi API key** dengan retry logic dan cooldown otomatis
- Filter topik cerdas — Ashley hanya membahas topik cuaca
- Fallback response manual saat API tidak tersedia
- Pembersihan format Markdown otomatis pada respons AI

### 🧠 Machine Learning — Prediksi Cuaca
- Model **Random Forest Regressor** untuk prediksi suhu
- Training otomatis dari data historis (30 hari) via Open-Meteo
- Feature engineering: lag features (1, 3, 6, 12, 24 jam), siklus harian, musiman
- Evaluasi model: **MAE, RMSE, R²**, dan Cross-Validation (5-fold)
- Visualisasi hasil prediksi interaktif di dashboard ML

### 📸 Computer Vision — Klasifikasi Cuaca dari Gambar
- Model **CNN (Convolutional Neural Network)** berbasis TensorFlow/Keras
- Arsitektur: 3 layer Conv2D → MaxPooling → Dropout → Dense
- Prediksi cuaca dari **upload gambar** atau **kamera langsung (webcam)**
- Download model pre-trained dari Google Drive (78MB)
- Training custom dari dataset lokal dengan ImageDataGenerator

### 📍 Manajemen Lokasi
- Simpan dan kelola **banyak lokasi** favorit
- Pencarian kota berdasarkan **nama** atau **koordinat (lat/lon)**
- Switch lokasi aktif dengan mudah dari sidebar
- Data lokasi tersimpan di database SQLite lokal

### ⭐ Sistem Ulasan
- Pengguna dapat memberikan ulasan dan rating (1–5 bintang)
- Tampilan ulasan publik di halaman `/ulasan`
- Admin dapat menghapus ulasan menggunakan password terproteksi

### 🎨 UI/UX Modern
- Desain **glassmorphism** dengan dark mode & light mode
- Fully **responsive** — mobile, tablet, desktop
- Animasi halus (page transition, cloud float, background shift)
- Bottom navigation untuk mobile
- Sidebar dengan scroll independen untuk daftar lokasi

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                      CLIENT (Browser)                   │
│              HTML + CSS + Vanilla JavaScript            │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP Request
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                       │
│                                                         │
│   ┌──────────────┐  ┌─────────────┐  ┌──────────────┐   │
│   │  Route       │  │  ML Module  │  │  CV Module   │   │
│   │  Handlers    │  │  (RF Model) │  │  (CNN Model) │   │
│   └──────┬───────┘  └──────┬──────┘  └──────┬───────┘   │
│          │                 │                 │          │
│   ┌──────▼─────────────────▼─────────────────▼───────┐  │
│   │               Core Services                      │  │
│   │  • WeatherService   • GeminiRotator              │  │
│   │  • AirQualityService • LocationService           │  │
│   │  • TimeService      • TestimonialService         │  │
│   └──────┬──────────────────────────────────┬────────┘  │
└──────────┼──────────────────────────────────┼───────────┘
           │                                  │
┌──────────▼──────────┐             ┌─────────▼──────────┐
│   External APIs     │             │   SQLite Databas   │
│                     │             │                    │
│ • Open-Meteo API    │             │ • saved_locations  │
│ • Air Quality API   │             │ • testimonials     │
│ • Nominatim/OSM     │             │                    │
│ • Google Gemini API │             └────────────────────┘
│ • Google Drive      │
└─────────────────────┘
```

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|---|---|
| **Backend Framework** | FastAPI, Uvicorn |
| **AI / LLM** | Google Gemini 2.5 Flash (`google-genai`) |
| **Machine Learning** | scikit-learn (RandomForestRegressor), pandas, numpy, joblib |
| **Computer Vision** | TensorFlow / Keras, OpenCV (`cv2`) |
| **Database** | SQLite (via `sqlite3`) |
| **Data Cuaca** | Open-Meteo API (gratis, tanpa API key) |
| **Geocoding** | Nominatim / OpenStreetMap |
| **Model Download** | gdown (Google Drive) |
| **Timezone** | pytz |
| **Frontend** | HTML5, CSS3, Vanilla JS (server-side render via FastAPI HTMLResponse) |
| **Styling** | Custom CSS (glassmorphism, CSS Variables, animasi) |
| **Fonts** | Google Fonts — Inter |

---

## 📁 Struktur Proyek

```
weatherai/
│
├── app.py                  # Aplikasi utama FastAPI (backend + frontend rendering)
├── styles.css              # Stylesheet utama (di-serve sebagai static file)
├── weather.db              # Database SQLite lokasi tersimpan (auto-generated)
├── testimonials.db         # Database SQLite ulasan (auto-generated)
├── weather_rf_model.pkl    # Model Random Forest terlatih (auto-generated)
├── weather_cnn_model.keras # Model CNN TensorFlow (download dari Google Drive)
│
├── .env                    # Konfigurasi environment variables (buat sendiri)
├── requirements.txt        # Daftar dependensi Python
└── README.md               # Dokumentasi proyek
```

---

## ⚙️ Instalasi & Setup

### Prasyarat

Pastikan sistem kamu memiliki:
- **Python 3.9** atau lebih baru
- **pip** (Python package manager)
- Koneksi internet (untuk data cuaca dan Gemini API)

### 1. Clone Repository

```bash
git clone https://github.com/aryamods/WeatherAI-App.git
cd weatherai
```

### 2. Buat Virtual Environment (Direkomendasikan)

```bash
# Buat environment
python -m venv venv

# Aktifkan — Windows
venv\Scripts\activate

# Aktifkan — macOS/Linux
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

Jika `requirements.txt` belum ada, install manual:

```bash
pip install fastapi uvicorn requests pandas numpy scikit-learn joblib \
            tensorflow opencv-python gdown pytz python-dotenv \
            google-genai pydantic
```

> **Catatan:** TensorFlow bersifat opsional. Jika tidak diinstall, fitur klasifikasi gambar (CNN) akan dinonaktifkan secara otomatis namun aplikasi tetap berjalan normal.

---

## 🔑 Konfigurasi Environment

Buat file `.env` di root proyek:

```env
# =============================================
# GOOGLE GEMINI API KEYS
# Dapatkan di: https://aistudio.google.com
# =============================================

GEMINI_API_KEY=your_primary_gemini_api_key_here
GEMINI_API_KEY_BACKUP=your_backup_key_1_here       # Opsional
GEMINI_API_KEY_BACKUP2=your_backup_key_2_here      # Opsional
GEMINI_API_KEY_BACKUP3=your_backup_key_3_here      # Opsional

# =============================================
# ADMIN PASSWORD (untuk hapus ulasan)
# =============================================

ADMIN_PASSWORD=your_secure_admin_password_here
```

> **Tips:** Gunakan beberapa API key Gemini untuk menghindari rate limiting. Sistem akan melakukan rotasi otomatis antar key dengan cooldown management.

> **Catatan:** Aplikasi tetap berjalan tanpa Gemini API key, menggunakan fallback response berbasis template.

---

## 🚀 Menjalankan Aplikasi

```bash
# Jalankan server development
python app.py
```

Atau langsung dengan uvicorn:

```bash
uvicorn app:app --host localhost --port 8080 --reload
```

Buka browser dan akses:

```
http://localhost:8080
```

Dokumentasi API otomatis tersedia di:

```
http://localhost:8080/docs        # Swagger UI
http://localhost:8080/redoc       # ReDoc
```

---

## 📡 API Endpoints

### Halaman Utama

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Halaman beranda — cuaca saat ini + AI insights |
| `GET` | `/main` | Dashboard ML — prediksi & klasifikasi gambar |
| `GET` | `/search` | Halaman pencarian lokasi |
| `GET` | `/ulasan` | Halaman ulasan pengguna |
| `GET` | `/about` | Halaman tentang aplikasi |

### API Fungsional

| Method | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/chat-ai` | Chat dengan asisten Ashley (JSON: `{message}`) |
| `POST` | `/search/city` | Cari kota berdasarkan nama |
| `POST` | `/search/coords` | Cari berdasarkan koordinat lat/lon |
| `GET` | `/select-location/{id}` | Set lokasi aktif |
| `GET` | `/delete-location/{id}` | Hapus lokasi tersimpan |
| `GET` | `/train-model` | Latih ulang model Random Forest |
| `POST` | `/predict-weather-image` | Prediksi cuaca dari upload gambar |
| `POST` | `/predict-weather-camera` | Prediksi cuaca dari frame kamera |
| `POST` | `/download-cnn-model` | Download model CNN dari Google Drive |
| `POST` | `/ulasan/submit` | Submit ulasan baru |
| `POST` | `/verify-delete-testimonial/{id}` | Hapus ulasan (butuh password admin) |
| `GET` | `/api-key-status` | Status penggunaan semua Gemini API key |

### Contoh Request — Chat AI

```bash
curl -X POST "http://localhost:8080/chat-ai" \
     -H "Content-Type: application/json" \
     -d '{"message": "Bagaimana cuaca hari ini?"}'
```

```json
{
  "reply": "☀️ Cerah Berawan di Jakarta\nSuhu 31°C (terasa 34°C), kelembaban 72% — cuaca hari ini cukup nyaman.\nAngin 15 km/j, tekanan udara 1009 hPa.\n😊 Kualitas udara: Baik (AQI 42, PM2.5 12.5 µg/m³)."
}
```

---

## 🤖 Modul AI & Machine Learning

### 1. GeminiRotator — Manajemen API Key

Sistem rotasi API key otomatis dengan fitur:

- **Load balancing** berdasarkan jumlah penggunaan harian
- **Cooldown otomatis** saat rate limit (429) terdeteksi — exponential backoff hingga 5 menit
- **Quota exhausted handling** — cooldown 1 jam jika quota habis
- **Daily counter reset** setiap 24 jam
- **Retry logic** hingga 3 kali percobaan per request

```python
# Inisialisasi dengan beberapa API key
rotator = GeminiRotator([key1, key2, key3, key4])
response = rotator.call_api("Prompt kamu di sini", model="gemini-2.5-flash")
```

### 2. WeatherPredictor — Random Forest

| Parameter | Nilai |
|---|---|
| Algoritma | `RandomForestRegressor` |
| Target | Suhu (temperature_2m) |
| Features | Suhu, kelembaban, curah hujan, kecepatan angin, tekanan, jam, hari dalam setahun, bulan, lag 1/3/6/12/24 jam |
| Data historis | 30 hari (Open-Meteo) |
| Train/Test split | 80% / 20% |
| Cross-validation | 5-fold KFold |
| Metrik evaluasi | MAE, RMSE, R² |

Training model pertama kali dilakukan otomatis saat server start. Model disimpan ke `weather_rf_model.pkl` menggunakan joblib.

### 3. WeatherImageClassifier — CNN TensorFlow

| Layer | Konfigurasi |
|---|---|
| Conv2D #1 | 32 filter, kernel 3×3, ReLU, input 128×128×3 |
| MaxPooling #1 | 2×2 |
| Conv2D #2 | 64 filter, kernel 3×3, ReLU |
| MaxPooling #2 | 2×2 |
| Conv2D #3 | 128 filter, kernel 3×3, ReLU |
| MaxPooling #3 | 2×2 |
| Dropout | 0.5 |
| Dense | 256 unit, ReLU |
| Output | Softmax (jumlah kelas) |

**Optimizer:** Adam | **Loss:** Categorical Crossentropy | **Epochs:** 15

---

## 🌐 Halaman & Tampilan

### 🏠 Beranda (`/`)
Dashboard utama yang menampilkan:
- Greeting dinamis berdasarkan waktu lokal (Pagi/Siang/Sore/Malam)
- Cuaca saat ini: suhu, feels like, kondisi langit, angin, kelembaban, tekanan
- Kualitas udara (AQI, PM2.5, PM10)
- UV Index dengan rekomendasi
- Prakiraan 6 hari ke depan
- AI Insights dari Gemini
- Chatbot Ashley (floating button)

### 📊 Dashboard ML (`/main`)
- Hasil prediksi suhu model Random Forest
- Grafik historis vs prediksi
- Metrik evaluasi model (MAE, RMSE, R², Cross-Val)
- Upload gambar untuk klasifikasi cuaca (CNN)
- Live kamera untuk klasifikasi real-time
- Tombol train ulang model & download CNN

### 🔍 Pencarian (`/search`)
- Cari lokasi via nama kota (Nominatim/OSM geocoding)
- Input koordinat manual (latitude/longitude)
- Tambah dan simpan lokasi ke database

### ⭐ Ulasan (`/ulasan`)
- Form submit ulasan: nama, peran, komentar, rating bintang
- Grid tampilan semua ulasan publik
- Fitur hapus ulasan dengan verifikasi password admin

---

## 📊 Database

Aplikasi menggunakan **dua database SQLite** yang dibuat otomatis:

### `weather.db` — Lokasi

```sql
CREATE TABLE saved_locations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    latitude   REAL NOT NULL,
    longitude  REAL NOT NULL,
    country    TEXT,
    timezone   TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `testimonials.db` — Ulasan

```sql
CREATE TABLE testimonials (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    role       TEXT NOT NULL,
    comment    TEXT NOT NULL,
    rating     INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🤝 Kontribusi

Kontribusi sangat disambut! Ikuti langkah berikut:

1. **Fork** repository ini
2. Buat branch fitur baru:
   ```bash
   git checkout -b feat/nama-fitur-kamu
   ```
3. Commit perubahan:
   ```bash
   git commit -m "feat: tambah fitur XYZ"
   ```
4. Push ke branch:
   ```bash
   git push origin feat/nama-fitur-kamu
   ```
5. Buat **Pull Request** ke branch `main`

### Panduan Commit Message

| Prefix | Digunakan untuk |
|---|---|
| `feat:` | Fitur baru |
| `fix:` | Bug fix |
| `docs:` | Perubahan dokumentasi |
| `style:` | Perubahan style/formatting |
| `refactor:` | Refactoring kode |
| `perf:` | Peningkatan performa |

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** — lihat file [LICENSE](LICENSE) untuk detail lengkap.

---

<div align="center">

Dibuat dengan ❤️ menggunakan Python & FastAPI

⭐ **Jika proyek ini bermanfaat, jangan lupa kasih star!** ⭐

</div>

from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import sqlite3
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
from pathlib import Path
import base64
import cv2

warnings.filterwarnings("ignore")

app = FastAPI(
    title="WeatherAI",
    description="Aplikasi Prediksi Cuaca Cerdas Berbasis AI",
    version="3.0.0",
)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


class PasswordRequest(BaseModel):
    password: str

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print("\033[92mINFO\033[0m:     TensorFlow tersedia.")
except ImportError:
    TF_AVAILABLE = False
    print("\033[92mINFO\033[0m     TensorFlow tidak tersedia.")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("\033[92mINFO\033[0m:     python-dotenv tersedia.")
except ImportError:
    print("\033[92mINFO\033[0m:     python-dotenv tidak tersedia.")

GEMINI_API_KEY = "AIzaSyD2pPh-rfKgqEeugy13BzWda-9ft3zWo6A"
GEMINI_API_KEY_BACKUP = "AIzaSyBD1VSdjPI81zOzqlnHHOgWeqIBauvn_hI"
GEMINI_API_KEY_BACKUP2 = "AIzaSyB0_TDsxSb2d0W3xZ_aY_BdZ4DuRi3h3ag"
GEMINI_API_KEY_BACKUP3 = "AIzaSyBKSoGybzrpLwFMEm9oukqxJcF25I7XAPs"

def initialize_gemini_client():
    api_keys = [GEMINI_API_KEY, GEMINI_API_KEY_BACKUP, GEMINI_API_KEY_BACKUP2, GEMINI_API_KEY_BACKUP3]
    api_keys = [key for key in api_keys if key]

    if not api_keys:
        print("\033[92mINFO:\033[0m     API key tidak tersedia.")
        return None, False

    for i, api_key in enumerate(api_keys, 1):
        try:
            client = genai.Client(api_key=api_key)
            test_response = client.models.generate_content(
                model="gemini-2.5-flash", contents="Test connection"
            )
            print(f"\033[92mINFO\033[0m:     Ashley siap digunakan dengan API Key {i}")
            print("   Model: gemini-2.5-flash")
            return client, True
        except Exception as e:
            print(f"\033[92mINFO\033[0m:     API Key {i} gagal: {e}")
            continue

    print("\033[92mINFO\033[0m:     Semua API key gagal. AI features akan dinonaktifkan.")
    return None, False

AI_AVAILABLE = False
client = None

try:
    from google import genai
    client, AI_AVAILABLE = initialize_gemini_client()
except ImportError:
    print("\033[92mINFO\033[0m:     Library google-genai belum terinstall.")
except Exception as e:
    print(f"\033[92mINFO\033[0m:     Error inisialisasi: {e}")
    AI_AVAILABLE = False

DB_PATH = "weather.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            country TEXT,
            timezone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(saved_locations)")
    columns = [column[1] for column in cursor.fetchall()]
    if "timezone" not in columns:
        print("\033[92mINFO\033[0m:     Menambahkan kolom timezone ke database...")
        cursor.execute("ALTER TABLE saved_locations ADD COLUMN timezone TEXT")
        print("\033[92mINFO\033[0m:     Kolom timezone berhasil ditambahkan")

    conn.commit()
    conn.close()
    print("\033[92mINFO\033[0m:     Database initialized")

def save_location(name: str, latitude: float, longitude: float, country: str = None, timezone: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO saved_locations (name, latitude, longitude, country, timezone)
        VALUES (?, ?, ?, ?, ?)
    """, (name, latitude, longitude, country, timezone))
    conn.commit()
    conn.close()

def get_saved_locations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, latitude, longitude, country, timezone FROM saved_locations ORDER BY created_at DESC")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        cursor.execute("SELECT id, name, latitude, longitude, country FROM saved_locations ORDER BY created_at DESC")
        rows = cursor.fetchall()
        locations = []
        for row in rows:
            locations.append({
                "id": row[0],
                "name": row[1],
                "latitude": row[2],
                "longitude": row[3],
                "country": row[4] or "Indonesia",
                "timezone": None,
            })
        conn.close()
        return locations

    locations = []
    for row in rows:
        locations.append({
            "id": row[0],
            "name": row[1],
            "latitude": row[2],
            "longitude": row[3],
            "country": row[4] or "Indonesia",
            "timezone": row[5] if len(row) > 5 else None,
        })
    conn.close()
    return locations

def delete_location(location_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_locations WHERE id = ?", (location_id,))
    conn.commit()
    conn.close()

def location_exists(name: str, latitude: float, longitude: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM saved_locations WHERE name = ? OR (latitude = ? AND longitude = ?)", (name, latitude, longitude))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

init_db()

# ============ FITUR TESTIMONIAL ============
TESTIMONIALS_DB_PATH = "testimonials.db"

def init_testimonials_db():
    """Inisialisasi database untuk testimonial"""
    conn = sqlite3.connect(TESTIMONIALS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS testimonials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            comment TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("\033[92mINFO\033[0m:     Database testimonials initialized")

def save_testimonial(name: str, role: str, comment: str, rating: int):
    """Menyimpan testimonial ke database"""
    conn = sqlite3.connect(TESTIMONIALS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO testimonials (name, role, comment, rating)
        VALUES (?, ?, ?, ?)
    """, (name, role, comment, rating))
    conn.commit()
    conn.close()

def get_all_testimonials():
    """Mendapatkan semua testimonial"""
    conn = sqlite3.connect(TESTIMONIALS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, role, comment, rating, created_at FROM testimonials ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    testimonials = []
    for row in rows:
        testimonials.append({
            "id": row[0],
            "name": row[1],
            "role": row[2],
            "comment": row[3],
            "rating": row[4],
            "created_at": row[5]
        })
    return testimonials

def delete_testimonial_by_id(testimonial_id: int):
    """Menghapus testimonial berdasarkan ID"""
    conn = sqlite3.connect(TESTIMONIALS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM testimonials WHERE id = ?", (testimonial_id,))
    conn.commit()
    conn.close()

# Inisialisasi database testimonial
init_testimonials_db()

def get_timezone_from_coords(latitude: float, longitude: float):
    if 95 <= longitude <= 141:
        if -8 <= latitude <= 6:
            if 95 <= longitude <= 120:
                return "Asia/Jakarta"
            elif 120 < longitude <= 128:
                return "Asia/Makassar"
            else:
                return "Asia/Jayapura"

    if longitude > 128 and longitude <= 141:
        if -10 <= latitude <= 0:
            return "Asia/Jayapura"

    offset = int((longitude + 7.5) / 15)
    offset = max(-12, min(12, offset))

    timezone_map = {
        -5: "America/New_York", -6: "America/Chicago", -7: "America/Denver",
        -8: "America/Los_Angeles", 0: "Europe/London", 1: "Europe/Paris",
        2: "Europe/Helsinki", 3: "Asia/Riyadh", 4: "Asia/Dubai",
        5: "Asia/Karachi", 5.5: "Asia/Kolkata", 6: "Asia/Dhaka",
        7: "Asia/Jakarta", 8: "Asia/Makassar", 9: "Asia/Jayapura",
        10: "Asia/Tokyo", 11: "Asia/Sakhalin", 12: "Pacific/Auckland",
    }
    return timezone_map.get(offset, "UTC")

def get_offset_display(timezone_str: str) -> str:
    try:
        tz = pytz.timezone(timezone_str)
        now = datetime.now(tz)
        offset = now.strftime("%z")
        hours = int(offset[:3])
        return f"UTC{hours:+d}"
    except:
        return "UTC"

def get_local_time(latitude: float, longitude: float, timezone_str: str = None):
    try:
        if not timezone_str:
            timezone_str = get_timezone_from_coords(latitude, longitude)
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz)
        offset_display = get_offset_display(timezone_str)

        hari_indonesia = {
            "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
            "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu",
        }
        day_name = hari_indonesia.get(local_time.strftime("%A"), local_time.strftime("%A"))

        bulan_indonesia = {
            "January": "Januari", "February": "Februari", "March": "Maret",
            "April": "April", "May": "Mei", "June": "Juni",
            "July": "Juli", "August": "Agustus", "September": "September",
            "October": "Oktober", "November": "November", "December": "Desember",
        }
        month_name = bulan_indonesia.get(local_time.strftime("%B"), local_time.strftime("%B"))

        return {
            "time": local_time.strftime("%H:%M:%S"),
            "date": f"{day_name}, {local_time.strftime('%d')} {month_name} {local_time.strftime('%Y')}",
            "day": day_name,
            "timezone": offset_display,
            "timezone_full": timezone_str,
            "hour": int(local_time.strftime("%H")),
            "minute": int(local_time.strftime("%M")),
            "offset": offset_display,
        }
    except Exception as e:
        print(f"Local time error: {e}")
        now = datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%d %B %Y"),
            "day": now.strftime("%A"),
            "timezone": "UTC+7",
            "timezone_full": "Asia/Jakarta",
            "hour": int(now.strftime("%H")),
            "minute": int(now.strftime("%M")),
            "offset": "UTC+7",
        }

def get_current_weather(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code", "wind_speed_10m", "surface_pressure"],
        "daily": ["uv_index_max"],
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        uv = daily.get("uv_index_max", [0])[0] if daily.get("uv_index_max") else 5

        return {
            "temperature": current.get("temperature_2m", 0),
            "feels_like": current.get("apparent_temperature", 0),
            "humidity": current.get("relative_humidity_2m", 0),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "pressure": current.get("surface_pressure", 0),
            "weather_code": current.get("weather_code", 0),
            "uv_index": uv,
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {
            "temperature": 28.5, "feels_like": 29.0, "humidity": 75,
            "precipitation": 0.5, "wind_speed": 12, "pressure": 1012,
            "weather_code": 0, "uv_index": 7,
        }

def get_air_quality(latitude: float, longitude: float):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["us_aqi", "pm10", "pm2_5"],
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        current = data.get("current", {})

        us_aqi = current.get("us_aqi", 0)
        pm25 = current.get("pm2_5", 0)
        pm10 = current.get("pm10", 0)

        if us_aqi <= 50:
            status, status_color, status_icon = "Baik", "#10b981", "fa-smile"
        elif us_aqi <= 100:
            status, status_color, status_icon = "Sedang", "#f59e0b", "fa-meh"
        elif us_aqi <= 150:
            status, status_color, status_icon = "Tidak Sehat", "#f97316", "fa-face-frown"
        elif us_aqi <= 200:
            status, status_color, status_icon = "Tidak Sehat", "#ef4444", "fa-face-angry"
        elif us_aqi <= 300:
            status, status_color, status_icon = "Sangat Tidak Sehat", "#8b5cf6", "fa-skull"
        else:
            status, status_color, status_icon = "Berbahaya", "#dc2626", "fa-biohazard"

        return {
            "aqi": us_aqi, "status": status, "status_color": status_color,
            "status_icon": status_icon, "pm25": round(pm25, 1), "pm10": round(pm10, 1), "available": True,
        }
    except Exception as e:
        print(f"Air Quality API error: {e}")
        return {
            "aqi": 42, "status": "Baik", "status_color": "#10b981",
            "status_icon": "fa-smile", "pm25": 12.5, "pm10": 25.0, "available": True,
        }

def get_6day_forecast(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code", "uv_index_max"],
        "timezone": "auto",
        "forecast_days": 6,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        daily = data.get("daily", {})

        hari_map = {"Mon": "Sen", "Tue": "Sel", "Wed": "Rab", "Thu": "Kam", "Fri": "Jum", "Sat": "Sab", "Sun": "Min"}

        forecast = []
        for i in range(6):
            date = datetime.now() + timedelta(days=i)
            day_eng = date.strftime("%a")
            day_ind = hari_map.get(day_eng, day_eng)

            forecast.append({
                "day": day_ind,
                "date": date.strftime("%d/%m"),
                "temp_max": daily.get("temperature_2m_max", [0])[i] if daily.get("temperature_2m_max") else 0,
                "temp_min": daily.get("temperature_2m_min", [0])[i] if daily.get("temperature_2m_min") else 0,
                "precipitation": daily.get("precipitation_sum", [0])[i] if daily.get("precipitation_sum") else 0,
                "weather_code": daily.get("weather_code", [0])[i] if daily.get("weather_code") else 0,
                "uv_index": daily.get("uv_index_max", [0])[i] if daily.get("uv_index_max") else 0,
            })
        return forecast
    except Exception as e:
        print(f"Forecast API error: {e}")
        return [
            {"day": "Sen", "date": "01/01", "temp_max": 30, "temp_min": 24, "precipitation": 2, "weather_code": 0, "uv_index": 7},
            {"day": "Sel", "date": "02/01", "temp_max": 29, "temp_min": 23, "precipitation": 5, "weather_code": 61, "uv_index": 6},
            {"day": "Rab", "date": "03/01", "temp_max": 28, "temp_min": 23, "precipitation": 3, "weather_code": 1, "uv_index": 8},
            {"day": "Kam", "date": "04/01", "temp_max": 29, "temp_min": 24, "precipitation": 1, "weather_code": 0, "uv_index": 9},
            {"day": "Jum", "date": "05/01", "temp_max": 31, "temp_min": 25, "precipitation": 0, "weather_code": 0, "uv_index": 10},
            {"day": "Sab", "date": "06/01", "temp_max": 30, "temp_min": 24, "precipitation": 1, "weather_code": 1, "uv_index": 8},
        ]

def search_city(city_name: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        results = data.get("results", [])
        if results:
            city = results[0]
            lat = city.get("latitude", 0)
            lon = city.get("longitude", 0)
            tz = get_timezone_from_coords(lat, lon)

            return {
                "name": city.get("name", city_name),
                "latitude": lat,
                "longitude": lon,
                "country": city.get("country", ""),
                "admin1": city.get("admin1", ""),
                "timezone": tz,
            }
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def search_by_coordinates(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m"],
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        location_name = f"Koordinat ({latitude:.4f}, {longitude:.4f})"

        try:
            geo_url = "https://nominatim.openstreetmap.org/reverse"
            geo_params = {"lat": latitude, "lon": longitude, "format": "json", "zoom": 10}
            geo_response = requests.get(geo_url, params=geo_params, headers={"User-Agent": "WeatherAI/1.0"}, timeout=10)
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if "display_name" in geo_data:
                    address = geo_data.get("address", {})
                    city = address.get("city") or address.get("town") or address.get("village") or address.get("state") or "Lokasi"
                    country = address.get("country", "")
                    location_name = city
                    return {
                        "name": location_name,
                        "latitude": latitude,
                        "longitude": longitude,
                        "country": country,
                        "admin1": address.get("state", ""),
                        "timezone": get_timezone_from_coords(latitude, longitude),
                        "from_coords": True,
                    }
        except Exception as e:
            print(f"Reverse geocoding error: {e}")

        tz = get_timezone_from_coords(latitude, longitude)
        return {
            "name": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "country": "",
            "admin1": "",
            "timezone": tz,
            "from_coords": True,
        }
    except Exception as e:
        print(f"Coordinate search error: {e}")
        return None

def validate_coordinates(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180

def get_weather_icon_html(weather_code: int) -> str:
    icons = {
        0: '<i class="fas fa-sun" style="color: #fbbf24;"></i>',
        1: '<i class="fas fa-cloud-sun" style="color: #fbbf24;"></i>',
        2: '<i class="fas fa-cloud" style="color: #94a3b8;"></i>',
        3: '<i class="fas fa-cloud" style="color: #64748b;"></i>',
        45: '<i class="fas fa-smog" style="color: #94a3b8;"></i>',
        51: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        53: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        55: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        61: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        63: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        65: '<i class="fas fa-cloud-showers-heavy" style="color: #2563eb;"></i>',
        80: '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
        81: '<i class="fas fa-cloud-showers-heavy" style="color: #3b82f6;"></i>',
        95: '<i class="fas fa-bolt" style="color: #f59e0b;"></i>',
    }
    return icons.get(weather_code, '<i class="fas fa-cloud-sun"></i>')

def get_condition_text(weather_code: int) -> str:
    conditions = {
        0: "Cerah", 1: "Sebagian Cerah", 2: "Berawan", 3: "Mendung",
        45: "Kabut", 51: "Gerimis", 53: "Gerimis Sedang", 55: "Gerimis Lebat",
        61: "Hujan Ringan", 63: "Hujan Sedang", 65: "Hujan Lebat",
        80: "Hujan Lokal", 81: "Hujan Sedang", 95: "Badai Petir",
    }
    return conditions.get(weather_code, "Cerah")

MODEL_PATH = "weather_model.pkl"

class WeatherPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()

    def fetch_historical_data(self, latitude=-6.2, longitude=106.816666, days=30):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "pressure_msl"],
            "timezone": "auto",
            "past_days": days,
            "forecast_days": 7,
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            hourly = data.get("hourly", {})

            df = pd.DataFrame({
                "temperature": hourly.get("temperature_2m", []),
                "humidity": hourly.get("relative_humidity_2m", []),
                "precipitation": hourly.get("precipitation", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "pressure": hourly.get("pressure_msl", []),
            })

            df["hour"] = [i % 24 for i in range(len(df))]
            df["day_of_year"] = [(datetime.now() - timedelta(days=len(df) - i - 1)).timetuple().tm_yday for i in range(len(df))]
            df["month"] = [(datetime.now() - timedelta(days=len(df) - i - 1)).month for i in range(len(df))]

            for lag in [1, 3, 6, 12, 24]:
                df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

            df = df.dropna()
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self.generate_synthetic_data()

    def generate_synthetic_data(self):
        np.random.seed(42)
        n_samples = 1000
        base_temp = 28
        hours = np.arange(n_samples)

        daily_cycle = 5 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2)
        seasonal = 3 * np.sin(2 * np.pi * (hours % 8760) / 8760)
        noise = np.random.normal(0, 1, n_samples)
        temperature = base_temp + daily_cycle + seasonal + noise

        df = pd.DataFrame({
            "temperature": temperature,
            "humidity": 70 + 15 * np.sin(2 * np.pi * hours / 48) + np.random.normal(0, 5, n_samples),
            "precipitation": np.random.exponential(0.5, n_samples),
            "wind_speed": 10 + 5 * np.random.randn(n_samples),
            "pressure": 1010 + 5 * np.random.randn(n_samples),
            "hour": hours % 24,
            "day_of_year": hours % 365,
            "month": (hours % 365) // 30 + 1,
        })

        for lag in [1, 3, 6, 12, 24]:
            df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

        df = df.dropna()
        return df

    def train_model(self, location_name="Jakarta", latitude=-6.2, longitude=106.816666):
        print(f"📊 Mengambil data historis untuk {location_name}...")
        df = self.fetch_historical_data(latitude, longitude)

        feature_cols = [
            "humidity", "precipitation", "wind_speed", "pressure", "hour",
            "day_of_year", "month", "temp_lag_1", "temp_lag_3", "temp_lag_6",
            "temp_lag_12", "temp_lag_24"
        ]

        X = df[feature_cols]
        y = df["temperature"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("🤖 Melatih model Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model selesai dilatih!")
        print(f"   📈 Mean Absolute Error: {mae:.2f}°C")
        print(f"   📊 R² Score: {r2:.3f}")

        joblib.dump({
            "model": self.model, "features": feature_cols, "feature_names": feature_cols,
            "mae": mae, "r2": r2, "location": location_name,
        }, MODEL_PATH)

        self.features = feature_cols
        return {"mae": round(mae, 6), "r2": round(r2, 6), "features": feature_cols}

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                saved = joblib.load(MODEL_PATH)
                self.model = saved["model"]
                self.features = saved["features"]
                print(f"✅ Model dimuat dari {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"⚠️ Gagal memuat model: {e}")
                try:
                    os.remove(MODEL_PATH)
                    print("🗑️ File model corrupt dihapus")
                except:
                    pass
                return False
        return False

    def predict_temperature(self, current_weather):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return self.fallback_prediction(current_weather)

        predictions = []
        last_temps = []

        for i in range(6):
            hour = (datetime.now().hour + i * 24) % 24

            if i == 0:
                current_temp = current_weather.get("temperature", 28)
                last_temps = [current_temp] * 25
            else:
                current_temp = predictions[-1]["temp_max"]

            features_dict = {
                "humidity": current_weather.get("humidity", 70),
                "precipitation": current_weather.get("precipitation", 0),
                "wind_speed": current_weather.get("wind_speed", 10),
                "pressure": current_weather.get("pressure", 1010),
                "hour": hour,
                "day_of_year": (datetime.now() + timedelta(days=i)).timetuple().tm_yday,
                "month": (datetime.now() + timedelta(days=i)).month,
                "temp_lag_1": last_temps[-1] if len(last_temps) >= 1 else current_temp,
                "temp_lag_3": last_temps[-3] if len(last_temps) >= 3 else current_temp,
                "temp_lag_6": last_temps[-6] if len(last_temps) >= 6 else current_temp,
                "temp_lag_12": last_temps[-12] if len(last_temps) >= 12 else current_temp,
                "temp_lag_24": last_temps[-24] if len(last_temps) >= 24 else current_temp,
            }

            feature_array = np.array([[features_dict[col] for col in self.features]])
            pred_temp = self.model.predict(feature_array)[0]

            hour_factor = 2 * np.sin(2 * np.pi * (hour - 14) / 24)
            final_temp = pred_temp + hour_factor

            predictions.append({
                "day": (datetime.now() + timedelta(days=i)).strftime("%a"),
                "date": (datetime.now() + timedelta(days=i)).strftime("%d/%m"),
                "temp_max": round(final_temp + 2, 1),
                "temp_min": round(final_temp - 2, 1),
                "precipitation": round(max(0, np.random.exponential(0.5)), 1),
                "weather_code": 0,
                "uv_index": round(5 + 3 * np.sin(i), 1),
                "is_ml": True,
            })

            last_temps.append(final_temp)
            if len(last_temps) > 25:
                last_temps.pop(0)

        return predictions

    def fallback_prediction(self, current_weather):
        current_temp = current_weather.get("temperature", 28)
        predictions = []

        for i in range(6):
            trend = i * 0.3
            daily_var = 2 * np.sin(2 * np.pi * i / 6)
            pred_temp = current_temp + trend + daily_var

            predictions.append({
                "day": (datetime.now() + timedelta(days=i)).strftime("%a"),
                "date": (datetime.now() + timedelta(days=i)).strftime("%d/%m"),
                "temp_max": round(pred_temp + 2, 1),
                "temp_min": round(pred_temp - 2, 1),
                "precipitation": round(max(0, np.random.exponential(0.5)), 1),
                "weather_code": 0,
                "uv_index": round(5 + 3 * np.sin(i), 1),
                "is_ml": False,
            })

        return predictions

    def get_model_info(self):
        if os.path.exists(MODEL_PATH):
            try:
                saved = joblib.load(MODEL_PATH)
                return {
                    "is_trained": True,
                    "mae": saved.get("mae", "N/A"),
                    "r2": saved.get("r2", "N/A"),
                    "location": saved.get("location", "Unknown"),
                }
            except:
                return {"is_trained": False}
        return {"is_trained": False}

weather_predictor = WeatherPredictor()

IMG_SIZE = (128, 128)
CLASS_NAMES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]
MODEL_CKPT_PATH = "weather_cnn_model.keras"

class WeatherImageClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

    def build_model(self):
        if not TF_AVAILABLE:
            return None
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_model(self, dataset_path: str, epochs=15):
        if not TF_AVAILABLE:
            return {"error": "TensorFlow tidak tersedia"}

        print(f"📂 Melatih model dari dataset: {dataset_path}")

        train_datagen = ImageDataGenerator(
            rescale=1.0/255, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=32,
            class_mode="categorical", subset="training", classes=CLASS_NAMES
        )

        validation_generator = train_datagen.flow_from_directory(
            dataset_path, target_size=IMG_SIZE, batch_size=32,
            class_mode="categorical", subset="validation", classes=CLASS_NAMES
        )

        self.model = self.build_model()

        history = self.model.fit(
            train_generator, validation_data=validation_generator,
            epochs=epochs, verbose=1
        )

        self.model.save(MODEL_CKPT_PATH)
        print(f"✅ Model CNN disimpan ke {MODEL_CKPT_PATH}")

        return {
            "accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
        }

    def load_model(self):
        if not TF_AVAILABLE:
            return False
        if Path(MODEL_CKPT_PATH).exists():
            try:
                self.model = models.load_model(MODEL_CKPT_PATH)
                print(f"✅ Model CNN dimuat dari {MODEL_CKPT_PATH}")
                return True
            except Exception as e:
                print(f"⚠️ Gagal memuat model CNN: {e}")
                return False
        return False

    def predict_image(self, image_bytes: bytes) -> dict:
        if not TF_AVAILABLE:
            return {"error": "TensorFlow tidak tersedia", "prediction": None, "confidence": 0}

        if self.model is None:
            self.load_model()
            if self.model is None:
                return {"error": "Model CNN belum dilatih. Jalankan training terlebih dahulu.", "prediction": None, "confidence": 0}

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Gambar tidak valid", "prediction": None, "confidence": 0}
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            predictions = self.model.predict(img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])

            condition_map = {
                "cloudy": "Berawan", "foggy": "Kabut", "rainy": "Hujan",
                "shine": "Cerah", "sunrise": "Matahari Terbit"
            }
            weather_code_map = {"cloudy": 2, "foggy": 45, "rainy": 61, "shine": 0, "sunrise": 0}

            pred_class = CLASS_NAMES[class_idx]

            return {
                "prediction": pred_class,
                "condition": condition_map.get(pred_class, pred_class),
                "confidence": round(confidence * 100, 2),
                "weather_code": weather_code_map.get(pred_class, 0),
                "all_scores": {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
            }
        except Exception as e:
            return {"error": str(e), "prediction": None, "confidence": 0}

weather_image_classifier = WeatherImageClassifier()

def get_ai_insights_fallback(weather, forecast, air_quality, location_name: str = None):
    temp = weather.get("temperature", 0)
    feels_like = weather.get("feels_like", 0)
    humidity = weather.get("humidity", 0)
    precip = weather.get("precipitation", 0)
    wind = weather.get("wind_speed", 0)
    weather_code = weather.get("weather_code", 0)
    uv = weather.get("uv_index", 5)
    pressure = weather.get("pressure", 1012)

    aqi = air_quality.get("aqi", 0)
    aqi_status = air_quality.get("status", "Baik")
    pm25 = air_quality.get("pm25", 0)
    pm10 = air_quality.get("pm10", 0)

    condition = get_condition_text(weather_code).lower()
    location = location_name or "Lokasi Anda"

    temps_next_days = [d["temp_max"] for d in forecast[:3]]
    max_temp_next = max(temps_next_days) if temps_next_days else temp
    min_temp_next = min([d["temp_min"] for d in forecast[:3]]) if temps_next_days else temp

    if condition == "cerah":
        p1 = f"Langit {location} sedang cerah tanpa awan berarti. Suhu saat ini {int(temp)}°C, namun karena kelembaban {int(humidity)}%, udara terasa lebih hangat yaitu {int(feels_like)}°C. Ini adalah cuaca yang cukup khas untuk wilayah ini."
    elif condition == "berawan":
        p1 = f"Hari ini {location} diliputi awan dengan suhu {int(temp)}°C. Rasa panasnya mencapai {int(feels_like)}°C karena kelembaban yang cukup tinggi ({int(humidity)}%). Meski berawan, sinar UV tetap bisa menembus awan."
    elif condition == "sebagian cerah":
        p1 = f"Cuaca di {location} cerah berawan, kombinasi antara sinar matahari dan awan tipis. Suhu udara {int(temp)}°C terasa seperti {int(feels_like)}°C. Kondisi ini sering terjadi di musim pancaroba."
    elif "hujan" in condition:
        p1 = f"Hujan {condition} sedang berlangsung di {location}. Suhu turun menjadi {int(temp)}°C dengan kelembaban mencapai {int(humidity)}%, membuat udara terasa lembab dan dingin seperti {int(feels_like)}°C."
    elif "kabut" in condition or weather_code == 45:
        p1 = f"Kabut menyelimuti {location} pagi ini dengan suhu {int(temp)}°C. Jarak pandang mungkin berkurang, jadi hati-hati jika berkendara. Kabut biasanya akan berkurang setelah jam 9 pagi."
    else:
        p1 = f" Cuaca {location} hari ini {condition} dengan suhu {int(temp)}°C. Kelembaban {int(humidity)}% membuat suhu terasa {int(feels_like)}°C. Tekanan udara tercatat {int(pressure)} hPa."

    if precip > 5:
        p2 = f" Dalam 24 jam terakhir, tercurah hujan lebat mencapai {precip:.1f} mm. Jalanan mungkin tergenang di beberapa titik. Angin bertiup dengan kecepatan {int(wind)} km/jam, cukup kencang dan mempercepat penguapan."
    elif precip > 1:
        p2 = f" Tercatat hujan ringan sebesar {precip:.1f} mm. Tidak terlalu mengganggu, namun tetap waspada karena permukaan jalan bisa licin. Angin bertiup sekitar {int(wind)} km/jam, masih tergolong normal."
    elif precip > 0:
        p2 = f" Ada gerimis tipis dengan curah hanya {precip:.1f} mm. Hampir tidak terasa. Angin bertiup pelan {int(wind)} km/jam, memberikan sirkulasi udara yang nyaman."
    else:
        if wind > 15:
            p2 = f" Tidak ada hujan yang tercatat. Angin bertiup cukup kencang {int(wind)} km/jam, membuat udara terasa lebih segar meski suhu cukup hangat."
        elif wind > 5:
            p2 = f" Sepanjang hari tidak ada hujan. Angin bertiup sepoi-sepoi dengan kecepatan {int(wind)} km/jam, sangat nyaman untuk aktivitas luar."
        else:
            p2 = f" Langit cerih tanpa hujan. Angin hampir tidak terasa ({int(wind)} km/jam), membuat udara terasa sedikit pengap terutama di daerah padat."

    if uv > 10:
        uv_part = f" Indeks UV sangat ekstrim ({uv:.1f})! Paparan sinar matahari langsung bisa membakar kulit dalam 10 menit."
    elif uv > 8:
        uv_part = f" Indeks UV sangat tinggi ({uv:.1f}). Gunakan tabir surya SPF 30+ jika beraktivitas di luar."
    elif uv > 6:
        uv_part = f" Indeks UV tinggi ({uv:.1f}). Meski tidak ekstrem, perlindungan tetap disarankan."
    elif uv > 3:
        uv_part = f" Indeks UV sedang ({uv:.1f}). Masih aman untuk beraktivitas normal."
    else:
        uv_part = f" Indeks UV rendah ({uv:.1f}). Sinar matahari tidak terlalu berbahaya saat ini."

    if aqi <= 50:
        aqi_part = f" Kabar baik! Kualitas udara di {location} masuk kategori BAIK dengan AQI {aqi}. Partikel halus PM2.5 tercatat {pm25} µg/m³ dan PM10 {pm10} µg/m³, masih di bawah batas aman WHO. Udara segar dan cocok untuk olahraga luar ruangan."
    elif aqi <= 100:
        aqi_part = f" Kualitas udara masuk kategori SEDANG (AQI {aqi}). PM2.5 {pm25} µg/m³ dan PM10 {pm10} µg/m³. Masih aman untuk umum, namun penderita asma atau alergi sebaiknya tidak beraktivitas berat di luar terlalu lama."
    elif aqi <= 150:
        aqi_part = f" Perlu perhatian! Kualitas udara TIDAK SEHAT untuk kelompok sensitif (AQI {aqi}). PM2.5 mencapai {pm25} µg/m³ yang bisa memicu iritasi saluran napas. Kelompok rentan seperti anak-anak, lansia, dan penderita penyakit paru-paru disarankan menggunakan masker saat keluar rumah."
    elif aqi <= 200:
        aqi_part = f" Peringatan! Kualitas udara TIDAK SEHAT (AQI {aqi}) dengan PM2.5 {pm25} µg/m³. Semua orang disarankan mengurangi aktivitas luar ruangan. Tutup jendela rumah dan gunakan penyaring udara jika tersedia."
    elif aqi <= 300:
        aqi_part = f" Kondisi darurat! Kualitas udara SANGAT TIDAK SEHAT (AQI {aqi}). PM2.5 {pm25} µg/m³ sangat berbahaya bagi kesehatan. Hindari keluar rumah, gunakan masker N95 jika terpaksa."
    else:
        aqi_part = f" BAHAYA! Kualitas udara BERBAHAYA (AQI {aqi}). Segera cari perlindungan di dalam ruangan dengan filtrasi udara yang baik. Ikuti arahan dari otoritas setempat."

    p3 = f"{uv_part}{aqi_part}"

    if max_temp_next > temp + 4:
        trend = f" Dalam 3 hari ke depan, suhu diprediksi akan MELONJAK hingga {int(max_temp_next)}°C, bahkan lebih panas dari hari ini."
    elif max_temp_next > temp + 2:
        trend = f" Suhu diperkirakan akan naik bertahap hingga mencapai {int(max_temp_next)}°C dalam beberapa hari ke depan."
    elif max_temp_next < temp - 2:
        trend = f" Ada penurunan suhu cukup signifikan hingga {int(min_temp_next)}°C dalam beberapa hari ke depan. Siapkan jaket atau selimut tambahan."
    else:
        trend = f" Suhu dalam 3-5 hari ke depan cenderung stabil di kisaran {int(temp-1)}-{int(temp+2)}°C, tidak banyak perubahan ekstrem."

    rainy_days = [d for d in forecast[:3] if d.get("precipitation", 0) > 5]
    if rainy_days:
        p4 = f"{trend} Waspada potensi hujan sedang hingga lebat pada hari {', '.join([d['day'] for d in rainy_days[:2]])}. Bawa payung atau jas hujan jika bepergian."
    elif any(d.get("precipitation", 0) > 2 for d in forecast[:3]):
        light_rain_days = [d["day"] for d in forecast[:3] if 2 < d.get("precipitation", 0) <= 5]
        if light_rain_days:
            p4 = f"{trend} Ada kemungkinan gerimis atau hujan ringan pada hari {', '.join(light_rain_days[:2])}. Tidak terlalu mengganggu, tapi tetap sedia payung lipat."
    else:
        p4 = f"{trend} Diprediksi tidak ada hujan signifikan dalam 3 hari ke depan, cocok untuk merencanakan kegiatan luar ruangan."

    return f"{p1}{p2} {p3} {p4}"

def get_ai_insights_real(weather, forecast, air_quality, location_name: str = None):
    if not AI_AVAILABLE or client is None:
        print("⚠️ AI tidak tersedia, menggunakan fallback")
        return get_ai_insights_fallback(weather, forecast, air_quality, location_name)

    location = location_name or "Lokasi Anda"
    temp = weather.get("temperature", 0)
    feels_like = weather.get("feels_like", 0)
    humidity = weather.get("humidity", 0)
    precip = weather.get("precipitation", 0)
    wind = weather.get("wind_speed", 0)
    weather_code = weather.get("weather_code", 0)
    uv = weather.get("uv_index", 5)
    pressure = weather.get("pressure", 1012)
    condition = get_condition_text(weather_code)

    aqi = air_quality.get("aqi", 0)
    aqi_status = air_quality.get("status", "Baik")
    pm25 = air_quality.get("pm25", 0)
    pm10 = air_quality.get("pm10", 0)

    forecast_summary = []
    for i, day in enumerate(forecast[:3]):
        day_name = "Hari ini" if i == 0 else "Besok" if i == 1 else day["day"]
        forecast_summary.append(
            f"- {day_name}: suhu {int(day['temp_max'])}°C / {int(day['temp_min'])}°C, "
            f"hujan {int(day['precipitation'])}mm, UV {day['uv_index']:.1f}"
        )
    forecast_text = "\n".join(forecast_summary)

    prompt = f"""Kamu adalah meteorolog yang ramah. Buat deskripsi cuaca dan kualitas udara untuk {location} dalam 4-5 kalimat (sekitar 100-150 kata). Gaya natural seperti sedang ngobrol.

DATA CUACA:
- Suhu: {int(temp)}°C (terasa {int(feels_like)}°C)
- Kondisi: {condition}
- Kelembaban: {int(humidity)}%
- Curah hujan: {precip:.1f} mm
- Angin: {int(wind)} km/jam
- Tekanan: {int(pressure)} hPa
- UV: {uv:.1f}

KUALITAS UDARA:
- AQI: {aqi} ({aqi_status})
- PM2.5: {pm25} µg/m³
- PM10: {pm10} µg/m³

PRAKIRAAN 3 HARI:
{forecast_text}

PANDUAN:
1. Bahasa Indonesia yang natural dan mengalir
2. Sertakan informasi kualitas udara secara alami
3. Beri konteks (misal: "suhu 32°C cukup panas untuk Jakarta")
4. Jika ada kondisi ekstrem (UV tinggi, AQI buruk, hujan lebat), sebutkan dengan bijak
5. Akhiri dengan 1 kalimat prakiraan singkat untuk besok
6. JANGAN gunakan bullet points, JANGAN terlalu panjang (maks 150 kata)
7. JANGAN gunakan emoji berlebihan, cukup 1-2 saja jika perlu

Mulai menulis:"""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        insights = response.text.strip()
        print(f"✅ Gemini response received for {location} (panjang: {len(insights.split())} kata)")

        word_count = len(insights.split())
        if word_count < 30 or word_count > 200:
            print(f"⚠️ Response tidak ideal ({word_count} kata), menggunakan fallback")
            return get_ai_insights_fallback(weather, forecast, air_quality, location_name)

        return insights
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return get_ai_insights_fallback(weather, forecast, air_quality, location_name)

app.mount("/static", StaticFiles(directory="static"), name="static")

selected_location = {
    "name": "Jakarta",
    "latitude": -6.2,
    "longitude": 106.816666,
    "timezone": "Asia/Jakarta",
}

# ============ FUNGSI RENDER PAGE ============
def render_page(content: str, active: str = "home", message: str = None, message_type: str = None, saved_locations: list = None, selected_location: dict = None):
    message_html = ""
    if message:
        icon = "check-circle" if message_type == "success" else "exclamation-circle"
        message_html = f'<div class="flash flash-{message_type}"><i class="fas fa-{icon}"></i> {message}</div>'

    sidebar_locations_html = ""
    if saved_locations:
        for loc in saved_locations:
            location_name = loc["name"]
            is_coords = location_name.startswith("Koordinat (")
            display_country = loc["country"] if loc["country"] else ("Koordinat" if is_coords else "Indonesia")

            sidebar_locations_html += f"""
            <div class="location-item" onclick="window.location.href='/select-location/{loc["id"]}'">
                <div class="location-info">
                    <i class="fas fa-map-marker-alt"></i>
                    <div>
                        <div class="location-name">{location_name}</div>
                        <div class="location-country">{display_country}</div>
                        <div class="location-coords">
                            <i class="fas fa-crosshairs"></i> {loc["latitude"]:.3f}, {loc["longitude"]:.3f}
                        </div>
                    </div>
                </div>
                <button class="delete-btn" onclick="event.stopPropagation(); window.location.href='/delete-location/{loc["id"]}'">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
            """
    else:
        sidebar_locations_html = '<div style="text-align:center;padding:32px;color:var(--text-tertiary);font-size:13px;"><i class="fas fa-star" style="font-size:32px;margin-bottom:12px;display:block;opacity:0.5;"></i>Belum ada lokasi tersimpan</div>'

    active_home = "active" if active == "home" else ""
    active_ml = "active" if active == "ml" else ""
    active_ulasan = "active" if active == "ulasan" else ""
    active_about = "active" if active == "about" else ""

    location_data = ""
    if selected_location:
        tz = selected_location.get("timezone", "Asia/Jakarta")
        location_data = f"""
        <script>
            window.currentLocation = {{
                lat: {selected_location.get("latitude", -6.2)},
                lng: {selected_location.get("longitude", 106.816666)},
                timezone: "{tz}",
                name: "{selected_location.get("name", "Jakarta")}"
            }};
        </script>
        """

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <meta name="theme-color" content="#0f172a">
    <meta name="description" content="WeatherAI - Aplikasi Prediksi Cuaca Cerdas Berbasis AI dengan akurasi tinggi. Informasi cuaca real-time, prakiraan 6 hari, dan analisis AI.">
    <meta name="keywords" content="cuaca, prediksi cuaca, weather, AI, weather forecast, Indonesia, gemini AI, machine learning">
    <meta name="author" content="WeatherAI">
    <title>WeatherAI | Aplikasi Prediksi Cuaca Cerdas Berbasis AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link rel="stylesheet" href="/static/styles.css">
    <style>
        /* ============ CHATBOT CSS INLINE ============ */
        
        /* Chat Toggle Button */
        .chat-toggle {{
            position: fixed;
            bottom: 28px;
            right: 100px;
            width: 52px;
            height: 52px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #a855f7);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 200;
            transition: all 0.3s ease;
            color: white;
            font-size: 24px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            border: none;
        }}
        .chat-toggle:hover {{
            transform: scale(1.1);
        }}

        /* Chat Bubble */
        .chat-bubble {{
            position: fixed;
            bottom: 100px;
            right: 28px;
            width: 380px;
            max-width: calc(100vw - 56px);
            height: 500px;
            max-height: calc(100vh - 140px);
            background: white;
            border-radius: 28px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
            z-index: 199;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transform: scale(0);
            opacity: 0;
            transform-origin: bottom right;
            transition: transform 0.3s ease, opacity 0.3s ease;
            visibility: hidden;
        }}
        body.dark .chat-bubble {{
            background: #1e293b;
        }}
        .chat-bubble.open {{
            transform: scale(1);
            opacity: 1;
            visibility: visible;
        }}

        /* Chat Header */
        .chat-header {{
            padding: 16px 20px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #a855f7);
            display: flex;
            align-items: center;
            gap: 12px;
            color: white;
            flex-shrink: 0;
        }}
        .chat-header-icon {{
            width: 36px;
            height: 36px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }}
        .chat-header-info {{
            flex: 1;
        }}
        .chat-header-info h4 {{
            font-size: 16px;
            font-weight: 700;
            margin: 0;
        }}
        .chat-header-info p {{
            font-size: 11px;
            opacity: 0.8;
            margin: 2px 0 0;
        }}
        .chat-close {{
            width: 30px;
            height: 30px;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .chat-close:hover {{
            background: rgba(255,255,255,0.3);
            transform: rotate(90deg);
        }}

        /* Chat Messages */
        .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .chat-message {{
            display: flex;
            gap: 10px;
            max-width: 85%;
        }}
        .chat-message.bot {{
            align-self: flex-start;
        }}
        .chat-message.user {{
            align-self: flex-end;
            flex-direction: row-reverse;
        }}
        .chat-avatar {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .chat-message.bot .chat-avatar {{
            background: linear-gradient(135deg, #8b5cf6, #a855f7);
            color: white;
        }}
        .chat-message.user .chat-avatar {{
            background: #3b82f6;
            color: white;
        }}
        .chat-bubble-text {{
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 13px;
            line-height: 1.5;
            word-wrap: break-word;
        }}
        .chat-message.bot .chat-bubble-text {{
            background: #f1f5f9;
            color: #0f172a;
            border-top-left-radius: 4px;
        }}
        body.dark .chat-message.bot .chat-bubble-text {{
            background: #334155;
            color: #f1f5f9;
        }}
        .chat-message.user .chat-bubble-text {{
            background: #3b82f6;
            color: white;
            border-top-right-radius: 4px;
        }}

        /* Typing Indicator */
        .chat-typing {{
            display: flex;
            gap: 4px;
            padding: 10px 14px;
            background: #f1f5f9;
            border-radius: 18px;
            border-top-left-radius: 4px;
            width: fit-content;
        }}
        body.dark .chat-typing {{
            background: #334155;
        }}
        .chat-typing span {{
            width: 8px;
            height: 8px;
            background: #94a3b8;
            border-radius: 50%;
            animation: typingBounce 1.4s infinite ease-in-out;
        }}
        body.dark .chat-typing span {{
            background: #cbd5e1;
        }}
        @keyframes typingBounce {{
            0%,60%,100% {{ transform: translateY(0); opacity: 0.5; }}
            30% {{ transform: translateY(-8px); opacity: 1; }}
        }}
        .chat-typing span:nth-child(1) {{ animation-delay: 0s; }}
        .chat-typing span:nth-child(2) {{ animation-delay: 0.2s; }}
        .chat-typing span:nth-child(3) {{ animation-delay: 0.4s; }}

        /* Chat Input */
        .chat-input-area {{
            padding: 16px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 10px;
            flex-shrink: 0;
        }}
        body.dark .chat-input-area {{
            border-top-color: #334155;
        }}
        .chat-input-area input {{
            flex: 1;
            padding: 12px 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 28px;
            color: #0f172a;
            font-family: inherit;
            font-size: 13px;
        }}
        body.dark .chat-input-area input {{
            background: #1e293b;
            border-color: #475569;
            color: #f1f5f9;
        }}
        .chat-input-area input:focus {{
            outline: none;
            border-color: #3b82f6;
        }}
        .chat-input-area button {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #a855f7);
            border: none;
            border-radius: 50%;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chat-input-area button:hover {{
            transform: scale(1.05);
        }}
        .chat-input-area button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        /* Chat Scrollbar */
        .chat-messages::-webkit-scrollbar {{
            width: 4px;
        }}
        .chat-messages::-webkit-scrollbar-track {{
            background: #e2e8f0;
            border-radius: 10px;
        }}
        .chat-messages::-webkit-scrollbar-thumb {{
            background: #3b82f6;
            border-radius: 10px;
        }}
        body.dark .chat-messages::-webkit-scrollbar-track {{
            background: #1e293b;
        }}

        @media (max-width: 768px) {{
            .chat-toggle {{
                bottom: 20px;
                right: 80px;
                width: 44px;
                height: 44px;
                font-size: 20px;
            }}
            .chat-bubble {{
                width: calc(100vw - 40px);
                right: 20px;
                bottom: 80px;
                max-height: calc(100vh - 100px);
            }}
        }}
        
        @media (max-width: 480px) {{
            .chat-bubble {{
                width: calc(100vw - 32px);
                right: 16px;
                bottom: 76px;
                border-radius: 24px;
            }}
            .chat-header {{
                padding: 12px 16px;
            }}
            .chat-message {{
                max-width: 90%;
            }}
        }}

        /* Custom Alert Styles */
        .custom-alert {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) scale(0.9);
            background: white;
            border-radius: 28px;
            padding: 28px 32px;
            max-width: 400px;
            width: 90%;
            z-index: 20000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            text-align: center;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
        }}
        body.dark .custom-alert {{
            background: #1e293b;
        }}
        .custom-alert.show {{
            opacity: 1;
            visibility: visible;
            transform: translate(-50%, -50%) scale(1);
        }}
        .custom-alert.error {{ border-top: 4px solid #ef4444; }}
        .custom-alert.success {{ border-top: 4px solid #10b981; }}
        .custom-alert.warning {{ border-top: 4px solid #f59e0b; }}
        .custom-alert.info {{ border-top: 4px solid #3b82f6; }}
        .custom-alert-icon {{ font-size: 56px; margin-bottom: 16px; }}
        .custom-alert-title {{ font-size: 20px; font-weight: 800; margin-bottom: 12px; }}
        .custom-alert-message {{ font-size: 14px; line-height: 1.6; margin-bottom: 24px; }}
        .custom-alert-buttons {{ display: flex; gap: 12px; justify-content: center; }}
        .custom-alert-btn {{ padding: 10px 24px; border-radius: 40px; font-weight: 600; cursor: pointer; border: none; background: #e2e8f0; }}
        .custom-alert-btn.primary {{ background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; }}
        .custom-alert-close {{ position: absolute; top: 16px; right: 20px; background: none; border: none; cursor: pointer; font-size: 18px; }}
        .custom-alert-input {{ width: 100%; padding: 12px 16px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 16px; margin-bottom: 20px; }}
        body.dark .custom-alert-input {{ background: #334155; border-color: #475569; color: white; }}
    </style>
    {location_data}
</head>
<body>
    <div class="loader-wrapper" id="loaderWrapper">
        <div class="loader">
            <div class="cloud-loader">
                <i class="fas fa-cloud-sun"></i>
            </div>
            <div class="loader-text">
                WeatherAI<span class="loader-dots"></span>
            </div>
        </div>
    </div>

    <div class="modal" id="trainingModal">
        <div class="modal-content">
            <div class="modal-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h3 class="modal-title">Melatih Model ML</h3>
            <p class="modal-message">Sedang melatih Random Forest Regressor dengan data historis...</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p class="modal-message" style="font-size: 12px;" id="trainingStatus">Mengambil data cuaca...</p>
        </div>
    </div>

    <div class="whatsapp-notification" id="whatsappNotif" onclick="openWhatsApp()">
        <div class="notification-close" onclick="event.stopPropagation(); closeNotificationPermanently()">
            <i class="fas fa-times"></i>
        </div>
        <div class="notification-content">
            <div class="notification-image">
                <img src="https://aryamods.rf.gd/images/iconashley.png" alt="Profile">
            </div>
            <div class="notification-text">
                <div class="notification-title">
                    Ashley
                </div>
                <div class="notification-desc">
                    Hai! Butuh bantuan? Klik di sini untuk chat langsung dengan Ashley di WhatsApp!
                </div>
            </div>
        </div>
    </div>

    <div class="modal-tim" id="modalTim">
        <div class="modal-tim-content">
            <div class="modal-tim-close" onclick="closeModalTim()">
                <i class="fas fa-times"></i>
            </div>

            <div class="modal-header">
                <div class="modal-header-icon">
                    <i class="fas fa-users"></i>
                </div>
                <h2 class="modal-title-glow">Tim Kami</h2>
                <p class="modal-subtitle">Dibangun dengan profesionalisme, kreativitas, dan inovasi teknologi untuk menghadirkan pengalaman prediksi cuaca yang lebih cerdas, modern, dan akurat</p>
            </div>

            <div class="team-grid-enhanced">
                <div class="team-card-enhanced">
                    <div class="card-glow"></div>
                    <div class="team-avatar-enhanced">
                        <img src="/static/images/arya.png" alt="WeatherAI Property">
                    </div>
                    <div class="team-info">
                        <h3 class="team-name-enhanced">Arya Satya</h3>
                        <p class="team-role">Lead Dev Engineer</p>
                        <div class="team-nim-enhanced">
                            <i class="fas fa-id-card"></i> 15240231
                        </div>
                        <div class="team-social">
                            <a href="https://wa.me/6288212733727" class="social-icon"><i class="fab fa-whatsapp"></i></a>
                            <a href="https://www.instagram.com/aryamods" class="social-icon"><i class="fab fa-instagram"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-card-enhanced">
                    <div class="card-glow"></div>
                    <div class="team-avatar-enhanced">
                        <img src="/static/images/kevin.jpeg" alt="WeatherAI Property">
                    </div>
                    <div class="team-info">
                        <h3 class="team-name-enhanced">Kevin Handerson</h3>
                        <p class="team-role">UI/UX Designer</p>
                        <div class="team-nim-enhanced">
                            <i class="fas fa-id-card"></i> 15240235
                        </div>
                        <div class="team-social">
                            <a href="https://wa.me/6283801011417" class="social-icon"><i class="fab fa-whatsapp"></i></a>
                            <a href="https://www.instagram.com/kevin071451?utm_source=qr&igsh=ZHExbTB1ZDF3aXJu" class="social-icon"><i class="fab fa-instagram"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-card-enhanced">
                    <div class="card-glow"></div>
                    <div class="team-avatar-enhanced">
                        <img src="/static/images/yoseph.jpeg" alt="WeatherAI Property">
                    </div>
                    <div class="team-info">
                        <h3 class="team-name-enhanced">Yosep Wai</h3>
                        <p class="team-role">Content Writer</p>
                        <div class="team-nim-enhanced">
                            <i class="fas fa-id-card"></i> 15240476
                        </div>
                        <div class="team-social">
                            <a href="https://wa.me/6281238606470" class="social-icon"><i class="fab fa-whatsapp"></i></a>
                            <a href="https://www.instagram.com/p/DSICYHjD-4M/?igsh=MW9yaXVqbGJlendvYw==" class="social-icon"><i class="fab fa-instagram"></i></a>
                        </div>
                    </div>
                </div>
                <div class="team-card-enhanced">
                    <div class="card-glow"></div>
                    <div class="team-avatar-enhanced">
                        <img src="/static/images/stenjo.jpeg" alt="WeatherAI Property">
                    </div>
                    <div class="team-info">
                        <h3 class="team-name-enhanced">Stanislaus Ratu</h3>
                        <p class="team-role">Research Writer</p>
                        <div class="team-nim-enhanced">
                            <i class="fas fa-id-card"></i> 15240782
                        </div>
                        <div class="team-social">
                            <a href="https://wa.me/6281237325309" class="social-icon"><i class="fab fa-whatsapp"></i></a>
                            <a href="https://www.instagram.com/sstenjo?igsh=MW0xbXF0eW8xdmM2Zg==" class="social-icon"><i class="fab fa-instagram"></i></a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="modal-footer-stats">
                <div class="stat-item">
                    <i class="fas fa-graduation-cap"></i>
                    <span>Universitas Bina Sarana Informatika</span>
                </div>
                <div class="stat-item">
                    <i class="fas fa-coffee"></i>
                    <span>Cengkareng</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Custom Alert Modal -->
    <div id="customAlert" class="custom-alert">
        <button class="custom-alert-close" onclick="closeCustomAlert()">
            <i class="fas fa-times"></i>
        </button>
        <div class="custom-alert-icon">
            <i class="fas fa-info-circle"></i>
        </div>
        <div class="custom-alert-title" id="alertTitle">Peringatan</div>
        <div class="custom-alert-message" id="alertMessage">Pesan</div>
        <div class="custom-alert-buttons" id="alertButtons">
            <button class="custom-alert-btn primary" onclick="closeCustomAlert()">OK</button>
        </div>
    </div>

    <!-- Custom Prompt Modal -->
    <div id="customPrompt" class="custom-alert">
        <button class="custom-alert-close" onclick="closeCustomPrompt()">
            <i class="fas fa-times"></i>
        </button>
        <div class="custom-alert-icon">
            <i class="fas fa-lock"></i>
        </div>
        <div class="custom-alert-title" id="promptTitle">Verifikasi Akses</div>
        <div class="custom-alert-message" id="promptMessage">masukkan kata kunci untuk menghapus komentar</div>
        <input type="password" id="promptInput" class="custom-alert-input" placeholder="Masukkan kata kunci...">
        <div class="custom-alert-buttons" id="promptButtons">
            <button class="custom-alert-btn" onclick="closeCustomPrompt()">Batal</button>
            <button class="custom-alert-btn primary" onclick="submitPrompt()">Verifikasi</button>
        </div>
    </div>

    <div class="aura-bg">
        <div class="aura-glow"></div>
    </div>

    <!-- Chat AI Ashley Button & Bubble -->
    <button class="chat-toggle" id="chatToggle" onclick="toggleChat()" aria-label="Chat AI Ashley">
        <i class="fas fa-comment-dots"></i>
    </button>

    <div class="chat-bubble" id="chatBubble">
        <div class="chat-header">
            <div class="chat-header-icon">
                <i class="fas fa-robot"></i>
            </div>
            <div class="chat-header-info">
                <h4>Ashley AI</h4>
                <p>Asisten Cuaca Cerdas</p>
            </div>
            <div class="chat-close" onclick="toggleChat()">
                <i class="fas fa-times"></i>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="chat-message bot">
                <div class="chat-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="chat-bubble-text">
                    Halo! Saya Ashley 👋<br>Saya bisa membantu Anda dengan informasi cuaca, prakiraan, analisis, dan tips terkait cuaca. Ada yang bisa saya bantu?
                </div>
            </div>
        </div>
        
        <div class="chat-input-area">
            <input type="text" id="chatInput" placeholder="Tanya tentang cuaca..." onkeypress="if(event.key === 'Enter') sendChatMessage()">
            <button id="chatSendBtn" onclick="sendChatMessage()">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>

    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle theme">
        <i class="fas fa-moon" id="themeIcon"></i>
    </button>

    <button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()" aria-label="Menu">
        <i class="fas fa-bars"></i>
    </button>

    <div class="app">
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-logo-icon">
                    <i class="fas fa-cloud-sun"></i>
                </div>
                <div>
                    <div class="sidebar-logo-text">WeatherAI</div>
                </div>
            </div>

            <div class="sidebar-content-wrapper">
                <nav class="sidebar-nav">
                    <a href="/" class="nav-item {active_home}" data-page="home">
                        <i class="fas fa-home"></i>
                        <span>Beranda</span>
                    </a>
                    <a href="/main" class="nav-item {active_ml}" data-page="ml">
                        <i class="fas fa-brain"></i>
                        <span>Main</span>
                    </a>
                    <a href="/ulasan" class="nav-item {active_ulasan}" data-page="ulasan">
                        <i class="fas fa-edit"></i>
                        <span>Tulis Ulasan</span>
                    </a>
                    <a href="/about" class="nav-item {active_about}" data-page="about">
                        <i class="fas fa-info-circle"></i>
                        <span>Tentang</span>
                    </a>
                </nav>

                <div class="sidebar-section">
                    <div class="sidebar-section-title">
                        <i class="fas fa-star"></i> Lokasi Tersimpan
                    </div>
                    <div class="sidebar-locations">
                        {sidebar_locations_html}
                    </div>
                    <a href="/search" style="display: block; margin-top: 20px; text-align: center; font-size: 13px; color: var(--accent); text-decoration: none; font-weight: 600;">
                        <i class="fas fa-plus-circle"></i> Tambah Lokasi
                    </a>
                </div>
            </div>
        </aside>

        <main class="main page-transition" id="mainContent">
            {message_html}
            {content}
            <div class="footer">
                <p>© 2026 WeatherAI · <a onclick="showModalTim()">Learn More</a></p>
            </div>
        </main>
    </div>

<script>
    window.addEventListener('load', function() {{
        setTimeout(function() {{
            var loader = document.getElementById('loaderWrapper');
            if (loader) loader.classList.add('hide');
        }}, 500);
    }});

    document.querySelectorAll('.nav-item').forEach(function(link) {{
        link.addEventListener('click', function(e) {{
            e.preventDefault();
            var href = this.getAttribute('href');
            var mainContent = document.getElementById('mainContent');
            if (mainContent) {{
                mainContent.style.opacity = '0';
                mainContent.style.transform = 'translateY(20px)';
            }}
            setTimeout(function() {{
                window.location.href = href;
            }}, 300);
        }});
    }});

    function showTrainingModal() {{
        var modal = document.getElementById('trainingModal');
        if (modal) modal.classList.add('active');

        var progressFill = document.getElementById('progressFill');
        var statusText = document.getElementById('trainingStatus');
        var progress = 0;

        var statuses = [
            'Mengambil data cuaca...',
            'Memproses fitur...',
            'Melatih Random Forest...',
            'Mengevaluasi model...',
            'Menyimpan model...'
        ];
        var statusIndex = 0;

        var interval = setInterval(function() {{
            progress += 2;
            if (progressFill) progressFill.style.width = progress + '%';

            if (progress % 20 === 0 && statusIndex < statuses.length - 1) {{
                statusIndex++;
                if (statusText) statusText.textContent = statuses[statusIndex];
            }}

            if (progress >= 100) {{
                clearInterval(interval);
            }}
        }}, 100);
    }}

    function hideTrainingModal() {{
        var modal = document.getElementById('trainingModal');
        if (modal) modal.classList.remove('active');
        var progressFill = document.getElementById('progressFill');
        if (progressFill) progressFill.style.width = '0%';
    }}

    var trainForms = document.querySelectorAll('form[action="/train-model"]');
    trainForms.forEach(function(form) {{
        form.addEventListener('submit', function(e) {{
            e.preventDefault();
            showTrainingModal();

            fetch('/train-model', {{
                method: 'GET',
                headers: {{
                    'X-Requested-With': 'XMLHttpRequest'
                }}
            }}).then(function(response) {{
                setTimeout(function() {{
                    hideTrainingModal();
                    window.location.href = '/main?message=✅ Model ML berhasil dilatih!&type=success';
                }}, 500);
            }}).catch(function(error) {{
                hideTrainingModal();
                window.location.href = '/main?message=❌ Gagal melatih model&type=error';
            }});
        }});
    }});

    function toggleTheme() {{
        var body = document.body;
        var icon = document.getElementById('themeIcon');

        if (body.classList.contains('dark')) {{
            body.classList.remove('dark');
            if (icon) {{
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }}
            localStorage.setItem('theme', 'light');
        }} else {{
            body.classList.add('dark');
            if (icon) {{
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }}
            localStorage.setItem('theme', 'dark');
        }}
        updateWhatsAppTheme();
    }}

    var savedTheme = localStorage.getItem('theme');
    var themeIcon = document.getElementById('themeIcon');

    if (savedTheme === 'dark') {{
        document.body.classList.add('dark');
        if (themeIcon) {{
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }}
    }}

    function toggleSidebar() {{
        var sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.toggle('open');
    }}

    document.addEventListener('click', function(event) {{
        var sidebar = document.getElementById('sidebar');
        var toggle = document.getElementById('menuToggle');
        if (window.innerWidth <= 768 && sidebar && toggle) {{
            if (!sidebar.contains(event.target) && !toggle.contains(event.target)) {{
                sidebar.classList.remove('open');
            }}
        }}
    }});

    var fixedOffsetMap = {{
        'Asia/Jayapura': 'UTC+9', 'Asia/Tokyo': 'UTC+9', 'Asia/Seoul': 'UTC+9',
        'Asia/Yakutsk': 'UTC+9', 'Pacific/Port_Moresby': 'UTC+9', 'Australia/Darwin': 'UTC+9:30',
        'Asia/Jakarta': 'UTC+7', 'Asia/Makassar': 'UTC+8', 'Asia/Singapore': 'UTC+8',
        'Asia/Bangkok': 'UTC+7', 'Asia/Ho_Chi_Minh': 'UTC+7', 'Asia/Shanghai': 'UTC+8',
        'Asia/Hong_Kong': 'UTC+8', 'Asia/Taipei': 'UTC+8', 'Australia/Perth': 'UTC+8',
        'Asia/Dili': 'UTC+9', 'Asia/Kolkata': 'UTC+5:30', 'Asia/Dubai': 'UTC+4',
        'Europe/London': 'UTC+0', 'Europe/Paris': 'UTC+1', 'Europe/Berlin': 'UTC+1',
        'America/New_York': 'UTC-5', 'America/Chicago': 'UTC-6', 'America/Denver': 'UTC-7',
        'America/Los_Angeles': 'UTC-8'
    }};

    var cachedOffset = null;
    var offsetInitialized = false;

    function updateRealTimeClock() {{
        var clockElement = document.getElementById('realtime-clock');
        var tzElement = document.getElementById('timezone-display');

        if (!clockElement) return;

        if (window.currentLocation && window.currentLocation.timezone) {{
            try {{
                var tzName = window.currentLocation.timezone;
                var options = {{
                    timeZone: tzName,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }};
                var formatter = new Intl.DateTimeFormat('id-ID', options);
                var timeStr = formatter.format(new Date());
                clockElement.textContent = timeStr;

                if (tzElement && !offsetInitialized) {{
                    offsetInitialized = true;
                    if (fixedOffsetMap[tzName]) {{
                        cachedOffset = fixedOffsetMap[tzName];
                    }} else {{
                        try {{
                            var now = new Date();
                            var utcDate = new Date(now.toLocaleString('en-US', {{ timeZone: 'UTC' }}));
                            var tzDate = new Date(now.toLocaleString('en-US', {{ timeZone: tzName }}));
                            var offsetHours = (tzDate - utcDate) / (1000 * 60 * 60);
                            var sign = offsetHours >= 0 ? '+' : '';
                            var offsetVal = Math.abs(offsetHours);
                            if (offsetVal % 1 !== 0) {{
                                var hours = Math.floor(offsetVal);
                                var minutes = (offsetVal % 1) * 60;
                                cachedOffset = 'UTC' + sign + hours + ':' + (minutes === 30 ? '30' : '00');
                            }} else {{
                                cachedOffset = 'UTC' + sign + offsetVal;
                            }}
                        }} catch(e) {{
                            cachedOffset = 'UTC';
                        }}
                    }}
                    tzElement.textContent = cachedOffset;
                }}
            }} catch(e) {{
                console.error('Timezone error:', e);
                var now = new Date();
                clockElement.textContent = now.toLocaleTimeString('id-ID');
                if (tzElement && !offsetInitialized) {{
                    offsetInitialized = true;
                    tzElement.textContent = cachedOffset || 'UTC+7';
                }}
            }}
        }} else if (clockElement) {{
            var now = new Date();
            clockElement.textContent = now.toLocaleTimeString('id-ID');
            if (tzElement && !offsetInitialized) {{
                offsetInitialized = true;
                tzElement.textContent = 'UTC+7';
            }}
        }}
    }}

    function initTimezone() {{
        if (window.currentLocation && window.currentLocation.timezone) {{
            var tzName = window.currentLocation.timezone;
            var tzElement = document.getElementById('timezone-display');
            var clockElement = document.getElementById('realtime-clock');

            if (tzElement) {{
                if (fixedOffsetMap[tzName]) {{
                    tzElement.textContent = fixedOffsetMap[tzName];
                }} else {{
                    try {{
                        var now = new Date();
                        var utcDate = new Date(now.toLocaleString('en-US', {{ timeZone: 'UTC' }}));
                        var tzDate = new Date(now.toLocaleString('en-US', {{ timeZone: tzName }}));
                        var offsetHours = (tzDate - utcDate) / (1000 * 60 * 60);
                        var sign = offsetHours >= 0 ? '+' : '';
                        var offsetVal = Math.abs(offsetHours);
                        if (offsetVal % 1 !== 0) {{
                            var hours = Math.floor(offsetVal);
                            var minutes = (offsetVal % 1) * 60;
                            tzElement.textContent = 'UTC' + sign + hours + ':' + (minutes === 30 ? '30' : '00');
                        }} else {{
                            tzElement.textContent = 'UTC' + sign + offsetVal;
                        }}
                    }} catch(e) {{
                        tzElement.textContent = 'UTC';
                    }}
                }}
                offsetInitialized = true;
            }}

            if (clockElement) {{
                try {{
                    var options = {{
                        timeZone: tzName,
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false
                    }};
                    var formatter = new Intl.DateTimeFormat('id-ID', options);
                    clockElement.textContent = formatter.format(new Date());
                }} catch(e) {{
                    clockElement.textContent = new Date().toLocaleTimeString('id-ID');
                }}
            }}
        }}
    }}

    setInterval(updateRealTimeClock, 1000);
    updateRealTimeClock();
    initTimezone();

    setTimeout(function() {{
        var flash = document.querySelector('.flash');
        if (flash) {{
            flash.style.opacity = '0';
            setTimeout(function() {{ if (flash) flash.remove(); }}, 500);
        }}
    }}, 3000);

    var notificationInterval;
    var isNotificationVisible = false;
    var isPermanentlyClosed = false;

    function showNotification() {{
        if (isPermanentlyClosed) return;
        var notif = document.getElementById('whatsappNotif');
        if (notif && !isNotificationVisible) {{
            notif.classList.add('show');
            isNotificationVisible = true;
            setTimeout(function() {{
                hideNotification();
            }}, 5000);
        }}
    }}

    function hideNotification() {{
        var notif = document.getElementById('whatsappNotif');
        if (notif && isNotificationVisible) {{
            notif.classList.remove('show');
            isNotificationVisible = false;
        }}
    }}

    function closeNotificationPermanently() {{
        isPermanentlyClosed = true;
        hideNotification();
        if (notificationInterval) {{
            clearInterval(notificationInterval);
            notificationInterval = null;
        }}
        localStorage.setItem('whatsappNotifClosed', 'true');
    }}

    function openWhatsApp() {{
        window.open('https://wa.me/6283168640385?text=menu', '_blank');
    }}

    function updateWhatsAppTheme() {{
        var notif = document.getElementById('whatsappNotif');
        if (notif) {{
            notif.style.textDecoration = 'none';
            var allChildren = notif.querySelectorAll('*');
            for (var i = 0; i < allChildren.length; i++) {{
                allChildren[i].style.textDecoration = 'none';
            }}
        }}
    }}

    var wasClosed = localStorage.getItem('whatsappNotifClosed');
    if (wasClosed !== 'true') {{
        setTimeout(function() {{
            showNotification();
            notificationInterval = setInterval(function() {{
                if (!isNotificationVisible && !isPermanentlyClosed) {{
                    showNotification();
                }}
            }}, 35000);
        }}, 15000);
    }}

    function showModalTim() {{
        var modal = document.getElementById('modalTim');
        if (modal) {{
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
    }}

    function closeModalTim() {{
        var modal = document.getElementById('modalTim');
        if (modal) {{
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }}
    }}

    var modalTim = document.getElementById('modalTim');
    if (modalTim) {{
        modalTim.addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeModalTim();
            }}
        }});
    }}

    // Custom Alert Functions
    let currentAlertCallback = null;
    let currentPromptCallback = null;

    function showCustomAlert(title, message, type = 'info', callback = null) {{
        const alert = document.getElementById('customAlert');
        const icon = alert.querySelector('.custom-alert-icon i');
        const titleEl = document.getElementById('alertTitle');
        const messageEl = document.getElementById('alertMessage');
        
        alert.classList.remove('error', 'success', 'warning', 'info');
        alert.classList.add(type);
        
        let iconClass = 'fa-info-circle';
        if (type === 'error') iconClass = 'fa-exclamation-circle';
        else if (type === 'success') iconClass = 'fa-check-circle';
        else if (type === 'warning') iconClass = 'fa-exclamation-triangle';
        
        icon.className = `fas ${{iconClass}}`;
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        
        currentAlertCallback = callback;
        
        alert.classList.add('show');
        
        if (type === 'success' || type === 'info') {{
            setTimeout(() => {{
                closeCustomAlert();
            }}, 3000);
        }}
    }}

    function closeCustomAlert() {{
        const alert = document.getElementById('customAlert');
        alert.classList.remove('show');
        if (currentAlertCallback) {{
            currentAlertCallback();
            currentAlertCallback = null;
        }}
    }}

    function showCustomPrompt(title, message, callback) {{
        const prompt = document.getElementById('customPrompt');
        const titleEl = document.getElementById('promptTitle');
        const messageEl = document.getElementById('promptMessage');
        const input = document.getElementById('promptInput');
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        input.value = '';
        input.classList.remove('error');
        
        currentPromptCallback = callback;
        
        prompt.classList.add('show');
        
        setTimeout(() => {{
            input.focus();
        }}, 100);
    }}

    function closeCustomPrompt() {{
        const prompt = document.getElementById('customPrompt');
        prompt.classList.remove('show');
        currentPromptCallback = null;
    }}

    function submitPrompt() {{
        const input = document.getElementById('promptInput');
        const password = input.value;
        
        if (currentPromptCallback) {{
            currentPromptCallback(password);
        }}
        closeCustomPrompt();
    }}

    function promptAndDeleteTestimonial(id) {{
        showCustomPrompt('Verifikasi Akses', 'Masukkan kata kunci untuk menghapus komentar', (password) => {{
            if (password && password !== '') {{
                fetch('/verify-delete-testimonial/' + id, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ password: password }})
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        showCustomAlert('Berhasil!', 'Ulasan berhasil dihapus!', 'success', () => {{
                            window.location.href = '/about?message=Ulasan berhasil dihapus&type=success';
                        }});
                    }} else {{
                        showCustomAlert('Gagal!', data.message || 'Password salah! Ulasan tidak dapat dihapus.', 'error');
                    }}
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    showCustomAlert('Error!', 'Terjadi kesalahan pada server.', 'error');
                }});
            }} else if (password === '') {{
                showCustomAlert('Gagal!', 'Password tidak boleh kosong!', 'warning');
            }}
        }});
    }}

    function initImageClassifier() {{
        var uploadArea = document.getElementById('image-upload-area');
        var fileInput = document.getElementById('weatherImageInput');
        var previewContainer = document.getElementById('image-preview-container');
        var imagePreview = document.getElementById('imagePreview');
        var resultDiv = document.getElementById('prediction-result');
        var loadingDiv = document.getElementById('prediction-loading');
        var clearBtn = document.getElementById('clearImageBtn');

        if (!uploadArea) return;

        function handleFile(file) {{
            if (file.size > 5 * 1024 * 1024) {{
                showCustomAlert('Ukuran Terlalu Besar', 'Ukuran gambar terlalu besar! Maksimal 5MB.', 'warning');
                return;
            }}

            var reader = new FileReader();
            reader.onload = function(event) {{
                if (imagePreview) imagePreview.src = event.target.result;
                if (previewContainer) previewContainer.style.display = 'block';
                if (uploadArea) uploadArea.style.display = 'none';
                if (resultDiv) resultDiv.style.display = 'none';
                predictImage(file);
            }};
            reader.readAsDataURL(file);
        }}

        uploadArea.addEventListener('click', function() {{
            if (fileInput) fileInput.click();
        }});

        uploadArea.addEventListener('dragover', function(e) {{
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--accent)';
            uploadArea.style.background = 'var(--accent-soft)';
        }});

        uploadArea.addEventListener('dragleave', function(e) {{
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.background = 'var(--bg-tertiary)';
        }});

        uploadArea.addEventListener('drop', function(e) {{
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--border-color)';
            uploadArea.style.background = 'var(--bg-tertiary)';
            var files = e.dataTransfer.files;
            if (files.length > 0 && fileInput) {{
                fileInput.files = files;
                handleFile(files[0]);
            }}
        }});

        if (fileInput) {{
            fileInput.addEventListener('change', function(e) {{
                if (e.target.files && e.target.files[0]) {{
                    handleFile(e.target.files[0]);
                }}
            }});
        }}

        if (clearBtn) {{
            clearBtn.addEventListener('click', function() {{
                if (fileInput) fileInput.value = '';
                if (previewContainer) previewContainer.style.display = 'none';
                if (uploadArea) uploadArea.style.display = 'block';
                if (resultDiv) resultDiv.style.display = 'none';
                if (loadingDiv) loadingDiv.style.display = 'none';
            }});
        }}
    }}

    async function predictImage(file) {{
        var resultDiv = document.getElementById('prediction-result');
        var loadingDiv = document.getElementById('prediction-loading');

        if (loadingDiv) loadingDiv.style.display = 'block';
        if (resultDiv) resultDiv.style.display = 'none';

        var formData = new FormData();
        formData.append('file', file);

        try {{
            var response = await fetch('/predict-weather-image', {{
                method: 'POST',
                body: formData
            }});

            var data = await response.json();
            if (loadingDiv) loadingDiv.style.display = 'none';

            if (data.success && resultDiv) {{
                var condition = data.condition;
                var confidence = data.confidence;

                var iconMap = {{
                    'Cerah': '<i class="fas fa-sun" style="color: #fbbf24;"></i>',
                    'Berawan': '<i class="fas fa-cloud" style="color: #94a3b8;"></i>',
                    'Kabut': '<i class="fas fa-smog" style="color: #94a3b8;"></i>',
                    'Hujan': '<i class="fas fa-cloud-rain" style="color: #60a5fa;"></i>',
                    'Matahari Terbit': '<i class="fas fa-sunrise" style="color: #f59e0b;"></i>'
                }};

                var iconElem = document.getElementById('prediction-icon');
                var conditionElem = document.getElementById('prediction-condition');
                var confidenceElem = document.getElementById('prediction-confidence');
                var scoresElem = document.getElementById('prediction-all-scores');

                if (iconElem) iconElem.innerHTML = iconMap[condition] || '<i class="fas fa-cloud-sun"></i>';
                if (conditionElem) conditionElem.innerHTML = condition;
                if (confidenceElem) confidenceElem.innerHTML = 'Confidence: ' + confidence + '%';

                if (scoresElem && data.all_scores) {{
                    var scoresHtml = '';
                    for (var key in data.all_scores) {{
                        if (data.all_scores.hasOwnProperty(key)) {{
                            var label = key.charAt(0).toUpperCase() + key.slice(1);
                            scoresHtml += '<span style="display: inline-block; margin-right: 12px;">' + label + ': ' + data.all_scores[key] + '%</span>';
                        }}
                    }}
                    scoresElem.innerHTML = scoresHtml;
                }}

                resultDiv.style.display = 'block';
            }} else if (resultDiv) {{
                resultDiv.style.display = 'block';
                var conditionElem = document.getElementById('prediction-condition');
                var confidenceElem = document.getElementById('prediction-confidence');
                if (conditionElem) conditionElem.innerHTML = 'Error';
                if (confidenceElem) confidenceElem.innerHTML = data.error || 'Gagal memproses gambar';
            }}
        }} catch (error) {{
            console.error('Prediction error:', error);
            if (loadingDiv) loadingDiv.style.display = 'none';
            var resultDivElem = document.getElementById('prediction-result');
            if (resultDivElem) {{
                resultDivElem.style.display = 'block';
                var conditionElem = document.getElementById('prediction-condition');
                var confidenceElem = document.getElementById('prediction-confidence');
                if (conditionElem) conditionElem.innerHTML = 'Error';
                if (confidenceElem) confidenceElem.innerHTML = 'Gagal terhubung ke server';
            }}
        }}
    }}

    function trainImageClassifier() {{
        showCustomPrompt('Training Model CNN', 'Masukkan path folder dataset (contoh: ./weather_dataset):', (datasetPath) => {{
            if (!datasetPath) return;
            
            var loadingDiv = document.getElementById('prediction-loading');
            if (loadingDiv) loadingDiv.style.display = 'block';
            
            fetch('/train-image-classifier?dataset_path=' + encodeURIComponent(datasetPath), {{
                method: 'POST'
            }})
            .then(function(response) {{ return response.json(); }})
            .then(function(data) {{
                if (loadingDiv) loadingDiv.style.display = 'none';
                if (data.success) {{
                    showCustomAlert(
                        'Berhasil!', 
                        '✅ Model berhasil dilatih!\\nAccuracy: ' + (data.accuracy * 100).toFixed(2) + '%\\nVal Accuracy: ' + (data.val_accuracy * 100).toFixed(2) + '%',
                        'success',
                        function() {{ location.reload(); }}
                    );
                }} else {{
                    showCustomAlert('Gagal', '❌ Gagal melatih model: ' + data.error, 'error');
                }}
            }})
            .catch(function(error) {{
                if (loadingDiv) loadingDiv.style.display = 'none';
                showCustomAlert('Error', '❌ Error: ' + error.message, 'error');
            }});
        }});
    }}

    // ============ AI CHAT ASHLEY ============
    let isTyping = false;

    function toggleChat() {{
        const bubble = document.getElementById('chatBubble');
        if (bubble) {{
            bubble.classList.toggle('open');
            if (bubble.classList.contains('open')) {{
                document.getElementById('chatInput')?.focus();
            }}
        }}
    }}

    async function sendChatMessage() {{
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || isTyping) return;
        
        input.value = '';
        
        addChatMessage('user', message);
        showTypingIndicator();
        isTyping = true;
        
        const sendBtn = document.getElementById('chatSendBtn');
        if (sendBtn) sendBtn.disabled = true;
        
        try {{
            const response = await fetch('/chat-ai', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{ message: message }})
            }});
            
            const data = await response.json();
            
            removeTypingIndicator();
            addChatMessage('bot', data.reply || 'Maaf, saya sedang mengalami gangguan. Silakan coba lagi nanti.');
            
        }} catch (error) {{
            console.error('Chat error:', error);
            removeTypingIndicator();
            addChatMessage('bot', 'Maaf, terjadi kesalahan koneksi. Silakan coba lagi.');
        }} finally {{
            isTyping = false;
            if (sendBtn) sendBtn.disabled = false;
            scrollChatToBottom();
        }}
    }}

    function addChatMessage(sender, text) {{
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const emptyState = messagesContainer.querySelector('.chat-empty-state');
        if (emptyState) emptyState.remove();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${{sender}}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        if (sender === 'bot') {{
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }} else {{
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        }}
        
        const bubbleText = document.createElement('div');
        bubbleText.className = 'chat-bubble-text';
        bubbleText.innerHTML = text.replace(/\\n/g, '<br>');
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubbleText);
        
        messagesContainer.appendChild(messageDiv);
        scrollChatToBottom();
    }}

    function showTypingIndicator() {{
        removeTypingIndicator();
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        
        const typingBubble = document.createElement('div');
        typingBubble.className = 'chat-typing';
        typingBubble.innerHTML = '<span></span><span></span><span></span>';
        
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(typingBubble);
        
        messagesContainer.appendChild(typingDiv);
        scrollChatToBottom();
    }}

    function removeTypingIndicator() {{
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }}

    function scrollChatToBottom() {{
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {{
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
    }}

    document.addEventListener('click', function(event) {{
        const bubble = document.getElementById('chatBubble');
        const toggle = document.getElementById('chatToggle');
        
        if (bubble && toggle && bubble.classList.contains('open')) {{
            if (!bubble.contains(event.target) && !toggle.contains(event.target)) {{
                bubble.classList.remove('open');
            }}
        }}
    }});

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initImageClassifier);
    }} else {{
        initImageClassifier();
    }}
</script>
</body>
</html>"""

# ============ ROUTE HOME ============
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    global selected_location

    saved_locations = get_saved_locations()
    weather = get_current_weather(selected_location["latitude"], selected_location["longitude"])
    air_quality = get_air_quality(selected_location["latitude"], selected_location["longitude"])
    forecast_api = get_6day_forecast(selected_location["latitude"], selected_location["longitude"])
    insights = get_ai_insights_real(weather, forecast_api, air_quality, selected_location["name"])

    weather_icon = get_weather_icon_html(weather.get("weather_code", 0))
    condition_text = get_condition_text(weather.get("weather_code", 0))

    local_info = get_local_time(selected_location["latitude"], selected_location["longitude"], selected_location.get("timezone"))

    hour = local_info["hour"]
    if 3 <= hour < 11:
        greeting = "Pagi"
        greeting_icon = "🌅"
    elif 11 <= hour < 15:
        greeting = "Siang"
        greeting_icon = "☀️"
    elif 15 <= hour < 18:
        greeting = "Sore"
        greeting_icon = "🌤️"
    else:
        greeting = "Malam"
        greeting_icon = "🌙"

    forecast_html = ""
    for day in forecast_api[:6]:
        forecast_html += f"""
        <div class="forecast-item">
            <div class="forecast-day">{day["day"]}</div>
            <div class="forecast-date">{day["date"]}</div>
            <div class="forecast-icon">{get_weather_icon_html(day["weather_code"])}</div>
            <div class="forecast-temp">{int(day["temp_max"])}°</div>
            <div class="forecast-temp-min">{int(day["temp_min"])}°</div>
            <div class="forecast-precip"><i class="fas fa-tint"></i> {int(day["precipitation"])}mm</div>
        </div>
        """

    uv = weather.get("uv_index", 5)
    if uv > 8:
        uv_color = "#f97316"
    elif uv > 6:
        uv_color = "#f59e0b"
    else:
        uv_color = "#eab308"

    aqi_color = air_quality["status_color"]
    aqi_icon = air_quality["status_icon"]
    aqi_status = air_quality["status"]

    tz_display = local_info["timezone"]

    content = f"""
    <div class="hero">
        <h1 class="hero-title">
            <span class="greeting-icon">{greeting_icon}</span> 
            Halo, Selamat {greeting}!
        </h1>
        <p class="hero-subtitle">Cuaca hari ini di {selected_location["name"]} untuk aktivitas Anda</p>
    </div>

    <div class="weather-hero">
        <div class="weather-main">
            <div class="weather-info">
                <div class="weather-icon">{weather_icon}</div>
                <div class="weather-temp">{int(weather["temperature"])}<span class="temp-unit">°C</span></div>
                <div class="weather-condition">{condition_text}</div>
                <div class="feels-like">Terasa seperti {int(weather["feels_like"])}°C</div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-tint" style="color: #38bdf8;"></i></div>
                    <div class="stat-label">Kelembaban</div>
                    <div class="stat-value">{int(weather["humidity"])}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-wind" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">Angin</div>
                    <div class="stat-value">{int(weather["wind_speed"])} km/j</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-cloud-rain" style="color: #60a5fa;"></i></div>
                    <div class="stat-label">Hujan</div>
                    <div class="stat-value">{weather["precipitation"]:.1f} mm</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-chart-line" style="color: #f59e0b;"></i></div>
                    <div class="stat-label">Tekanan</div>
                    <div class="stat-value">{int(weather["pressure"])} hPa</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-sun" style="color: {uv_color};"></i></div>
                    <div class="stat-label">UV Index</div>
                    <div class="stat-value">{uv:.1f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas {aqi_icon}" style="color: {aqi_color};"></i></div>
                    <div class="stat-label">Kualitas Udara</div>
                    <div class="stat-value">{aqi_status}</div>
                </div>
            </div>
        </div>
    </div>

    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-robot"></i> Analisa Cuaca (Ashley)</span>
            </div>
            <div class="insights-text">
                {insights}
            </div>
        </div>

        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-info-circle"></i> Tanggal & Waktu</span>
            </div>
            <div class="info-row">
                <div class="info-item">
                    <i class="fas fa-calendar-alt info-icon"></i>
                    <div>
                        <div class="info-label">TANGGAL</div>
                        <div class="info-value">{local_info["date"]}</div>
                    </div>
                </div>
                <div class="info-item">
                    <i class="fas fa-clock info-icon"></i>
                    <div>
                        <div class="info-label">WAKTU LOKAL</div>
                        <div class="info-value">
                        <div class="timezone-badge" id="timezone-display" style="margin-top: 6px; font-size: 12px; opacity: 1;transform: translateY(-2px);">{tz_display}</div>
                        <span id="realtime-clock">{local_info["time"]}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="glass-card">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-calendar-week"></i> Prakiraan 6 Hari</span>
        </div>
        <div class="forecast-grid">{forecast_html}</div>
    </div>
    """

    return HTMLResponse(content=render_page(content, active="home", saved_locations=saved_locations, selected_location=selected_location))


# ============ ROUTE MAIN (ML) ============
@app.get("/main", response_class=HTMLResponse)
async def ml_dashboard(request: Request):
    global selected_location

    saved_locations = get_saved_locations()
    weather = get_current_weather(selected_location["latitude"], selected_location["longitude"])

    ml_predictions = weather_predictor.predict_temperature(weather)
    model_info = weather_predictor.get_model_info()

    local_info = get_local_time(selected_location["latitude"], selected_location["longitude"], selected_location.get("timezone"))

    ml_forecast_html = ""
    for day in ml_predictions[:6]:
        precip_value = day.get("precipitation", 0)
        ml_forecast_html += f"""
        <div class="forecast-item">
            <div class="forecast-day">{day["day"]}</div>
            <div class="forecast-date">{day["date"]}</div>
            <div class="forecast-icon"><i class="fas fa-chart-line" style="color: #8b5cf6;"></i></div>
            <div class="forecast-temp">{int(day["temp_max"])}°</div>
            <div class="forecast-temp-min">{int(day["temp_min"])}°</div>
            <div class="forecast-precip"><i class="fas fa-tint"></i> {precip_value}mm</div>
            <div class="ml-badge"><i class="fas fa-brain"></i> ML</div>
        </div>
        """

    if model_info["is_trained"]:
        model_status = f"""
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-chart-simple"></i> Metrik Model (Random Forest)</span>
            </div>
            <div class="stats-grid" style="margin-bottom: 0;">
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-chart-line" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">MAE (Mean Absolute Error)</div>
                    <div class="stat-value" style="color: #8b5cf6;">{model_info['mae']:.6f}°C</div>
                </div>
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-chart-bar" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">R² Score (Akurasi)</div>
                    <div class="stat-value" style="color: #8b5cf6;">{model_info['r2']:.6f}</div>
                </div>
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-icon"><i class="fas fa-map-marker-alt" style="color: #8b5cf6;"></i></div>
                    <div class="stat-label">Lokasi Training</div>
                    <div class="stat-value" style="color: #8b5cf6; font-size: 16px;">{model_info['location']}</div>
                </div>
            </div>
            <form method="GET" action="/train-model" style="margin-top: 24px;" id="trainForm">
                <button type="submit" class="train-btn-ml" style="width: 100%;">
                    <i class="fas fa-sync-alt"></i> Latih Ulang Model ML
                </button>
            </form>
        </div>
        """
    else:
        model_status = f"""
        <div class="glass-card" style="text-align: center;">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-exclamation-triangle"></i> Model Belum Dilatih</span>
            </div>
            <div style="padding: 32px 0;">
                <i class="fas fa-brain" style="font-size: 64px; color: var(--ml-purple); margin-bottom: 24px; display: block;"></i>
                <p style="color: var(--text-tertiary); margin-bottom: 24px;">Klik tombol di bawah untuk melatih model Random Forest Regressor</p>
                <form method="GET" action="/train-model" id="trainForm">
                    <button type="submit" class="train-btn-ml">
                        <i class="fas fa-play"></i> Latih Model ML Sekarang
                    </button>
                </form>
            </div>
        </div>
        """

    image_classifier_html = f"""
    <div class="glass-card" style="height: 100%;">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-camera"></i> Pendeteksi Cuaca dari Gambar (CNN)</span>
        </div>
        <div style="padding: 16px;">
            <div id="image-upload-area" style="
                border: 2px dashed var(--border-color);
                border-radius: 24px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: var(--bg-tertiary);
                margin-bottom: 20px;
            ">
                <i class="fas fa-cloud-upload-alt" style="font-size: 48px; color: var(--accent); margin-bottom: 12px;"></i>
                <p style="margin-bottom: 8px;">Klik atau drag & drop gambar di sini</p>
                <p style="font-size: 12px; color: var(--text-tertiary);">Format: JPG, PNG (Max 5MB)</p>
                <input type="file" id="weatherImageInput" accept="image/*" style="display: none;">
            </div>

            <div id="image-preview-container" style="display: none; margin-bottom: 20px;">
                <img id="imagePreview" style="max-width: 100%; max-height: 250px; border-radius: 20px; margin-bottom: 12px;">
                <button id="clearImageBtn" class="search-btn" style="background: #6b7280; padding: 8px 16px; font-size: 12px;">
                    <i class="fas fa-times"></i> Hapus Gambar
                </button>
            </div>

            <div id="prediction-result" style="display: none; background: var(--accent-soft); border-radius: 20px; padding: 20px; margin-top: 16px;">
                <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                    <div id="prediction-icon" style="font-size: 48px;"></div>
                    <div>
                        <div style="font-size: 12px; color: var(--text-tertiary);">Hasil Deteksi</div>
                        <div id="prediction-condition" style="font-size: 24px; font-weight: 700;"></div>
                        <div id="prediction-confidence" style="font-size: 14px; color: var(--accent);"></div>
                    </div>
                </div>
                <div id="prediction-all-scores" style="margin-top: 16px; font-size: 12px;"></div>
            </div>

            <div id="prediction-loading" style="display: none; text-align: center; padding: 20px;">
                <i class="fas fa-spinner fa-pulse" style="font-size: 32px; color: var(--accent);"></i>
                <p style="margin-top: 12px;">Menganalisis gambar...</p>
            </div>

            <button onclick="trainImageClassifier()" class="train-btn-green" style="margin-top: 20px; width: 100%;">
                <i class="fas fa-database"></i> Latih Model CNN (Upload Dataset)
            </button>
            <p style="font-size: 11px; color: var(--text-tertiary); text-align: center; margin-top: 12px;">
                *pastikan gambar memiliki resolusi yang cukup dan jelas untuk hasil terbaik. Dataset harus memiliki struktur folder dengan subfolder untuk setiap kelas cuaca (misal: ./weather_dataset/cerah, ./weather_dataset/hujan, dll) dan masing-masing subfolder berisi gambar-gambar terkait.
            </p>
        </div>
    </div>
    """

    tz_display = local_info["timezone"]

    content = f"""
    <div class="hero">
        <h1 class="hero-title">Machine Learning</h1>
        <p class="hero-subtitle">Prediksi cuaca berdasarkan gambar menggunakan teknologi deep learning</p>
    </div>

    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-chart-line"></i> Prediksi 6 Hari (Random Forest)</span>
            </div>
            <div class="forecast-grid">{ml_forecast_html}</div>
        </div>

        {model_status}
    </div>

    <div class="bento-grid" style="grid-template-columns: repeat(2, 1fr);">
        {image_classifier_html}

        <div class="glass-card" style="height: 100%;">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-info-circle"></i> Tanggal & Waktu</span>
            </div>
            <div class="info-row">
                <div class="info-item">
                    <i class="fas fa-calendar-alt info-icon"></i>
                    <div>
                        <div class="info-label">TANGGAL</div>
                        <div class="info-value">{local_info["date"]}</div>
                    </div>
                </div>
                <div class="info-item">
                    <i class="fas fa-clock info-icon"></i>
                    <div>
                        <div class="info-label">WAKTU LOKAL</div>
                        <div class="info-value">
                        <div class="timezone-badge" id="timezone-display" style="margin-top: 6px; font-size: 12px; opacity: 1;transform: translateY(-2px);">{tz_display}</div>
                        <span id="realtime-clock">{local_info["time"]}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    return HTMLResponse(content=render_page(content, active="ml", saved_locations=saved_locations, selected_location=selected_location))


# ============ ROUTE TULIS ULASAN ============
@app.get("/ulasan", response_class=HTMLResponse)
async def ulasan_page(request: Request, message: str = None, type: str = None):
    saved_locations = get_saved_locations()
    
    content = f"""
    <style>
        .rating-input {{
            display: flex;
            gap: 12px;
            justify-content: center;
            margin: 20px 0;
        }}
        .rating-star {{
            font-size: 48px;
            cursor: pointer;
            color: #cbd5e1;
            transition: all 0.2s ease;
        }}
        .rating-star:hover,
        .rating-star.active {{
            color: #fbbf24;
            transform: scale(1.1);
        }}
        .rating-star.selected {{
            color: #fbbf24;
        }}
        .char-counter {{
            text-align: right;
            font-size: 12px;
            color: var(--text-tertiary);
            margin-top: 8px;
        }}
        .char-counter.warning {{
            color: #f59e0b;
        }}
        .char-counter.danger {{
            color: #ef4444;
        }}
        .review-form {{
            max-width: 600px;
            margin: 0 auto;
        }}
        .form-group {{
            margin-bottom: 24px;
        }}
        .form-group label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        .form-group input,
        .form-group textarea {{
            width: 100%;
            padding: 14px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            color: var(--text-primary);
            font-family: inherit;
            transition: all 0.2s ease;
        }}
        .form-group input:focus,
        .form-group textarea:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }}
        .submit-btn {{
            width: 100%;
            padding: 14px;
            background: var(--accent-gradient);
            border: none;
            border-radius: 16px;
            color: white;
            font-weight: 700;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        }}
    </style>

    <div class="hero">
        <h1 class="hero-title">Tulis Ulasan</h1>
        <p class="hero-subtitle">Bagikan pengalaman Anda menggunakan WeatherAI</p>
    </div>

    <div class="glass-card">
        <div class="card-header">
            <span class="card-title"><i class="fas fa-pen"></i> Form Ulasan</span>
        </div>
        
        <form class="review-form" method="POST" action="/ulasan/submit" onsubmit="return validateForm()">
            <div class="form-group">
                <label><i class="fas fa-user"></i> Nama Anda</label>
                <input type="text" name="name" placeholder="Contoh: John Doe" required maxlength="100">
            </div>
            
            <div class="form-group">
                <label><i class="fas fa-briefcase"></i> Pekerjaan</label>
                <input type="text" name="role" placeholder="Contoh: Mahasiswa, Programmer, Petani, dll" required maxlength="100">
            </div>
            
            <div class="form-group">
                <label><i class="fas fa-star"></i> Rating</label>
                <div class="rating-input" id="ratingInput">
                    <i class="fas fa-star rating-star" data-value="1"></i>
                    <i class="fas fa-star rating-star" data-value="2"></i>
                    <i class="fas fa-star rating-star" data-value="3"></i>
                    <i class="fas fa-star rating-star" data-value="4"></i>
                    <i class="fas fa-star rating-star" data-value="5"></i>
                </div>
                <input type="hidden" name="rating" id="ratingValue" required>
            </div>
            
            <div class="form-group">
                <label><i class="fas fa-comment"></i> Komentar</label>
                <textarea name="comment" rows="5" placeholder="Tulis pengalaman Anda menggunakan WeatherAI..." required maxlength="250"></textarea>
                <div class="char-counter">
                    <span id="charCount">0</span> / 250 karakter
                </div>
            </div>
            
            <button type="submit" class="submit-btn">
                <i class="fas fa-paper-plane"></i> Kirim Ulasan
            </button>
        </form>
    </div>

    <script>
        const stars = document.querySelectorAll('.rating-star');
        const ratingInput = document.getElementById('ratingValue');
        let selectedRating = 0;
        
        stars.forEach(star => {{
            star.addEventListener('click', function() {{
                selectedRating = parseInt(this.dataset.value);
                ratingInput.value = selectedRating;
                
                stars.forEach(s => {{
                    const val = parseInt(s.dataset.value);
                    if (val <= selectedRating) {{
                        s.classList.add('selected');
                    }} else {{
                        s.classList.remove('selected');
                    }}
                }});
            }});
            
            star.addEventListener('mouseenter', function() {{
                const hoverVal = parseInt(this.dataset.value);
                stars.forEach(s => {{
                    const val = parseInt(s.dataset.value);
                    if (val <= hoverVal) {{
                        s.style.color = '#fbbf24';
                    }} else {{
                        s.style.color = '#cbd5e1';
                    }}
                }});
            }});
            
            star.addEventListener('mouseleave', function() {{
                stars.forEach(s => {{
                    const val = parseInt(s.dataset.value);
                    if (val <= selectedRating) {{
                        s.style.color = '#fbbf24';
                    }} else {{
                        s.style.color = '#cbd5e1';
                    }}
                }});
            }});
        }});
        
        const textarea = document.querySelector('textarea[name="comment"]');
        const charCount = document.getElementById('charCount');
        
        textarea.addEventListener('input', function() {{
            const length = this.value.length;
            charCount.textContent = length;
            const counter = document.querySelector('.char-counter');
            if (length > 200) {{
                counter.classList.add('warning');
            }} else {{
                counter.classList.remove('warning');
            }}
            if (length > 240) {{
                counter.classList.add('danger');
            }} else {{
                counter.classList.remove('danger');
            }}
        }});
        
        function validateForm() {{
            if (selectedRating === 0) {{
                showCustomAlert('Rating Belum Dipilih', 'Silakan pilih rating bintang untuk ulasan Anda.', 'warning');
                return false;
            }}
            return true;
        }}
    </script>
    """
    
    return HTMLResponse(content=render_page(content, active="ulasan", saved_locations=saved_locations, selected_location=selected_location, message=message, message_type=type))


@app.post("/ulasan/submit")
async def submit_ulasan(name: str = Form(...), role: str = Form(...), comment: str = Form(...), rating: int = Form(...)):
    if len(comment) > 250:
        return RedirectResponse(url="/ulasan?message=Komentar terlalu panjang! Maksimal 250 karakter&type=error", status_code=303)
    
    save_testimonial(name, role, comment, rating)
    return RedirectResponse(url="/ulasan?message=Terima kasih! Ulasan Anda telah disimpan&type=success", status_code=303)


# ============ ROUTE DELETE TESTIMONIAL ============
@app.post("/verify-delete-testimonial/{testimonial_id}")
async def verify_delete_testimonial(testimonial_id: int, request: PasswordRequest):
    if not ADMIN_PASSWORD:
        return {"success": False, "message": "Password belum dikonfigurasi di server"}
    
    if request.password == ADMIN_PASSWORD:
        delete_testimonial_by_id(testimonial_id)
        return {"success": True, "message": "Ulasan berhasil dihapus"}
    else:
        return {"success": False, "message": "Password salah!"}


# ============ ROUTE SEARCH ============
@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, message: str = None, type: str = None):
    saved_locations = get_saved_locations()

    content = f"""
    <div class="hero">
        <h1 class="hero-title">Cari Lokasi</h1>
        <p class="hero-subtitle">Cari berdasarkan nama kota <strong>atau</strong> koordinat (Latitude, Longitude)</p>
    </div>

    <div class="bento-grid">
        <div class="glass-card">
            <div class="card-header">
                <span class="card-title">
                    <i class="fas fa-city"></i> Cari berdasarkan Nama Kota
                </span>
            </div>
            <form method="POST" action="/search/city">
                <div class="search-container" style="margin-bottom: 0;">
                    <input type="text" name="city_name" class="search-input" 
                           placeholder="Contoh: Jakarta, New York, London, Tokyo" required>
                    <button type="submit" class="search-btn">
                        <i class="fas fa-search"></i> Cari
                    </button>
                </div>
            </form>
            <p style="margin-top: 16px; font-size: 12px; color: var(--text-tertiary);">
                <i class="fas fa-info-circle"></i> Gunakan nama kota atau daerah. Mendukung semua kota di dunia.
            </p>
        </div>

        <div class="glass-card">
            <div class="card-header">
                <span class="card-title">
                    <i class="fas fa-crosshairs"></i> Cari berdasarkan Koordinat
                </span>
            </div>
            <form method="POST" action="/search/coords">
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;">
                    <div style="flex: 1;">
                        <label style="display: block; font-size: 12px; margin-bottom: 6px; color: var(--text-tertiary);">
                            <i class="fas fa-arrow-up"></i> Latitude (Lintang)
                        </label>
                        <input type="number" step="any" name="latitude" class="search-input" 
                               placeholder="Contoh: -6.2" required
                               style="width: 100%; border-radius: 50px; padding: 12px 16px;">
                    </div>
                    <div style="flex: 1;">
                        <label style="display: block; font-size: 12px; margin-bottom: 6px; color: var(--text-tertiary);">
                            <i class="fas fa-arrow-right"></i> Longitude (Bujur)
                        </label>
                        <input type="number" step="any" name="longitude" class="search-input" 
                               placeholder="Contoh: 106.816666" required
                               style="width: 100%; border-radius: 50px; padding: 12px 16px;">
                    </div>
                </div>
                <button type="submit" class="search-btn" style="width: 100%;">
                    <i class="fas fa-location-dot"></i> Cari & Simpan dari Koordinat
                </button>
            </form>
            <p style="margin-top: 16px; font-size: 12px; color: var(--text-tertiary);">
                <i class="fas fa-lightbulb"></i> <strong>Tips:</strong> Buka Google Maps, klik kanan pada lokasi, 
                pilih "What's here?" untuk mendapatkan koordinat.
            </p>
        </div>
    </div>

    <div class="glass-card" style="margin-top: 20px;">
        <div class="card-header">
            <span class="card-title">
                <i class="fas fa-location-arrow"></i> Contoh Koordinat Populer
            </span>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">
            <button onclick="setCoordinates(-6.2, 106.816666)" class="train-btn-ml" 
                    style="padding: 8px 16px; font-size: 12px;">
                🇮🇩 Jakarta (-6.2, 106.82)
            </button>
            <button onclick="setCoordinates(40.7128, -74.0060)" class="train-btn-ml" 
                    style="padding: 8px 16px; font-size: 12px;">
                🇺🇸 New York (40.71, -74.01)
            </button>
            <button onclick="setCoordinates(51.5074, -0.1278)" class="train-btn-ml" 
                    style="padding: 8px 16px; font-size: 12px;">
                🇬🇧 London (51.51, -0.13)
            </button>
            <button onclick="setCoordinates(35.6895, 139.6917)" class="train-btn-ml" 
                    style="padding: 8px 16px; font-size: 12px;">
                🇯🇵 Tokyo (35.69, 139.69)
            </button>
            <button onclick="setCoordinates(-33.8688, 151.2093)" class="train-btn-ml" 
                    style="padding: 8px 16px; font-size: 12px;">
                🇦🇺 Sydney (-33.87, 151.21)
            </button>
        </div>
    </div>

    <script>
        function setCoordinates(lat, lon) {{
            const latInput = document.querySelector('input[name="latitude"]');
            const lonInput = document.querySelector('input[name="longitude"]');

            if (latInput && lonInput) {{
                latInput.value = lat;
                lonInput.value = lon;

                latInput.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                latInput.style.borderColor = '#10b981';
                lonInput.style.borderColor = '#10b981';

                setTimeout(() => {{
                    latInput.style.borderColor = '';
                    lonInput.style.borderColor = '';
                }}, 2000);
            }}
        }}
    </script>
    """

    return HTMLResponse(content=render_page(content=content, active="search", message=message, message_type=type, saved_locations=saved_locations, selected_location=selected_location))


@app.post("/search/city", response_class=HTMLResponse)
async def search_city_post(city_name: str = Form(...)):
    result = search_city(city_name)
    if result:
        if location_exists(result["name"], result["latitude"], result["longitude"]):
            return RedirectResponse(url="/search?message=Lokasi sudah tersimpan&type=error", status_code=303)

        save_location(result["name"], result["latitude"], result["longitude"], result["country"], result["timezone"])
        return RedirectResponse(url=f"/search?message={result['name']} berhasil ditambahkan ke favorit&type=success", status_code=303)
    else:
        return RedirectResponse(url=f"/search?message=Kota '{city_name}' tidak ditemukan. Periksa ejaan Anda.&type=error", status_code=303)


@app.post("/search/coords", response_class=HTMLResponse)
async def search_coords_post(latitude: float = Form(...), longitude: float = Form(...)):
    if not validate_coordinates(latitude, longitude):
        return RedirectResponse(url="/search?message=❌ Koordinat tidak valid! Latitude: -90 s/d 90, Longitude: -180 s/d 180&type=error", status_code=303)

    result = search_by_coordinates(latitude, longitude)
    if result:
        if location_exists(result["name"], result["latitude"], result["longitude"]):
            return RedirectResponse(url="/search?message=Lokasi dengan koordinat ini sudah tersimpan&type=error", status_code=303)

        save_location(result["name"], result["latitude"], result["longitude"], result["country"], result["timezone"])

        coords_text = f"{latitude:.4f}, {longitude:.4f}"
        return RedirectResponse(url=f"/search?message=📍 {result['name']} ({coords_text}) berhasil ditambahkan ke favorit&type=success", status_code=303)
    else:
        return RedirectResponse(url=f"/search?message=❌ Gagal mendapatkan informasi dari koordinat ({latitude}, {longitude}). Periksa kembali koordinat Anda.&type=error", status_code=303)


# ============ ROUTE ABOUT ============
@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request, message: str = None, type: str = None):
    saved_locations = get_saved_locations()
    testimonials = get_all_testimonials()
    
    testimonial_items = ""
    if testimonials:
        for t in testimonials:
            stars_html = ""
            for i in range(5):
                if i < t["rating"]:
                    stars_html += '<i class="fas fa-star" style="color: #fbbf24;"></i>'
                else:
                    stars_html += '<i class="fas fa-star" style="color: #cbd5e1;"></i>'
            
            testimonial_items += f"""
            <div class="testimonial-scroll-item">
                <div class="testimonial-card-horizontal">
                    <div class="testimonial-header">
                        <div class="testimonial-avatar">
                            <span>{t['name'][0].upper()}</span>
                        </div>
                        <div class="testimonial-info">
                            <div class="testimonial-name">{t['name']}</div>
                            <div class="testimonial-role">{t['role']}</div>
                            <div class="testimonial-rating">{stars_html}</div>
                        </div>
                        <button class="testimonial-delete" onclick="promptAndDeleteTestimonial({t['id']})">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="testimonial-comment">
                        "{t['comment']}"
                    </div>
                    <div class="testimonial-date">
                        <i class="far fa-calendar-alt"></i> {t['created_at'][:10]}
                    </div>
                </div>
            </div>
            """
    else:
        testimonial_items = '<div class="no-testimonials">Belum ada ulasan. Jadilah yang pertama memberikan ulasan!</div>'

    content = f"""
    <style>
        .about-container {{
            display: flex;
            flex-direction: column;
            gap: 28px;
        }}
        .about-card {{
            margin-bottom: 0;
        }}
        .section-content {{
            padding: 24px;
        }}
        .section-content p {{
            text-align: justify;
            text-justify: inter-word;
            margin-bottom: 16px;
            line-height: 1.6;
        }}
        .section-content p:last-child {{
            margin-bottom: 0;
        }}
        
        .scrollable-testimonials {{
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            padding: 20px 0 30px 0;
            scroll-behavior: smooth;
            cursor: grab;
            user-select: none;
            width: 100%;
        }}
        .scrollable-testimonials:active {{
            cursor: grabbing;
        }}
        .testimonials-container {{
            display: inline-flex;
            gap: 24px;
            padding: 0 8px;
        }}
        .testimonial-scroll-item {{
            display: inline-block;
            white-space: normal;
            width: 350px;
            flex-shrink: 0;
        }}
        .testimonial-card-horizontal {{
            background: var(--bg-tertiary);
            border-radius: 24px;
            padding: 20px;
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        .testimonial-card-horizontal:hover {{
            transform: translateY(-4px);
            border-color: var(--accent);
            box-shadow: var(--shadow-lg);
        }}
        .testimonial-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            position: relative;
        }}
        .testimonial-avatar {{
            width: 50px;
            height: 50px;
            background: var(--accent-gradient);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            color: white;
            flex-shrink: 0;
        }}
        .testimonial-info {{
            flex: 1;
        }}
        .testimonial-name {{
            font-weight: 800;
            font-size: 16px;
            color: var(--text-primary);
        }}
        .testimonial-role {{
            font-size: 12px;
            color: var(--accent);
            margin-top: 2px;
        }}
        .testimonial-rating {{
            margin-top: 4px;
            font-size: 12px;
        }}
        .testimonial-comment {{
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-secondary);
            margin-bottom: 16px;
            flex: 1;
            word-wrap: break-word;
            white-space: normal;
        }}
        .testimonial-date {{
            font-size: 11px;
            color: var(--text-tertiary);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .testimonial-delete {{
            background: rgba(239, 68, 68, 0.1);
            border: none;
            color: var(--danger);
            width: 30px;
            height: 30px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .testimonial-delete:hover {{
            background: var(--danger);
            color: white;
            transform: scale(1.05);
        }}
        .no-testimonials {{
            text-align: center;
            padding: 40px;
            color: var(--text-tertiary);
        }}
        .scroll-indicator {{
            text-align: center;
            margin-top: 12px;
            color: var(--text-tertiary);
            font-size: 12px;
        }}
        .tech-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            row-gap: 32px;
            margin-top: 16px;
        }}
        .tech-item {{
            text-align: center;
            padding: 16px 12px;
        }}
        @media (max-width: 768px) {{
            .testimonial-scroll-item {{ width: 300px; }}
            .tech-grid {{ grid-template-columns: repeat(2, 1fr); gap: 16px; }}
            .section-content {{ padding: 16px; }}
        }}
        @media (max-width: 480px) {{
            .testimonial-scroll-item {{ width: 280px; }}
        }}
    </style>

    <div class="hero">
        <h1 class="hero-title">WeatherAI</h1>
        <p class="hero-subtitle">Aplikasi prediksi cuaca cerdas berbasis AI untuk informasi akurat dan real-time</p>
    </div>

    <div class="about-container">
        <div class="glass-card about-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-info-circle"></i> introduction</span>
            </div>
            <div class="section-content">
                <p><strong>WeatherAI</strong> adalah aplikasi prediksi cuaca berbasis Artificial Intelligence yang menyajikan informasi meteorologi real-time, prakiraan jangka pendek, serta analisis cuaca dalam bahasa alami. Aplikasi ini mengintegrasikan data dari Open-Meteo API, model Machine Learning Random Forest untuk prediksi suhu, dan Ashley untuk menghasilkan wawasan cuaca yang kontekstual. Dengan antarmuka modern, dark mode, serta fitur penyimpanan lokasi favorit, WeatherAI hadir sebagai solusi prediksi cuaca yang praktis, akurat, dan ramah pengguna.</p>
                <p><strong>Fitur utama</strong> meliputi cuaca real-time (suhu, kelembaban, angin, tekanan, UV), prakiraan 6 hari, kualitas udara (AQI, PM2.5, PM10), serta pencarian lokasi berdasarkan nama kota atau koordinat GPS. Fitur unggulan lainnya adalah deteksi kondisi cuaca dari gambar menggunakan Deep Learning CNN, dan analisis Ashley yang memberikan deskripsi natural serta peringatan dini kondisi ekstrem.</p>
                <p><strong>Cara kerja</strong> aplikasi: sistem mengambil data cuaca dari API berdasarkan koordinat lokasi, lalu model Random Forest memprediksi suhu 6 hari ke depan. Untuk deteksi gambar, CNN mengklasifikasikan foto pemandangan ke kategori cuaca. Semua data kemudian diproses Gemini AI menjadi narasi analisis yang mudah dipahami. Kombinasi real-time data, machine learning, deep learning, dan AI generatif menjadikan WeatherAI lebih cerdas dan akurat dibanding aplikasi cuaca konvensional.</p>
            </div>
        </div>

        <div class="glass-card about-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-code"></i> Teknologi & Framework</span>
            </div>
            <div class="section-content">
                <div class="tech-grid">
                    <div class="tech-item"><i class="fab fa-python" style="font-size: 36px; color: #3776ab;"></i><div style="font-weight:700;">Python</div><div style="font-size:12px;">Backend Utama</div></div>
                    <div class="tech-item"><i class="fas fa-rocket" style="font-size: 36px; color: #00c8ff;"></i><div style="font-weight:700;">FastAPI</div><div style="font-size:12px;">Web Framework</div></div>
                    <div class="tech-item"><i class="fas fa-brain" style="font-size: 36px; color: #8b5cf6;"></i><div style="font-weight:700;">Scikit-learn</div><div style="font-size:12px;">Random Forest</div></div>
                    <div class="tech-item"><i class="fas fa-chart-line" style="font-size: 36px; color: #ff6b6b;"></i><div style="font-weight:700;">TensorFlow</div><div style="font-size:12px;">CNN</div></div>
                    <div class="tech-item"><i class="fas fa-cloud-sun" style="font-size: 36px; color: #f59e0b;"></i><div style="font-weight:700;">Open-Meteo</div><div style="font-size:12px;">Weather API</div></div>
                    <div class="tech-item"><i class="fas fa-robot" style="font-size: 36px; color: #4285f4;"></i><div style="font-weight:700;">Ashley</div><div style="font-size:12px;">AI Assistant</div></div>
                </div>
            </div>
        </div>

        <div class="glass-card about-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-book-open"></i> Ulasan Pengguna</span>
            </div>
            <div class="section-content" style="padding: 0 24px 24px 24px;">
                <div class="scrollable-testimonials" id="testimonialsScroll">
                    <div class="testimonials-container">{testimonial_items}</div>
                </div>
                <div class="scroll-indicator"><i class="fas fa-chevron-right"></i> Geser ke kanan untuk melihat lebih banyak ulasan <i class="fas fa-chevron-left"></i></div>
            </div>
        </div>

        <div class="glass-card about-card">
            <div class="card-header">
                <span class="card-title"><i class="fas fa-shield-alt"></i> Privasi & Keamanan</span>
            </div>
            <div class="section-content">
                <p>WeatherAI menghormati dan melindungi privasi setiap pengguna. Aplikasi ini <strong>tidak mengumpulkan, menyimpan, atau membagikan data pribadi</strong> seperti nama, alamat email, nomor telepon, atau lokasi spesifik pengguna ke pihak manapun. Semua data cuaca yang ditampilkan diperoleh secara langsung dari API publik Open-Meteo dan diproses secara anonim serta real-time tanpa disimpan dalam database server.</p>
                <p><strong>Data lokasi yang Anda simpan</strong> (seperti nama kota favorit dan koordinat GPS) hanya tersimpan secara lokal di dalam database perangkat Anda sendiri menggunakan SQLite. Data ini tidak pernah dikirim ke server eksternal manapun dan dapat Anda hapus kapan saja melalui antarmuka aplikasi dengan menekan tombol hapus pada sidebar lokasi tersimpan.</p>
                <p>Untuk fitur <strong>AI Assistant (Ashley)</strong>, data cuaca dan kualitas udara secara real-time hanya digunakan untuk menghasilkan analisis cuaca. Data tersebut tidak disimpan oleh Google untuk keperluan pelatihan model. Kami tidak menggunakan cookie pelacak, tidak menanamkan tracker iklan, dan tidak memonetisasi data pengguna. Keamanan dan kenyamanan Anda adalah prioritas utama kami.</p>
            </div>
        </div>
    </div>

    <script>
        const scrollContainer = document.getElementById('testimonialsScroll');
        if (scrollContainer) {{
            let isDown = false, startX, scrollLeft;
            scrollContainer.addEventListener('mousedown', (e) => {{ isDown = true; scrollContainer.style.cursor = 'grabbing'; startX = e.pageX - scrollContainer.offsetLeft; scrollLeft = scrollContainer.scrollLeft; }});
            scrollContainer.addEventListener('mouseleave', () => {{ isDown = false; scrollContainer.style.cursor = 'grab'; }});
            scrollContainer.addEventListener('mouseup', () => {{ isDown = false; scrollContainer.style.cursor = 'grab'; }});
            scrollContainer.addEventListener('mousemove', (e) => {{ if (!isDown) return; e.preventDefault(); const x = e.pageX - scrollContainer.offsetLeft; scrollContainer.scrollLeft = scrollLeft - (x - startX) * 2; }});
        }}
    </script>
    """

    return HTMLResponse(content=render_page(content, active="about", saved_locations=saved_locations, message=message, message_type=type))


# ============ ROUTE LOCATION ============
@app.get("/select-location/{location_id}")
async def select_location(location_id: int):
    global selected_location
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, latitude, longitude, country, timezone FROM saved_locations WHERE id = ?", (location_id,))
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        cursor.execute("SELECT id, name, latitude, longitude, country FROM saved_locations WHERE id = ?", (location_id,))
        row = cursor.fetchone()
        if row:
            row = list(row) + [None]

    conn.close()
    if row:
        selected_location = {
            "name": row[1],
            "latitude": row[2],
            "longitude": row[3],
            "timezone": row[5] if len(row) > 5 and row[5] else get_timezone_from_coords(row[2], row[3]),
        }
    return RedirectResponse(url="/", status_code=303)


@app.get("/delete-location/{location_id}")
async def delete_location_route(location_id: int):
    delete_location(location_id)
    return RedirectResponse(url="/?message=Lokasi berhasil dihapus&type=success", status_code=303)


# ============ ROUTE TRAIN MODEL ============
@app.get("/train-model")
async def train_model_route(request: Request):
    global selected_location

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    try:
        result = weather_predictor.train_model(selected_location["name"], selected_location["latitude"], selected_location["longitude"])
        if is_ajax:
            return {"success": True, "mae": result["mae"], "r2": result["r2"]}
        return RedirectResponse(url=f"/main?message=✅ Model ML berhasil dilatih! MAE: {result['mae']:.6f}°C, R²: {result['r2']}&type=success", status_code=303)
    except Exception as e:
        if is_ajax:
            return {"success": False, "error": str(e)}
        return RedirectResponse(url=f"/main?message=❌ Gagal melatih model: {str(e)}&type=error", status_code=303)


# ============ ROUTE PREDICT IMAGE ============
@app.post("/predict-weather-image")
async def predict_weather_from_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"success": False, "error": "File harus berupa gambar"}

        contents = await file.read()
        result = weather_image_classifier.predict_image(contents)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        img_preview = base64.b64encode(contents).decode("utf-8")

        return {
            "success": True,
            "prediction": result["prediction"],
            "condition": result["condition"],
            "confidence": result["confidence"],
            "weather_code": result["weather_code"],
            "all_scores": result["all_scores"],
            "preview": f"data:{file.content_type};base64,{img_preview}",
        }
    except Exception as e:
        print(f"Error predict image: {e}")
        return {"success": False, "error": str(e)}


@app.post("/train-image-classifier")
async def train_image_classifier_route(dataset_path: str):
    try:
        if not TF_AVAILABLE:
            return {"success": False, "error": "TensorFlow tidak tersedia. Install dengan: pip install tensorflow"}

        if not Path(dataset_path).exists():
            return {"success": False, "error": f"Dataset path tidak ditemukan: {dataset_path}"}

        result = weather_image_classifier.train_model(dataset_path, epochs=15)
        if "error" in result:
            return {"success": False, "error": result["error"]}
        return {"success": True, "accuracy": result["accuracy"], "val_accuracy": result["val_accuracy"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ AI CHAT ASHLEY ENDPOINT ============
class ChatMessage(BaseModel):
    message: str

@app.post("/chat-ai")
async def chat_ai(chat: ChatMessage):
    """Endpoint untuk chat AI Ashley dengan batasan topik cuaca"""
    user_message = chat.message.strip()
    
    if not user_message:
        return {"reply": "Silakan tulis pesan Anda terlebih dahulu."}
    
    # Cek apakah pertanyaan terkait cuaca
    weather_keywords = [
        "cuaca", "weather", "hujan", "rain", "cerah", "sunny", "berawan", "cloudy",
        "suhu", "temperature", "angin", "wind", "lembab", "humidity", "tekanan", "pressure",
        "uv", "ultraviolet", "kualitas udara", "air quality", "aqi", "pm2.5", "pm10",
        "prakiraan", "forecast", "prediksi", "prediction", "musim", "season", "iklim", "climate",
        "badai", "storm", "petir", "thunder", "topan", "cyclone", "panas", "hot", "dingin", "cold",
        "kabut", "fog", "salju", "snow", "gemini", "ashley", "weather ai", "cuaca hari ini",
        "besok", "tomorrow", "hari ini", "hari", "minggu", "week", "update cuaca"
    ]
    
    message_lower = user_message.lower()
    
    # Kata kunci yang menandakan topik di luar cuaca
    off_topic_keywords = [
        "politik", "presiden", "pemilu", "korupsi", "agama", "islam", "kristen", "hindu", "buddha",
        "seks", "porn", "narkoba", "judi", "togel", "sabung", "ayam", "senjata", "bom", "teroris",
        "crypto", "investasi", "saham", "forex", "bisnis", "uang", "kaya", "miskin", "pekerjaan",
        "gaji", "hutang", "pinjaman", "kredit", "resep", "masak", "makanan", "film", "aktor", "artis",
        "gosip", "skandal", "perselingkuhan", "cerai", "nikah", "pernikahan", "sepak bola", "bola"
    ]
    
    # Cek apakah pertanyaan off-topic
    is_off_topic = any(kw in message_lower for kw in off_topic_keywords)
    
    # Cek apakah ada keyword cuaca
    has_weather_keyword = any(kw in message_lower for kw in weather_keywords)
    
    # Pertanyaan sapaan umum yang masih diperbolehkan
    greetings = ["hai", "hello", "halo", "hey", "assalamualaikum", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"]
    is_greeting = any(msg in message_lower for msg in greetings)
    
    # Pertanyaan tentang Ashley
    ashley_questions = ["siapa kamu", "kamu siapa", "nama kamu", "anda siapa", "ashley", "perkenalan"]
    is_ashley_question = any(q in message_lower for q in ashley_questions)
    
    if is_off_topic and not has_weather_keyword and not is_greeting:
        return {
            "reply": "🌤️ *Maaf, saya hanya bisa membantu pertanyaan tentang cuaca!*\n\nSaya adalah asisten cuaca cerdas Ashley. Silakan tanyakan hal-hal seperti:\n• Cuaca hari ini / besok\n• Prakiraan cuaca\n• Suhu, kelembaban, angin\n• Kualitas udara\n• Tips menghadapi cuaca tertentu\n\nAda yang bisa saya bantu terkait cuaca?"
        }
    
    if is_ashley_question:
        return {
            "reply": "Halo! Saya *Ashley* 👋\n\nSaya adalah asisten cuaca cerdas berbasis AI yang siap membantu Anda dengan:\n✅ Informasi cuaca real-time\n✅ Prakiraan cuaca\n✅ Analisis kualitas udara\n✅ Tips & rekomendasi cuaca\n\nTanyakan apapun tentang cuaca, dan saya akan dengan senang hati membantu! ☁️🌤️"
        }
    
    # Jika hanya sapaan
    if is_greeting and not has_weather_keyword and len(user_message.split()) < 4:
        return {
            "reply": "Halo! Selamat datang di WeatherAI ☁️\n\nAda yang bisa saya bantu tentang cuaca hari ini? Silakan tanyakan prakiraan cuaca, suhu, atau kondisi cuaca di lokasi Anda!"
        }
    
    # Dapatkan data cuaca saat ini untuk konteks
    try:
        lat = selected_location.get("latitude", -6.2)
        lon = selected_location.get("longitude", 106.816666)
        location_name = selected_location.get("name", "lokasi Anda")
        
        weather = get_current_weather(lat, lon)
        air_quality = get_air_quality(lat, lon)
        forecast = get_6day_forecast(lat, lon)
        
        condition = get_condition_text(weather.get("weather_code", 0))
        
        # Buat prompt untuk Gemini dengan konteks cuaca
        prompt = f"""Kamu adalah Ashley, asisten cuaca yang ramah dan profesional. Jawab pertanyaan pengguna tentang cuaca dengan singkat, padat, dan informatif (maksimal 3 kalimat jika memungkinkan). Gunakan bahasa Indonesia yang natural.

DATA CUACA REAL-TIME ({location_name}):
- Suhu: {int(weather.get('temperature', 0))}°C
- Kondisi: {condition}
- Kelembaban: {int(weather.get('humidity', 0))}%
- Angin: {int(weather.get('wind_speed', 0))} km/j
- Kualitas Udara: {air_quality.get('status', 'Baik')} (AQI {air_quality.get('aqi', 0)})

PERTANYAAN PENGGUNA: "{user_message}"

PANDUAN:
1. Jawab hanya terkait cuaca
2. Gunakan data cuaca di atas jika relevan
3. Jika ditanya prakiraan, beri gambaran singkat
4. Jika ditanya di luar cuaca, tolak dengan sopan
5. Jawaban singkat, maksimal 50 kata
6. Sertakan emoji yang relevan (maksimal 2)

JAWABAN:"""

        if AI_AVAILABLE and client:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            reply = response.text.strip()
            
            if len(reply) > 300:
                reply = reply[:300] + "..."
        else:
            # Fallback response
            reply = f"☁️ Cuaca di {location_name} saat ini {condition} dengan suhu {int(weather.get('temperature', 0))}°C. Kelembaban {int(weather.get('humidity', 0))}%. Ada yang ingin ditanyakan lagi?"
            
    except Exception as e:
        print(f"Chat AI error: {e}")
        reply = "Maaf, saya sedang mengalami gangguan teknis. Silakan coba lagi nanti."
    
    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

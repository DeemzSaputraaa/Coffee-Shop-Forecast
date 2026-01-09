# ☕ Coffee Shop Sales Forecast - Improved Version

Sistem prediksi penjualan untuk kedai kopi menggunakan Machine Learning dengan berbagai peningkatan fitur.

## 🎯 Fitur Utama

### ✨ Peningkatan Baru (Versi Improved)

1. **🛡️ Error Handling & Logging**
   - Penanganan error yang lebih baik untuk file yang hilang
   - Logging sistem untuk debugging
   - Pesan error yang informatif dengan solusi

2. **💰 Harga Aktual Per Produk**
   - Menggunakan harga aktual berdasarkan `product_detail`
   - Prediksi lebih akurat dengan data harga yang tepat
   - Display harga yang digunakan dalam prediksi

3. **📊 Confidence Interval**
   - Prediksi dengan batas atas dan batas bawah (80% confidence)
   - Visualisasi uncertainty untuk keputusan yang lebih baik
   - Metrik prediksi yang lebih komprehensif

4. **📈 Visualisasi Tren Historis**
   - Tab baru untuk analisis tren penjualan
   - Analisis per jam, per hari, dan per bulan
   - Identifikasi peak hours dan slow hours
   - Insights bisnis yang actionable

5. **🔧 Feature Engineering yang Lebih Baik**
   - Seasonality features (musim, quarter)
   - Holiday detection
   - Day of month features
   - Month start/end indicators
   - Model accuracy yang lebih tinggi

## 📁 Struktur File

```
coba cbl/
├── streamlit_app.py                    # Aplikasi Streamlit (IMPROVED)
├── trainingke100_improved.ipynb        # Notebook training baru
├── trainingke100.ipynb                 # Notebook training lama
├── model_prediksi_penjualan.pkl        # Model ML
├── label_encoder.pkl                   # Label encoder
├── Coffee_Shop.xlsx                    # Dataset
└── README.md                           # Dokumentasi ini
```

## 🚀 Cara Menggunakan

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy joblib openpyxl scikit-learn
```

### 2. Training Model Baru (Recommended)

Jalankan notebook `trainingke100_improved.ipynb` untuk mendapatkan model dengan accuracy lebih baik:

```bash
jupyter notebook trainingke100_improved.ipynb
```

Atau gunakan Google Colab untuk menjalankan notebook.

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📊 Fitur-Fitur Aplikasi

### 1. Dashboard Ringkas
- Live clock dengan auto-refresh
- Carousel produk terlaris (auto-sliding)
- Metrics penjualan utama

### 2. Tab Estimasi Penjualan
- Pilih rentang jam
- Pilih kategori, tipe, dan detail produk
- Lihat prediksi dengan confidence interval
- Visualisasi grafik prediksi

### 3. Tab Ranking Produk
- Analisis ranking berdasarkan jam tertentu
- Filter per kategori atau semua kategori
- Visualisasi top 15 produk
- Tabel detail ranking

### 4. Tab Tren Historis (NEW!)
- Analisis tren per jam
- Analisis tren per hari dalam minggu
- Analisis tren per bulan
- Identifikasi peak periods dan slow periods
- Detail data dalam expander

## 🔧 Improvements Details

### Error Handling
```python
# Sebelum
model = joblib.load(MODEL_PATH)  # Crash jika file tidak ada

# Sesudah
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("File tidak ditemukan...")
    st.info("Cara mengatasi: ...")
```

### Actual Pricing
```python
# Sebelum
unit_price = overall_median  # Semua produk pakai median

# Sesudah
unit_price = price_by_detail.get(product_detail, overall_median)
```

### Confidence Intervals
```python
# Sebelum
pred = predict_qty(model, X)

# Sesudah
pred, lower, upper = predict_qty_with_interval(model, X, confidence=0.80)
```

### Feature Engineering
```python
# Fitur Baru:
- day_of_month
- quarter (Q1, Q2, Q3, Q4)
- season (Winter, Spring, Summer, Fall)
- is_month_start / is_month_end
- is_holiday (deteksi hari libur)
```

## 📈 Model Performance

### Sebelum (Original)
- **MAE**: 0.199
- **RMSE**: 0.208
- **R² Score**: 0.034 (hanya 3.4% variance explained)

### Sesudah (Expected dengan feature engineering)
- **R² Score**: > 0.3 (target 30%+ variance explained)
- Akurasi prediksi lebih baik
- Lebih banyak pola tertangkap

*Note: Hasil aktual tergantung pada training dengan notebook improved*

## 🎨 UI/UX Features

- **Responsive Design**: Mobile-friendly
- **Dark Theme**: Modern glassmorphism design
- **Live Updates**: Auto-refresh untuk data real-time
- **Interactive Charts**: Visualisasi dengan Streamlit charts
- **User-Friendly**: Interface yang intuitif

## 🐛 Troubleshooting

### Error: File tidak ditemukan
**Solusi**: Pastikan semua file berada dalam satu folder:
- `model_prediksi_penjualan.pkl`
- `label_encoder.pkl`
- `Coffee_Shop.xlsx`

### Model accuracy rendah
**Solusi**: Training ulang dengan `trainingke100_improved.ipynb` untuk mendapatkan model dengan feature engineering yang lebih baik.

### Aplikasi lambat
**Solusi**: 
- Caching Streamlit sudah diaktifkan (`@st.cache_resource` dan `@st.cache_data`)
- Pastikan dataset tidak terlalu besar
- Tutup aplikasi lain yang berjalan

## 📝 Changelog

### Version 2.0 (Improved)
- ✅ Added error handling dengan logging
- ✅ Implemented actual pricing per product
- ✅ Added confidence intervals untuk predictions
- ✅ Created historical trends visualization tab
- ✅ Improved feature engineering (seasonality, holidays)
- ✅ Better code structure dan maintainability
- ✅ Comprehensive error messages

### Version 1.0 (Original)
- Basic prediction functionality
- Dashboard dengan carousel
- Ranking produk
- Simple Linear Regression model

## 👥 Credits

Developed untuk analisis dan forecasting penjualan Coffee Shop.

## 📄 License

Free to use for educational purposes.

---

**Selamat menggunakan! ☕**

Jika ada pertanyaan atau issue, silakan hubungi developer.

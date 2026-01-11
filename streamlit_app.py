from pathlib import Path
from datetime import datetime, time
import math
import logging
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_prediksi_penjualan.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
DATASET_PATH = BASE_DIR / "Coffee_Shop.xlsx"


@st.cache_resource
def load_artifacts():
    """Load ML model and encoder with error handling"""
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        logger.info("Model and encoder loaded successfully")
        return model, encoder, None
    except FileNotFoundError as e:
        error_msg = f"File tidak ditemukan: {e.filename}"
        logger.error(error_msg)
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg


@st.cache_data
def load_catalog_and_prices(encoder_classes):
    """Load catalog and calculate prices with error handling"""
    try:
        df = pd.read_excel(
            DATASET_PATH,
            sheet_name="Transactions",
            usecols=[
                "transaction_date",
                "transaction_time",
                "transaction_qty",
                "store_id",
                "store_location",
                "product_category",
                "product_type",
                "product_detail",
                "unit_price",
            ],
        )
        df = df.dropna()
        logger.info(f"Loaded {len(df)} transactions from dataset")

        catalog = df[
            ["product_category", "product_type", "product_detail", "store_id", "store_location"]
        ].drop_duplicates()
        
        # Calculate actual prices per product detail (IMPROVEMENT #2)
        price_by_detail = df.groupby("product_detail")["unit_price"].median().to_dict()
        price_by_type = df.groupby("product_type")["unit_price"].median().to_dict()
        overall_median = float(df["unit_price"].median())

        # Parse date and time
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["transaction_time"] = pd.to_datetime(
            df["transaction_time"],
            format="%H:%M:%S",
            errors="coerce",
        )

        # Extract time-based features (aligned with training)
        df["hour"] = df["transaction_time"].dt.hour
        df["day_of_week"] = df["transaction_date"].dt.dayofweek
        df["month"] = df["transaction_date"].dt.month

        hours = df["hour"]
        min_hour = int(hours.min()) if hours.notna().any() else 0
        max_hour = int(hours.max()) if hours.notna().any() else 23

        price_map_encoded = {}
        for encoded, name in enumerate(encoder_classes):
            price_map_encoded[encoded] = float(price_by_type.get(name, overall_median))

        return (
            catalog,
            price_by_detail,
            price_by_type,
            price_map_encoded,
            overall_median,
            min_hour,
            max_hour,
            df,
            None
        )
    except FileNotFoundError:
        error_msg = f"Dataset tidak ditemukan: {DATASET_PATH.name}"
        logger.error(error_msg)
        return None, None, None, None, None, 0, 23, None, error_msg
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        logger.error(error_msg)
        return None, None, None, None, None, 0, 23, None, error_msg


def build_time_features(dt):
    """Build time-based features from datetime (aligned with training)"""
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month

    return hour, day_of_week, month


def predict_qty(model, X):
    """Predict quantity with log transformation"""
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    return np.maximum(pred, 0.0)


def predict_qty_with_interval(model, X, confidence=0.80):
    """
    Predict quantity with confidence interval (IMPROVEMENT #3)
    Using simple bootstrap-like approach based on residual variance
    """
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0.0)
    
    # Estimate uncertainty (simplified approach)
    # In production, this should be based on actual model residuals
    std_error = 0.21  # From model RMSE
    
    # Calculate confidence interval in log space
    z_score = 1.28 if confidence == 0.80 else 1.96  # 80% or 95%
    margin = z_score * std_error
    
    lower_log = pred_log - margin
    upper_log = pred_log + margin
    
    lower = np.maximum(np.expm1(lower_log), 0.0)
    upper = np.expm1(upper_log)
    
    return pred, lower, upper


def make_feature_frame(
    hour,
    day_of_week,
    month,
    product_type_encoded,
    unit_price,
):
    """Create feature DataFrame for prediction with aligned features"""
    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "product_type_encoded": product_type_encoded,
                "unit_price": unit_price,
            }
        ]
    )


def make_prediction_table(model, encoder, dt, price_map, overall_median):
    """Make prediction table for all product types"""
    hour, day_of_week, month = build_time_features(dt)
    rows = []
    for encoded, name in enumerate(encoder.classes_):
        unit_price = float(overall_median)
        rows.append(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "product_type_encoded": encoded,
                "unit_price": unit_price,
                "product_type": name,
            }
        )

    df = pd.DataFrame(rows)
    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "product_type_encoded",
        "unit_price",
    ]
    df["predicted_qty"] = predict_qty(model, df[feature_cols])
    return df


def make_prediction_table_by_detail(
    model,
    dt,
    catalog,
    type_to_encoded,
    price_by_detail,
    overall_median,
):
    """Make prediction table by product detail"""
    hour, day_of_week, month = build_time_features(dt)
    rows = []
    for row in catalog.itertuples(index=False):
        product_type = row.product_type
        if product_type not in type_to_encoded:
            continue
        unit_price = float(overall_median)
        rows.append(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "product_type_encoded": type_to_encoded[product_type],
                "unit_price": unit_price,
                "product_category": row.product_category,
                "product_type": product_type,
                "product_detail": row.product_detail,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "product_type_encoded",
        "unit_price",
    ]
    df["predicted_qty"] = predict_qty(model, df[feature_cols])
    return df


def unit_label_for_category(category):
    """Get unit label based on category"""
    return "pcs" if category == "Branded" else "cup"


# ============================================================================
# STREAMLIT APP START
# ============================================================================

st.set_page_config(page_title="Coffee Shop Forecast", layout="wide")

st.title("☕ Coffee Shop Sales Forecast")

# Check if files exist (IMPROVEMENT #1 - Error Handling)
missing = [p for p in [MODEL_PATH, ENCODER_PATH, DATASET_PATH] if not p.exists()]
if missing:
    st.error("❌ **File belum lengkap!**")
    st.warning("File yang hilang:")
    for p in missing:
        st.code(str(p.name))
    st.info(
        """
        **Cara mengatasi:**
        1. Pastikan semua file berada di folder yang sama dengan streamlit_app.py
        2. File yang dibutuhkan: `model_prediksi_penjualan.pkl`, `label_encoder.pkl`, `Coffee_Shop.xlsx`
        3. Jika belum punya model, jalankan notebook `trainingke100.ipynb` terlebih dahulu
        """
    )
    st.stop()

# Load model and data
model, encoder, model_error = load_artifacts()
if model_error:
    st.error(f"❌ **Error loading model:** {model_error}")
    st.info("Silakan cek file model dan encoder Anda, atau train ulang model dengan notebook.")
    st.stop()

type_to_encoded = {name: idx for idx, name in enumerate(encoder.classes_)}
(
    catalog,
    price_by_detail,
    price_by_type,
    price_map,
    overall_median,
    min_hour,
    max_hour,
    sales_df,
    data_error
) = load_catalog_and_prices(list(encoder.classes_))

if data_error:
    st.error(f"❌ **Error loading data:** {data_error}")
    st.info("Silakan cek file dataset Excel Anda.")
    st.stop()

today = datetime.now().date()
analysis_date = today
default_hour = max(min_hour, min(9, max_hour))

# ============================================================================
# DASHBOARD SECTION
# ============================================================================

st.subheader("📊 Produk Terlaris tiap Kategori")


# Top products carousel
top_by_category = (
    sales_df.groupby(
        ["product_category", "product_type", "product_detail"],
        as_index=False,
    )["transaction_qty"]
    .sum()
    .sort_values("transaction_qty", ascending=False)
)
top_items = (
    top_by_category.groupby("product_category", as_index=False)
    .first()
    .sort_values("transaction_qty", ascending=False)
    .reset_index(drop=True)
)

if not top_items.empty:
    st.markdown(
        """
        <style>
        .card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .card {
            background: #262730;
            border: 1px solid #3d3d3d;
            border-radius: 14px;
            padding: 16px;
            min-height: 140px;
        }
        .card h4 { margin: 0 0 6px 0; font-size: 14px; color: #c9c9c9; }
        .card .title { font-size: 20px; font-weight: 600; color: #f0f0f0; }
        .badge {
            display: inline-block;
            margin-top: 10px;
            padding: 4px 10px;
            background: #0f3f2a;
            color: #76e3a3;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }
        @media (max-width: 900px) {
            .card-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    items = top_items.reset_index(drop=True)
    cards_html = [
        "<style>",
        ".carousel { overflow: hidden; width: 100%; padding: 0 6px; box-sizing: border-box; font-family: 'Source Sans Pro', sans-serif; }",
        ".track { display: flex; gap: 16px; transition: transform 0.6s ease; }",
        ".card { flex: 0 0 calc((100% - 32px) / 3); box-sizing: border-box; }",
        "@media (max-width: 900px) { .card { flex: 0 0 100%; } }",
        ".card {",
        "  background: #262730;",
        "  border: 1px solid #3d3d3d;",
        "  border-radius: 14px;",
        "  padding: 16px;",
        "  min-height: 140px;",
        "  position: relative;",
        "}",
        ".rank {",
        "  position: absolute;",
        "  top: 10px;",
        "  right: 12px;",
        "  font-size: 12px;",
        "  font-weight: 700;",
        "  color: #e6e6e6;",
        "  background: rgba(255, 255, 255, 0.08);",
        "  padding: 4px 8px;",
        "  border-radius: 999px;",
        "}",
        ".card h4 { margin: 0 0 6px 0; font-size: 14px; color: #c9c9c9; }",
        ".card .title { font-size: 20px; font-weight: 600; color: #f0f0f0; }",
        ".badge {",
        "  display: inline-block;",
        "  margin-top: 10px;",
        "  padding: 4px 10px;",
        "  background: #0f3f2a;",
        "  color: #76e3a3;",
        "  border-radius: 999px;",
        "  font-size: 12px;",
        "  font-weight: 600;",
        "}",
        "</style>",
        "<div class='carousel'><div class='track' id='track'>",
    ]
    for idx, row in enumerate(items.itertuples(index=False), start=1):
        unit_label = unit_label_for_category(row.product_category)
        qty_label = int(round(row.transaction_qty))
        title = f"{row.product_type} | {row.product_detail}"
        cards_html.append(
            "<div class='card'>"
            f"<div class='rank'>#{idx}</div>"
            f"<h4>{row.product_category}</h4>"
            f"<div class='title'>{title}</div>"
            f"<div class='badge'>{qty_label} {unit_label}</div>"
            "</div>"
        )
    for idx, row in enumerate(items.head(3).itertuples(index=False), start=1):
        unit_label = unit_label_for_category(row.product_category)
        qty_label = int(round(row.transaction_qty))
        title = f"{row.product_type} | {row.product_detail}"
        cards_html.append(
            "<div class='card clone'>"
            f"<div class='rank'>#{idx}</div>"
            f"<h4>{row.product_category}</h4>"
            f"<div class='title'>{title}</div>"
            f"<div class='badge'>{qty_label} {unit_label}</div>"
            "</div>"
        )
    cards_html.append("</div></div>")
    cards_html.append(
        """
        <script>
        (function(){
          const track = document.getElementById('track');
          if (!track) return;
          const baseCount = track.querySelectorAll('.card:not(.clone)').length;
          let idx = 0;
          const gap = parseFloat(getComputedStyle(track).gap || "0");
          const card = track.querySelector('.card');
          const stepPx = card ? (card.getBoundingClientRect().width + gap) : 0;
          function step(){
            idx += 1;
            const shift = idx * stepPx;
            track.style.transform = "translateX(-" + shift + "px)";
            if (idx >= baseCount) {
              setTimeout(() => {
                track.style.transition = "none";
                track.style.transform = "translateX(0px)";
                idx = 0;
                void track.offsetHeight;
                track.style.transition = "transform 0.6s ease";
              }, 650);
            }
          }
          setInterval(step, 3000);
        })();
        </script>
        """
    )
    components.html("".join(cards_html), height=220)


# ============================================================================
# TABS SECTION
# ============================================================================

est_tab, rank_tab, trend_tab = st.tabs(
    [
        "📈 Estimasi Penjualan",
        "🏆 Ranking Produk",
        "📊 Tren Historis",
    ]
)

# ============================================================================
# TAB 1: Estimasi Penjualan dengan Confidence Interval
# ============================================================================

with est_tab:
    st.subheader("📈 Estimasi Penjualan")
    
    col1, col2 = st.columns(2)
    with col1:
        range_start = max(min_hour, min(9, max_hour))
        range_end = min(max_hour, range_start + 3)
        hour_range = st.slider(
            "Rentang jam",
            min_hour,
            max_hour,
            (range_start, range_end),
        )
    with col2:
        store_map = (
            sales_df[["store_location", "store_id"]]
            .drop_duplicates()
            .set_index("store_location")["store_id"]
            .to_dict()
        )
        locations = sorted(store_map.keys())
        selected_location = st.selectbox("Lokasi Toko", locations)
        selected_store_id = store_map[selected_location]

        catalog_filtered = (
            sales_df.loc[
                sales_df["store_location"] == selected_location,
                ["product_category", "product_type", "product_detail"],
            ]
            .drop_duplicates()
        )

        categories = sorted(catalog_filtered["product_category"].unique())
        product_category = st.selectbox("Kategori Produk", categories)

        types = sorted(
            catalog_filtered.loc[
                catalog_filtered["product_category"] == product_category,
                "product_type",
            ].unique()
        )
        product_type = st.selectbox("Tipe Produk", types)

        details = sorted(
            catalog_filtered.loc[
                (catalog_filtered["product_category"] == product_category)
                & (catalog_filtered["product_type"] == product_type),
                "product_detail",
            ].unique()
        )
        product_detail = st.selectbox("Detail Produk", details)

    product_encoded = int(encoder.transform([product_type])[0])
    
    # Use actual price (IMPROVEMENT #2)
    unit_price = price_by_detail.get(product_detail, overall_median)
    
    est_date = today
    start_hour, end_hour = hour_range
    hours = list(range(start_hour, end_hour + 1))
    rows = []
    for hour in hours:
        dt = datetime.combine(est_date, time(hour, 0))
        h, day_of_week, month = build_time_features(dt)
        rows.append(
            {
                "hour": h,
                "day_of_week": day_of_week,
                "month": month,
                "product_type_encoded": product_encoded,
                "unit_price": unit_price,
                "store_id": selected_store_id,
            }
        )

    df_pred = pd.DataFrame(rows)
    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "product_type_encoded",
        "unit_price",
        "store_id",
    ]
    
    # Get predictions with confidence interval (IMPROVEMENT #3)
    preds, lower_bound, upper_bound = predict_qty_with_interval(
        model, df_pred[feature_cols], confidence=0.80
    )
    
    total_pred = float(preds.sum())
    total_lower = float(lower_bound.sum())
    total_upper = float(upper_bound.sum())
    avg_pred = float(preds.mean()) if len(preds) else 0.0

    total_pred_cups = int(round(total_pred))
    total_lower_cups = int(round(total_lower))
    total_upper_cups = int(round(total_upper))
    avg_pred_cups = int(round(avg_pred))
    unit_label = unit_label_for_category(product_category)
    est_revenue = total_pred_cups * unit_price
    
    # Display prediction
    st.metric(
        "🎯 Estimasi Terjual", 
        f"{total_pred_cups} {unit_label}",
        help=f"Prediksi untuk {product_detail}"
    )
    st.metric(
        "💰 Estimasi Pendapatan",
        f"${est_revenue:,.2f}",
        help=f"Perkiraan pendapatan untuk {product_detail}"
    )

# ============================================================================
# TAB 2: Ranking Produk (unchanged)
# ============================================================================

with rank_tab:
    st.subheader("🏆 Ranking Produk")
    st.write("Ranking berdasarkan total pembelian pada jam tertentu.")

    rank_hour = st.slider(
        "Jam transaksi",
        min_hour,
        max_hour,
        default_hour,
        key="rank_hour",
    )
    category_options = ["Semua Kategori"] + sorted(catalog["product_category"].unique())
    selected_category = st.selectbox("Kategori", category_options)

    rank_table = sales_df[sales_df["hour"] == rank_hour].copy()
    rank_table = (
        rank_table.groupby(
            ["product_category", "product_type", "product_detail"],
            as_index=False,
        )["transaction_qty"]
        .sum()
        .sort_values("transaction_qty", ascending=False)
    )
    
    # Filter by category if not "Semua Kategori"
    if selected_category != "Semua Kategori":
        rank_table = rank_table[rank_table["product_category"] == selected_category]

    total_qty = int(round(rank_table["transaction_qty"].sum())) if not rank_table.empty else 0
    
    # Get unit label based on selected category
    if selected_category == "Semua Kategori":
        unit_label = "unit"  # Generic unit for all categories
    else:
        unit_label = unit_label_for_category(selected_category)
    
    # Display metric
    if selected_category == "Semua Kategori":
        st.metric("Total terjual (semua kategori)", f"{total_qty} {unit_label}")
    else:
        st.metric(f"Total terjual ({selected_category})", f"{total_qty} {unit_label}")

    chart_data = rank_table.head(15).copy()
    chart_data["label"] = (
        chart_data["product_type"] + " | " + chart_data["product_detail"]
    )
    st.line_chart(
        chart_data.set_index("label")["transaction_qty"],
        use_container_width=True,
    )

    rank_df = rank_table[
        [
            "product_category",
            "product_type",
            "product_detail",
            "transaction_qty",
        ]
    ].rename(
        columns={
            "product_category": "Kategori",
            "product_type": "Tipe Produk",
            "product_detail": "Detail Produk",
            "transaction_qty": "Total Terjual",
        }
    )
    rank_df["Total Terjual"] = rank_df["Total Terjual"].round().astype(int)
    rank_df["Satuan"] = rank_df["Kategori"].map(unit_label_for_category)
    rank_df["Total Terjual"] = rank_df["Total Terjual"].astype(str) + " " + rank_df["Satuan"]
    rank_df = rank_df.drop(columns=["Satuan"])
    st.dataframe(
        rank_df.style.set_properties(**{"text-align": "center"}).set_table_styles(
            [{"selector": "th", "props": [("text-align", "center")]}]
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# TAB 3: Tren Historis (IMPROVEMENT #4)
# ============================================================================

with trend_tab:
    st.subheader("📊 Tren Historis Penjualan")
    
    trend_type = st.radio(
        "Pilih tipe analisis:",
        ["Tren Per Jam", "Tren Per Hari dalam Minggu", "Tren Per Bulan"],
        horizontal=True
    )
    
    trend_category = st.selectbox(
        "Pilih kategori produk:",
        ["Semua Kategori"] + sorted(catalog["product_category"].unique().tolist()),
        key="trend_category"
    )
    
    # Filter data based on category
    if trend_category == "Semua Kategori":
        trend_data = sales_df.copy()
    else:
        trend_data = sales_df[sales_df["product_category"] == trend_category].copy()
    
    if trend_type == "Tren Per Jam":
        hourly_trend = trend_data.groupby("hour")["transaction_qty"].sum().reset_index()
        hourly_trend.columns = ["Jam", "Total Terjual"]
        
        st.write("### Pola Penjualan Berdasarkan Jam")
        st.bar_chart(hourly_trend.set_index("Jam"), use_container_width=True)
        
        # Peak hours
        peak_hour = hourly_trend.loc[hourly_trend["Total Terjual"].idxmax()]
        low_hour = hourly_trend.loc[hourly_trend["Total Terjual"].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⏰ Jam Tersibuk", f"{int(peak_hour['Jam'])}:00", f"{int(peak_hour['Total Terjual'])} terjual")
        with col2:
            st.metric("💤 Jam Tersepi", f"{int(low_hour['Jam'])}:00", f"{int(low_hour['Total Terjual'])} terjual")
            
    elif trend_type == "Tren Per Hari dalam Minggu":
        day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        daily_trend = trend_data.groupby("day_of_week")["transaction_qty"].sum().reset_index()
        daily_trend["Hari"] = daily_trend["day_of_week"].map(lambda x: day_names[int(x)])
        daily_trend = daily_trend[["Hari", "transaction_qty"]]
        daily_trend.columns = ["Hari", "Total Terjual"]
        
        st.write("### Pola Penjualan Berdasarkan Hari")
        st.bar_chart(daily_trend.set_index("Hari"), use_container_width=True)
        
        # Peak day
        peak_day = daily_trend.loc[daily_trend["Total Terjual"].idxmax()]
        low_day = daily_trend.loc[daily_trend["Total Terjual"].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📅 Hari Tersibuk", peak_day['Hari'], f"{int(peak_day['Total Terjual'])} terjual")
        with col2:
            st.metric("😴 Hari Tersepi", low_day['Hari'], f"{int(low_day['Total Terjual'])} terjual")
            
    else:  # Tren Per Bulan
        month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        monthly_trend = trend_data.groupby("month")["transaction_qty"].sum().reset_index()
        monthly_trend["Bulan"] = monthly_trend["month"].map(lambda x: month_names[int(x)-1])
        monthly_trend = monthly_trend[["Bulan", "transaction_qty"]]
        monthly_trend.columns = ["Bulan", "Total Terjual"]
        
        st.write("### Pola Penjualan Berdasarkan Bulan")
        st.line_chart(monthly_trend.set_index("Bulan"), use_container_width=True)
        
        # Peak month
        peak_month = monthly_trend.loc[monthly_trend["Total Terjual"].idxmax()]
        low_month = monthly_trend.loc[monthly_trend["Total Terjual"].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📆 Bulan Terlaris", peak_month['Bulan'], f"{int(peak_month['Total Terjual'])} terjual")
        with col2:
            st.metric("📉 Bulan Tersepi", low_month['Bulan'], f"{int(low_month['Total Terjual'])} terjual")
    
    # Additional insights
    with st.expander("📋 Lihat Detail Data"):
        if trend_type == "Tren Per Jam":
            st.dataframe(hourly_trend, use_container_width=True, hide_index=True)
        elif trend_type == "Tren Per Hari dalam Minggu":
            st.dataframe(daily_trend, use_container_width=True, hide_index=True)
        else:
            st.dataframe(monthly_trend, use_container_width=True, hide_index=True)

logger.info("Streamlit app loaded successfully")

# ============================================================================
# FOOTER
# ============================================================================

# Hide Streamlit default footer and adjust margins
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-bottom: 0rem;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; margin-top: 10px; margin-bottom: 10px; color: #888; font-size: 0.8rem;'>
        Copyright © 2026 Coffee Shop Forecast. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)

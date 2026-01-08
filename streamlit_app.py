from pathlib import Path
from datetime import datetime, time
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_prediksi_penjualan.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
DATASET_PATH = BASE_DIR / "Coffee_Shop.xlsx"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


@st.cache_data
def load_catalog_and_prices(encoder_classes):
    df = pd.read_excel(
        DATASET_PATH,
        sheet_name="Transactions",
        usecols=[
            "product_category",
            "product_type",
            "product_detail",
            "unit_price",
            "transaction_time",
            "transaction_qty",
        ],
    )
    df = df.dropna()

    catalog = df[["product_category", "product_type", "product_detail"]].drop_duplicates()
    price_by_detail = df.groupby("product_detail")["unit_price"].median()
    price_by_type = df.groupby("product_type")["unit_price"].median()
    overall_median = float(df["unit_price"].median())

    time_parsed = pd.to_datetime(
        df["transaction_time"].astype(str),
        format="%H:%M:%S",
        errors="coerce",
    )
    df["hour"] = time_parsed.dt.hour
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
    )


def build_time_features(dt):
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    weekend = 1 if day_of_week >= 5 else 0
    return hour, day_of_week, month, weekend


def predict_qty(model, X):
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    return np.maximum(pred, 0.0)


def make_feature_frame(hour, day_of_week, month, weekend, product_type_encoded, unit_price):
    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "weekend": weekend,
                "product_type_encoded": product_type_encoded,
                "unit_price": unit_price,
            }
        ]
    )


def make_prediction_table(model, encoder, dt, price_map, overall_median):
    hour, day_of_week, month, weekend = build_time_features(dt)
    rows = []
    for encoded, name in enumerate(encoder.classes_):
        unit_price = float(overall_median)
        rows.append(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "weekend": weekend,
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
        "weekend",
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
    hour, day_of_week, month, weekend = build_time_features(dt)
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
                "weekend": weekend,
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
        "weekend",
        "product_type_encoded",
        "unit_price",
    ]
    df["predicted_qty"] = predict_qty(model, df[feature_cols])
    return df


st.set_page_config(page_title="Coffee Shop Forecast", layout="wide")

st.title("Coffee Shop Sales Forecast")
# st.write("Model menggunakan fitur waktu dan produk untuk estimasi jumlah terjual.")

missing = [p for p in [MODEL_PATH, ENCODER_PATH, DATASET_PATH] if not p.exists()]
if missing:
    st.error("File belum lengkap: " + ", ".join(str(p.name) for p in missing))
    st.stop()

model, encoder = load_artifacts()
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
) = load_catalog_and_prices(list(encoder.classes_))

today = datetime.now().date()
analysis_date = today
default_hour = max(min_hour, min(9, max_hour))

def unit_label_for_category(category):
    return "pcs" if category == "Branded" else "cup"

st.subheader("Dashboard ringkas")
if st_autorefresh is not None:
    st_autorefresh(interval=3000, key="clock_refresh")
components.html(
    """
    <div id="clock" style="color:#c9c9c9; font-size:0.9rem; margin-bottom:0.5rem;"></div>
    <script>
      function pad(n){return n.toString().padStart(2,'0');}
      function tick(){
        const d = new Date();
        const t = pad(d.getHours())+":"+pad(d.getMinutes())+":"+pad(d.getSeconds());
        document.getElementById("clock").innerText = "Jam sekarang: " + t;
      }
      tick();
      setInterval(tick, 1000);
    </script>
    """,
    height=32,
)

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
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
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
        ".carousel { overflow: hidden; width: 100%; padding: 0 6px; box-sizing: border-box; }",
        ".track { display: flex; gap: 16px; transition: transform 0.6s ease; }",
        ".card { flex: 0 0 calc((100% - 32px) / 3); box-sizing: border-box; }",
        "@media (max-width: 900px) { .card { flex: 0 0 100%; } }",
        ".card {",
        "  background: rgba(255, 255, 255, 0.04);",
        "  border: 1px solid rgba(255, 255, 255, 0.08);",
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


est_tab, rank_tab = st.tabs(
    [
        "Estimasi penjualan",
        "Ranking produk",
    ]
)


with est_tab:
    st.subheader("Estimasi penjualan")
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
        categories = sorted(catalog["product_category"].unique())
        product_category = st.selectbox("Product category", categories)

        types = sorted(
            catalog.loc[catalog["product_category"] == product_category, "product_type"].unique()
        )
        product_type = st.selectbox("Product type", types)

        details = sorted(
            catalog.loc[
                (catalog["product_category"] == product_category)
                & (catalog["product_type"] == product_type),
                "product_detail",
            ].unique()
        )
        product_detail = st.selectbox("Product detail", details)

    product_encoded = int(encoder.transform([product_type])[0])
    unit_price = float(overall_median)

    est_date = today
    start_hour, end_hour = hour_range
    hours = list(range(start_hour, end_hour + 1))
    rows = []
    for hour in hours:
        dt = datetime.combine(est_date, time(hour, 0))
        h, day_of_week, month, weekend = build_time_features(dt)
        rows.append(
            {
                "hour": h,
                "day_of_week": day_of_week,
                "month": month,
                "weekend": weekend,
                "product_type_encoded": product_encoded,
                "unit_price": unit_price,
            }
        )

    df_pred = pd.DataFrame(rows)
    preds = predict_qty(
        model,
        df_pred[
            [
                "hour",
                "day_of_week",
                "month",
                "weekend",
                "product_type_encoded",
                "unit_price",
            ]
        ],
    )
    total_pred = float(preds.sum())
    avg_pred = float(preds.mean()) if len(preds) else 0.0

    total_pred_cups = int(round(total_pred))
    avg_pred_cups = int(round(avg_pred))
    unit_label = unit_label_for_category(product_category)
    st.metric("Estimasi jumlah terjual (rentang jam)", f"{total_pred_cups} {unit_label}")
    # st.caption(f"Rata-rata per jam: {avg_pred_cups} {unit_label}")

with rank_tab:
    st.subheader("Ranking produk")
    st.write("Ranking berdasarkan total pembelian pada jam tertentu.")

    rank_hour = st.slider(
        "Jam transaksi",
        min_hour,
        max_hour,
        default_hour,
        key="rank_hour",
    )
    category_options = sorted(catalog["product_category"].unique())
    selected_category = st.selectbox("Category", category_options)

    if "rank_mode" not in st.session_state:
        st.session_state["rank_mode"] = "category"

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Lihat ranking produk keseluruhan"):
            st.session_state["rank_mode"] = "all"
    with btn_col2:
        if st.button("Lihat ranking kategori terpilih"):
            st.session_state["rank_mode"] = "category"

    rank_table = sales_df[sales_df["hour"] == rank_hour].copy()
    rank_table = (
        rank_table.groupby(
            ["product_category", "product_type", "product_detail"],
            as_index=False,
        )["transaction_qty"]
        .sum()
        .sort_values("transaction_qty", ascending=False)
    )
    if st.session_state["rank_mode"] == "category":
        rank_table = rank_table[rank_table["product_category"] == selected_category]

    total_qty = int(round(rank_table["transaction_qty"].sum())) if not rank_table.empty else 0
    unit_label = unit_label_for_category(selected_category)
    if st.session_state["rank_mode"] == "all":
        st.metric("Total terjual (semua kategori)", f"{total_qty} {unit_label}")
    else:
        st.metric("Total terjual kategori", f"{total_qty} {unit_label}")

    chart_data = rank_table.head(15).copy()
    chart_data["label"] = (
        chart_data["product_type"] + " | " + chart_data["product_detail"]
    )
    st.line_chart(
        chart_data.set_index("label")["transaction_qty"],
        width="stretch",
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
        width="stretch",
        hide_index=True,
    )

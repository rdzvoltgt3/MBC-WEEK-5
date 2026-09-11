"""
app.py — Prakiraan Suhu Udara Jena (versi 1.0, fungsi dasar)

Gaya tampilan sama dengan v2 (komponen bawaan Streamlit, font standar), tetapi cakupan fitur
sengaja minimal. Fitur lain ditambahkan di v2:
  * satu model saja (LSTM di app_lstm.py, GRU di app_gru.py)
  * data contoh periode test, pilih rentang tanggal (maks 7 hari)
  * grafik prediksi vs aktual + MAE/RMSE + prakiraan 10 menit berikutnya
  * cache model & data (st.cache_resource / st.cache_data)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inference as inf

APP_VERSION = "1.0"
BASE_DIR = Path(__file__).resolve().parent
MAX_DAYS = 7
STEPS_PER_DAY = 24 * 60 // inf.STEP_MINUTES

# Warna grafik sama dengan v2: biru-tinta untuk data aktual, warna berbeda per model.
INK = "#1B2F45"
LINE = "#D5DEE7"
GRID = "#E6ECF1"
MODEL_STYLE = {"LSTM": {"color": "#B45309", "dash": "solid"}, "GRU": {"color": "#0F766E", "dash": "dot"}}
MODEL_NAMES = {"LSTM": "Long Short-Term Memory", "GRU": "Gated Recurrent Unit"}
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


# ----------------------------------------------------------------------------------------------
# Format angka & waktu (gaya Indonesia: koma desimal), sama dengan v2
# ----------------------------------------------------------------------------------------------
def num(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def fmt_dt(ts: pd.Timestamp) -> str:
    return f"{HARI[ts.weekday()]}, {ts.day} {BULAN[ts.month - 1]} {ts.year} pukul {ts:%H:%M}"


def fmt_date(ts: pd.Timestamp) -> str:
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"


# ----------------------------------------------------------------------------------------------
# Loader ber-cache
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    scaler = inf.MinMaxParams.from_json(BASE_DIR / "models" / "scaler.json")
    meta = json.loads((BASE_DIR / "models" / "metadata.json").read_text())
    return scaler, meta


@st.cache_resource(show_spinner="Memuat model…")
def load_model(name: str) -> inf.OnnxForecaster:
    _, meta = load_artifacts()
    return inf.OnnxForecaster(BASE_DIR / meta["models"][name]["onnx_file"], name, meta["seq_len"])


@st.cache_data(show_spinner="Menyiapkan data…")
def load_data():
    data, _ = inf.clean_dataframe(pd.read_csv(BASE_DIR / "data" / "jena_test_period.csv.gz"))
    scaler, _ = load_artifacts()
    return data, scaler.transform(data.to_numpy(dtype=np.float64))


# ----------------------------------------------------------------------------------------------
# Grafik
# ----------------------------------------------------------------------------------------------
def forecast_figure(times, actual, pred, primary: str) -> go.Figure:
    style = MODEL_STYLE[primary]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=actual, name="Suhu aktual",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.2f} °C"))
    fig.add_trace(go.Scatter(x=times, y=pred, name=f"Prediksi {primary}",
                             line=dict(color=style["color"], width=2, dash=style["dash"]),
                             hovertemplate="%{y:.2f} °C"))
    fig.update_layout(
        height=450, margin=dict(l=10, r=10, t=30, b=10), separators=",.",
        font=dict(color=INK, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=LINE, zeroline=False, title_text="Suhu (°C)")
    return fig


# ----------------------------------------------------------------------------------------------
# Aplikasi
# ----------------------------------------------------------------------------------------------
def main(primary: str = "LSTM") -> None:
    primary = primary.upper()
    st.set_page_config(page_title=f"Prakiraan Suhu Jena · {primary}", page_icon="🌡️",
                       layout="wide", initial_sidebar_state="expanded")
    scaler, meta = load_artifacts()
    seq_len = meta["seq_len"]
    model = load_model(primary)
    data, scaled = load_data()
    idx = data.index

    with st.sidebar:
        st.markdown(f"**Aplikasi model {primary}**")
        st.caption(MODEL_NAMES[primary])
        st.subheader("1. Pilih rentang waktu")
        first_ok, last = idx[seq_len].date(), idx[-1].date()
        end_default = (idx[-1] - pd.Timedelta(minutes=inf.STEP_MINUTES)).date()
        picked = st.date_input(
            "Tanggal awal dan akhir", value=(end_default - pd.Timedelta(days=2), end_default),
            min_value=first_ok, max_value=last, format="DD/MM/YYYY",
            help=f"Maksimal {MAX_DAYS} hari. Model butuh 72 jam riwayat sebelum tanggal awal.")
        st.caption(f"Data contoh tersedia {fmt_date(idx[0])} s.d. {fmt_date(idx[-1])}.")
        st.divider()
        st.caption(f"Versi {APP_VERSION}. Model {primary} dari eksperimen Week 3 (Jena Climate).")

    st.title("Berapa suhu udara di Jena 10 menit lagi?")
    st.write(f"Model {primary} ({MODEL_NAMES[primary]}) membaca data cuaca 72 jam terakhir dari "
             "stasiun Institut Max Planck di Jena, Jerman, lalu menebak suhu udara 10 menit "
             "berikutnya. Pilih rentang waktu di panel kiri, hasilnya muncul di bawah.")

    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pilih tanggal akhir di kalender panel kiri untuk menyelesaikan rentang waktu.")
        st.stop()

    lo = max(int(idx.searchsorted(pd.Timestamp(picked[0]))), seq_len)
    hi = int(idx.searchsorted(pd.Timestamp(picked[1]) + pd.Timedelta(days=1)))
    hi = min(hi, lo + MAX_DAYS * STEPS_PER_DAY)
    if hi <= lo:
        st.warning("Rentang yang dipilih tidak berisi data. Coba pilih tanggal lain.")
        st.stop()

    pred, _ = inf.rolling_forecast(model, scaled, scaler, np.arange(lo, hi))
    actual = data[inf.TARGET_COL].to_numpy()[lo:hi]
    m = inf.regression_metrics(actual, pred)
    next_value = inf.forecast_next(model, scaled[:hi], scaler)
    next_time = idx[hi - 1] + pd.Timedelta(minutes=inf.STEP_MINUTES)

    with st.container(border=True):
        left, right = st.columns([1, 2])
        left.metric(f"Prakiraan {primary}", f"{num(next_value, 1)} °C",
                    help="Tebakan suhu untuk 10 menit setelah akhir rentang yang dipilih.")
        right.markdown(f"**{fmt_dt(next_time)}**")

    c1, c2 = st.columns(2)
    c1.metric("Rata-rata meleset (MAE)", f"{num(m['MAE'], 3)} °C",
              help="Rata-rata selisih antara tebakan model dan suhu yang tercatat.")
    c2.metric("RMSE", f"{num(m['RMSE'], 3)} °C",
              help="Seperti MAE, tetapi kesalahan besar dihitung lebih berat.")

    st.plotly_chart(forecast_figure(idx[lo:hi], actual, pred, primary), use_container_width=True)

    with st.expander("Tentang model"):
        info = meta["models"][primary]
        st.markdown(
            f"- Arsitektur: {primary}, hidden size {info['hidden_size']}, {info['num_layers']} layer, "
            f"{num(info['params'], 0)} parameter.\n"
            f"- Input: 72 jam terakhir (432 langkah × 12 fitur cuaca), output: suhu 10 menit ke depan.\n"
            f"- Performa di seluruh data test: MAE {num(info['test_mae'], 4)} °C, "
            f"RMSE {num(info['test_rmse'], 4)} °C.")


if __name__ == "__main__":
    main("LSTM")

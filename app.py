#v2
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inference as inf

APP_VERSION = "2.0"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEMO_PATH = BASE_DIR / "data" / "jena_test_period.csv.gz"
MAX_DAYS = 14
STEPS_PER_DAY = 24 * 60 // inf.STEP_MINUTES
MAX_STEPS = MAX_DAYS * STEPS_PER_DAY

# Token warna: biru-tinta stasiun untuk teks & data aktual, aksen per model agar 4 deployment
# mudah dibedakan (LSTM = amber, GRU = teal). Warna tidak pernah menjadi satu-satunya pembawa
# makna: setiap garis juga punya nama di legenda & gaya garis berbeda.
INK = "#1B2F45"
MUTED = "#566779"
LINE = "#D5DEE7"
GRID = "#E6ECF1"
BASELINE_COLOR = "#8A99A8"
MODEL_STYLE = {
    "LSTM": {"color": "#B45309", "fill": "rgba(180, 83, 9, 0.15)", "dash": "solid"},
    "GRU": {"color": "#0F766E", "fill": "rgba(15, 118, 110, 0.15)", "dash": "dot"},
}
MODEL_INFO = {
    "LSTM": {
        "full": "Long Short-Term Memory",
        "plain": ("Jaringan saraf yang membaca data secara berurutan dan punya semacam buku catatan "
                  "internal. Tiga gerbang (lupa, masuk, keluar) memutuskan informasi mana dari 72 jam "
                  "terakhir yang perlu diingat, diperbarui, atau dibuang."),
    },
    "GRU": {
        "full": "Gated Recurrent Unit",
        "plain": ("Saudara yang lebih ringkas dari LSTM. Hanya memakai dua gerbang (perbarui dan reset), "
                  "sehingga parameternya lebih sedikit dan lebih ringan dijalankan, namun sering sama "
                  "akuratnya untuk data yang polanya teratur seperti suhu."),
    },
}
INTERVAL_OPTIONS = {"Tanpa rentang": None, "80%": "0.80", "90%": "0.90", "95%": "0.95"}
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
html, body, .stApp, .stMarkdown, h1, h2, h3, h4, p, li, label, input, textarea, button,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    font-family: 'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
}}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 1280px; }}
h2, h3 {{ color: {INK}; letter-spacing: -0.005em; }}

.hero {{ display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(300px, 1fr);
        gap: 2.2rem; align-items: stretch; margin: 0 0 1.4rem; }}
.hero-title {{ font-size: 2.15rem; line-height: 1.15; font-weight: 700; color: {INK};
              margin: 0 0 .7rem; letter-spacing: -0.015em; }}
.hero-text p {{ color: {MUTED}; font-size: 1.03rem; line-height: 1.6; max-width: 62ch; margin: 0 0 .55rem; }}
.hero-text .model-line {{ color: {INK}; }}
.hero-text .swatch {{ display: inline-block; width: .75rem; height: .75rem; border-radius: 2px;
                     background: var(--accent); margin-right: .4rem; vertical-align: baseline; }}

.readout {{ background: #FFFFFF; border: 1px solid {LINE}; border-left: 6px solid var(--accent);
           border-radius: 10px; padding: 1.1rem 1.35rem 1rem; }}
.readout-label {{ color: {MUTED}; font-size: .93rem; }}
.readout-value {{ font-size: 3.3rem; font-weight: 700; color: {INK}; line-height: 1.05;
                 font-variant-numeric: tabular-nums; margin: .3rem 0 .35rem; }}
.readout-value span {{ font-size: 1.45rem; font-weight: 400; color: {MUTED}; margin-left: .15rem; }}
.readout-sub {{ font-size: .95rem; color: {INK}; line-height: 1.5; }}
.strip {{ position: relative; height: 10px; border-radius: 5px; margin: 1rem 0 .35rem;
         background: linear-gradient(90deg, #3B6EA5 0%, #A9C1D6 45%, #E9C46A 75%, #B45309 100%); }}
.strip-marker {{ position: absolute; top: -5px; width: 4px; height: 20px; background: {INK};
                border-radius: 2px; transform: translateX(-50%); outline: 2px solid #fff; }}
.strip-labels {{ display: flex; justify-content: space-between; font-size: .8rem; color: {MUTED}; }}

.model-badge {{ display: flex; gap: .7rem; align-items: center; padding: .75rem .85rem;
               border: 1px solid {LINE}; border-left: 5px solid var(--accent); border-radius: 8px;
               background: #FFFFFF; margin-bottom: .4rem; }}
.model-badge strong {{ color: {INK}; font-size: 1rem; }}
.model-badge small {{ color: {MUTED}; }}
.step {{ display: flex; align-items: center; gap: .55rem; font-weight: 700; color: {INK};
        margin: 1.3rem 0 .35rem; font-size: 1.02rem; }}
.step-num {{ display: inline-flex; align-items: center; justify-content: center; width: 1.6rem;
            height: 1.6rem; border-radius: 50%; background: {INK}; color: #fff; font-size: .85rem; }}

[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 8px;
                           padding: .75rem 1rem .6rem; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; color: {INK}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1rem; padding-left: .2rem; padding-right: .2rem; }}
.note {{ color: {MUTED}; font-size: .92rem; line-height: 1.55; }}
.verdict {{ font-size: 1.08rem; line-height: 1.6; color: {INK}; max-width: 75ch; margin: .2rem 0 1rem; }}
@media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; gap: 1.2rem; }}
                             .hero-title {{ font-size: 1.7rem; }} }}
</style>
"""


# ----------------------------------------------------------------------------------------------
# Format angka & waktu (gaya Indonesia: koma desimal)
# ----------------------------------------------------------------------------------------------
def num(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def fmt_dt(ts: pd.Timestamp, with_day: bool = True) -> str:
    base = f"{ts.day} {BULAN[ts.month - 1]} {ts.year} pukul {ts:%H:%M}"
    return f"{HARI[ts.weekday()]}, {base}" if with_day else base


def fmt_date(ts: pd.Timestamp) -> str:
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"


# ----------------------------------------------------------------------------------------------
# Loader ber-cache (optimasi: model & data hanya dimuat sekali per server, bukan setiap klik)
# ----------------------------------------------------------------------------------------------
REQUIRED_FILES = ["models/scaler.json", "models/metadata.json", "models/lstm_best.onnx",
                  "models/gru_best.onnx", "data/jena_test_period.csv.gz"]


def missing_artifacts() -> list[str]:
    return [f for f in REQUIRED_FILES if not (BASE_DIR / f).exists()]


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[inf.MinMaxParams, dict]:
    scaler = inf.MinMaxParams.from_json(MODELS_DIR / "scaler.json")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    return scaler, metadata


@st.cache_resource(show_spinner="Memuat model…")
def load_model(name: str) -> inf.OnnxForecaster:
    _, meta = load_artifacts()
    return inf.OnnxForecaster(BASE_DIR / meta["models"][name]["onnx_file"], name, meta["seq_len"])


@st.cache_data(show_spinner="Menyiapkan data contoh…")
def load_demo() -> tuple[pd.DataFrame, dict]:
    return inf.clean_dataframe(pd.read_csv(DEMO_PATH))


@st.cache_data(show_spinner="Membaca file yang diunggah…", max_entries=5)
def load_upload(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
        if raw.shape[1] == 1:  # kemungkinan pemisah ';' atau tab
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - pesan diteruskan ke pengguna
        raise inf.DataValidationError(f"File tidak bisa dibaca sebagai CSV ({exc}).") from exc
    return inf.clean_dataframe(raw)


@st.cache_data(show_spinner=False, max_entries=5)
def scale_data(data_key: str, _data: pd.DataFrame) -> np.ndarray:
    scaler, _ = load_artifacts()
    return scaler.transform(_data.to_numpy(dtype=np.float64))


@st.cache_data(show_spinner=False, max_entries=64)
def run_backtest(model_name: str, data_key: str, lo: int, hi: int,
                 _scaled: np.ndarray) -> tuple[np.ndarray, float]:
    """Hasil disimpan per (model, data, rentang): mengganti opsi tampilan tidak memicu prediksi ulang."""
    scaler, _ = load_artifacts()
    return inf.rolling_forecast(load_model(model_name), _scaled, scaler, np.arange(lo, hi))


@st.cache_data(show_spinner=False)
def template_csv() -> bytes:
    data, _ = load_demo()
    sample = data.tail(600).copy()
    sample.index = sample.index.strftime(inf.DATETIME_FORMAT)
    sample.index.name = inf.DATETIME_COL
    return sample.to_csv().encode("utf-8")


# ----------------------------------------------------------------------------------------------
# Grafik
# ----------------------------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 420, y_title: str = "Suhu (°C)") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), separators=",.",
        font=dict(family="Atkinson Hyperlegible, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family="Atkinson Hyperlegible, sans-serif"),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=LINE, zeroline=False, title_text=y_title)
    return fig


PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def forecast_figure(res: pd.DataFrame, primary: str, q: float | None, show_baseline: bool) -> go.Figure:
    style = MODEL_STYLE[primary]
    pred = res[f"Prediksi {primary} (°C)"]
    fig = go.Figure()
    if q is not None:
        fig.add_trace(go.Scatter(x=res["Waktu"], y=pred + q, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=res["Waktu"], y=pred - q, line=dict(width=0), fill="tonexty",
                                 fillcolor=style["fill"], hoverinfo="skip", name="Rentang kemungkinan"))
    fig.add_trace(go.Scatter(x=res["Waktu"], y=res["Suhu aktual (°C)"], name="Suhu aktual",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.2f} °C"))
    fig.add_trace(go.Scatter(x=res["Waktu"], y=pred, name=f"Prediksi {primary}",
                             line=dict(color=style["color"], width=2, dash=style["dash"]),
                             hovertemplate="%{y:.2f} °C"))
    if show_baseline:
        fig.add_trace(go.Scatter(x=res["Waktu"], y=res["Tebakan naif (°C)"], name="Tebakan naif",
                                 line=dict(color=BASELINE_COLOR, width=1.5, dash="dash"),
                                 hovertemplate="%{y:.2f} °C"))
    fig = style_fig(fig, 460)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor="#F3F6F9"))
    return fig


def comparison_figure(res: pd.DataFrame, order: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["Waktu"], y=res["Suhu aktual (°C)"], name="Suhu aktual",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.2f} °C"))
    for name in order:
        s = MODEL_STYLE[name]
        fig.add_trace(go.Scatter(x=res["Waktu"], y=res[f"Prediksi {name} (°C)"], name=f"Prediksi {name}",
                                 line=dict(color=s["color"], width=2, dash=s["dash"]),
                                 hovertemplate="%{y:.2f} °C"))
    return style_fig(fig, 420)


def error_time_figure(res: pd.DataFrame, order: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in order:
        s = MODEL_STYLE[name]
        fig.add_trace(go.Scatter(x=res["Waktu"], y=res[f"Selisih {name} (°C)"], name=name,
                                 line=dict(color=s["color"], width=1.4, dash=s["dash"]),
                                 hovertemplate="%{y:+.2f} °C"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    return style_fig(fig, 340, "Prediksi − aktual (°C)")


def error_hist_figure(res: pd.DataFrame, order: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in order:
        fig.add_trace(go.Histogram(x=res[f"Selisih {name} (°C)"], name=name, opacity=0.6, nbinsx=60,
                                   marker_color=MODEL_STYLE[name]["color"]))
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig = style_fig(fig, 340, "Jumlah titik")
    fig.update_xaxes(title_text="Prediksi − aktual (°C)")
    return fig


def error_hour_figure(by_hour: pd.DataFrame, order: list[str]) -> go.Figure:
    fig = go.Figure()
    for name in order:
        fig.add_trace(go.Bar(x=by_hour.index, y=by_hour[name], name=name,
                             marker_color=MODEL_STYLE[name]["color"], hovertemplate="%{y:.3f} °C"))
    fig.update_layout(barmode="group")
    fig = style_fig(fig, 340, "Rata-rata meleset (°C)")
    fig.update_xaxes(title_text="Jam dalam sehari", dtick=2)
    return fig


# ----------------------------------------------------------------------------------------------
# Potongan UI
# ----------------------------------------------------------------------------------------------
def step_heading(n: int, text: str) -> None:
    st.markdown(f'<div class="step"><span class="step-num">{n}</span>{text}</div>', unsafe_allow_html=True)


def hero(primary: str, forecast_value: float, forecast_time: pd.Timestamp, q: float | None,
         level: str, actual: float | None, lo72: float, hi72: float) -> None:
    color = MODEL_STYLE[primary]["color"]
    span = max(hi72 - lo72, 1e-6)
    pos = float(np.clip((forecast_value - lo72) / span, 0, 1) * 100)
    range_line = (f"Kemungkinan besar antara {num(forecast_value - q)} dan {num(forecast_value + q)} °C "
                  f"(keyakinan {level})." if q is not None else "")
    actual_line = (f"<br>Tercatat sebenarnya: <strong>{num(actual)} °C</strong> "
                   f"(meleset {num(abs(forecast_value - actual))} °C)." if actual is not None else "")
    html = (
        f'<div class="hero" style="--accent:{color}">'
        '<div class="hero-text">'
        '<div class="hero-title" role="heading" aria-level="1">Berapa suhu udara di Jena 10 menit lagi?</div>'
        f'<p>Aplikasi ini membaca data cuaca 72 jam terakhir dari stasiun Institut Max Planck di Jena, '
        f'Jerman (suhu, tekanan, kelembapan, angin, dan beberapa ukuran lain), lalu menebak suhu '
        f'udara 10 menit berikutnya.</p>'
        f'<p class="model-line"><span class="swatch"></span>Model utama: <strong>{primary}</strong> '
        f'({MODEL_INFO[primary]["full"]}). Pilih data dan rentang waktu di panel kiri, hasilnya muncul di bawah.</p>'
        '</div>'
        '<div class="readout">'
        f'<div class="readout-label">Prakiraan untuk {fmt_dt(forecast_time)}</div>'
        f'<div class="readout-value">{num(forecast_value, 1)}<span>°C</span></div>'
        f'<div class="readout-sub">{range_line}{actual_line}</div>'
        f'<div class="strip" aria-hidden="true"><div class="strip-marker" style="left:{pos:.1f}%"></div></div>'
        f'<div class="strip-labels"><span>Terendah 72 jam: {num(lo72, 1)} °C</span>'
        f'<span>Tertinggi: {num(hi72, 1)} °C</span></div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_setup_error(missing: list[str]) -> None:
    st.error("Artefak model belum lengkap, sehingga aplikasi belum bisa membuat prakiraan.")
    st.markdown("File yang belum ada: " + ", ".join(f"`{m}`" for m in missing))
    st.markdown("Jalankan `training/train_export.py` (lihat README), lalu salin isi folder "
                "`deploy_artifacts/` ke root repositori.")


def data_quality_panel(report: dict, oor: dict[str, float]) -> None:
    with st.expander("Pemeriksaan kualitas data"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris dipakai", num(report["rows_used"], 0))
        c2.metric("Error sensor diperbaiki", num(report["sensor_errors_fixed"], 0),
                  help="Nilai angin -9999 (kode error sensor) diganti hasil interpolasi, sama seperti saat training.")
        c3.metric("Celah waktu", num(report["gaps"], 0),
                  help="Jumlah loncatan waktu lebih dari 10 menit antarbaris.")
        c4.metric("Baris dibuang", num(report["rows_dropped_missing"] + report["bad_timestamps"], 0),
                  help="Baris dengan waktu tidak terbaca atau nilai kosong.")
        if report["gaps"] or report["duplicate_timestamps"]:
            st.warning(f"Ditemukan {report['gaps']} celah waktu dan {report['duplicate_timestamps']} waktu "
                       "ganda. Model mengira data selalu berjarak 10 menit, jadi prediksi di sekitar celah "
                       "bisa kurang akurat.")
        if oor:
            names = ", ".join(inf.FEATURE_LABELS[c].split(" (")[0].lower() for c in oor)
            st.warning(f"Sebagian nilai {names} berada di luar jangkauan data latih (2009–2014). "
                       "Prediksi untuk kondisi yang belum pernah dilihat model perlu dibaca dengan hati-hati.")
        if not (report["gaps"] or report["duplicate_timestamps"] or oor):
            st.success("Data berurutan setiap 10 menit dan berada dalam jangkauan data latih.")


def guide_tab(primary: str, other: str, meta: dict) -> None:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Cara memakai")
        st.markdown(
            "1. **Pilih data** di panel kiri. Data contoh berisi pengukuran Okt 2015 – Des 2016, periode "
            "yang tidak pernah dilihat model saat belajar, jadi hasilnya jujur.\n"
            "2. **Pilih rentang tanggal** (maksimal 14 hari). Untuk setiap titik 10 menit di rentang itu, "
            "model menebak suhu memakai 72 jam data sebelumnya, lalu tebakannya dibandingkan dengan suhu "
            "yang benar-benar tercatat.\n"
            "3. **Baca hasilnya** di tab *Prakiraan*. Kotak di kanan atas menampilkan tebakan untuk 10 menit "
            "setelah akhir rentang yang dipilih.\n"
            f"4. **Bandingkan** {primary} dengan {other} di tab *Bandingkan model*, dan unduh hasil prediksi "
            "sebagai CSV bila perlu."
        )
        st.subheader("Cara membaca angka")
        st.markdown(
            "- **Rata-rata meleset (MAE)**: rata-rata jarak antara tebakan dan suhu sebenarnya. "
            "MAE 0,13 °C artinya tebakan rata-rata meleset 0,13 derajat.\n"
            "- **RMSE**: mirip MAE, tetapi kesalahan besar dihukum lebih berat. Kalau RMSE jauh di atas "
            "MAE, berarti sesekali ada tebakan yang meleset jauh.\n"
            "- **Tebakan naif**: pembanding paling sederhana, yaitu menganggap suhu 10 menit lagi sama "
            "dengan suhu sekarang. Model yang berguna harus lebih baik dari ini.\n"
            "- **Rentang kemungkinan**: pita di sekitar garis prediksi. Rentang 90% dibuat dari kesalahan "
            "model pada data validasi, sehingga sekitar 9 dari 10 nilai aktual diharapkan jatuh di dalamnya."
        )
    with right:
        st.subheader("Apa itu LSTM dan GRU?")
        st.markdown(
            "Keduanya adalah *recurrent neural network*: jaringan saraf yang membaca data satu per satu "
            "sesuai urutan waktu, seperti kita membaca kalimat kata demi kata, sambil menyimpan ringkasan "
            "hal penting yang sudah dibaca."
        )
        for name in (primary, other):
            st.markdown(f"**{name} ({MODEL_INFO[name]['full']})**. {MODEL_INFO[name]['plain']}")
        st.subheader("Batasan")
        st.markdown(
            "- Model hanya dilatih untuk menebak **10 menit ke depan**. Untuk jam atau hari berikutnya "
            "perlu model lain.\n"
            "- Model dilatih dengan data satu stasiun di Jena (2009–2014). Data dari lokasi lain dengan "
            "iklim berbeda kemungkinan menghasilkan tebakan yang kurang akurat.\n"
            "- Data yang diunggah harus berjarak 10 menit dan memuat minimal 72 jam (432 baris)."
        )

    st.subheader("Format CSV untuk diunggah")
    st.markdown("Kolom `Date Time` (contoh: `31.12.2016 23:50:00`) ditambah 12 kolom berikut, dengan nama "
                "persis sama. Kolom lain boleh ada dan akan diabaikan.")
    st.dataframe(pd.DataFrame({"Nama kolom": inf.FEATURE_COLS,
                               "Arti": [inf.FEATURE_LABELS[c] for c in inf.FEATURE_COLS]}),
                 hide_index=True, use_container_width=True)
    st.download_button("Unduh contoh file CSV", template_csv(), "contoh_format_jena.csv", "text/csv")

    with st.expander("Detail teknis untuk developer"):
        rows = []
        for name, m in meta["models"].items():
            rows.append({"Model": name, "Hidden size": m["hidden_size"], "Layer": m["num_layers"],
                         "Dropout": m["dropout"], "Learning rate": m["lr"], "Parameter": m["params"],
                         "Epoch dilatih": m["epochs_trained"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown(
            f"- Input: {meta['seq_len']} langkah × {len(meta['feature_cols'])} fitur (72 jam), "
            "target: `T (degC)` 1 langkah (10 menit) ke depan.\n"
            "- Preprocessing: nilai angin −9999 diinterpolasi, MinMaxScaler di-fit **hanya** pada 70% data "
            "awal (train) dan disimpan di `models/scaler.json`; saat inference scaler tidak di-fit ulang.\n"
            f"- Split berurutan waktu 70/15/15. Periode test: {meta['split']['test_start']} s.d. "
            f"{meta['split']['test_end']}.\n"
            "- Dilatih dengan PyTorch, disimpan sebagai `.pt` (state_dict), dan diekspor ke ONNX untuk "
            "deployment. Output ONNX diverifikasi sama dengan PyTorch (selisih < 1e-4).\n"
            f"- Diekspor: {meta['exported_at']} (PyTorch {meta['torch_version']})."
        )


# ----------------------------------------------------------------------------------------------
# Aplikasi
# ----------------------------------------------------------------------------------------------
def main(primary: str = "LSTM") -> None:
    primary = primary.upper()
    other = "GRU" if primary == "LSTM" else "LSTM"
    order = [primary, other]

    st.set_page_config(page_title=f"Prakiraan Suhu Jena · {primary}", page_icon="🌡️",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    missing = missing_artifacts()
    if missing:
        show_setup_error(missing)
        st.stop()
    scaler, meta = load_artifacts()
    seq_len = meta["seq_len"]

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.markdown(
            f'<div class="model-badge" style="--accent:{MODEL_STYLE[primary]["color"]}">'
            f'<div><strong>Aplikasi model {primary}</strong><br>'
            f'<small>{MODEL_INFO[primary]["full"]}, dibandingkan dengan {other}</small></div></div>',
            unsafe_allow_html=True)

        step_heading(1, "Pilih data")
        source = st.radio(
            "Sumber data", ["Data contoh", "Unggah CSV"], label_visibility="collapsed",
            captions=["Stasiun Jena, Okt 2015 – Des 2016", "Data Anda sendiri, format sama dengan dataset Jena"],
            help="Data contoh adalah periode test: model tidak pernah melihatnya saat training.")

        data = report = None
        data_key = "demo"
        if source == "Data contoh":
            data, report = load_demo()
        else:
            uploaded = st.file_uploader("File CSV data cuaca", type=["csv"],
                                        help="Minimal 432 baris (72 jam) berjarak 10 menit. "
                                             "Lihat tab Panduan untuk daftar kolom.")
            st.download_button("Unduh contoh format CSV", template_csv(), "contoh_format_jena.csv",
                               "text/csv", use_container_width=True)
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                data_key = hashlib.md5(file_bytes).hexdigest()
                try:
                    data, report = load_upload(file_bytes)
                except inf.DataValidationError as exc:
                    st.error(f"{exc} Periksa nama kolom di tab Panduan, atau unduh contoh format di atas.")
                    data = None

    if data is None:
        st.markdown('<div class="hero-title" role="heading" aria-level="1">'
                    'Berapa suhu udara di Jena 10 menit lagi?</div>', unsafe_allow_html=True)
        st.info("Unggah file CSV di panel kiri untuk mulai. Belum punya file? Unduh contoh format di "
                "panel kiri, atau pilih **Data contoh**.")
        guide_tab(primary, other, meta)
        st.stop()

    if len(data) < seq_len:
        st.error(f"Data hanya berisi {len(data)} baris yang valid. Model butuh minimal {seq_len} baris "
                 "(72 jam data berjarak 10 menit) untuk membuat satu prakiraan.")
        st.stop()

    idx = data.index
    can_backtest = len(data) > seq_len

    with st.sidebar:
        step_heading(2, "Pilih rentang waktu")
        if can_backtest:
            first_ok, last = idx[seq_len], idx[-1]
            end_default = max((last - pd.Timedelta(minutes=inf.STEP_MINUTES)).date(), first_ok.date())
            start_default = max(end_default - pd.Timedelta(days=2), first_ok.date())
            picked = st.date_input(
                "Tanggal awal dan akhir", value=(start_default, end_default),
                min_value=first_ok.date(), max_value=last.date(), format="DD/MM/YYYY", key=f"range_{data_key}",
                help=f"Maksimal {MAX_DAYS} hari. Tanggal paling awal yang bisa dipilih adalah 3 hari setelah "
                     "data dimulai, karena model butuh 72 jam riwayat.")
            st.caption(f"Data tersedia {fmt_date(idx[0])} s.d. {fmt_date(last)}.")
        else:
            picked = None
            st.caption("Data pas 72 jam: hanya prakiraan 10 menit berikutnya yang bisa dibuat.")

        step_heading(3, "Atur tampilan")
        level = st.segmented_control(
            "Rentang kemungkinan", list(INTERVAL_OPTIONS), default="90%",
            help="Pita di sekitar garis prediksi. 90% artinya sekitar 9 dari 10 nilai aktual diharapkan "
                 "berada di dalam pita.") or "Tanpa rentang"
        show_baseline = st.toggle("Tampilkan tebakan naif", value=False,
                                  help="Pembanding sederhana: suhu 10 menit lagi dianggap sama dengan suhu sekarang.")
        st.divider()
        st.caption(f"Versi {APP_VERSION}. Model {primary} dan {other} dari eksperimen Week 3 (Jena Climate).")

    # ------------------------------------------------------------------ hitung
    scaled = scale_data(data_key, data)
    q_key = INTERVAL_OPTIONS[level]
    q = meta["models"][primary].get("val_abs_error_quantiles", {}).get(q_key) if q_key else None

    lo = hi = None
    truncated = False
    if can_backtest:
        if not isinstance(picked, (tuple, list)) or len(picked) != 2:
            st.info("Pilih tanggal akhir di kalender panel kiri untuk menyelesaikan rentang waktu.")
            st.stop()
        lo = max(int(idx.searchsorted(pd.Timestamp(picked[0]))), seq_len)
        hi = int(idx.searchsorted(pd.Timestamp(picked[1]) + pd.Timedelta(days=1)))
        if hi - lo > MAX_STEPS:
            lo, truncated = hi - MAX_STEPS, True
        if hi <= lo:
            st.warning("Rentang yang dipilih tidak berisi data. Coba pilih tanggal lain.")
            st.stop()
    context_end = hi if hi is not None else len(data)

    primary_model = load_model(primary)
    next_value = inf.forecast_next(primary_model, scaled[:context_end], scaler)
    next_time = idx[context_end - 1] + pd.Timedelta(minutes=inf.STEP_MINUTES)
    next_actual = None
    if context_end < len(data) and idx[context_end] == next_time:
        next_actual = float(data[inf.TARGET_COL].iloc[context_end])
    last72 = data[inf.TARGET_COL].iloc[context_end - seq_len:context_end]

    hero(primary, next_value, next_time, q, level, next_actual, float(last72.min()), float(last72.max()))

    if not can_backtest:
        st.info("Tambahkan data lebih dari 72 jam untuk melihat grafik prediksi dan perbandingan model.")
        data_quality_panel(report, inf.out_of_range_share(scaled))
        guide_tab(primary, other, meta)
        return

    with st.spinner("Menjalankan kedua model…"):
        backtests = {name: run_backtest(name, data_key, lo, hi, scaled) for name in order}
    positions = np.arange(lo, hi)
    y_true = data[inf.TARGET_COL].to_numpy()[lo:hi]
    baseline = inf.persistence_forecast(data, positions)

    res = pd.DataFrame({"Waktu": idx[lo:hi], "Suhu aktual (°C)": y_true})
    for name in order:
        res[f"Prediksi {name} (°C)"] = backtests[name][0]
    res["Tebakan naif (°C)"] = baseline
    for name in order:
        res[f"Selisih {name} (°C)"] = res[f"Prediksi {name} (°C)"] - y_true

    metrics = {name: inf.regression_metrics(y_true, backtests[name][0]) for name in order}
    base_m = inf.regression_metrics(y_true, baseline)
    n_points = hi - lo

    tab_fc, tab_cmp, tab_err, tab_guide = st.tabs(
        ["📈 Prakiraan", "⚖️ Bandingkan model", "🔍 Analisis kesalahan", "📘 Panduan"])

    # ------------------------------------------------------------------ tab: prakiraan
    with tab_fc:
        if truncated:
            st.info(f"Rentang dipotong menjadi {MAX_DAYS} hari terakhir agar aplikasi tetap cepat.")
        m = metrics[primary]
        gain = (1 - m["MAE"] / base_m["MAE"]) * 100 if base_m["MAE"] > 0 else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rata-rata meleset (MAE)", f"{num(m['MAE'], 3)} °C",
                  help="Rata-rata selisih antara tebakan model dan suhu yang tercatat.")
        c2.metric("RMSE", f"{num(m['RMSE'], 3)} °C",
                  help="Seperti MAE, tetapi kesalahan besar dihitung lebih berat.")
        c3.metric("Dibanding tebakan naif", f"{num(abs(gain), 0)}% {'lebih akurat' if gain >= 0 else 'kurang akurat'}",
                  help=f"Tebakan naif (suhu tetap) meleset rata-rata {num(base_m['MAE'], 3)} °C pada rentang ini.")
        c4.metric("Titik yang dievaluasi", num(n_points, 0),
                  help=f"{num(n_points / STEPS_PER_DAY, 1)} hari × 144 titik per hari (setiap 10 menit).")

        st.plotly_chart(forecast_figure(res, primary, q, show_baseline), use_container_width=True,
                        config=PLOT_CONFIG)
        notes = ["Geser atau perkecil kotak di bawah grafik untuk memperbesar bagian tertentu."]
        if q is not None:
            coverage = float(np.mean(np.abs(res[f"Selisih {primary} (°C)"]) <= q) * 100)
            notes.append(f"Pada rentang ini, {num(coverage, 0)}% nilai aktual berada di dalam pita "
                         f"{level} (lebar ±{num(q, 2)} °C).")
        st.markdown(f'<p class="note">{" ".join(notes)}</p>', unsafe_allow_html=True)

        export = res.copy()
        if q is not None:
            export[f"Batas bawah {level} (°C)"] = export[f"Prediksi {primary} (°C)"] - q
            export[f"Batas atas {level} (°C)"] = export[f"Prediksi {primary} (°C)"] + q
        export["Waktu"] = export["Waktu"].dt.strftime("%Y-%m-%d %H:%M")
        dl, _ = st.columns([1, 2])
        dl.download_button(
            "Unduh hasil prediksi (CSV)", export.round(4).to_csv(index=False).encode("utf-8"),
            f"prakiraan_{primary.lower()}_{idx[lo]:%Y%m%d}_{idx[hi - 1]:%Y%m%d}.csv", "text/csv",
            use_container_width=True, help="Berisi waktu, suhu aktual, prediksi kedua model, tebakan naif, dan selisih.")
        data_quality_panel(report, inf.out_of_range_share(scaled[lo - seq_len:hi]))

    # ------------------------------------------------------------------ tab: bandingkan
    with tab_cmp:
        a, b = metrics[primary]["MAE"], metrics[other]["MAE"]
        better, worse = (primary, other) if a <= b else (other, primary)
        rel = abs(a - b) / max(a, b) * 100
        if rel < 5:
            verdict = (f"Pada rentang ini **{primary} dan {other} praktis setara**: rata-rata meleset "
                       f"{num(a, 3)} °C vs {num(b, 3)} °C (beda {num(abs(a - b), 3)} °C). "
                       "Perbedaan sekecil ini tidak terasa dalam pemakaian sehari-hari.")
        else:
            verdict = (f"Pada rentang ini **{better} lebih akurat**: rata-rata meleset "
                       f"{num(min(a, b), 3)} °C dibanding {num(max(a, b), 3)} °C untuk {worse} "
                       f"({num(rel, 0)}% lebih kecil).")
        st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

        rows = []
        for name in order:
            mm, secs = metrics[name], backtests[name][1]
            info = meta["models"][name]
            rows.append({
                "Model": name,
                "MAE (°C)": num(mm["MAE"], 3), "RMSE (°C)": num(mm["RMSE"], 3),
                "Meleset terbesar (°C)": num(mm["MaxAE"], 2),
                "Vs tebakan naif": f"{num((1 - mm['MAE'] / base_m['MAE']) * 100, 0)}% lebih akurat",
                "Waktu per 1.000 prediksi": f"{num(secs / n_points * 1e6, 0)} ms",
                "Parameter": num(info["params"], 0),
                "Ukuran file": f"{num(load_model(name).file_size_kb, 0)} KB",
            })
        rows.append({"Model": "Tebakan naif", "MAE (°C)": num(base_m["MAE"], 3),
                     "RMSE (°C)": num(base_m["RMSE"], 3), "Meleset terbesar (°C)": num(base_m["MaxAE"], 2),
                     "Vs tebakan naif": "–", "Waktu per 1.000 prediksi": "–", "Parameter": "–",
                     "Ukuran file": "–"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Waktu prediksi diukur di server ini saat rentang pertama kali dihitung (CPU, ONNX Runtime). "
                   "Hasil perhitungan disimpan di cache, jadi membuka rentang yang sama lagi terasa instan.")

        st.plotly_chart(comparison_figure(res, order), use_container_width=True, config=PLOT_CONFIG)

        with st.expander("Uji kecepatan ulang"):
            st.markdown('<p class="note">Mengukur waktu memprediksi 500 window sekaligus, diulang 3 kali, '
                        'lalu diambil nilai tengahnya.</p>', unsafe_allow_html=True)
            if st.button("Jalankan uji kecepatan"):
                st.session_state["bench"] = {name: inf.benchmark(load_model(name), scaled) for name in order}
            if "bench" in st.session_state:
                cols = st.columns(len(order))
                for col, name in zip(cols, order):
                    col.metric(f"{name}: 500 prediksi", f"{num(st.session_state['bench'][name], 1)} ms")

        st.subheader("Hasil resmi di seluruh periode test")
        official = []
        for name in order:
            info = meta["models"][name]
            official.append({"Model": name, "MAE test (°C)": num(info["test_mae"], 4),
                             "RMSE test (°C)": num(info["test_rmse"], 4),
                             "MAE di notebook Week 3": num(info["notebook_mae"], 4),
                             "RMSE di notebook Week 3": num(info["notebook_rmse"], 4)})
        bm = meta["baseline_persistence_test"]
        official.append({"Model": "Tebakan naif", "MAE test (°C)": num(bm["MAE"], 4),
                         "RMSE test (°C)": num(bm["RMSE"], 4), "MAE di notebook Week 3": "–",
                         "RMSE di notebook Week 3": "–"})
        st.dataframe(pd.DataFrame(official), hide_index=True, use_container_width=True)
        st.caption("Dihitung pada seluruh data test (Okt 2015 – Des 2016). Angka notebook adalah hasil "
                   "eksperimen awal; selisih kecil wajar karena training di GPU tidak sepenuhnya deterministik.")

    # ------------------------------------------------------------------ tab: analisis kesalahan
    with tab_err:
        by_hour = pd.DataFrame({name: res[f"Selisih {name} (°C)"].abs().groupby(res["Waktu"].dt.hour).mean()
                                for name in order})
        worst_hour = int(by_hour[primary].idxmax())
        best_hour = int(by_hour[primary].idxmin())
        bias = metrics[primary]["Bias"]
        bias_txt = ("cenderung menebak sedikit terlalu tinggi" if bias > 0.01 else
                    "cenderung menebak sedikit terlalu rendah" if bias < -0.01 else "tidak condong ke atas atau ke bawah")
        st.markdown(
            f'<div class="verdict">{primary} paling sering meleset sekitar pukul {worst_hour:02d}.00 '
            f'dan paling tepat sekitar pukul {best_hour:02d}.00. Secara keseluruhan model {bias_txt} '
            f'(rata-rata selisih {num(bias, 3)} °C).</div>', unsafe_allow_html=True)
        st.markdown("**Rata-rata meleset per jam**. Suhu biasanya berubah paling cepat saat pagi dan sore, "
                    "sehingga kesalahan cenderung lebih besar pada jam-jam tersebut.")
        st.plotly_chart(error_hour_figure(by_hour, order), use_container_width=True, config=PLOT_CONFIG)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Selisih dari waktu ke waktu**. Di atas nol berarti tebakan terlalu tinggi.")
            st.plotly_chart(error_time_figure(res, order), use_container_width=True, config=PLOT_CONFIG)
        with right:
            st.markdown("**Sebaran selisih**. Makin rapat di sekitar nol, makin baik.")
            st.plotly_chart(error_hist_figure(res, order), use_container_width=True, config=PLOT_CONFIG)

    with tab_guide:
        guide_tab(primary, other, meta)


if __name__ == "__main__":
    main("LSTM")

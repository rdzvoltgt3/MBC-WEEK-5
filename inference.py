"""
inference.py — Pipeline inference yang konsisten dengan preprocessing notebook Week 3.

Dipisah dari app.py agar:
  * logika data/model bisa diuji tanpa Streamlit,
  * v1 dan v2 aplikasi memakai pipeline yang PERSIS sama (yang berubah hanya UI/fitur).

Aturan penting yang dijaga di sini:
  1. Scaler TIDAK pernah di-fit ulang. Parameter min/max dibaca dari models/scaler.json
     (hasil fit pada train split di notebook).
  2. Urutan fitur, pembersihan nilai -9999, dan panjang window (432 langkah = 72 jam)
     sama dengan saat training.
  3. Model memprediksi 1 langkah ke depan (10 menit) — sesuai target training.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATETIME_COL = "Date Time"
DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S"
TARGET_COL = "T (degC)"
FEATURE_COLS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "rho (g/m**3)", "wv (m/s)", "wd (deg)",
]
WIND_COLS = ["wv (m/s)", "max. wv (m/s)"]
STEP_MINUTES = 10

# Label ramah orang awam untuk tiap kolom (dipakai di UI)
FEATURE_LABELS = {
    "p (mbar)": "Tekanan udara (milibar)",
    "T (degC)": "Suhu udara (°C) — yang diprediksi",
    "Tpot (K)": "Suhu potensial (Kelvin)",
    "Tdew (degC)": "Titik embun (°C)",
    "rh (%)": "Kelembapan relatif (%)",
    "VPmax (mbar)": "Tekanan uap jenuh (milibar)",
    "VPact (mbar)": "Tekanan uap aktual (milibar)",
    "VPdef (mbar)": "Defisit tekanan uap (milibar)",
    "sh (g/kg)": "Kelembapan spesifik (g/kg)",
    "rho (g/m**3)": "Kerapatan udara (g/m³)",
    "wv (m/s)": "Kecepatan angin (m/s)",
    "wd (deg)": "Arah angin (derajat)",
}


class DataValidationError(ValueError):
    """Error yang pesannya aman ditampilkan langsung ke pengguna."""


# ----------------------------------------------------------------------------------------------
# Scaler
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MinMaxParams:
    """Setara sklearn MinMaxScaler (feature_range 0–1) tetapi tanpa pickle.

    Alasan JSON, bukan pickle: pickle sklearn terikat versi library; JSON bisa dibaca
    di versi Python/sklearn apa pun dan isinya bisa diaudit manusia.
    """
    feature_cols: tuple
    data_min: np.ndarray
    data_max: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path) -> "MinMaxParams":
        cfg = json.loads(Path(path).read_text())
        if list(cfg["feature_cols"]) != FEATURE_COLS:
            raise ValueError("Urutan fitur di scaler.json berbeda dari FEATURE_COLS.")
        return cls(tuple(cfg["feature_cols"]),
                   np.asarray(cfg["data_min"], dtype=np.float64),
                   np.asarray(cfg["data_max"], dtype=np.float64))

    @property
    def _range(self) -> np.ndarray:
        rng = self.data_max - self.data_min
        return np.where(rng == 0, 1.0, rng)  # perilaku sama dengan sklearn untuk kolom konstan

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.data_min) / self._range

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        i = FEATURE_COLS.index(TARGET_COL)
        return scaled * self._range[i] + self.data_min[i]


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def missing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in [DATETIME_COL, *FEATURE_COLS] if c not in df.columns]


def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce")
    if parsed.isna().mean() > 0.5:  # bukan format asli Jena -> coba ISO / format umum
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.5:  # terakhir: format tanggal-dulu (31/12/2016 23:50)
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def clean_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan data mentah dengan langkah yang sama seperti notebook.

    Return: (data bersih ber-index waktu berisi FEATURE_COLS, laporan kualitas data).
    """
    missing = missing_columns(df_raw)
    if missing:
        raise DataValidationError("Kolom berikut tidak ditemukan: " + ", ".join(missing))

    df = df_raw.copy()
    rows_in = len(df)
    df[DATETIME_COL] = _parse_datetime(df[DATETIME_COL])
    bad_time = int(df[DATETIME_COL].isna().sum())
    df = df.dropna(subset=[DATETIME_COL]).set_index(DATETIME_COL).sort_index()

    for col in set(FEATURE_COLS) | {c for c in WIND_COLS if c in df.columns}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sensor_errors = 0
    for col in WIND_COLS:  # nilai -9999 = error sensor -> interpolasi (sama dengan notebook)
        if col in df.columns:
            bad = df[col] <= -9000
            sensor_errors += int(bad.sum())
            df[col] = df[col].where(~bad, np.nan).interpolate()

    data = df[FEATURE_COLS].dropna()
    steps = data.index.to_series().diff().dt.total_seconds().div(60).iloc[1:]
    report = {
        "rows_in": rows_in,
        "rows_used": len(data),
        "bad_timestamps": bad_time,
        "sensor_errors_fixed": sensor_errors,
        "rows_dropped_missing": rows_in - bad_time - len(data),
        "duplicate_timestamps": int((steps == 0).sum()),
        "gaps": int((steps > STEP_MINUTES).sum()),
        "start": data.index.min() if len(data) else None,
        "end": data.index.max() if len(data) else None,
    }
    return data, report


def out_of_range_share(scaled: np.ndarray, tol: float = 0.1) -> dict[str, float]:
    """Porsi nilai yang jauh di luar jangkauan data training (skala < -tol atau > 1+tol)."""
    share = ((scaled < -tol) | (scaled > 1 + tol)).mean(axis=0)
    return {c: float(s) for c, s in zip(FEATURE_COLS, share) if s > 0}


def build_windows(scaled: np.ndarray, target_positions: np.ndarray, seq_len: int) -> np.ndarray:
    """Window input untuk tiap posisi target t: baris [t-seq_len, t) -> bentuk (n, seq_len, fitur).

    Memakai sliding_window_view (tanpa menyalin seluruh data) lalu hanya mengambil posisi
    yang dibutuhkan — hemat memori di server Streamlit Cloud.
    """
    target_positions = np.asarray(target_positions)
    if len(target_positions) and target_positions.min() < seq_len:
        raise ValueError("Posisi target membutuhkan riwayat minimal seq_len baris.")
    view = np.lib.stride_tricks.sliding_window_view(scaled, seq_len, axis=0)  # (N-L+1, F, L)
    return np.ascontiguousarray(view[target_positions - seq_len].transpose(0, 2, 1), dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class OnnxForecaster:
    """Pembungkus model ONNX (hasil ekspor dari PyTorch) untuk inference CPU."""

    def __init__(self, onnx_path: str | Path, name: str, seq_len: int):
        import onnxruntime as ort  # import di sini agar modul tetap bisa diuji tanpa onnxruntime

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # server Community Cloud hanya punya sedikit core
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.name = name
        self.seq_len = seq_len
        self.file_size_kb = Path(onnx_path).stat().st_size / 1024

    def predict_scaled(self, windows: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: windows})[0].reshape(-1)


def rolling_forecast(model, scaled: np.ndarray, scaler: MinMaxParams,
                     target_positions: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, float]:
    """Prediksi 1-langkah untuk setiap posisi target (tiap prediksi memakai data aktual sebelumnya).

    Return: (prediksi dalam °C, waktu inference murni dalam detik).
    """
    preds, elapsed = [], 0.0
    for i in range(0, len(target_positions), batch_size):
        windows = build_windows(scaled, target_positions[i:i + batch_size], model.seq_len)
        t0 = time.perf_counter()
        preds.append(model.predict_scaled(windows))
        elapsed += time.perf_counter() - t0
    out = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    return scaler.inverse_target(out.astype(np.float64)), elapsed


def forecast_next(model, scaled: np.ndarray, scaler: MinMaxParams) -> float:
    """Prakiraan suhu 10 menit setelah baris terakhir yang diberikan."""
    window = np.ascontiguousarray(scaled[-model.seq_len:][None, ...], dtype=np.float32)
    return float(scaler.inverse_target(model.predict_scaled(window))[0])


def benchmark(model, scaled: np.ndarray, n_windows: int = 500, repeats: int = 3) -> float:
    """Median waktu (ms) untuk memprediksi n_windows window."""
    n_windows = min(n_windows, len(scaled) - model.seq_len)
    positions = np.arange(len(scaled) - n_windows, len(scaled))
    windows = build_windows(scaled, positions, model.seq_len)
    model.predict_scaled(windows[:8])  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_scaled(windows)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


# ----------------------------------------------------------------------------------------------
# Evaluasi
# ----------------------------------------------------------------------------------------------
def persistence_forecast(data: pd.DataFrame, target_positions: np.ndarray) -> np.ndarray:
    """Baseline naif: suhu 10 menit lagi = suhu saat ini."""
    return data[TARGET_COL].to_numpy()[np.asarray(target_positions) - 1]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MaxAE": float(np.max(np.abs(err))),
        "Bias": float(np.mean(err)),
    }

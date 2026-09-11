"""
check_setup.py — Cek apakah repo siap dijalankan, SEBELUM `streamlit run`.

    python check_setup.py

Memeriksa: versi Python, library, kelengkapan file artefak, lalu menjalankan kedua model ONNX
pada 1 hari data contoh dan membandingkan hasilnya dengan metrik di metadata.json.
"""
import importlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OK, WARN, FAIL = "[OK]  ", "[CEK] ", "[GAGAL]"
problems = 0


def say(tag, msg):
    global problems
    problems += tag == FAIL
    print(f"{tag} {msg}")


# 1. Python
v = sys.version_info
if (3, 10) <= (v.major, v.minor) <= (3, 13):
    say(OK, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    say(WARN, f"Python {v.major}.{v.minor}: disarankan 3.12 (sama dengan Streamlit Cloud). "
              "Versi library di requirements.txt mungkin belum punya wheel untuk versi ini.")

# 2. Library
pinned = {}
for line in (BASE / "requirements.txt").read_text().splitlines():
    if "==" in line and not line.strip().startswith("#"):
        name, ver = line.strip().split("==")
        pinned[name] = ver
for pkg, ver in pinned.items():
    try:
        mod = importlib.import_module(pkg)
        got = getattr(mod, "__version__", "?")
        say(OK if got == ver else WARN, f"{pkg} {got}" + ("" if got == ver else f" (requirements: {ver})"))
    except ImportError:
        say(FAIL, f"{pkg} belum terpasang -> pip install -r requirements.txt")

# 3. File artefak
required = ["app.py", "app_lstm.py", "app_gru.py", "inference.py", "models/scaler.json",
            "models/metadata.json", "models/lstm_best.onnx", "models/gru_best.onnx",
            "data/jena_test_period.csv.gz"]
optional = ["models/lstm_best.pt", "models/gru_best.pt", "models/scaler.joblib", ".streamlit/config.toml"]
for f in required:
    say(OK if (BASE / f).exists() else FAIL, f + ("" if (BASE / f).exists() else
        "  <- belum ada (jalankan training/train_export.py di Kaggle, salin folder models/)"))
for f in optional:
    if not (BASE / f).exists():
        say(WARN, f"{f} tidak ada (tidak wajib untuk menjalankan app, tapi termasuk deliverable)")

# 4. Uji model
if problems == 0:
    import numpy as np
    import pandas as pd
    import inference as inf

    scaler = inf.MinMaxParams.from_json(BASE / "models/scaler.json")
    meta = json.loads((BASE / "models/metadata.json").read_text())
    data, _ = inf.clean_dataframe(pd.read_csv(BASE / "data/jena_test_period.csv.gz"))
    scaled = scaler.transform(data.to_numpy(dtype=np.float64))
    pos = np.arange(len(data) - 144, len(data))  # 1 hari terakhir
    y = data[inf.TARGET_COL].to_numpy()[pos]
    for name, info in meta["models"].items():
        model = inf.OnnxForecaster(BASE / info["onnx_file"], name, meta["seq_len"])
        pred, secs = inf.rolling_forecast(model, scaled, scaler, pos)
        mae = inf.regression_metrics(y, pred)["MAE"]
        tag = OK if mae < 1.0 else FAIL  # MAE 1 hari normalnya ~0,1–0,3 °C
        say(tag, f"{name}: MAE 1 hari terakhir {mae:.3f} °C (test penuh: {info['test_mae']}), "
                 f"{secs * 1000:.0f} ms untuk 144 prediksi")

print()
print("Siap dijalankan: streamlit run app_lstm.py" if problems == 0 else
      f"Ada {problems} masalah yang perlu diperbaiki dulu (lihat [GAGAL] di atas).")
sys.exit(1 if problems else 0)

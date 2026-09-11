"""
train_export.py — Melatih ulang 2 model terbaik Week 3 lalu mengekspor artefak deployment.

Cara menjalankan (pilih salah satu):
  A. Kaggle, mode Script (GPU): File -> Editor Type -> Script, tambahkan dataset "jena-climate"
     (mnassrib), aktifkan GPU & Internet, tempel isi file ini, jalankan (±10–15 menit).
  B. Kaggle, mode Notebook: tempel isi file ini ke satu cell, jalankan.
  C. Lokal: letakkan jena_climate_2009_2016.csv di root repo, lalu
         pip install -r training/requirements-training.txt
         python training/train_export.py
     (tanpa GPU NVIDIA bisa memakan waktu beberapa jam).
Hasil: deploy_artifacts.zip (di /kaggle/working saat di Kaggle, di folder kerja saat lokal).
onnx & onnxruntime dipasang otomatis bila belum ada.

Output (ke folder deploy_artifacts/):
    models/lstm_best.pt, models/gru_best.pt       -> bobot asli PyTorch (state_dict + config)
    models/lstm_best.onnx, models/gru_best.onnx   -> format deployment (dipakai app.py)
    models/scaler.json, models/scaler.joblib      -> scaler yang di-fit HANYA pada train split
    models/metadata.json                          -> metrik test, interval, info arsitektur
    data/jena_test_period.csv.gz                  -> data demo (periode test, tidak pernah dilihat model)

Preprocessing, split, arsitektur, dan fungsi training disalin dari notebook Week 3
(Soal_1_final.ipynb) TANPA perubahan logika, agar model hasil ekspor = model yang dilaporkan.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# Pasang onnx/onnxruntime otomatis bila belum ada (mode Script di Kaggle tidak bisa memakai `!pip`).
_missing = [p for p in ("onnx", "onnxruntime") if importlib.util.find_spec(p) is None]
if _missing:
    print("Memasang:", ", ".join(_missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_missing])

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------------------------
# Konfigurasi (sesuaikan DATA_PATH bila lokasi dataset di Kaggle berbeda)
# ----------------------------------------------------------------------------------------------
DATA_PATH = "/kaggle/input/datasets/mnassrib/jena-climate/jena_climate_2009_2016.csv"
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if not os.path.exists(DATA_PATH):  # fallback: path Kaggle lain, folder kerja, folder skrip, root repo
    for cand in ["/kaggle/input/jena-climate/jena_climate_2009_2016.csv",
                 "jena_climate_2009_2016.csv",
                 os.path.join(_HERE, "jena_climate_2009_2016.csv"),
                 os.path.join(_HERE, "..", "jena_climate_2009_2016.csv")]:
        if os.path.exists(cand):
            DATA_PATH = cand
            break

OUT_DIR = "/kaggle/working/deploy_artifacts" if os.path.exists("/kaggle/working") else "deploy_artifacts"
SEED = 42
SEQ_LEN = 432            # 72 jam konteks — seq_len terbaik untuk KEDUA arsitektur di notebook
BATCH_SIZE = 128
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 4

# Konfigurasi terbaik per arsitektur (hasil tabel eksperimen notebook, diurutkan RMSE):
#   GRU  | seq 432 | small -> MAE 0.1295, RMSE 0.1973  (peringkat 1)
#   LSTM | seq 432 | small -> MAE 0.1288, RMSE 0.1974  (peringkat 2)
BEST_CONFIGS = {
    "LSTM": {"hidden_size": 32, "num_layers": 1, "dropout": 0.1, "lr": 1e-3,
             "notebook_mae": 0.1288, "notebook_rmse": 0.1974},
    "GRU": {"hidden_size": 32, "num_layers": 1, "dropout": 0.1, "lr": 1e-3,
            "notebook_mae": 0.1295, "notebook_rmse": 0.1973},
}

TARGET_COL = "T (degC)"
FEATURE_COLS = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
                "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
                "rho (g/m**3)", "wv (m/s)", "wd (deg)"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------
# 1. Preprocessing — identik dengan notebook
# ----------------------------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        if col in df.columns:
            df[col] = df[col].where(df[col] > -9000, np.nan)
            df[col] = df[col].interpolate()
    return df[FEATURE_COLS].dropna()   # index waktu dipertahankan untuk data demo


# ----------------------------------------------------------------------------------------------
# 2. Dataset & model — identik dengan notebook
# ----------------------------------------------------------------------------------------------
class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, arr, target_idx, seq_len):
        self.arr = arr.astype(np.float32)
        self.target_idx = target_idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.arr) - self.seq_len

    def __getitem__(self, idx):
        x = self.arr[idx: idx + self.seq_len]
        y = self.arr[idx + self.seq_len, self.target_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)


MODEL_TYPES = {"LSTM": LSTMForecaster, "GRU": GRUForecaster}


def train_model(model, train_loader, val_loader, lr):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl.append(criterion(model(xb), yb).item())
        history["train_loss"].append(float(np.mean(tl)))
        history["val_loss"].append(float(np.mean(vl)))
        print(f"  epoch {epoch + 1:02d} train {history['train_loss'][-1]:.6f} val {history['val_loss'][-1]:.6f}")
        if history["val_loss"][-1] < best_val:
            best_val, best_state, no_improve = history["val_loss"][-1], copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  early stopping di epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    return model, history, time.time() - start


def predict_degc(model, loader, t_min, t_max):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    p = np.concatenate(preds) * (t_max - t_min) + t_min
    y = np.concatenate(trues) * (t_max - t_min) + t_min
    return y, p


def export_onnx(model, path, n_features):
    """Ekspor ke ONNX. Batch dinamis, seq_len tetap (model dilatih untuk 432 langkah)."""
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy = torch.randn(1, SEQ_LEN, n_features)
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=17,
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    try:  # PyTorch >= 2.5: pakai eksporter TorchScript klasik (paling stabil untuk LSTM/GRU)
        torch.onnx.export(model_cpu, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # PyTorch lama belum punya argumen `dynamo`
        torch.onnx.export(model_cpu, dummy, path, **kwargs)
    return model_cpu


def verify_onnx(model_cpu, path, sample):
    """Pastikan output ONNX Runtime == output PyTorch (toleransi numerik float32)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": sample})[0].reshape(-1)
    with torch.no_grad():
        torch_out = model_cpu(torch.from_numpy(sample)).numpy().reshape(-1)
    max_diff = float(np.max(np.abs(ort_out - torch_out)))
    assert max_diff < 1e-4, f"Output ONNX berbeda dari PyTorch (max diff {max_diff})"
    return max_diff


def main():
    set_seed(SEED)
    os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    print("Device:", device, "| dataset:", DATA_PATH)

    data = load_and_clean(DATA_PATH)
    target_idx = FEATURE_COLS.index(TARGET_COL)
    n = len(data)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    train_df, val_df, test_df = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]

    # Scaler di-fit HANYA pada train -> disimpan -> dipakai ulang saat inference (tidak di-fit ulang)
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train_df.values)
    val_s, test_s = scaler.transform(val_df.values), scaler.transform(test_df.values)
    t_min, t_max = float(scaler.data_min_[target_idx]), float(scaler.data_max_[target_idx])

    joblib.dump(scaler, f"{OUT_DIR}/models/scaler.joblib")
    with open(f"{OUT_DIR}/models/scaler.json", "w") as f:
        json.dump({"feature_cols": FEATURE_COLS,
                   "data_min": scaler.data_min_.tolist(),
                   "data_max": scaler.data_max_.tolist(),
                   "fitted_on": f"train split: 70 persen baris pertama ({data.index[0]} s.d. {data.index[train_end - 1]})",
                   "n_train_rows": train_end}, f, indent=2)

    loaders = {
        name: DataLoader(WindowedTimeSeriesDataset(arr, target_idx, SEQ_LEN), batch_size=BATCH_SIZE,
                         shuffle=(name == "train"), num_workers=2)
        for name, arr in [("train", train_s), ("val", val_s), ("test", test_s)]
    }

    # Baseline naif (persistence): "suhu 10 menit lagi = suhu sekarang"
    t_test = test_df[TARGET_COL].values
    y_b, p_b = t_test[SEQ_LEN:], t_test[SEQ_LEN - 1:-1]
    baseline = {"MAE": float(np.mean(np.abs(y_b - p_b))), "RMSE": float(np.sqrt(np.mean((y_b - p_b) ** 2)))}
    print("Baseline persistence (test):", baseline)

    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "feature_cols": FEATURE_COLS, "target_col": TARGET_COL,
        "seq_len": SEQ_LEN, "step_minutes": 10, "horizon_steps": 1,
        "split": {"train_end": str(data.index[train_end - 1]), "val_end": str(data.index[val_end - 1]),
                  "test_start": str(data.index[val_end]), "test_end": str(data.index[-1])},
        "baseline_persistence_test": baseline,
        "models": {},
    }

    sample = test_s[:SEQ_LEN + 256].astype(np.float32)
    sample_windows = np.stack([sample[i:i + SEQ_LEN] for i in range(256)])

    for name, cfg in BEST_CONFIGS.items():
        print(f"\n=== {name} | seq_len={SEQ_LEN} | hidden={cfg['hidden_size']} ===")
        set_seed(SEED)
        model = MODEL_TYPES[name](len(FEATURE_COLS), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        model, history, train_time = train_model(model, loaders["train"], loaders["val"], cfg["lr"])

        y_te, p_te = predict_degc(model, loaders["test"], t_min, t_max)
        y_va, p_va = predict_degc(model, loaders["val"], t_min, t_max)
        abs_res_val = np.abs(y_va - p_va)
        mae = float(mean_absolute_error(y_te, p_te))
        rmse = float(np.sqrt(mean_squared_error(y_te, p_te)))
        print(f"  TEST MAE {mae:.4f} | RMSE {rmse:.4f} (notebook: {cfg['notebook_mae']} / {cfg['notebook_rmse']})")

        slug = name.lower()
        pt_path = f"{OUT_DIR}/models/{slug}_best.pt"
        onnx_path = f"{OUT_DIR}/models/{slug}_best.onnx"
        arch = {k: cfg[k] for k in ("hidden_size", "num_layers", "dropout")}
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "arch": name, "config": arch, "input_size": len(FEATURE_COLS),
                    "seq_len": SEQ_LEN}, pt_path)
        model_cpu = export_onnx(model, onnx_path, len(FEATURE_COLS))
        max_diff = verify_onnx(model_cpu, onnx_path, sample_windows)
        print(f"  ONNX terverifikasi (selisih maks vs PyTorch: {max_diff:.2e})")

        metadata["models"][name] = {
            "onnx_file": f"models/{slug}_best.onnx", "pt_file": f"models/{slug}_best.pt",
            **arch, "lr": cfg["lr"],
            "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "epochs_trained": len(history["train_loss"]), "train_time_sec": round(train_time, 1),
            "test_mae": round(mae, 4), "test_rmse": round(rmse, 4),
            "notebook_mae": cfg["notebook_mae"], "notebook_rmse": cfg["notebook_rmse"],
            # Interval prediksi ala split-conformal: kuantil |error| pada data VALIDASI
            "val_abs_error_quantiles": {q: round(float(np.quantile(abs_res_val, float(q))), 4)
                                        for q in ("0.80", "0.90", "0.95")},
            "onnx_max_abs_diff": max_diff,
        }

    with open(f"{OUT_DIR}/models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Data demo = periode TEST (tidak dipakai untuk training/early stopping) -> evaluasi di app jujur
    demo = test_df.copy()
    demo.index = demo.index.strftime("%d.%m.%Y %H:%M:%S")
    demo.index.name = "Date Time"
    demo.round(4).to_csv(f"{OUT_DIR}/data/jena_test_period.csv.gz", compression="gzip")

    shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
    print(f"\nSelesai. Unduh: {OUT_DIR}.zip")
    print(json.dumps({k: {m: v[m] for m in ('test_mae', 'test_rmse')} for k, v in metadata['models'].items()}, indent=2))


if __name__ == "__main__":
    main()

"""SP-05 Conservative Risk-Aware Policy: Complete Training, Calibration, Evaluation and Backtesting Pipeline.
Trained from scratch using Indian Stock Market Data (NSE/BSE equities + NIFTY 50 benchmark).
"""
import os
import sys
import json
import time
import shutil
import pickle
import random
import platform
import warnings
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix
)

warnings.filterwarnings("ignore")

# Set working directory and import local modules
ROOT_DIR = Path(".").resolve()
sys.path.insert(0, str(ROOT_DIR))

from sp05_features import (
    ACTION_NAMES, HOLD, BUY, SELL, EXIT, N_ACTIONS,
    FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, EXTRA_FEATURE_COLUMNS,
    N_FEATURES, N_BASE_FEATURES, N_STATIC, N_SHARED_STATIC,
    STATIC_FEATURE_NAMES, SHARED_STATIC_NAMES, PROFILE_FEATURES,
    DEFAULT_PROFILE, RiskFeatureAdapter, risk_features,
    build_features, build_feature_schema, encode_static,
    valid_action_mask, apply_action, scale_features,
    WARMUP_BARS, OHLCV_COLUMNS
)
from sp05_risk import RiskController, RiskLimits, REASON_NAMES, FORCED_EXIT_REASONS, REQUIRED_FEATURES
from sp05_model import ConservativePolicyNet, ConservativePolicy, mask_logits

# ----------------------------------------------------------------------------
# 1. Configuration & Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "seed": SEED,
    "policy_name": "SP-05 Conservative Risk-Aware Policy",
    "market": "Indian Equity Market (NSE/BSE)",
    "benchmark": "NIFTY 50",
    "raw_csv_path": "Data/Raw/indian-market-historical-ohlcv.csv",
    "seq_len": 30,
    "strict_window_isolation": True,
    "allow_short": True,
    # Dates: Exact 63% / 7% / 30% trading day split
    "train_end": "2021-09-03",
    "val_end": "2022-10-24",
    "test_end": "2026-10-23",
    # Objective weights (conservative proxy labels)
    "objective": {
        "w_return": 1.0,
        "w_drawdown": 1.0,
        "w_turnover": 0.3,
        "entry_threshold": 0.3,
        "exit_threshold": -0.25,
        "short_penalty": 0.2,
    },
    # Soft limits (decision-time)
    "soft_limits": {
        "portfolio_drawdown": 0.05,
        "gross_exposure": 0.50,
        "recent_loss": 0.04,
        "vol_exit_percentile": 0.90,
        "abnormal_ret_z": 3.0,
        "gap_atr": 2.5,
        "market_stress_dd": 0.10,
    },
    "hold_sample_max": 60,
    "eval_horizon": 10,
    # Backtest capital parameters
    "initial_capital": 1000000.0, # 10 Lakh INR
    "dd_lookback_bars": 120,
    "position_fraction": 0.10,
    "max_position_size": 0.20,
    "max_gross_exposure": 1.0,
    "min_trade_fraction": 0.005,
    "fee_bps": 5.0,
    "slippage_bps": 5.0,
    "short_borrow_bps_annual": 50.0,
    "risk_free_annual": 0.06, # 6% RBI repo rate approx
    "confidence_sizing": True,
    # Hard limits for deterministic RiskController
    "risk_limits": {
        "allow_short": True,
        "max_position_fraction": 0.10,
        "max_gross_exposure": 0.60,
        "position_stop_loss": 0.05,
        "max_holding_bars": 120,
        "halt_drawdown": 0.08,
        "flatten_drawdown": 0.15,
        "recent_loss_halt": 0.06,
        "max_vol_percentile": 0.90,
        "max_abs_gap_atr": 3.0,
        "max_abnormal_ret_z": 4.0,
        "market_stress_dd": 0.12,
        "min_entry_confidence": 0.50,
    },
    # Model architecture
    "model": {
        "encoder": "gru",
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.25,
        "static_hidden": 16,
        "head_hidden": 64,
    },
    # Training parameters
    "train": {
        "epochs": 40,
        "batch_size": 512,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "patience": 7,
        "grad_clip": 1.0,
        "input_noise_std": 0.02,
        "max_class_weight": 20.0,
        "max_fit_samples": 360000,
        "scaler_clip": 5.0,
    },
    "output_dir": "outputs/sp05_conservative_policy/indian_market_run_001",
}

OUT_DIR = Path(CONFIG["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"SP-05 Conservative Risk-Aware Policy Training Pipeline")
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
print(f"Artifact directory: {OUT_DIR.resolve()}")
print("=" * 80, flush=True)

# ----------------------------------------------------------------------------
# 2. Load Processed Datasets and Panel
# ----------------------------------------------------------------------------
cache_dir = Path("scratch/features_cache")
main_syms = pd.read_csv("scratch/main_universe_1795.csv")["symbol"].astype(str).tolist()
unseen_syms = pd.read_csv("scratch/unseen_cohort_100.csv")["symbol"].astype(str).tolist()
top20_bluechips = pd.read_csv("scratch/top20_bluechips.csv")["symbol"].astype(str).tolist()
bench_df = pd.read_parquet(cache_dir / "benchmark_nifty50.parquet")
bench_df.index = pd.to_datetime(bench_df.index).normalize()

# The feature cache is derived from the supplied local CSV. Rebuild any missing approved symbol cache
# directly from that file; never substitute or download market data.
missing_syms = sorted((set(main_syms) | set(unseen_syms)) - {p.stem for p in cache_dir.glob("*.parquet")})
if missing_syms:
    raw_path = Path(CONFIG["raw_csv_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Required local Indian market dataset is missing: {raw_path}")
    print(f"Building {len(missing_syms)} missing approved symbol cache(s) from local CSV...", flush=True)
    parts = {sym: [] for sym in missing_syms}
    usecols = ["asset_type", "date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=300000,
                             dtype={"asset_type": "string", "symbol": "string"}):
        keep = chunk["symbol"].isin(missing_syms) & chunk["asset_type"].str.lower().isin(["equities", "equity", "stocks"])
        for sym, grp in chunk.loc[keep].groupby("symbol", sort=False):
            parts[str(sym)].append(grp.copy())
    exclusion_rows = []
    for sym in missing_syms:
        if not parts[sym]:
            exclusion_rows.append({"symbol": sym, "cohort": "main" if sym in main_syms else "unseen",
                                   "reason": "no matching equity records in supplied local CSV"})
            continue
        raw = pd.concat(parts[sym], ignore_index=True)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=["date", "open", "high", "low", "close", "adj_close"])
        raw = raw[(raw[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1)]
        raw = raw[(raw["high"] >= raw[["open", "close", "low"]].max(axis=1)) &
                  (raw["low"] <= raw[["open", "close", "high"]].min(axis=1))]
        raw = raw.sort_values("date").drop_duplicates("date", keep="last")
        if len(raw) < 800:
            exclusion_rows.append({"symbol": sym, "cohort": "main" if sym in main_syms else "unseen",
                                   "reason": f"only {len(raw)} clean OHLC observations; minimum is 800"})
            continue
        factor = raw["adj_close"] / raw["close"]
        ohlcv = pd.DataFrame({"Open": raw["open"].to_numpy() * factor.to_numpy(), "High": raw["high"].to_numpy() * factor.to_numpy(),
                              "Low": raw["low"].to_numpy() * factor.to_numpy(), "Close": raw["adj_close"].to_numpy(),
                              "Volume": raw["volume"].fillna(0.0).clip(lower=0.0).to_numpy()},
                             index=pd.DatetimeIndex(raw["date"], name="date"))
        f = build_features(ohlcv, bench_df)
        firsts = [f[c].first_valid_index() for c in FEATURE_COLUMNS]
        valid_firsts = [x for x in firsts if x is not None]
        if not valid_firsts:
            exclusion_rows.append({"symbol": sym, "cohort": "main" if sym in main_syms else "unseen",
                                   "reason": "no valid feature observations after cleaning"})
            continue
        f = f.loc[max(valid_firsts):].copy()
        f[FEATURE_COLUMNS] = f[FEATURE_COLUMNS].ffill(limit=3)
        H = CONFIG["eval_horizon"]
        f["fwd_ret_h"] = f["Open"].shift(-(H + 1)) / f["Open"].shift(-1) - 1.0
        f["symbol"] = sym
        f.to_parquet(cache_dir / f"{sym}.parquet")
    if exclusion_rows:
        pd.DataFrame(exclusion_rows).to_csv(OUT_DIR / "reports" / "universe_exclusions.csv", index=False)

print(f"  Main training universe: {len(main_syms)} approved candidate stocks")
print(f"  Unseen holdout cohort:  {len(unseen_syms)} stocks")
print(f"  Top-20 liquid bluechips: {len(top20_bluechips)} stocks")
print(f"  NIFTY 50 Benchmark: {len(bench_df)} bars ({bench_df.index.min().date()} to {bench_df.index.max().date()})")

print("\nLoading precomputed feature panels for main universe...", flush=True)
t0 = time.time()
frames = []
missing_main = []
for sym in main_syms:
    fpath = cache_dir / f"{sym}.parquet"
    if fpath.exists():
        df = pd.read_parquet(fpath)
        df["symbol"] = sym
        df = df.reset_index().rename(columns={"index": "date"})
        frames.append(df)
    else:
        missing_main.append(sym)
if missing_main:
    raise RuntimeError(f"Approved main symbols lack local feature caches: {missing_main}")
panel = pd.concat(frames, ignore_index=True)
panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
for _col in FEATURE_COLUMNS + ["Open", "High", "Low", "Close", "Volume", "fwd_ret_h"]:
    panel[_col] = pd.to_numeric(panel[_col], errors="coerce").astype(np.float32)
panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
panel["row_in_sym"] = panel.groupby("symbol").cumcount()
print(f"Loaded {panel['symbol'].nunique()} stocks: {panel.shape[0]:,} bars in {time.time() - t0:.1f}s", flush=True)

# Save the exact SP-05 training and holdout manifests, including the per-symbol clean bar counts.
_bars_by_symbol = panel.groupby("symbol").size().to_dict()
universe_rows = [{"symbol": sym, "cohort": "main", "clean_feature_bars": int(_bars_by_symbol.get(sym, 0)), "eligible": True,
                  "training_use": "train/validation/test chronological"} for sym in main_syms]
for sym in unseen_syms:
    fp = cache_dir / f"{sym}.parquet"
    bars = len(pd.read_parquet(fp)) if fp.exists() else 0
    universe_rows.append({"symbol": sym, "cohort": "unseen", "clean_feature_bars": bars, "eligible": bool(bars >= 800),
                          "training_use": "test only; excluded from all fitting and calibration"})
for sym in top20_bluechips:
    universe_rows.append({"symbol": sym, "cohort": "top20_bluechip", "clean_feature_bars": int(_bars_by_symbol.get(sym, 0)),
                          "eligible": sym in main_syms, "training_use": "separate evaluation cohort"})
(OUT_DIR / "reports" / "eligible_symbols.csv").write_text(
    pd.DataFrame(universe_rows).to_csv(index=False), encoding="utf-8")
pd.DataFrame(columns=["symbol", "cohort", "reason"]).to_csv(OUT_DIR / "reports" / "universe_exclusions.csv", index=False)

# ----------------------------------------------------------------------------
# 3. Chronological 70:30 Split with Strict Window Isolation
# ----------------------------------------------------------------------------
d_train_end = pd.Timestamp(CONFIG["train_end"])
d_val_end = pd.Timestamp(CONFIG["val_end"])
L = CONFIG["seq_len"]

win_start = panel.groupby("symbol")["date"].shift(L - 1)
panel["win_start"] = win_start

SPLIT_BOUNDS = {
    "train": (None, d_train_end),
    "val": (d_train_end + pd.Timedelta(days=1), d_val_end),
    "test": (d_val_end + pd.Timedelta(days=1), None)
}

def eligible_rows(df_panel, split: str) -> np.ndarray:
    start, end = SPLIT_BOUNDS[split]
    ok = df_panel["win_start"].notna().to_numpy().copy()
    if start is not None:
        ok &= (df_panel["date"] >= start).to_numpy()
        if CONFIG["strict_window_isolation"]:
            ok &= (df_panel["win_start"] >= start).to_numpy()
    if end is not None:
        ok &= (df_panel["date"] <= end).to_numpy()
    return np.where(ok)[0]

train_rows_all = eligible_rows(panel, "train")
val_rows_all = eligible_rows(panel, "val")
test_rows_all = eligible_rows(panel, "test")

print("\n[Chronological Split with Strict Window Isolation (L=30)]")
print(f"  Train: <= {d_train_end.date()} | {len(train_rows_all):,} bars ({len(train_rows_all)/len(panel)*100:.2f}%)")
print(f"  Val:   {d_train_end.date() + pd.Timedelta(days=1)} to {d_val_end.date()} | {len(val_rows_all):,} bars ({len(val_rows_all)/len(panel)*100:.2f}%)")
print(f"  Test:  > {d_val_end.date()} | {len(test_rows_all):,} bars ({len(test_rows_all)/len(panel)*100:.2f}%)")

# ----------------------------------------------------------------------------
# 4. Scaler Fitting Exclusively on Training Observations
# ----------------------------------------------------------------------------
print("\nFitting preprocessing scaler on training observations exclusively...", flush=True)
fit_mask = (panel["date"] <= d_train_end).to_numpy()
RAW_MAT = panel[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

med = np.nanmedian(RAW_MAT[fit_mask], axis=0)
imp = np.where(np.isfinite(RAW_MAT[fit_mask]), RAW_MAT[fit_mask], med)
std = imp.std(axis=0)
std = np.where((std < 1e-8) | np.isnan(std) | np.isinf(std), 1.0, std)
mean = np.where(np.isnan(imp.mean(axis=0)) | np.isinf(imp.mean(axis=0)), 0.0, imp.mean(axis=0))
med = np.where(np.isnan(med) | np.isinf(med), 0.0, med)

PREPROCESSING = {
    "feature_columns": list(FEATURE_COLUMNS),
    "median": med.astype(np.float32),
    "mean": mean.astype(np.float32),
    "std": std.astype(np.float32),
    "clip": float(CONFIG["train"]["scaler_clip"]),
    "fit_rows_end_date": str(d_train_end.date()),
    "n_fit_rows": int(fit_mask.sum()),
}

print(f"Scaler fit on {PREPROCESSING['n_fit_rows']:,} rows with dates <= {PREPROCESSING['fit_rows_end_date']}")
X_SCALED = scale_features(RAW_MAT, PREPROCESSING["median"], PREPROCESSING["mean"], PREPROCESSING["std"], PREPROCESSING["clip"])
print(f"Scaled feature matrix: shape {X_SCALED.shape}, dtype {X_SCALED.dtype}, min {X_SCALED.min():.2f}, max {X_SCALED.max():.2f}")
del RAW_MAT
gc.collect()

# ----------------------------------------------------------------------------
# 5. Conservative Objective and Label Generation
# ----------------------------------------------------------------------------
OBJ = CONFIG["objective"]
SOFT = CONFIG["soft_limits"]
RAW = {c: panel[c].to_numpy(dtype=np.float32, copy=False) for c in FEATURE_COLUMNS}
OPEN_ARR = panel["Open"].to_numpy(dtype=float)
CLOSE_ARR = panel["Close"].to_numpy(dtype=float)
FWD_RET = panel["fwd_ret_h"].to_numpy(dtype=float)
ROW_IN_SYM = panel["row_in_sym"].to_numpy()

def cons_params(profile: np.ndarray) -> dict:
    risk, pat, agg = profile[:, 0], profile[:, 1], profile[:, 2]
    return {
        "w_ret": OBJ["w_return"],
        "w_dd": OBJ["w_drawdown"] * (1.5 - risk),
        "w_turn": OBJ["w_turnover"] * (0.5 + pat),
        "thr": OBJ["entry_threshold"] + 0.5 * (1.0 - agg),
        "soft_stop": 0.015 + 0.03 * risk,
        "vol_cap": 0.6 + 0.2 * risk,
    }

def decision_scores(feat: dict):
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe60 = np.clip(feat["ret_60"] / (feat["rv_60"] * np.sqrt(60.0 / 252.0) + 1e-8), -3.0, 3.0)
        struct = 0.5 * np.sign(feat["sma50_sma200"]) + 0.5 * np.sign(feat["close_ema50"])
        edge = 0.6 * sharpe60 + 0.6 * struct
        vol_pct = 0.5 * (feat["atr_pct_pctile_252"] + feat["rv20_pctile_252"])
        risk = vol_pct + 0.5 * np.clip(-feat["dd_from_high_60"] / 0.10, 0.0, 2.0)
    return edge, risk, vol_pct

def conservative_rule_labels(feat, pos, profile, hold, pnl, port, allow_short=True):
    p = cons_params(profile)
    edge, risk, vol_pct = decision_scores(feat)
    pdd, gross, rloss = (np.asarray(port[k], dtype=float) for k in ("pdd", "gross", "rloss"))
    with np.errstate(invalid="ignore"):
        prohibited = (
            (vol_pct >= p["vol_cap"]) |
            (np.abs(feat["abnormal_ret_z"]) >= SOFT["abnormal_ret_z"]) |
            (feat["gap_abs_atr"] >= SOFT["gap_atr"]) |
            (feat["bench_dd_60"] <= -SOFT["market_stress_dd"])
        )
        portfolio_block = (
            (pdd >= SOFT["portfolio_drawdown"]) |
            (gross >= SOFT["gross_exposure"]) |
            (rloss >= SOFT["recent_loss"])
        )
        u_long = p["w_ret"] * edge - p["w_dd"] * risk - p["w_turn"]
        u_short = -p["w_ret"] * edge - p["w_dd"] * risk - p["w_turn"] - OBJ["short_penalty"]
        long_entry = (u_long >= p["thr"]) & ~prohibited & ~portfolio_block
        short_entry = (u_short >= p["thr"]) & ~prohibited & ~portfolio_block
        h_long = p["w_ret"] * edge - p["w_dd"] * risk
        h_short = -p["w_ret"] * edge - p["w_dd"] * risk
        vol_spike = vol_pct >= SOFT["vol_exit_percentile"]
        derisk = (pdd >= SOFT["portfolio_drawdown"]) | ((rloss >= SOFT["recent_loss"]) & (pnl < 0))
        long_exit = (
            (pnl <= -p["soft_stop"]) |
            (h_long <= OBJ["exit_threshold"]) |
            vol_spike |
            (feat["abnormal_ret_z"] <= -SOFT["abnormal_ret_z"]) |
            derisk
        )
        short_exit = (
            (pnl <= -p["soft_stop"]) |
            (h_short <= OBJ["exit_threshold"]) |
            vol_spike |
            (feat["abnormal_ret_z"] >= SOFT["abnormal_ret_z"]) |
            derisk
        )
    y = np.zeros(len(pos), dtype=np.int64)
    flat = pos == 0
    y[flat & long_entry] = BUY
    if allow_short:
        y[flat & ~long_entry & short_entry] = SELL
    y[(pos == 1) & long_exit] = EXIT
    y[(pos == -1) & short_exit] = EXIT
    return y

def finalize_samples(rows, pos, prof, hold, pnl, port, y, allow_short=True):
    valid = valid_action_mask(pos, allow_short)
    assert valid[np.arange(len(y)), y].all(), "Invalid action in labels!"
    static = encode_static(pos, prof, pnl, hold, port["pdd"], port["gross"], port["size"], port["rloss"])
    return {
        "row": np.asarray(rows, dtype=np.int64),
        "pos": np.asarray(pos, dtype=np.int64),
        "profile": np.asarray(prof, dtype=np.float32),
        "hold": np.asarray(hold, dtype=np.float32),
        "pnl": np.asarray(pnl, dtype=np.float32),
        "port": port,
        "y": np.asarray(y, dtype=np.int64),
        "valid": valid,
        "static": static,
    }

def build_split_samples(rows, rng, allow_short=True):
    r = np.repeat(rows, 3)
    n = len(r)
    pos = np.tile(np.array([0, 1, -1]), len(rows))
    prof = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    cap = np.minimum(CONFIG["hold_sample_max"], ROW_IN_SYM[r] + 1)
    hold = np.where(pos == 0, 0, 1 + np.floor(rng.random(n) * cap).astype(int))
    entry_px = OPEN_ARR[r - np.maximum(hold, 1) + 1]
    pnl = np.where(pos == 0, 0.0, pos * (CLOSE_ARR[r] / np.where(entry_px > 0, entry_px, 1.0) - 1.0))
    size = np.where(pos == 0, 0.0, rng.uniform(0.02, 0.15, n))
    port = {
        "pdd": np.where(rng.random(n) < 0.55, rng.random(n) * 0.02, rng.random(n) * 0.15),
        "gross": np.maximum(rng.random(n) * 0.9, size),
        "size": size,
        "rloss": np.where(rng.random(n) < 0.6, 0.0, rng.random(n) * 0.08),
    }
    feat_slice = {k: v[r] for k, v in RAW.items()}
    y = conservative_rule_labels(feat_slice, pos, prof, hold, pnl, port, allow_short)
    return finalize_samples(r, pos, prof, hold, pnl, port, y, allow_short)

print("\nGenerating training, validation, and testing proxy samples...", flush=True)
rng = np.random.default_rng(CONFIG["seed"])

# Every eligible bar contributes flat, long and short position-state samples.
SAMPLES = {
    "train": build_split_samples(train_rows_all, rng, CONFIG["allow_short"]),
    "val": build_split_samples(val_rows_all, rng, CONFIG["allow_short"]),
    "test": build_split_samples(test_rows_all, rng, CONFIG["allow_short"]),
}

LABEL_SOURCE = "strategy_derived_proxy (conservative objective, decision-time)"
print("=" * 80)
print(f"LABEL SOURCE: {LABEL_SOURCE}")
print("No future price used for any label. All proxy labels causal at bar t.")
print("=" * 80)

dist = pd.DataFrame({s: np.bincount(S["y"], minlength=N_ACTIONS) for s, S in SAMPLES.items()}, index=ACTION_NAMES)
dist_pct = (100 * dist / dist.sum(axis=0)).round(2)
print("\n[Action Label Distribution]")
for c in dist.columns:
    print(f"  {c:>5s}: " + ", ".join([f"{act}: {dist.loc[act, c]:,} ({dist_pct.loc[act, c]:.1f}%)" for act in ACTION_NAMES]))

for s, S in SAMPLES.items():
    dates_s = panel.loc[S["row"], "date"]
    print(f"  {s:5s}: {len(S['y']):>8,} samples | {dates_s.min().date()} -> {dates_s.max().date()}")

assert panel.loc[SAMPLES["train"]["row"], "date"].max() < panel.loc[SAMPLES["val"]["row"], "date"].min()
assert panel.loc[SAMPLES["val"]["row"], "date"].max() < panel.loc[SAMPLES["test"]["row"], "date"].min()
print("Chronology check passed: train < val < test (strict, no temporal overlap)")

# Persist exact date boundaries, realized row/sample percentages, and per-stock/state/action counts.
_split_rows = {"train": train_rows_all, "val": val_rows_all, "test": test_rows_all}
_assigned_bar_count = sum(len(_rows) for _rows in _split_rows.values())
_split_dates = {"candidate_boundaries": {"train_end": str(d_train_end.date()), "validation_end": str(d_val_end.date())},
                "strict_sequence_isolation": bool(CONFIG["strict_window_isolation"]), "sequence_length": int(L),
                "panel_bars": int(len(panel)), "assigned_window_complete_bars": int(_assigned_bar_count),
                "unassigned_bars_due_to_warmup_or_boundary_isolation": int(len(panel) - _assigned_bar_count), "splits": {}}
for _name, _rows in _split_rows.items():
    _dates = panel.loc[_rows, "date"]
    _split_dates["splits"][_name] = {"first_decision_date": str(_dates.min().date()), "last_decision_date": str(_dates.max().date()),
                           "eligible_bars": int(len(_rows)), "eligible_bar_pct_of_panel": float(100 * len(_rows) / len(panel)),
                           "eligible_bar_pct_of_assigned": float(100 * len(_rows) / _assigned_bar_count)}
(OUT_DIR / "reports" / "chronological_split.json").write_text(json.dumps(_split_dates, indent=2))
_sample_counts = []
for _split, _S in SAMPLES.items():
    _sym = panel.loc[_S["row"], "symbol"].to_numpy()
    _counts = pd.DataFrame({"symbol": _sym, "position": _S["pos"], "action": _S["y"]}).groupby(["symbol", "position", "action"]).size()
    for (_symbol, _position, _action), _count in _counts.items():
        _sample_counts.append({"split": _split, "symbol": _symbol, "position": {0: "flat", 1: "long", -1: "short"}[_position],
                               "action": ACTION_NAMES[_action], "samples": int(_count)})
pd.DataFrame(_sample_counts).to_csv(OUT_DIR / "reports" / "sample_counts_per_stock_state_action.csv", index=False)

# Class weighting
cnt_tr = np.bincount(SAMPLES["train"]["y"], minlength=N_ACTIONS).astype(float)
w_tr = np.where(cnt_tr > 0, cnt_tr.sum() / (N_ACTIONS * np.maximum(cnt_tr, 1)), 0.0)
w_tr = np.minimum(w_tr, CONFIG["train"]["max_class_weight"])
CLASS_WEIGHTS = np.minimum(w_tr * (cnt_tr.sum() / (w_tr * cnt_tr).sum()), CONFIG["train"]["max_class_weight"]).astype(np.float32)
print("\nClass weights (capped inverse-frequency):", {n: round(float(w), 3) for n, w in zip(ACTION_NAMES, CLASS_WEIGHTS)})

# The full proxy sample set above is retained for reporting and class-weight estimation. Fit on a
# deterministic per-stock stratified subset so CPU-only training remains tractable without dropping stocks.
MAX_TRAIN_SAMPLES = CONFIG["train"]["max_fit_samples"]
_full_train_n = len(SAMPLES["train"]["y"])
if _full_train_n > MAX_TRAIN_SAMPLES:
    _full = SAMPLES["train"]
    _bar_rows = _full["row"][::3]
    _bar_symbols = panel.loc[_bar_rows, "symbol"].to_numpy()
    _groups = {sym: np.flatnonzero(_bar_symbols == sym) for sym in np.unique(_bar_symbols)}
    _target_bars = MAX_TRAIN_SAMPLES // 3
    _quota = {sym: int(np.floor(_target_bars * len(ix) / len(_bar_rows))) for sym, ix in _groups.items()}
    _left = _target_bars - sum(_quota.values())
    for sym in sorted(_groups, key=lambda k: (-len(_groups[k]) * _target_bars / len(_bar_rows) + _quota[k], k))[:_left]:
        _quota[sym] += 1
    _chosen = np.concatenate([ix[np.linspace(0, len(ix) - 1, min(len(ix), _quota[sym]), dtype=int)]
                              for sym, ix in _groups.items() if _quota[sym] > 0])
    _sample_idx = (_chosen[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
    SAMPLES["train"] = {k: ({pk: pv[_sample_idx] for pk, pv in v.items()} if k == "port" else v[_sample_idx])
                        for k, v in _full.items()}
    del _full, _bar_rows, _bar_symbols, _groups, _quota, _chosen, _sample_idx
    gc.collect()
print(f"Training subset: {_full_train_n:,} proxy labels generated; {len(SAMPLES['train']['y']):,} deterministic stratified samples used for fitting across {len(main_syms)} stocks.")
(OUT_DIR / "reports" / "training_sample_selection.json").write_text(json.dumps({
    "full_training_proxy_samples": int(_full_train_n), "fit_samples": int(len(SAMPLES["train"]["y"])),
    "max_fit_samples": MAX_TRAIN_SAMPLES, "method": "deterministic equal-weight proportional allocation by stock; evenly spaced sample bars within each stock",
    "class_weights_fit_on": "all full training labels before fit subset selection"
}, indent=2))

# ----------------------------------------------------------------------------
# 6. Neural Network Architecture & Setup
# ----------------------------------------------------------------------------
set_seed(CONFIG["seed"])
MODEL_CFG = {
    "n_features": N_FEATURES,
    "n_static": N_STATIC,
    "n_actions": N_ACTIONS,
    **CONFIG["model"]
}

net = ConservativePolicyNet(**MODEL_CFG).to(DEVICE)
print("\n[Model Architecture - ConservativePolicyNet]")
print(net)
n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_trainable:,}")

# Sequence gathering on the fly
X_ALL = torch.from_numpy(X_SCALED).to(DEVICE)
OFFSETS = torch.arange(-(L - 1), 1, device=DEVICE)
OFFSETS_NP = np.arange(-(L - 1), 1)

def to_torch(S):
    return {
        "row": torch.from_numpy(S["row"]).long().to(DEVICE),
        "static": torch.from_numpy(S["static"]).to(DEVICE),
        "y": torch.from_numpy(S["y"]).long().to(DEVICE),
        "valid": torch.from_numpy(S["valid"]).to(DEVICE)
    }

TS = {k: to_torch(v) for k, v in SAMPLES.items()}

def get_batch(T, idx):
    win = T["row"][idx].unsqueeze(1) + OFFSETS
    return X_ALL[win], T["static"][idx], T["valid"][idx], T["y"][idx]

@torch.no_grad()
def forward_all(model, T, batch_size=2048):
    model.eval()
    out = []
    for i in range(0, len(T["y"]), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(T["y"])), device=DEVICE)
        seq, st, valid, _ = get_batch(T, idx)
        out.append(mask_logits(model(seq, st), valid).cpu())
    return torch.cat(out)

# ----------------------------------------------------------------------------
# 7. Model Training Loop
# ----------------------------------------------------------------------------
tc = CONFIG["train"]
class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=DEVICE)
optimizer = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
y_val_cpu = TS["val"]["y"].cpu()
gen = torch.Generator().manual_seed(CONFIG["seed"])
n_train = len(TS["train"]["y"])
batch_sz = tc["batch_size"]

print(f"\nStarting training on {DEVICE} ({n_train:,} samples, batch_size={batch_sz}, max_epochs={tc['epochs']})...", flush=True)

if "--eval-only" in sys.argv:
    # Continue the same completed from-scratch run after a downstream evaluation failure.
    ckpt_file = OUT_DIR / "best_checkpoint.pt"
    hist_file = OUT_DIR / "training_history.csv"
    if not ckpt_file.exists() or not hist_file.exists():
        raise FileNotFoundError("--eval-only needs a best_checkpoint.pt and training_history.csv from this run")
    best_state = torch.load(ckpt_file, map_location=DEVICE, weights_only=True)
    net.load_state_dict(best_state)
    if "--eval-only-recover-log" in sys.argv:
        pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+) \| Val Macro-F1: ([0-9.]+) \| LR: ([0-9.eE+-]+) \| Time: ([0-9.]+)s")
        records = []
        for line in (OUT_DIR / "execution.log").read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                ep, _, tr, vl, f1, lr, dur = match.groups()
                records.append({"epoch": int(ep), "train_loss": float(tr), "val_loss": float(vl),
                                "val_macro_f1": float(f1), "lr": float(lr), "duration_sec": float(dur)})
        if not records:
            raise RuntimeError("No completed epochs could be recovered from this run's execution.log")
        history_df = pd.DataFrame(records)
        history_df.to_csv(hist_file, index=False)
    else:
        history_df = pd.read_csv(hist_file)
    BEST_EPOCH = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    best_val = float(history_df["val_loss"].min())
    torch.save(best_state, OUT_DIR / "final_checkpoint.pt")
    net.eval()
    print(f"\nEvaluation continuation loaded this run's epoch {BEST_EPOCH} best checkpoint (val loss {best_val:.4f}).")
else:
    # Always initialize from the seed-42 random weights above. Existing artifacts are never loaded.
    resume_interrupted = "--resume-interrupted" in sys.argv
    if resume_interrupted:
        # Resume only this run's seed-42 checkpoint after an interrupted training process.
        # The run log is authoritative for epochs completed before interruption.
        ckpt_file = OUT_DIR / "best_checkpoint.pt"
        log_file = OUT_DIR / "execution.log"
        if not ckpt_file.exists() or not log_file.exists():
            raise FileNotFoundError("--resume-interrupted requires this run's checkpoint and execution.log")
        pattern = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+) \| Val Macro-F1: ([0-9.]+) \| LR: ([0-9.eE+-]+) \| Time: ([0-9.]+)s")
        history = []
        for line in log_file.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                ep, _, tr, vl, f1, lr, dur = match.groups()
                history.append({"epoch": int(ep), "train_loss": float(tr), "val_loss": float(vl),
                                "val_macro_f1": float(f1), "lr": float(lr), "duration_sec": float(dur)})
        if not history:
            raise RuntimeError("No completed epochs could be recovered from this run's execution.log")
        best_state = torch.load(ckpt_file, map_location=DEVICE, weights_only=True)
        net.load_state_dict(best_state)
        best_val = min(row["val_loss"] for row in history)
        best_epoch_seen = min(history, key=lambda row: row["val_loss"])["epoch"]
        bad_epochs = sum(row["epoch"] > best_epoch_seen for row in history)
        start_epoch = max(row["epoch"] for row in history) + 1
        optimizer.param_groups[0]["lr"] = history[-1]["lr"]
        gen = torch.Generator().manual_seed(CONFIG["seed"])
        for _ in range(start_epoch - 1):
            torch.randperm(n_train, generator=gen)
        print(f"Resuming same fresh-initialization run at epoch {start_epoch}; recovered {len(history)} epoch records.")
    else:
        history = []
        start_epoch = 1
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

    train_t0 = time.time()

    for epoch in range(start_epoch, tc["epochs"] + 1):
        ep_t0 = time.time()
        net.train()
        perm = torch.randperm(n_train, generator=gen).to(DEVICE)
        total_loss, seen = 0.0, 0
    
        for i in range(0, n_train, batch_sz):
            idx = perm[i:i + batch_sz]
            seq, st, valid, y = get_batch(TS["train"], idx)
            if tc["input_noise_std"] > 0:
                seq = seq + tc["input_noise_std"] * torch.randn_like(seq)
            logits = mask_logits(net(seq, st), valid)
            loss = Fnn.cross_entropy(logits, y, weight=class_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
        
        train_loss = total_loss / seen
        val_logits = forward_all(net, TS["val"])
        val_loss = Fnn.cross_entropy(val_logits, y_val_cpu, weight=class_w.cpu()).item()
        val_pred = val_logits.argmax(1).numpy()
        val_f1 = f1_score(y_val_cpu.numpy(), val_pred, average="macro", labels=list(range(N_ACTIONS)), zero_division=0)
        scheduler.step(val_loss)
    
        cur_lr = optimizer.param_groups[0]["lr"]
        ep_dur = time.time() - ep_t0
        improved = val_loss < best_val - 1e-5
    
        if improved:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            # Save best checkpoint
            torch.save(best_state, OUT_DIR / "best_checkpoint.pt")
        else:
            bad_epochs += 1
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "lr": cur_lr,
            "duration_sec": ep_dur
        })
    
        star = " * BEST" if improved else ""
        print(f"Epoch {epoch:2d}/{tc['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.3f} | LR: {cur_lr:.1e} | Time: {ep_dur:.1f}s{star}", flush=True)
    
        if bad_epochs >= tc["patience"]:
            print(f"Early stopping at epoch {epoch} (no validation loss improvement for {tc['patience']} epochs)", flush=True)
            break

    # Restore the lowest-validation-loss checkpoint before calibration and final evaluation.
    if best_state is None:
        raise RuntimeError("Training ended without producing a validation checkpoint")
    net.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "final_checkpoint.pt")
    net.eval()
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_DIR / "training_history.csv", index=False)
    BEST_EPOCH = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    print(f"\nTraining Complete! Best Epoch: {BEST_EPOCH} | Best Val Loss: {best_val:.4f} | Total Time: {time.time() - train_t0:.1f}s")

# ----------------------------------------------------------------------------
# 8. Temperature Calibration & Classification Evaluation
# ----------------------------------------------------------------------------
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = Fnn.cross_entropy(logits / torch.exp(log_t), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 10.0).item())

def ece_score(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc = float(correct[m].mean())
            c_mean = float(conf[m].mean())
            ece += m.mean() * abs(acc - c_mean)
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()), "confidence": c_mean, "accuracy": acc})
    return float(ece), rows

def classification_summary(logits, y, temperature):
    y = np.asarray(y)
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    labels = list(range(N_ACTIONS))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    ece, r_bins = ece_score(conf, (pred == y).astype(float))
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        "majority_class_baseline_accuracy": float(np.bincount(y, minlength=N_ACTIONS).max() / len(y)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "ece": ece,
        "reliability_bins": r_bins,
        "temperature": float(temperature),
        "per_class": {
            ACTION_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            } for i in labels
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "pred": pred,
        "conf": conf,
    }

LOGITS = {k: forward_all(net, TS[k]) for k in ("val", "test")}
Y = {k: TS[k]["y"].cpu().numpy() for k in ("val", "test")}
TEMPERATURE = fit_temperature(LOGITS["val"], TS["val"]["y"].cpu())
print(f"\nFitted Temperature (Validation Only): {TEMPERATURE:.3f}")

CLS = {k: classification_summary(LOGITS[k], Y[k], TEMPERATURE) for k in ("val", "test")}
CLS_UNCAL_TEST = classification_summary(LOGITS["test"], Y["test"], 1.0)

for k in ("val", "test"):
    c = CLS[k]
    print(f"\n[{k.upper()}] Acc: {c['accuracy']:.3f} | Bal Acc: {c['balanced_accuracy']:.3f} | Macro-F1: {c['macro_f1']:.3f} | Weighted-F1: {c['weighted_f1']:.3f}")
    print(f"        NLL: {c['nll']:.4f} | ECE: {c['ece']:.4f}" + (f" (Uncalibrated: NLL {CLS_UNCAL_TEST['nll']:.4f}, ECE {CLS_UNCAL_TEST['ece']:.4f})" if k == "test" else ""))
    print(pd.DataFrame(c["per_class"]).T.round(3).to_string())

# Instantiate and build ConservativePolicy wrapper
RISK_LIMITS = RiskLimits(**CONFIG["risk_limits"])
GATE = RiskController(RISK_LIMITS)

# Required deterministic adversarial checks: blocked BUY during a portfolio halt and hard-stop EXIT.
_adversarial_features = {"atr_pct_pctile_252": np.array([0.2]), "rv20_pctile_252": np.array([0.2]),
                         "gap_abs_atr": np.array([0.1]), "abnormal_ret_z": np.array([0.0]),
                         "bench_dd_60": np.array([-0.01])}
_blocked_action, _blocked_reasons = GATE.review_batch(
    [BUY], [0.99], [0], [0.0], [0], _adversarial_features,
    {"portfolio_drawdown": RISK_LIMITS.halt_drawdown, "gross_exposure": 0.0, "recent_loss": 0.0,
     "entry_fraction": RISK_LIMITS.max_position_fraction})
assert int(_blocked_action[0]) == HOLD, "RiskController failed to block BUY during prohibited portfolio drawdown"
assert bool(_blocked_reasons[0, REASON_NAMES.index("drawdown_halt")]), "Drawdown halt reason was not recorded"
_stop_action, _stop_reasons = GATE.review_batch(
    [HOLD], [0.0], [1], [-RISK_LIMITS.position_stop_loss], [1], _adversarial_features,
    {"portfolio_drawdown": 0.0, "gross_exposure": 0.1, "recent_loss": 0.0})
assert int(_stop_action[0]) == EXIT, "RiskController failed to force EXIT at the hard position stop"
assert bool(_stop_reasons[0, REASON_NAMES.index("position_stop_loss")]), "Hard stop reason was not recorded"
(OUT_DIR / "reports" / "risk_controller_adversarial_checks.json").write_text(json.dumps({
    "portfolio_halt_blocks_high_confidence_buy": {"passed": True, "final_action": ACTION_NAMES[int(_blocked_action[0])], "reason": "drawdown_halt"},
    "hard_stop_forces_exit_at_zero_confidence": {"passed": True, "final_action": ACTION_NAMES[int(_stop_action[0])], "reason": "position_stop_loss"}
}, indent=2))
print("RiskController adversarial checks passed: halted BUY blocked; hard stop forced EXIT.")

policy_config = {
    "model_name": "SP-05 Conservative Risk-Aware Policy",
    "policy_id": "SP-05",
    "version": "1.0.0",
    "artifact": "conservative_policy.pt",
    "market": CONFIG["market"],
    "benchmark": CONFIG["benchmark"],
    "seq_len": L,
    "allow_short": CONFIG["allow_short"],
    "temperature": TEMPERATURE,
    "model": MODEL_CFG,
    "action_space": {str(i): n for i, n in enumerate(ACTION_NAMES)},
    "label_source": LABEL_SOURCE,
    "label_note": "Labels are strategy-derived PROXY labels from an explicit conservative objective: NOT human demonstrations.",
    "confidence_definition": "temperature-calibrated probability of the chosen action after masking invalid actions",
    "authority": "This policy only PROPOSES actions. sp05_risk.RiskController has final authority over execution.",
    "risk_limits": RISK_LIMITS.to_dict(),
    "execution_convention": "decision at close of bar t, fill at open of bar t+1",
    "training": {
        **CONFIG["train"],
        "seed": CONFIG["seed"],
        "best_epoch": BEST_EPOCH,
        "class_weights": CLASS_WEIGHTS.tolist(),
        "n_train_samples": int(n_train),
    },
}

policy = ConservativePolicy(net, policy_config, PREPROCESSING, DEVICE)

# ----------------------------------------------------------------------------
# 9. Risk Condition Evaluation
# ----------------------------------------------------------------------------
COND_NAMES = {
    0: "flat: calm, portfolio within soft limits",
    1: "flat: entry PROHIBITED (volatility / abnormal move / market stress)",
    2: "flat: portfolio soft limit breached (drawdown / exposure / recent loss)",
    3: "holding: risk elevated (vol spike / de-risk trigger)",
    4: "holding: calm"
}

def condition_codes(S):
    r, pos, prof = S["row"], S["pos"], S["profile"]
    feat = {k: v[r] for k, v in RAW.items()}
    p = cons_params(prof)
    edge, risk, vol_pct = decision_scores(feat)
    port = S["port"]
    with np.errstate(invalid="ignore"):
        prohibited = (
            (vol_pct >= p["vol_cap"]) |
            (np.abs(feat["abnormal_ret_z"]) >= SOFT["abnormal_ret_z"]) |
            (feat["gap_abs_atr"] >= SOFT["gap_atr"]) |
            (feat["bench_dd_60"] <= -SOFT["market_stress_dd"])
        )
        pblock = (port["pdd"] >= SOFT["portfolio_drawdown"]) | (port["gross"] >= SOFT["gross_exposure"]) | (port["rloss"] >= SOFT["recent_loss"])
        elevated = (vol_pct >= SOFT["vol_exit_percentile"]) | (port["pdd"] >= SOFT["portfolio_drawdown"])
    cond = np.zeros(len(r), dtype=int)
    cond[(pos == 0) & pblock] = 2
    cond[(pos == 0) & prohibited] = 1
    cond[(pos != 0) & ~elevated] = 4
    cond[(pos != 0) & elevated] = 3
    return cond

COND_CLS, VIOLATIONS = {}, {}
for split in ("val", "test"):
    S = SAMPLES[split]
    y = Y[split]
    pred = CLS[split]["pred"]
    cond = condition_codes(S)
    rows_ = []
    for cd_, name in COND_NAMES.items():
        m = cond == cd_
        if m.sum() == 0:
            continue
        present = np.unique(np.concatenate([y[m], pred[m]]))
        rows_.append({
            "condition": name,
            "n": int(m.sum()),
            "accuracy": float((y[m] == pred[m]).mean()),
            "macro_f1": float(f1_score(y[m], pred[m], average="macro", labels=present, zero_division=0)),
            **{f"true_{a}": int((y[m] == i).sum()) for i, a in enumerate(ACTION_NAMES)},
            **{f"pred_{a}": int((pred[m] == i).sum()) for i, a in enumerate(ACTION_NAMES)}
        })
    COND_CLS[split] = pd.DataFrame(rows_).set_index("condition")
    forb = np.isin(cond, [1, 2])
    VIOLATIONS[split] = {
        "flat_forbidden_states": int(forb.sum()),
        "network_proposes_entry_in_forbidden_states_pct": float(100 * np.isin(pred[forb], [BUY, SELL]).mean()) if forb.any() else 0.0,
        "labels_with_entry_in_forbidden_states_pct": float(100 * np.isin(y[forb], [BUY, SELL]).mean()) if forb.any() else 0.0,
    }
    print(f"\n[{split.upper()}] Classification by Risk Condition:")
    print(COND_CLS[split][["n", "accuracy", "macro_f1", "pred_HOLD", "pred_BUY", "pred_SELL", "pred_EXIT"]].round(3).to_string())
    print(f"[{split.upper()}] Entry Proposals in Forbidden States: {VIOLATIONS[split]}")

# ----------------------------------------------------------------------------
# 10. Generalization Experiments (Unseen Holdout, Bluechips, Profiles, Regimes)
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RUNNING GENERALIZATION EXPERIMENTS")
print("=" * 80)

# Experiment B: Unseen-Stock Holdout Cohort (100 stocks)
print("\n--- Experiment B: Unseen-Stock Holdout Cohort (100 stocks) ---", flush=True)
unseen_frames = []
for sym in unseen_syms:
    fpath = cache_dir / f"{sym}.parquet"
    if fpath.exists():
        df = pd.read_parquet(fpath)
        df["symbol"] = sym
        df = df.reset_index().rename(columns={"index": "date"})
        unseen_frames.append(df)

unseen_panel = pd.concat(unseen_frames, ignore_index=True)
unseen_panel["date"] = pd.to_datetime(unseen_panel["date"]).dt.normalize()
unseen_panel = unseen_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
unseen_panel["row_in_sym"] = unseen_panel.groupby("symbol").cumcount()
unseen_win_start = unseen_panel.groupby("symbol")["date"].shift(L - 1)
unseen_panel["win_start"] = unseen_win_start

# Evaluate unseen stocks on test split
unseen_test_rows = eligible_rows(unseen_panel, "test")
print(f"Unseen cohort test rows: {len(unseen_test_rows):,} across {len(unseen_syms)} stocks")

# Scale unseen features with training scaler
unseen_raw_mat = unseen_panel[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
unseen_scaled = scale_features(unseen_raw_mat, PREPROCESSING["median"], PREPROCESSING["mean"], PREPROCESSING["std"], PREPROCESSING["clip"])
unseen_x_all = torch.from_numpy(unseen_scaled).to(DEVICE)

# Sample unseen test set
unseen_rng = np.random.default_rng(CONFIG["seed"] + 101)
unseen_samples = build_split_samples(unseen_test_rows, unseen_rng, CONFIG["allow_short"])

@torch.no_grad()
def forward_custom(model, T, x_tensor, batch_size=2048):
    model.eval()
    out = []
    for i in range(0, len(T["y"]), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(T["y"])), device=DEVICE)
        win = T["row"][idx].unsqueeze(1) + OFFSETS
        seq = x_tensor[win]
        st = T["static"][idx]
        valid = T["valid"][idx]
        out.append(mask_logits(model(seq, st), valid).cpu())
    return torch.cat(out)

unseen_ts = {
    "row": torch.from_numpy(unseen_samples["row"]).long().to(DEVICE),
    "static": torch.from_numpy(unseen_samples["static"]).to(DEVICE),
    "y": torch.from_numpy(unseen_samples["y"]).long().to(DEVICE),
    "valid": torch.from_numpy(unseen_samples["valid"]).to(DEVICE)
}
unseen_logits = forward_custom(net, unseen_ts, unseen_x_all)
unseen_cls = classification_summary(unseen_logits, unseen_samples["y"], TEMPERATURE)
print(f"Unseen Cohort [TEST]: Acc {unseen_cls['accuracy']:.3f} | Bal Acc {unseen_cls['balanced_accuracy']:.3f} | Macro-F1 {unseen_cls['macro_f1']:.3f} | ECE {unseen_cls['ece']:.4f}")
print(pd.DataFrame(unseen_cls["per_class"]).T.round(3).to_string())

# Experiment C: Top-20 Liquid Bluechips Cohort
print("\n--- Experiment C: Top-20 Liquid Bluechips Cohort ---", flush=True)
bluechip_mask = panel["symbol"].isin(top20_bluechips).to_numpy()
bluechip_test_rows = test_rows_all[bluechip_mask[test_rows_all]]
print(f"Bluechip test rows: {len(bluechip_test_rows):,} across {len(top20_bluechips)} bluechips")
bluechip_samples = build_split_samples(bluechip_test_rows, rng, CONFIG["allow_short"])
bluechip_ts = {
    "row": torch.from_numpy(bluechip_samples["row"]).long().to(DEVICE),
    "static": torch.from_numpy(bluechip_samples["static"]).to(DEVICE),
    "y": torch.from_numpy(bluechip_samples["y"]).long().to(DEVICE),
    "valid": torch.from_numpy(bluechip_samples["valid"]).to(DEVICE)
}
bluechip_logits = forward_all(net, bluechip_ts)
bluechip_cls = classification_summary(bluechip_logits, bluechip_samples["y"], TEMPERATURE)
print(f"Top-20 Bluechips [TEST]: Acc {bluechip_cls['accuracy']:.3f} | Bal Acc {bluechip_cls['balanced_accuracy']:.3f} | Macro-F1 {bluechip_cls['macro_f1']:.3f} | ECE {bluechip_cls['ece']:.4f}")
print(pd.DataFrame(bluechip_cls["per_class"]).T.round(3).to_string())

# Experiment D: Trader Profile Sensitivity
PROFILES = {
    "balanced": (0.5, 0.5, 0.5),
    "conservative": (0.2, 0.8, 0.2),
    "aggressive": (0.8, 0.3, 0.8)
}

print("\n--- Experiment D: Trader Profile Sensitivity ---", flush=True)
profile_eval = {}
for pname, prof_vals in PROFILES.items():
    # Evaluate test samples with pinned profile
    test_rows_sub = test_rows_all[:15000]
    n_sub = len(test_rows_sub) * 3
    r_sub = np.repeat(test_rows_sub, 3)
    pos_sub = np.tile(np.array([0, 1, -1]), len(test_rows_sub))
    prof_sub = np.tile(np.array(prof_vals, dtype=np.float32), (n_sub, 1))
    hold_sub = np.where(pos_sub == 0, 0, 10)
    pnl_sub = np.where(pos_sub == 0, 0.0, 0.01 * pos_sub)
    port_sub = {"pdd": np.zeros(n_sub), "gross": np.full(n_sub, 0.3), "size": np.full(n_sub, 0.08), "rloss": np.zeros(n_sub)}
    y_sub = conservative_rule_labels({k: v[r_sub] for k, v in RAW.items()}, pos_sub, prof_sub, hold_sub, pnl_sub, port_sub, CONFIG["allow_short"])
    final_sub = finalize_samples(r_sub, pos_sub, prof_sub, hold_sub, pnl_sub, port_sub, y_sub, CONFIG["allow_short"])
    ts_sub = {
        "row": torch.from_numpy(final_sub["row"]).long().to(DEVICE),
        "static": torch.from_numpy(final_sub["static"]).to(DEVICE),
        "y": torch.from_numpy(final_sub["y"]).long().to(DEVICE),
        "valid": torch.from_numpy(final_sub["valid"]).to(DEVICE)
    }
    sub_logits = forward_all(net, ts_sub)
    sub_cls = classification_summary(sub_logits, final_sub["y"], TEMPERATURE)
    sub_pred = sub_cls["pred"]
    profile_eval[pname] = {
        "profile": prof_vals,
        "accuracy": sub_cls["accuracy"],
        "macro_f1": sub_cls["macro_f1"],
        "buy_prop_pct": float(100 * (sub_pred == BUY).mean()),
        "sell_prop_pct": float(100 * (sub_pred == SELL).mean()),
        "hold_prop_pct": float(100 * (sub_pred == HOLD).mean()),
        "exit_prop_pct": float(100 * (sub_pred == EXIT).mean()),
    }
    print(f"Profile: {pname:<12s} | HOLD: {profile_eval[pname]['hold_prop_pct']:.1f}% | BUY: {profile_eval[pname]['buy_prop_pct']:.1f}% | SELL: {profile_eval[pname]['sell_prop_pct']:.1f}% | EXIT: {profile_eval[pname]['exit_prop_pct']:.1f}% | F1: {sub_cls['macro_f1']:.3f}")

# Experiment E: Past-Only Market Regime Evaluation
print("\n--- Experiment E: Market Regime Evaluation (Test Split) ---", flush=True)
bench_feat = build_features(bench_df, bench_df)
bench_close = bench_df["Close"].reindex(panel["date"]).ffill(limit=3).to_numpy()
bench_sma200 = bench_df["Close"].rolling(200).mean().reindex(panel["date"]).ffill(limit=3).to_numpy()
bench_ret60 = (bench_df["Close"] / bench_df["Close"].shift(60) - 1.0).reindex(panel["date"]).ffill(limit=3).to_numpy()
bench_rv20_p = bench_feat["bench_rv20_pctile_252"].reindex(panel["date"]).ffill(limit=3).to_numpy()
bench_dd252 = (bench_df["Close"] / bench_df["Close"].rolling(252).max() - 1.0).reindex(panel["date"]).ffill(limit=3).to_numpy()

regimes = {}
test_rows_arr = SAMPLES["test"]["row"]
y_test = SAMPLES["test"]["y"]
pred_test = CLS["test"]["pred"]

m_bull = (bench_close[test_rows_arr] > bench_sma200[test_rows_arr]) & (bench_ret60[test_rows_arr] > 0)
m_bear = (bench_close[test_rows_arr] < bench_sma200[test_rows_arr]) & (bench_ret60[test_rows_arr] < 0)
m_side = ~m_bull & ~m_bear
m_high_vol = bench_rv20_p[test_rows_arr] >= 0.85
m_low_vol = bench_rv20_p[test_rows_arr] < 0.30
m_severe_dd = bench_dd252[test_rows_arr] <= -0.10

regime_masks = {
    "Bull Market": m_bull,
    "Bear Market": m_bear,
    "Sideways/Chop": m_side,
    "High Volatility (>=85%ile)": m_high_vol,
    "Low Volatility (<30%ile)": m_low_vol,
    "Severe Benchmark Drawdown (<= -10%)": m_severe_dd,
}

regime_results = []
for rname, rmask in regime_masks.items():
    if rmask.sum() > 0:
        r_acc = float(accuracy_score(y_test[rmask], pred_test[rmask]))
        r_f1 = float(f1_score(y_test[rmask], pred_test[rmask], average="macro", zero_division=0))
        regime_results.append({
            "regime": rname,
            "samples": int(rmask.sum()),
            "accuracy": r_acc,
            "macro_f1": r_f1,
            "buy_pct": float(100 * (pred_test[rmask] == BUY).mean()),
            "hold_pct": float(100 * (pred_test[rmask] == HOLD).mean()),
            "exit_pct": float(100 * (pred_test[rmask] == EXIT).mean()),
        })
        print(f"Regime: {rname:<35s} | N: {rmask.sum():>6,} | Acc: {r_acc:.3f} | Macro-F1: {r_f1:.3f} | HOLD: {100 * (pred_test[rmask] == HOLD).mean():.1f}%")

df_regimes = pd.DataFrame(regime_results).set_index("regime")

# ----------------------------------------------------------------------------
# 11. Portfolio Backtesting System
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PORTFOLIO BACKTESTING (Top-20 Liquid Indian Bluechips)")
print("Decision at close t, Fill at open t+1, 5 bps fee + 5 bps slippage")
print("=" * 80, flush=True)

FEE = CONFIG["fee_bps"] / 1e4
SLIP = CONFIG["slippage_bps"] / 1e4
BORROW_DAILY = CONFIG["short_borrow_bps_annual"] / 1e4 / 252.0
CAP0 = float(CONFIG["initial_capital"])
DD_LOOKBACK = int(CONFIG["dd_lookback_bars"])

def prepare_backtest(split: str, symbols: list) -> dict:
    start, end = SPLIT_BOUNDS[split]
    sub = panel[panel["symbol"].isin(symbols)].copy()
    if start is not None:
        sub = sub[sub["date"] >= start]
        if CONFIG["strict_window_isolation"]:
            sub = sub[sub["win_start"] >= start]
    if end is not None:
        sub = sub[sub["date"] <= end]
        
    active_symbols = [s for s in symbols if s in sub["symbol"].unique()]
    piv = sub.pivot(index="date", columns="symbol", values="row_in_sym")[active_symbols]
    valid_dates = piv.dropna(how="any").index
    
    sub = sub[sub["date"].isin(valid_dates)]
    piv_open = sub.pivot(index="date", columns="symbol", values="Open")[active_symbols]
    piv_close = sub.pivot(index="date", columns="symbol", values="Close")[active_symbols]
    
    # Global row IDs for feature lookup
    sub_key = sub.assign(global_row=sub.index)
    piv_global = sub_key.pivot(index="date", columns="symbol", values="global_row")[active_symbols]
    
    bclose = bench_df["Close"].reindex(valid_dates).ffill(limit=3).to_numpy(dtype=float)
    return {
        "split": split,
        "dates": valid_dates,
        "symbols": active_symbols,
        "rows": piv_global.to_numpy().astype(np.int64),
        "open": piv_open.to_numpy(dtype=float),
        "close": piv_close.to_numpy(dtype=float),
        "bench_close": bclose
    }

def run_portfolio_backtest(bt: dict, decide_fn, allow_short: bool, gate=None, conf_sizing: bool = False) -> dict:
    T, S = bt["open"].shape
    cash = CAP0
    shares = np.zeros(S)
    entry_px = np.zeros(S)
    entry_k = np.zeros(S, dtype=int)
    entry_conf = np.zeros(S)
    hold = np.zeros(S, dtype=int)
    eq_hist = [CAP0]
    ret_hist, eq_out, expo_hist, turn_hist, pdd_hist, trades, events = [], [], [], [], [], [], []
    reason_counts = np.zeros(len(REASON_NAMES), dtype=int)
    override_days = []
    n_entries = n_skipped = 0
    cap_gross = CONFIG["max_gross_exposure"] if gate is None else min(CONFIG["max_gross_exposure"], gate.limits.max_gross_exposure)
    entry_frac = min(CONFIG["position_fraction"], CONFIG["max_position_size"])
    if gate is not None:
        entry_frac = min(entry_frac, gate.limits.max_position_fraction)
        
    for k in range(T - 1):
        prev_eq = eq_hist[-1]
        peak = max(eq_hist[-(DD_LOOKBACK + 1):])
        pdd = max(0.0, 1.0 - prev_eq / peak)
        values = shares * bt["close"][k]
        gross_now = float(np.abs(values).sum() / prev_eq)
        size_frac = np.abs(values) / prev_eq
        rloss = max(0.0, 1.0 - prev_eq / eq_hist[max(0, len(eq_hist) - 21)])
        pos = np.sign(shares).astype(int)
        pnl = np.where(pos != 0, pos * (bt["close"][k] / np.where(entry_px > 0, entry_px, 1.0) - 1.0), 0.0)
        ctx = {"portfolio_drawdown": pdd, "gross_exposure": gross_now, "size_frac": size_frac, "recent_loss": rloss}
        
        probs = np.asarray(decide_fn(k, bt["rows"][k], pos, pnl, hold, ctx), dtype=float)
        valid = valid_action_mask(pos, allow_short)
        proposed = np.where(valid, probs, -1.0).argmax(axis=1)
        conf = probs[np.arange(S), proposed]
        reasons = np.zeros((S, len(REASON_NAMES)), dtype=bool)
        
        if gate is not None:
            feats = {n: RAW[n][bt["rows"][k]] for n in REQUIRED_FEATURES}
            action, reasons = gate.review_batch(
                proposed, conf, pos, pnl, hold, feats,
                {"portfolio_drawdown": pdd, "gross_exposure": gross_now, "entry_fraction": entry_frac, "recent_loss": rloss}
            )
            reason_counts += reasons.sum(axis=0)
            changed = np.where(action != proposed)[0]
            override_days.append(len(changed))
            for s in changed:
                events.append({
                    "k": k, "symbol": bt["symbols"][s], "row": int(bt["rows"][k][s]),
                    "proposed": int(proposed[s]), "final": int(action[s]),
                    "reasons": [REASON_NAMES[j] for j in np.where(reasons[s])[0]]
                })
        else:
            action = proposed
            override_days.append(0)
            
        o = bt["open"][k + 1]
        traded = 0.0
        entered = np.zeros(S, dtype=bool)
        
        # Exits first: free up cash and room
        for s in np.where(action == EXIT)[0]:
            sh = shares[s]
            if sh != 0:
                px = o[s] * (1 - SLIP) if sh > 0 else o[s] * (1 + SLIP)
                cash += sh * px - FEE * abs(sh) * px
                traded += abs(sh) * px
                forced = [REASON_NAMES[j] for j in np.where(reasons[s])[0] if REASON_NAMES[j] in FORCED_EXIT_REASONS]
                trades.append({
                    "symbol": bt["symbols"][s],
                    "side": int(np.sign(sh)),
                    "entry_date": bt["dates"][entry_k[s]],
                    "exit_date": bt["dates"][k + 1],
                    "decision_date": bt["dates"][entry_k[s] - 1],
                    "bars": int(hold[s]),
                    "ret": float(np.sign(sh) * (px / entry_px[s] - 1.0)),
                    "exit_by": ("gate: " + forced[0]) if forced else "policy",
                    "entry_conf": float(entry_conf[s])
                })
                shares[s] = 0.0
                entry_px[s] = 0.0
                
        equity_open = cash + float((shares * o).sum())
        
        # Entries, highest confidence first
        for s in [i for i in np.argsort(-conf) if action[i] in (BUY, SELL)]:
            scale = float(np.clip((conf[s] - 0.5) / 0.4, 0.5, 1.0)) if conf_sizing else 1.0
            target = entry_frac * equity_open * scale
            room = cap_gross * equity_open - float(np.abs(shares * o).sum())
            if action[s] == BUY:
                px = o[s] * (1 + SLIP)
                notional = min(target, room, cash / (1 + FEE))
            else:
                px = o[s] * (1 - SLIP)
                notional = min(target, room)
            if notional < CONFIG["min_trade_fraction"] * equity_open:
                n_skipped += 1
                continue
            if action[s] == BUY:
                shares[s] = notional / px
                cash -= notional * (1 + FEE)
            else:
                shares[s] = -notional / px
                cash += notional * (1 - FEE)
            entry_px[s] = px
            entry_k[s] = k + 1
            entry_conf[s] = conf[s]
            entered[s] = True
            traded += notional
            n_entries += 1
            
        c1 = bt["close"][k + 1]
        short_mask = shares < 0
        cash -= float(-(shares[short_mask] * c1[short_mask]).sum()) * BORROW_DAILY
        eq = max(cash + float((shares * c1).sum()), 1e-9)
        hold = np.where(entered, 1, np.where(shares != 0, hold + 1, 0))
        ret_hist.append(eq / prev_eq - 1.0)
        eq_hist.append(eq)
        eq_out.append(eq / CAP0)
        expo_hist.append(float(np.abs(shares * c1).sum()) / eq)
        turn_hist.append(traded / prev_eq)
        pdd_hist.append(max(0.0, 1.0 - eq / max(eq_hist[-(DD_LOOKBACK + 1):])))
        
    tr = pd.DataFrame(trades)
    return {
        "dates": bt["dates"][1:],
        "ret": np.array(ret_hist),
        "equity": np.array(eq_out),
        "exposure": np.array(expo_hist),
        "turnover": np.array(turn_hist),
        "portfolio_drawdown": np.array(pdd_hist),
        "n_entries": n_entries,
        "n_closed_trades": len(tr),
        "n_skipped_entries": n_skipped,
        "open_positions_at_end": int((shares != 0).sum()),
        "trades": tr,
        "win_rate": float((tr["ret"] > 0).mean()) if len(tr) else float("nan"),
        "avg_trade_return": float(tr["ret"].mean()) if len(tr) else float("nan"),
        "gate_reason_counts": dict(zip(REASON_NAMES, reason_counts.tolist())),
        "gate_events": events,
        "gate_override_days": np.array(override_days) if gate is not None else np.zeros(T - 1, dtype=int),
    }

def perf_metrics(r, turnover=None, exposure=None) -> dict:
    r = np.asarray(r, dtype=float)
    n = len(r)
    eq = np.cumprod(1.0 + r)
    years = n / 252.0
    sd = r.std(ddof=1) if n > 1 else np.nan
    ex = r - CONFIG["risk_free_annual"] / 252.0
    dd_sd = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) if n else np.nan
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else float("nan")
    var5 = float(np.percentile(r, 5)) if n else float("nan")
    out = {
        "n_days": int(n),
        "total_return": float(eq[-1] - 1.0),
        "cagr": cagr,
        "ann_vol": float(sd * np.sqrt(252)) if sd == sd else float("nan"),
        "sharpe": float(ex.mean() / sd * np.sqrt(252)) if sd and sd > 0 else float("nan"),
        "sortino": float(ex.mean() / dd_sd * np.sqrt(252)) if dd_sd and dd_sd > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and cagr == cagr else float("nan"),
        "downside_deviation_ann": float(dd_sd * np.sqrt(252)) if dd_sd == dd_sd else float("nan"),
        "var_95_daily": var5,
        "cvar_95_daily": float(r[r <= var5].mean()) if n and (r <= var5).any() else float("nan"),
        "worst_day": float(r.min()) if n else float("nan"),
        "ulcer_index": float(np.sqrt(np.mean((100 * dd) ** 2))),
        "share_days_underwater": float((dd < 0).mean()),
    }
    out["annualized_turnover"] = float(np.mean(turnover) * 252.0) if turnover is not None else 0.0
    out["avg_gross_exposure"] = float(np.mean(exposure)) if exposure is not None else 1.0
    out["max_gross_exposure"] = float(np.max(exposure)) if exposure is not None else 1.0
    return out

def _ctx_args(ctx, n):
    return {
        "portfolio_drawdown": ctx["portfolio_drawdown"],
        "gross_exposure": ctx["gross_exposure"],
        "position_size": ctx["size_frac"],
        "recent_loss": ctx["recent_loss"]
    }

def model_decide(profile):
    prof = np.asarray(profile, dtype=np.float32)
    def fn(k, rows, pos, pnl, hold, ctx):
        seq = np.stack([panel.iloc[int(row) - L + 1:int(row) + 1][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                        for row in rows], axis=0)
        return policy.predict({
            "sequence": seq, "position": pos, "profile": prof,
            "unrealized_pnl": pnl, "holding_bars": hold,
            **_ctx_args(ctx, len(rows))
        })["action_probs"]
    return fn

def label_rule_decide(profile, allow_short):
    prof = np.asarray(profile, dtype=np.float32)
    def fn(k, rows, pos, pnl, hold, ctx):
        port = {"pdd": ctx["portfolio_drawdown"], "gross": ctx["gross_exposure"], "size": ctx["size_frac"], "rloss": ctx["recent_loss"]}
        y = conservative_rule_labels({c: v[rows] for c, v in RAW.items()}, pos, np.tile(prof, (len(rows), 1)), hold, pnl, port, allow_short)
        return np.eye(N_ACTIONS)[y]
    return fn

def rule_risk_controller_decide(k, rows, pos, pnl, hold, ctx):
    vp = 0.5 * (RAW["atr_pct_pctile_252"][rows] + RAW["rv20_pctile_252"][rows])
    y = np.zeros(len(rows), dtype=int)
    with np.errstate(invalid="ignore"):
        y[(pos == 0) & (RAW["sma50_sma200"][rows] > 0) & (RAW["close_sma200"][rows] > 0) & (vp < 0.8) & (RAW["dd_from_high_60"][rows] > -0.08)] = BUY
        y[(pos == 1) & ((RAW["sma50_sma200"][rows] < 0) | (vp >= 0.9) | (RAW["dd_from_high_60"][rows] < -0.12))] = EXIT
    return np.eye(N_ACTIONS)[y]

def always_long_decide(k, rows, pos, pnl, hold, ctx):
    y = np.zeros(len(rows), dtype=int)
    y[pos == 0] = BUY
    return np.eye(N_ACTIONS)[y]

LS = "long/short" if CONFIG["allow_short"] else "long-only"
MAIN = f"SP-05 policy + gate (balanced, {LS})"
LONG_ONLY = "SP-05 policy + gate (balanced, long-only)"
NOGATE = f"SP-05 policy ALONE, no gate (balanced, {LS})"
RULECTL = "Rule-based risk controller + gate (long-only)"
GATEONLY = "Always-long + gate (gate-only baseline)"
LABELRULE = f"Conservative label rule + gate (balanced, {LS})"

BT, BT_RESULTS, BT_TABLE = {}, {}, {}

for split in ("val", "test"):
    bt = prepare_backtest(split, top20_bluechips)
    BT[split] = bt
    runs = {}
    for pname, prof in PROFILES.items():
        runs[f"SP-05 policy + gate ({pname}, {LS})"] = run_portfolio_backtest(
            bt, model_decide(prof), CONFIG["allow_short"], GATE, CONFIG["confidence_sizing"]
        )
    if CONFIG["allow_short"]:
        runs[LONG_ONLY] = run_portfolio_backtest(
            bt, model_decide(PROFILES["balanced"]), False, GATE, CONFIG["confidence_sizing"]
        )
    runs[NOGATE] = run_portfolio_backtest(
        bt, model_decide(PROFILES["balanced"]), CONFIG["allow_short"], None, CONFIG["confidence_sizing"]
    )
    runs[RULECTL] = run_portfolio_backtest(bt, rule_risk_controller_decide, False, GATE, False)
    runs[GATEONLY] = run_portfolio_backtest(bt, always_long_decide, False, GATE, False)
    runs[LABELRULE] = run_portfolio_backtest(
        bt, label_rule_decide(PROFILES["balanced"], CONFIG["allow_short"]), CONFIG["allow_short"], GATE, False
    )
    
    dates = runs[MAIN]["dates"]
    b_ret = bt["bench_close"][1:] / bt["bench_close"][:-1] - 1.0
    ew_ret = (bt["close"][1:] / bt["close"][:-1] - 1.0).mean(axis=1)
    
    metrics = {n: perf_metrics(r["ret"], r["turnover"], r["exposure"]) for n, r in runs.items()}
    metrics[f"{CONFIG['benchmark']} buy & hold"] = perf_metrics(b_ret)
    metrics["Equal-weight universe buy & hold"] = perf_metrics(ew_ret)
    
    EXPO_MATCHED = f"{CONFIG['benchmark']} scaled to the main run's average exposure (rest in cash)"
    _me = float(np.mean(runs[MAIN]["exposure"]))
    metrics[EXPO_MATCHED] = perf_metrics(b_ret * _me, None, np.full(len(b_ret), _me))
    
    BT_RESULTS[split] = {"runs": runs, "bench_ret": b_ret, "ew_ret": ew_ret, "dates": dates}
    extra = pd.DataFrame({
        n: {
            "trade_entries": r["n_entries"],
            "closed_trades": r["n_closed_trades"],
            "skipped_entries": r["n_skipped_entries"],
            "win_rate": r["win_rate"],
            "avg_trade_return": r["avg_trade_return"],
            "open_at_end": r["open_positions_at_end"],
            "gate_overrides": int(sum(r["gate_override_days"])),
        } for n, r in runs.items()
    }).T
    BT_TABLE[split] = pd.DataFrame(metrics).T.join(extra)
    print(f"\n===== {split.upper()} Backtest ({dates[0].date()} to {dates[-1].date()} | {len(dates)} days | Capital INR {CAP0:,.0f}) =====")
    print(BT_TABLE[split][["total_return", "sharpe", "sortino", "max_drawdown", "cvar_95_daily", "worst_day", "annualized_turnover", "avg_gross_exposure", "trade_entries", "gate_overrides"]].round(3).to_string())

# ----------------------------------------------------------------------------
# 12. Gate Overrides Analysis & Stress Testing
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GATE OVERRIDES & STRESS TESTING ANALYSIS")
print("=" * 80)

gate_tab = pd.DataFrame({n: r["gate_reason_counts"] for n, r in BT_RESULTS["test"]["runs"].items() if n != NOGATE}).T
print("\nGate Overrides by Reason (TEST split):")
print(gate_tab.loc[:, (gate_tab != 0).any(axis=0)].to_string())

BLOCK_REASONS = {
    "risk_data_missing", "drawdown_halt", "recent_loss_halt", "volatility_prohibited",
    "abnormal_gap_or_move", "market_stress", "low_confidence_entry", "gross_exposure_limit", "short_not_allowed"
}

def blocked_entry_analysis(events):
    rows_ = []
    for e in events:
        if e["proposed"] in (BUY, SELL) and e["final"] == HOLD:
            f = FWD_RET[e["row"]]
            if np.isfinite(f):
                for rsn in (e["reasons"] or ["(other)"]):
                    if rsn in BLOCK_REASONS or rsn == "(other)":
                        rows_.append({"reason": rsn, "dir_adj_fwd_return": (1 if e["proposed"] == BUY else -1) * f})
    if not rows_:
        return pd.DataFrame()
    d = pd.DataFrame(rows_)
    return d.groupby("reason")["dir_adj_fwd_return"].agg(
        blocked_entries="size",
        mean_dir_adj_fwd_return="mean",
        share_that_would_have_lost=lambda s: (s < 0).mean()
    )

BLOCKED = {s: blocked_entry_analysis(BT_RESULTS[s]["runs"][MAIN]["gate_events"]) for s in ("val", "test")}
print(f"\nBlocked Entries Hindsight {CONFIG['eval_horizon']}-bar Forward Return Analysis:")
for s in ("val", "test"):
    print(f"[{s.upper()}] (Negative mean return = gate avoided a loss):")
    print(BLOCKED[s].round(4).to_string() if len(BLOCKED[s]) else "No blocked entries")

FORCED = {}
for s in ("val", "test"):
    tr = BT_RESULTS[s]["runs"][MAIN]["trades"]
    if len(tr):
        FORCED[s] = tr.groupby("exit_by").agg(
            trades=("ret", "size"),
            mean_return=("ret", "mean"),
            win_rate=("ret", lambda x: (x > 0).mean()),
            median_bars=("bars", "median")
        )
        print(f"\n[{s.upper()}] Closed Trades by Exit Authority:")
        print(FORCED[s].round(4).to_string())

# Stress tests
BENCH_STRESS = pd.DataFrame({
    "high_vol": bench_feat["bench_rv20_pctile_252"] >= 0.85,
    "severe_dd": bench_feat["bench_dd_60"] <= -0.10
})
BENCH_STRESS["stress"] = BENCH_STRESS["high_vol"] | BENCH_STRESS["severe_dd"]
STRESS_STRATS = [MAIN, NOGATE, RULECTL, GATEONLY, f"{CONFIG['benchmark']} buy & hold", "Equal-weight universe buy & hold"]

def collect(split):
    R = BT_RESULTS[split]
    flags = BENCH_STRESS.reindex(BT[split]["dates"][:-1]).fillna(False).astype(bool)
    series = {n: R["runs"][n]["ret"] for n in STRESS_STRATS if n in R["runs"]}
    series[f"{CONFIG['benchmark']} buy & hold"] = R["bench_ret"]
    series["Equal-weight universe buy & hold"] = R["ew_ret"]
    expo = {n: R["runs"][n]["exposure"] for n in STRESS_STRATS if n in R["runs"]}
    return series, expo, flags

def group_stats(r, e, m):
    if m.sum() < 3:
        return None
    rr = r[m]
    eq = np.cumprod(1 + rr)
    pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    v5 = np.percentile(rr, 5)
    return {
        "days": int(m.sum()),
        "compounded_return": float(eq[-1] - 1),
        "mean_daily_pct": float(100 * rr.mean()),
        "ann_vol": float(rr.std(ddof=1) * np.sqrt(252)),
        "worst_day": float(rr.min()),
        "cvar5_daily": float(rr[rr <= v5].mean()),
        "max_dd_within_group": float((eq / pk - 1).min()),
        "avg_gross_exposure": float(np.mean(e[m])) if e is not None else 1.0
    }

STRESS = {}
for scope in ("val", "test", "val+test"):
    parts = [collect(s) for s in (("val", "test") if scope == "val+test" else (scope,))]
    rows_ = []
    for name in STRESS_STRATS:
        r = np.concatenate([p[0][name] for p in parts if name in p[0]])
        e = np.concatenate([p[1][name] for p in parts]) if name in parts[0][1] else None
        fl = pd.concat([p[2] for p in parts])
        for gname, m in (("normal", ~fl["stress"].to_numpy()), ("high_vol", fl["high_vol"].to_numpy()), ("severe_drawdown", fl["severe_dd"].to_numpy()), ("any stress", fl["stress"].to_numpy())):
            st = group_stats(r, e, m)
            if st:
                rows_.append({"strategy": name, "period": gname, **st})
    STRESS[scope] = pd.DataFrame(rows_)

print("\nCompounded Return by Stress Period (val+test):")
t_stress = STRESS["val+test"]
if len(t_stress):
    print(t_stress.pivot(index="strategy", columns="period", values="compounded_return").reindex(STRESS_STRATS).round(3).to_string())

# Sharp down days on NIFTY 50
rets_all, spy_all = {n: [] for n in STRESS_STRATS if n in BT_RESULTS["test"]["runs"]}, []
for s in ("val", "test"):
    R = BT_RESULTS[s]
    spy_all.append(R["bench_ret"])
    for n in rets_all:
        rets_all[n].append(R["runs"][n]["ret"])
spy_all = np.concatenate(spy_all)
DOWN = {}
for thr in (-0.015, -0.02, -0.03):
    m = spy_all <= thr
    if m.sum() >= 3:
        DOWN[f"NIFTY 50 <= {thr:.1%}"] = {
            "days": int(m.sum()),
            "NIFTY mean return": float(spy_all[m].mean()),
            **{n: float(np.concatenate(v)[m].mean()) for n, v in rets_all.items()}
        }
DOWN_T = pd.DataFrame(DOWN).T
print("\nDownside Capture on Sharp Market Down Days:")
print(DOWN_T.round(4).to_string())

# Gap-loss budgets
GAP = {}
for n in (MAIN, NOGATE, RULECTL, GATEONLY):
    e = np.concatenate([BT_RESULTS[s]["runs"][n]["exposure"] for s in ("val", "test")])
    GAP[n] = {
        "mean_exposure": float(e.mean()),
        "p95_exposure": float(np.percentile(e, 95)),
        "max_exposure": float(e.max()),
        **{f"loss_at_max_exposure_gap_{int(g*100)}%": float(-g * e.max()) for g in (0.05, 0.10, 0.20)}
    }
GAP_T = pd.DataFrame(GAP).T
print("\nGap-Loss Budget on Open Gross Exposure:")
print(GAP_T.round(3).to_string())

# Failure Analysis
def drawdown_episodes(eq, dates, top=3):
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1.0
    eps, in_dd = [], False
    for t in range(len(eq)):
        if dd[t] < 0 and not in_dd:
            in_dd, start, trough = True, max(t - 1, 0), t
        if in_dd:
            if dd[t] < dd[trough]:
                trough = t
            if dd[t] >= 0:
                eps.append((start, trough, t))
                in_dd = False
    if in_dd:
        eps.append((start, trough, None))
    rows_ = []
    for s, tr_, e in sorted(eps, key=lambda x: dd[x[1]])[:top]:
        rows_.append({
            "peak_date": dates[s].date(),
            "trough_date": dates[tr_].date(),
            "depth": float(dd[tr_]),
            "days_peak_to_trough": int(tr_ - s),
            "recovered_on": dates[e].date() if e is not None else "NOT RECOVERED",
            "days_underwater": int((e if e is not None else len(eq) - 1) - s)
        })
    return pd.DataFrame(rows_)

R_test = BT_RESULTS["test"]
dts_test = R_test["dates"]
main_test = R_test["runs"][MAIN]
named_test = {
    MAIN: main_test["ret"],
    NOGATE: R_test["runs"][NOGATE]["ret"],
    RULECTL: R_test["runs"][RULECTL]["ret"],
    f"{CONFIG['benchmark']} buy & hold": R_test["bench_ret"]
}

FAIL = {
    "drawdown_episodes": {name: drawdown_episodes(np.cumprod(1 + r), dts_test) for name, r in named_test.items()},
    "calendar_years": pd.DataFrame({n: pd.Series(r, index=dts_test).groupby(dts_test.year).apply(lambda x: np.prod(1 + x) - 1) for n, r in named_test.items()}),
    "worst_days": None,
    "worst_21d": {name: float(pd.Series(r).rolling(21).apply(lambda x: np.prod(1 + x) - 1, raw=True).dropna().min()) for name, r in named_test.items()},
    "worst_trades": None,
    "forced_exits": FORCED.get("test")
}

wd = pd.DataFrame({
    "date": dts_test,
    "main_return": main_test["ret"],
    "bench_return": R_test["bench_ret"],
    "gross_exposure": main_test["exposure"],
    "portfolio_drawdown": main_test["portfolio_drawdown"],
    "gate_overrides_that_day": main_test["gate_override_days"]
})
FAIL["worst_days"] = wd.nsmallest(5, "main_return")
tr_main = main_test["trades"]
if len(tr_main):
    FAIL["worst_trades"] = tr_main.nsmallest(5, "ret")[["symbol", "side", "decision_date", "exit_date", "bars", "ret", "exit_by"]]

print("\nWorst Drawdown Episodes (Main Run):")
print(FAIL["drawdown_episodes"][MAIN].to_string(index=False))
print("\nWorst 5 Trading Days (Main Run):")
print(FAIL["worst_days"].round(4).to_string(index=False))

# ----------------------------------------------------------------------------
# 13. Generate and Save Standard Visualizations
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING EVALUATION PLOTS")
print("=" * 80)

# Plot 1: Learning curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="royalblue", lw=1.5)
ax[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="crimson", lw=1.5)
ax[0].axvline(BEST_EPOCH, color="grey", ls="--", lw=1, label=f"Best Epoch ({BEST_EPOCH})")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Weighted Cross-Entropy")
ax[0].set_title("SP-05 Learning Curves")
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(history_df["epoch"], history_df["val_macro_f1"], color="forestgreen", lw=1.5)
ax[1].axvline(BEST_EPOCH, color="grey", ls="--", lw=1, label=f"Best Epoch ({BEST_EPOCH})")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Macro-F1")
ax[1].set_title("Validation Macro-F1 Progression")
ax[1].legend()
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "learning_curves.png", dpi=140)
plt.close()

# Plot 2: Confusion matrices (Counts and Row-normalized)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cm = np.array(CLS["test"]["confusion_matrix"])
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
for a, mat, title, fmt in ((ax[0], cm, "Test Confusion Matrix (Counts)", "d"),
                           (ax[1], cmn, "Test Confusion Matrix (Row-Normalized)", ".2f")):
    im = a.imshow(mat, cmap="Blues")
    a.set_title(title, fontweight="bold")
    a.set_xticks(range(N_ACTIONS))
    a.set_yticks(range(N_ACTIONS))
    a.set_xticklabels(ACTION_NAMES)
    a.set_yticklabels(ACTION_NAMES)
    a.set_xlabel("Predicted Action")
    a.set_ylabel("True Proxy Label")
    for i in range(N_ACTIONS):
        for j in range(N_ACTIONS):
            a.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                   color="white" if mat[i, j] > mat.max() / 2 else "black")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_test.png", dpi=140)
plt.close()

# Plot 3: Reliability diagram
fig, ax = plt.subplots(figsize=(5.5, 4.8))
r_bins = pd.DataFrame(CLS["test"]["reliability_bins"])
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
ax.plot(r_bins["confidence"], r_bins["accuracy"], "o-", color="darkorange", label=f"Calibrated (T={TEMPERATURE:.2f}, ECE={CLS['test']['ece']:.3f})")
ax.set_xlabel("Mean Confidence")
ax.set_ylabel("Empirical Accuracy vs Proxy Labels")
ax.set_title("Reliability Diagram (SP-05 Calibrated Test)", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "reliability_test.png", dpi=140)
plt.close()

# Plot 4: Backtest Performance (4-panel)
d = R_test["dates"]
fl_plot = BENCH_STRESS.reindex(BT["test"]["dates"][:-1]).fillna(False).astype(bool)["stress"].to_numpy()
fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.4, 1.2]})
for a in ax:
    start_idx = None
    for i, v in enumerate(list(fl_plot) + [False]):
        if v and start_idx is None:
            start_idx = i
        if not v and start_idx is not None:
            a.axvspan(d[start_idx], d[min(i, len(d) - 1)], color="red", alpha=0.10, lw=0)
            start_idx = None

ax[0].plot(d, main_test["equity"], label=MAIN, lw=2.2, color="navy")
ax[0].plot(d, R_test["runs"][NOGATE]["equity"], label=NOGATE, lw=1.2, color="gray", ls="--")
ax[0].plot(d, R_test["runs"][RULECTL]["equity"], label=RULECTL, lw=1.2, color="teal")
ax[0].plot(d, R_test["runs"][GATEONLY]["equity"], label=GATEONLY, lw=1.0, color="purple")
ax[0].plot(d, np.cumprod(1 + R_test["bench_ret"]), label=f"{CONFIG['benchmark']} Buy & Hold", lw=1.4, color="crimson")
ax[0].plot(d, np.cumprod(1 + R_test["ew_ret"]), label="Equal-Weight Buy & Hold", lw=1.2, color="orange")
ax[0].set_ylabel("Growth of INR 1.0", fontweight="bold")
ax[0].set_title("Out-Of-Sample Test Backtest: Equity After Fees & Slippage (Shaded Red = Market Stress)", fontweight="bold")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].grid(True, alpha=0.25)

for name, eq, col in ((MAIN, main_test["equity"], "navy"),
                      (NOGATE, R_test["runs"][NOGATE]["equity"], "gray"),
                      (CONFIG["benchmark"], np.cumprod(1 + R_test["bench_ret"]), "crimson")):
    ax[1].plot(d, eq / np.maximum.accumulate(eq) - 1, label=name, color=col, lw=1.2)
ax[1].axhline(-RISK_LIMITS.halt_drawdown, color="orange", ls="--", lw=1, label=f"Gate Halt ({RISK_LIMITS.halt_drawdown:.0%})")
ax[1].axhline(-RISK_LIMITS.flatten_drawdown, color="red", ls="--", lw=1, label=f"Gate Flatten ({RISK_LIMITS.flatten_drawdown:.0%})")
ax[1].set_ylabel("Drawdown", fontweight="bold")
ax[1].legend(fontsize=7, loc="lower left")
ax[1].grid(True, alpha=0.25)

ax[2].plot(d, main_test["exposure"], color="tab:purple", label="Policy + Gate Exposure", lw=1.4)
ax[2].plot(d, R_test["runs"][NOGATE]["exposure"], color="tab:gray", lw=0.9, ls=":", label="Policy Alone Exposure")
ax[2].axhline(RISK_LIMITS.max_gross_exposure, color="red", ls="--", lw=1, label=f"Gate Gross Cap ({RISK_LIMITS.max_gross_exposure:.0%})")
ax[2].set_ylabel("Gross Exposure", fontweight="bold")
ax[2].legend(fontsize=7, loc="upper right")
ax[2].grid(True, alpha=0.25)

ax[3].bar(d, main_test["gate_override_days"], color="tab:red", width=1.5)
ax[3].set_ylabel("Gate Overrides/Day", fontweight="bold")
ax[3].grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "backtest_test.png", dpi=140)
plt.close()

# Plot 5: Gate and Stress Bar Charts
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
gt = gate_tab.loc[[n for n in gate_tab.index if "policy + gate" in n or "Rule-based" in n or "Always" in n or "label rule" in n], :]
gt = gt.loc[:, (gt != 0).any(axis=0)]
if gt.shape[1]:
    gt.T.plot(kind="barh", ax=ax[0])
    ax[0].set_title("Test Gate Overrides by Reason", fontweight="bold")
    ax[0].set_xlabel("Override Count")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, alpha=0.25)

st_plot = STRESS["val+test"]
if len(st_plot):
    pv = st_plot.pivot(index="period", columns="strategy", values="compounded_return")[[c for c in STRESS_STRATS if c in st_plot["strategy"].values]]
    pv.plot(kind="bar", ax=ax[1])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Compounded Return by Benchmark Stress State", fontweight="bold")
    ax[1].set_ylabel("Compounded Return")
    ax[1].legend(fontsize=7)
    ax[1].grid(True, alpha=0.25)
    plt.setp(ax[1].get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "gate_and_stress.png", dpi=140)
plt.close()

# Plot 6: Label distribution
fig, ax = plt.subplots(figsize=(7, 3.8))
dist_pct.plot(kind="bar", ax=ax, colormap="tab10")
ax.set_ylabel("% of Samples")
ax.set_title("SP-05 Action Label Distribution by Split", fontweight="bold")
ax.grid(True, alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "label_distribution.png", dpi=140)
plt.close()

# Plot 7: Regime Comparison Plot
fig, ax = plt.subplots(figsize=(9, 4.2))
df_regimes[["accuracy", "macro_f1"]].plot(kind="bar", ax=ax, color=["royalblue", "forestgreen"])
ax.set_ylabel("Score")
ax.set_title("SP-05 Performance Across Market Regimes (Past-Only Defined)", fontweight="bold")
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.25)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "regime_comparison.png", dpi=140)
plt.close()

# Plot 8: Profile Sensitivity Plot
fig, ax = plt.subplots(figsize=(8, 4))
df_prof = pd.DataFrame(profile_eval).T[["hold_prop_pct", "buy_prop_pct", "sell_prop_pct", "exit_prop_pct"]]
df_prof.plot(kind="bar", stacked=True, ax=ax, color=["lightgray", "forestgreen", "crimson", "gold"])
ax.set_ylabel("Proposal Share (%)")
ax.set_title("Action Proposal Share by Trader Profile", fontweight="bold")
ax.legend(["HOLD", "BUY", "SELL", "EXIT"], loc="upper right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "profile_sensitivity.png", dpi=140)
plt.close()

# Plot 9: Cohort Comparison (Main vs Unseen Holdout vs Bluechips)
fig, ax = plt.subplots(figsize=(7, 4))
cohort_comp = pd.DataFrame({
    "Main Universe": [CLS["test"]["accuracy"], CLS["test"]["macro_f1"], CLS["test"]["ece"]],
    "Unseen Holdout": [unseen_cls["accuracy"], unseen_cls["macro_f1"], unseen_cls["ece"]],
    "Top-20 Bluechips": [bluechip_cls["accuracy"], bluechip_cls["macro_f1"], bluechip_cls["ece"]]
}, index=["Accuracy", "Macro-F1", "ECE"]).T
cohort_comp.plot(kind="bar", ax=ax, color=["royalblue", "forestgreen", "darkorange"])
ax.set_ylabel("Metric Value")
ax.set_title("Cohort Generalization Comparison (Test Split)", fontweight="bold")
ax.grid(True, alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "cohort_comparison.png", dpi=140)
plt.close()

print("All 9 evaluation figures generated and saved to outputs/figures!")

# ----------------------------------------------------------------------------
# 14. Standardized Artifacts Persistence
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING STANDARDIZED ARTIFACTS")
print("=" * 80)

# 1. Authoritative model checkpoint
policy.save(OUT_DIR) # saves conservative_policy.pt, conservative_config.json, feature_schema.json, preprocessing.pkl

# Copy python modules
for fname in ("sp05_features.py", "sp05_risk.py", "sp05_model.py"):
    shutil.copy(fname, OUT_DIR / fname)

# Save equity curves CSV
curves = []
for split in ("val", "test"):
    R_ = BT_RESULTS[split]
    m_ = R_["runs"][MAIN]
    fl_ = BENCH_STRESS.reindex(BT[split]["dates"][:-1]).fillna(False).astype(bool)
    curves.append(pd.DataFrame({
        "date": R_["dates"],
        "split": split,
        "strategy_return": m_["ret"],
        "strategy_equity": m_["equity"],
        "no_gate_equity": R_["runs"][NOGATE]["equity"],
        "rule_controller_equity": R_["runs"][RULECTL]["equity"],
        "gate_only_equity": R_["runs"][GATEONLY]["equity"],
        "benchmark_return": R_["bench_ret"],
        "benchmark_equity": np.cumprod(1 + R_["bench_ret"]),
        "equal_weight_bh_equity": np.cumprod(1 + R_["ew_ret"]),
        "gross_exposure": m_["exposure"],
        "portfolio_drawdown": m_["portfolio_drawdown"],
        "turnover": m_["turnover"],
        "gate_overrides": m_["gate_override_days"],
        "benchmark_stress_prev_close": fl_["stress"].to_numpy()
    }))
pd.concat(curves, ignore_index=True).to_csv(OUT_DIR / "equity_curve.csv", index=False)

def to_py(o):
    if isinstance(o, pd.DataFrame):
        return to_py(o.reset_index().to_dict(orient="records"))
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)) or hasattr(o, "isoformat"):
        return str(o)
    return o

def strip_pred(c):
    return {k: v for k, v in c.items() if k not in ("pred", "conf")}

evaluation_data = {
    "model": "SP-05 Conservative Risk-Aware Policy",
    "market": CONFIG["market"],
    "benchmark": CONFIG["benchmark"],
    "label_source": LABEL_SOURCE,
    "label_warning": policy_config["label_note"],
    "authority": policy_config["authority"],
    "confidence_limitations": [
        "Calibrated against proxy labels on the validation period, not against loss avoidance.",
        "Not calibrated for rare high-stress states.",
        "Not comparable across policies.",
        "The hard risk gate does not depend on confidence for safety decisions."
    ],
    "objective": CONFIG["objective"],
    "soft_limits": CONFIG["soft_limits"],
    "hard_risk_limits": RISK_LIMITS.to_dict(),
    "n_samples": {k: int(len(v["y"])) for k, v in SAMPLES.items()},
    "label_distribution": dist.to_dict(),
    "class_weights": dict(zip(ACTION_NAMES, CLASS_WEIGHTS.tolist())),
    "classification": {
        "validation": strip_pred(CLS["val"]),
        "test": strip_pred(CLS["test"]),
        "test_uncalibrated": strip_pred(CLS_UNCAL_TEST),
        "unseen_holdout_cohort": strip_pred(unseen_cls),
        "top20_bluechips_cohort": strip_pred(bluechip_cls)
    },
    "classification_by_risk_condition": {k: v.to_dict(orient="index") for k, v in COND_CLS.items()},
    "entry_proposals_in_forbidden_states": VIOLATIONS,
    "generalization_experiments": {
        "experiment_a_main_test": strip_pred(CLS["test"]),
        "experiment_b_unseen_holdout": strip_pred(unseen_cls),
        "experiment_c_top20_bluechips": strip_pred(bluechip_cls),
        "experiment_d_profile_sensitivity": profile_eval,
        "experiment_e_market_regimes": df_regimes.to_dict(orient="index")
    },
    "backtest": {s: BT_TABLE[s].to_dict(orient="index") for s in ("val", "test")},
    "gate_overrides_by_reason_test": gate_tab.to_dict(orient="index"),
    "blocked_entries_hindsight": {k: v.to_dict(orient="index") if len(v) else {} for k, v in BLOCKED.items()},
    "trades_by_exit_authority": {k: v.to_dict(orient="index") if len(v) else {} for k, v in FORCED.items()},
    "stress_tests": {k: v.to_dict(orient="records") for k, v in STRESS.items()},
    "market_down_days": DOWN_T.to_dict(orient="index"),
    "gap_loss_budget": GAP_T.to_dict(orient="index"),
    "failure_analysis_test": {
        "drawdown_episodes": {name: df_ep.to_dict(orient="records") for name, df_ep in FAIL["drawdown_episodes"].items()},
        "calendar_years": FAIL["calendar_years"].to_dict(orient="index"),
        "worst_days": FAIL["worst_days"].to_dict(orient="records"),
        "worst_21d": FAIL["worst_21d"],
        "worst_trades": FAIL["worst_trades"].to_dict(orient="records") if FAIL["worst_trades"] is not None else [],
        "forced_exits": FAIL["forced_exits"].to_dict(orient="index") if FAIL["forced_exits"] is not None else {}
    },
    "training_history": history_df.to_dict(orient="records"),
    "best_epoch": BEST_EPOCH,
}

(OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(to_py(evaluation_data), indent=2))
print("Saved evaluation_metrics.json")

# Deterministic reload check and gate limits round-trip
print("\nVerifying reload check and round-trip assertions...", flush=True)
reloaded = ConservativePolicy.load(OUT_DIR, DEVICE)
test_rows_sample = test_rows_all[:128]
win_sample = np.asarray([[panel.loc[r + off, FEATURE_COLUMNS].to_numpy(dtype=np.float32) for off in OFFSETS_NP]
                         for r in test_rows_sample], dtype=np.float32)
state_sample = {"sequence": win_sample, "position": 0, "profile": PROFILES["balanced"]}

pred_orig = policy.predict(state_sample)
pred_reloaded = reloaded.predict(state_sample)

assert np.allclose(pred_orig["action_probs"], pred_reloaded["action_probs"], atol=1e-5), "Reload predictions mismatch!"
assert (pred_orig["action"] == pred_reloaded["action"]).all(), "Reload actions mismatch!"
assert RiskController.from_policy_config(reloaded.config).limits.to_dict() == RISK_LIMITS.to_dict(), "Risk limits round-trip failed!"
print("Reload check and round-trip assertions PASSED!")

# Verify all required artifacts exist
REQUIRED_ARTIFACTS = [
    "conservative_policy.pt",
    "best_checkpoint.pt",
    "final_checkpoint.pt",
    "conservative_config.json",
    "feature_schema.json",
    "preprocessing.pkl",
    "evaluation_metrics.json",
    "training_history.csv",
    "equity_curve.csv",
    "sp05_features.py",
    "sp05_risk.py",
    "sp05_model.py",
]

print("\n[Artifact Verification]")
for fname in REQUIRED_ARTIFACTS:
    fpath = OUT_DIR / fname
    assert fpath.exists(), f"Missing artifact: {fname}"
    size_kb = fpath.stat().st_size / 1024
    print(f"  [OK] {fname:<30s} ({size_kb:8.1f} KiB)")

print("\n" + "=" * 80)
print("TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80, flush=True)

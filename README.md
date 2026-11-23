# SP-05 Conservative Risk-Aware Trading Policy

SP-05 is a PyTorch policy that proposes `HOLD`, `BUY`, `SELL`, or `EXIT` actions. The separate deterministic `RiskController` has final authority to reject entries or force exits. It is a research artifact, not a live-trading system or a promise of returns.

## Load the trained policy

The authoritative trained bundle is `outputs/sp05_conservative_policy/indian_market_run_001/conservative_policy.pt`. It contains model weights, architecture/configuration, fitted preprocessing, feature schema, action mapping, and calibrated temperature. The sidecar config/schema/preprocessing files are also included for compatibility and inspection.

Install runtime dependencies with `python -m pip install -r requirements.txt`, then provide local OHLCV history and a local NIFTY 50 benchmark. This repository intentionally excludes all raw market data and locally generated feature caches.

```python
import pandas as pd
from sp05_features import FEATURE_COLUMNS, build_features
from sp05_model import ConservativePolicy
from sp05_risk import RiskController

run_dir = "outputs/sp05_conservative_policy/indian_market_run_001"
policy = ConservativePolicy.load(run_dir, device="cpu")
risk_controller = RiskController.from_policy_config(policy.config)

# `ohlcv` is indexed by date and has Open, High, Low, Close, Volume columns.
# `nifty50` is indexed by date and has a Close column. Provide adjusted OHLC
# consistently with the price history used to compute returns.
features = build_features(ohlcv, nifty50)
i = len(features) - 1  # decision after the latest completed close
seq_len = policy.seq_len
state = {
    "sequence": features.iloc[i-seq_len+1:i+1][FEATURE_COLUMNS].to_numpy(),
    "position": 0,                       # -1 short, 0 flat, +1 long
    "profile": (0.5, 0.5, 0.5),          # balanced profile
    "unrealized_pnl": 0.0,               # position-signed fractional P&L
    "holding_bars": 0,
    "portfolio_drawdown": 0.0,           # actual portfolio drawdown fraction
    "gross_exposure": 0.0,               # actual gross exposure / equity
    "position_size": 0.0,                # this position notional / equity
    "recent_loss": 0.0,                  # actual recent portfolio loss fraction
}
proposal = policy.predict(state)
decision = risk_controller.review(proposal, state, FEATURE_COLUMNS)
print(proposal["action_name"], proposal["confidence"], decision)
```

Use actual portfolio state in production integration. `decision` is the controller-approved action; the model proposal alone must never be executed. The configured convention is decision at close *t* and execution at open *t+1*. Risk limits and short-sale restrictions are in the saved policy config.

## Artifacts

- `sp05_features.py`, `sp05_model.py`, `sp05_risk.py`: feature contract, inference model, deterministic risk gate.
- `outputs/sp05_conservative_policy/indian_market_run_001/`: trained policy, checkpoints, preprocessing, schema, evaluation metrics, history, equity curve, execution log, reports, and plots.
- `05_conservative_policy.ipynb` and `run_sp05_training.py`: local-data training and evaluation workflow. To rerun training, provide the approved local market dataset at the configured path; it is deliberately not committed.
- `outputs/sp05_conservative_policy/bse_ns_hard_test_20261123/`: separate single-stock temporal-test results. These do not represent an unseen-stock test or live performance.

## Reproduce training

The training workflow expects `Data/Raw/indian-market-historical-ohlcv.csv` plus the approved cohort manifests under `scratch/`. Neither market data nor derived caches are included in this repository. The notebook and runner use the local NIFTY 50 benchmark from that dataset and do not fetch market prices online.

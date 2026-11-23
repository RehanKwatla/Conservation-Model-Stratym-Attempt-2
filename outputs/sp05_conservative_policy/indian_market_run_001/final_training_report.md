# SP-05 Conservative Risk-Aware Policy — Final Training Report

Run directory: `outputs/sp05_conservative_policy/indian_market_run_001/`  
Data source: local Indian-market OHLCV dataset only; no replacement or external price feed. Local benchmark: NIFTY 50.  
Generated from the final completed run on 2026-11-23.

## Executive summary

SP-05 was trained from fresh seed-42 initialization with the notebook's two-layer GRU architecture. Early stopping selected **epoch 25** with best class-weighted validation loss 0.0534 (validation calibrated classification NLL 0.0419) after **32 epochs**. It used a 360,000 example deterministic proportional fit subset from 10,366,761 generated training proxy samples across 1,795 stocks. The best checkpoint was restored before validation-only temperature calibration and final evaluation.

On the untouched chronological test classification set, calibrated accuracy was 98.15%, balanced accuracy 91.97%, macro-F1 0.820, NLL 0.0459, and ECE 0.0009. The balanced long/short SP-05 + gate backtest returned -5.52% over 495 days, with maximum drawdown -8.35%, Sharpe -1.58, and average gross exposure 25.39%. This was a negative result; classification quality is not evidence of profitability or capital protection.

## Data, cohort, and chronology

- Main eligible cohort: 1,795 symbols. Separate unseen-stock cohort: 100. Dedicated top-20 bluechip cohort: 20. Exact assigned symbols are in `reports/eligible_symbols.csv`; excluded candidates and reasons are in `reports/universe_exclusions.csv`.
- Local NIFTY 50 benchmark: 4,684 observations, 2007-09-17 through 2026-10-23.
- Minimum clean history: 800 observations; eligibility and feature cleaning use the local dataset. Historical delisted stocks remain if eligible. No bars/prices were fabricated.
- Chronological common split boundaries: train through 2021-09-03; validation 2021-09-04 through 2022-10-24; test after 2022-10-24. Strict 30-bar sequence-window isolation is enabled. Per-stock decision date ranges exclude boundary windows and feature warmup.

| Split | First decision | Last decision | Complete bars | Proxy samples | % assigned bars | % all panel bars |
|---|---|---|---|---|---|---|
| train | 2008-12-04 | 2021-09-03 | 3,455,587 | 10,366,761 | 63.022% | 61.452% |
| val | 2021-10-19 | 2022-10-24 | 382,123 | 1,146,369 | 6.969% | 6.795% |
| test | 2022-12-07 | 2026-10-23 | 1,645,424 | 4,936,272 | 30.009% | 29.261% |

Percentages are based on complete, assigned decision bars; the all-panel column separately includes warmup and boundary-isolated bars. Exact panel totals and counts are in `reports/chronological_split.json`. Per-stock, position-state, action, and split label counts are in `reports/sample_counts_per_stock_state_action.csv`. Training weights were fitted on all training labels; fitting itself used the documented deterministic subset. Validation and test classification use the complete generated samples.

## Model, objective, and safeguards

- GRU(43,64), two layers, dropout 0.25; static Linear(12,16)+GELU; 80-wide combined representation; Linear(80,64)+GELU+Dropout(0.25)+Linear(64,4). 51,540 parameters. Sequence length 30; action map HOLD/BUY/SELL/EXIT.
- Labels are **strategy-derived decision-time proxy labels**, not human demonstrations and not future-outcome-confirmed labels. Confidence measures agreement with this proxy objective, not probability of avoiding a loss.
- The network proposes actions only. Deterministic `RiskController` has execution authority. Configured limits include 10% max position, 60% max gross, 5% position stop, 120-bar maximum hold, 8% drawdown halt, 15% flatten, 6% recent-loss halt, and 90th percentile volatility ceiling.
- Adversarial checks passed: a high-confidence BUY during drawdown halt was blocked; a hard stop forced EXIT at zero confidence. EXIT remains independent of confidence. These checks are in `reports/risk_controller_adversarial_checks.json`.
- Decision at close *t*, fill at open *t+1*, 5 bps fee plus 5 bps slippage; short borrowing cost follows the configured backtest convention. A deterministic stop does not prevent gap losses beyond the configured stop.

## Classification and calibration

Validation-only fitted temperature: **1.2499**. Test rows are 4,936,272; uncalibrated and calibrated values below come from separate saved test metrics. Per-action counts and scores:

| Action | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| HOLD | 2,126,502 | 0.978 | 0.980 | 0.979 |
| BUY | 37,607 | 0.658 | 0.985 | 0.789 |
| SELL | 1,461 | 0.410 | 0.731 | 0.525 |
| EXIT | 2,770,702 | 0.992 | 0.983 | 0.987 |

| Test metric | Calibrated | Uncalibrated |
|---|---:|---:|
| Accuracy | 98.15% | 98.15% |
| Balanced accuracy | 91.97% | 91.97% |
| Macro-F1 | 0.8200 | 0.8200 |
| Weighted-F1 | 0.9819 | 0.9819 |
| NLL | 0.04592 | 0.04730 |
| ECE | 0.00089 | 0.00363 |

Confusion matrix (rows actual, columns predicted):

| Actual / Predicted | HOLD | BUY | SELL | EXIT |
|---|---|---|---|---|
| HOLD | 2,083,013 | 19,258 | 1,538 | 22,693 |
| BUY | 561 | 37,046 | 0 | 0 |
| SELL | 393 | 0 | 1,068 | 0 |
| EXIT | 46,958 | 0 | 0 | 2,723,744 |

Validation metrics, per-position-state metrics, full per-action statistics, reliability bins, and probability confidence distributions are in `evaluation_metrics.json`. Reliability diagram: `figures/reliability_test.png`; confusion matrix plot: `figures/confusion_matrix_test.png`.

## Generalization and regime experiments

- Unseen-stock cohort (100 stocks; held out of scaler, class weights, fitting, calibration, and model selection): 271,134 test samples; accuracy 86.42%, balanced accuracy 46.02%, macro-F1 0.457, ECE 0.1201. Performance is materially weaker than main test and SELL support is sparse.
- Top-20 bluechip cohort: 56,229 test samples across 20 symbols; accuracy 97.61%, balanced accuracy 84.70%, macro-F1 0.769, ECE 0.0018.
- Profile sensitivity, regimes, and risk-condition classification are fully recorded under `generalization_experiments`, `classification_by_risk_condition`, and `entry_proposals_in_forbidden_states` in `evaluation_metrics.json`; plotted in `figures/profile_sensitivity.png`, `figures/regime_comparison.png`, and `figures/gate_and_stress.png`.

## Portfolio backtest results

Backtesting is on the common 20-stock liquid bluechip basket with available aligned bars, not on all 1,795 equities. The final test backtest spans 495 sessions (2024-10-18 to 2026-10-23); the earlier full test classification window begins 2022-12-07. Therefore the portfolio results and classifier metrics have different test coverage. `Equal-weight universe buy & hold` in the implementation is the equal-weight common bluechip basket used by this backtest, not all 1,795 stocks. Long/short results are theoretical; Indian cash-equity shorting is restricted and these figures are not a SEBI-compliant cash-equity execution claim.

| Strategy | Return | CAGR | Max DD | Sharpe | Sortino | Ann. vol. | Avg gross | Turnover | Entries | Closed | Overrides |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SP-05 + gate, balanced L/S | -5.52% | -2.85% | -8.35% | -1.58 | -2.08 | 5.54% | 25.39% | 9.44x | 122 | 120 | 90 |
| SP-05 + gate, conservative L/S | -8.21% | -4.27% | -9.65% | -2.54 | -3.08 | 4.04% | 16.21% | 8.01x | 114 | 114 | 31 |
| SP-05 + gate, aggressive L/S | 0.15% | 0.08% | -8.87% | -0.93 | -1.33 | 6.18% | 25.16% | 11.22x | 157 | 152 | 251 |
| SP-05 + gate, balanced long-only | -5.54% | -2.86% | -8.12% | -1.57 | -2.08 | 5.56% | 25.10% | 9.35x | 120 | 119 | 90 |
| SP-05 without gate, balanced L/S | -7.56% | -3.92% | -9.86% | -1.79 | -2.30 | 5.51% | 25.31% | 12.31x | 164 | 163 | 0 |
| Rule controller + gate, long-only | 4.95% | 2.49% | -9.85% | -0.39 | -0.53 | 8.15% | 42.16% | 5.51x | 56 | 52 | 1,446 |
| Always long + gate | -2.70% | -1.38% | -11.22% | -0.83 | -1.15 | 8.50% | 50.80% | 4.88x | 50 | 46 | 7,560 |
| Conservative label rule + gate | -5.08% | -2.62% | -8.29% | -1.52 | -1.99 | 5.58% | 22.41% | 5.29x | 52 | 52 | 35 |
| NIFTY 50 buy & hold | -8.69% | -4.52% | -15.56% | -0.73 | -1.01 | 13.28% | 100.00% | 0.00x | — | — | — |
| Equal-weight common bluechip basket* | 6.77% | 3.39% | -16.96% | -0.07 | -0.09 | 17.43% | 100.00% | 0.00x | — | — | — |
| Exposure-matched NIFTY 50 | -1.96% | -1.00% | -4.05% | -2.06 | -2.67 | 3.37% | 25.39% | 0.00x | — | — | — |

*The common-basket equal-weight reference is labelled accurately above. Detailed reports include VaR/CVaR, worst day, ulcer index, underwater share, maximum exposure, trade win rate/mean return, and validation-period values in `evaluation_metrics.json`; plotted equity comparison is `figures/backtest_test.png`. Fees, slippage, and short borrowing costs are applied to fills, but this pipeline does not separately itemize cumulative charges in the final metrics JSON.*

## Risk-controller, stress, and failure analysis

The balanced long/short test strategy had 90 gate overrides; total overrides by strategy and reason are in the table and `reports/gate_overrides_test.json`.

| Strategy | Stop loss | Max hold | DD halt | Recent loss | Volatility | Abnormal move/gap | Market stress | Gross exposure |
|---|---|---|---|---|---|---|---|---|
| SP-05 policy + gate (balanced, long/short) | 11 | 0 | 0 | 0 | 20 | 0 | 0 | 70 |
| SP-05 policy + gate (conservative, long/short) | 8 | 0 | 0 | 0 | 1 | 0 | 0 | 30 |
| SP-05 policy + gate (aggressive, long/short) | 8 | 0 | 0 | 0 | 26 | 0 | 0 | 225 |
| SP-05 policy + gate (balanced, long-only) | 11 | 0 | 0 | 0 | 21 | 0 | 0 | 69 |
| Rule-based risk controller + gate (long-only) | 26 | 3 | 91 | 2 | 72 | 8 | 1 | 1,260 |
| Always-long + gate (gate-only baseline) | 36 | 10 | 1,054 | 212 | 1,190 | 44 | 164 | 5,541 |
| Conservative label rule + gate (balanced, long/short) | 4 | 0 | 0 | 0 | 14 | 0 | 0 | 21 |

Hindsight-only blocked-entry forward-return analysis distinguishes losses avoided from profitable entries blocked. For the balanced test policy, gross-exposure blocks averaged +0.87% direction-adjusted 10-bar return (43.36% would have lost), while volatility blocks averaged +2.49% (29.63% would have lost); those observed cases show that gate blocks were not automatically beneficial. Full breakdowns, stress periods, severe drawdowns, worst days, forced exits, and gap-loss budgets are in `blocked_entries_hindsight.json`, `stress_tests.json`, `failure_analysis.json`, and `evaluation_metrics.json`.

## Training record and artifacts

- Seed 42; fresh initialization; AdamW, initial LR 0.001, weight decay 0.0001, batch 512, gradient clip 1.0, input noise 0.02, capped inverse-frequency weights, ReduceLROnPlateau, patience 7, max 40 epochs. Early stopping at epoch 32; best checkpoint epoch 25.
- All 32 epoch losses, validation macro-F1, LR and durations are in `training_history.csv` and `reports/training_history_full_precision.csv`. The figure is `figures/learning_curves.png`.
- Main deliverables: `conservative_policy.pt`, `best_checkpoint.pt`, `final_checkpoint.pt`, `conservative_config.json`, `feature_schema.json`, `preprocessing.pkl`, `evaluation_metrics.json`, `training_history.csv`, `equity_curve.csv`, copied `sp05_features.py`, `sp05_risk.py`, `sp05_model.py`, and `execution.log`.
- Nine figures are in `figures/`. Machine-readable backtest summary, confusion matrix, gate override, stress, failure, blocked-entry and split reports are in `reports/`.
- The final deterministic reload and risk-limit round-trip assertions passed.

## Limitations

Results are historical, proxy-label-driven and dependent on the supplied OHLCV coverage, modeled execution costs, and the common bluechip portfolio backtest. The fit subset was 360,000 of 10.37 million generated training proxy samples for CPU/memory feasibility; all training labels still informed class weights. SELL examples are rare. The unseen-stock generalization result is weak. The evaluated SP-05 balanced configuration lost money over the 495-session portfolio test. These findings do not establish profitability, durable risk control, or capital preservation.

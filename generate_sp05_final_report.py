"""Build the human-readable SP-05 final training report from run artifacts."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/sp05_conservative_policy/indian_market_run_001"
REPORTS = OUT / "reports"
MET = json.loads((OUT / "evaluation_metrics.json").read_text())
CFG = json.loads((OUT / "conservative_config.json").read_text())
SPLIT = json.loads((REPORTS / "chronological_split.json").read_text())
TRAINING = json.loads((REPORTS / "training_sample_selection.json").read_text())

def pct(x, n=2):
    return f"{100*x:.{n}f}%"

def num(x):
    if x is None:
        return "—"
    return f"{x:,.0f}"

def md_table(rows, headers):
    if not rows:
        return "_No rows._"
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)

cls = MET["classification"]
test = cls["test"]
uncal = cls["test_uncalibrated"]
val = cls["validation"]
unseen = cls["unseen_holdout_cohort"]
blue = cls["top20_bluechips_cohort"]
bt = MET["backtest"]["test"]
balanced = bt["SP-05 policy + gate (balanced, long/short)"]
main_names = {
    "SP-05 policy + gate (balanced, long/short)": "SP-05 + gate, balanced L/S",
    "SP-05 policy + gate (conservative, long/short)": "SP-05 + gate, conservative L/S",
    "SP-05 policy + gate (aggressive, long/short)": "SP-05 + gate, aggressive L/S",
    "SP-05 policy + gate (balanced, long-only)": "SP-05 + gate, balanced long-only",
    "SP-05 policy ALONE, no gate (balanced, long/short)": "SP-05 without gate, balanced L/S",
    "Rule-based risk controller + gate (long-only)": "Rule controller + gate, long-only",
    "Always-long + gate (gate-only baseline)": "Always long + gate",
    "Conservative label rule + gate (balanced, long/short)": "Conservative label rule + gate",
    "NIFTY 50 buy & hold": "NIFTY 50 buy & hold",
    "Equal-weight universe buy & hold": "Equal-weight common bluechip basket*",
    "NIFTY 50 scaled to the main run's average exposure (rest in cash)": "Exposure-matched NIFTY 50",
}
backtest_rows = []
for key, label in main_names.items():
    m = bt[key]
    backtest_rows.append([label, pct(m["total_return"]), pct(m["cagr"]), pct(m["max_drawdown"]),
                          f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}", pct(m["ann_vol"]),
                          pct(m["avg_gross_exposure"]), f"{m['annualized_turnover']:.2f}x",
                          num(m["trade_entries"]), num(m["closed_trades"]), num(m["gate_overrides"])])

actions = ["HOLD", "BUY", "SELL", "EXIT"]
test_actions = [[a, num(test["per_class"][a]["support"]),
                 f"{test['per_class'][a]['precision']:.3f}", f"{test['per_class'][a]['recall']:.3f}",
                 f"{test['per_class'][a]['f1']:.3f}"] for a in actions]
cm_rows = [[actions[i]] + [num(v) for v in test["confusion_matrix"][i]] for i in range(4)]
confusion_table = md_table(cm_rows, ["Actual / Predicted"] + actions)
pd.DataFrame(test["confusion_matrix"], index=actions, columns=actions).rename_axis("actual\\predicted").to_csv(REPORTS / "confusion_matrix_test.csv")
pd.DataFrame(backtest_rows, columns=["strategy", "total_return", "CAGR", "max_drawdown", "Sharpe", "Sortino", "annualized_volatility", "avg_gross_exposure", "annualized_turnover", "entries", "closed_trades", "gate_overrides"]).to_csv(REPORTS / "backtest_test_summary.csv", index=False)
pd.DataFrame(MET["training_history"]).to_csv(REPORTS / "training_history_full_precision.csv", index=False)
(REPORTS / "gate_overrides_test.json").write_text(json.dumps(MET["gate_overrides_by_reason_test"], indent=2))
(REPORTS / "stress_tests.json").write_text(json.dumps(MET["stress_tests"], indent=2))
(REPORTS / "failure_analysis.json").write_text(json.dumps(MET["failure_analysis_test"], indent=2))
(REPORTS / "blocked_entries_hindsight.json").write_text(json.dumps(MET["blocked_entries_hindsight"], indent=2))

split_rows = []
for name, d in SPLIT["splits"].items():
    count = {"train": TRAINING["full_training_proxy_samples"], "val": MET["n_samples"]["val"], "test": MET["n_samples"]["test"]}[name]
    split_rows.append([name, d["first_decision_date"], d["last_decision_date"], num(d["eligible_bars"]), num(count),
                       f"{d['eligible_bar_pct_of_assigned']:.3f}%", f"{d['eligible_bar_pct_of_panel']:.3f}%"])

profile = MET["generalization_experiments"]["experiment_d_profile_sensitivity"]
regimes = MET["generalization_experiments"]["experiment_e_market_regimes"]
controller = MET["gate_overrides_by_reason_test"]
override_rows = []
for strategy, reasons in controller.items():
    override_rows.append([strategy] + [num(reasons.get(k, 0)) for k in ["position_stop_loss", "max_holding_period", "drawdown_halt", "recent_loss_halt", "volatility_prohibited", "abnormal_gap_or_move", "market_stress", "gross_exposure_limit"]])

report = f'''# SP-05 Conservative Risk-Aware Policy — Final Training Report

Run directory: `outputs/sp05_conservative_policy/indian_market_run_001/`  
Data source: local Indian-market OHLCV dataset only; no replacement or external price feed. Local benchmark: NIFTY 50.  
Generated from the final completed run on 2026-11-23.

## Executive summary

SP-05 was trained from fresh seed-42 initialization with the notebook's two-layer GRU architecture. Early stopping selected **epoch {MET['best_epoch']}** with best class-weighted validation loss {min(x['val_loss'] for x in MET['training_history']):.4f} (validation calibrated classification NLL {val['nll']:.4f}) after **32 epochs**. It used a 360,000 example deterministic proportional fit subset from {num(TRAINING['full_training_proxy_samples'])} generated training proxy samples across 1,795 stocks. The best checkpoint was restored before validation-only temperature calibration and final evaluation.

On the untouched chronological test classification set, calibrated accuracy was {pct(test['accuracy'])}, balanced accuracy {pct(test['balanced_accuracy'])}, macro-F1 {test['macro_f1']:.3f}, NLL {test['nll']:.4f}, and ECE {test['ece']:.4f}. The balanced long/short SP-05 + gate backtest returned {pct(balanced['total_return'])} over {int(balanced['n_days'])} days, with maximum drawdown {pct(balanced['max_drawdown'])}, Sharpe {balanced['sharpe']:.2f}, and average gross exposure {pct(balanced['avg_gross_exposure'])}. This was a negative result; classification quality is not evidence of profitability or capital protection.

## Data, cohort, and chronology

- Main eligible cohort: 1,795 symbols. Separate unseen-stock cohort: 100. Dedicated top-20 bluechip cohort: 20. Exact assigned symbols are in `reports/eligible_symbols.csv`; excluded candidates and reasons are in `reports/universe_exclusions.csv`.
- Local NIFTY 50 benchmark: 4,684 observations, 2007-09-17 through 2026-10-23.
- Minimum clean history: 800 observations; eligibility and feature cleaning use the local dataset. Historical delisted stocks remain if eligible. No bars/prices were fabricated.
- Chronological common split boundaries: train through 2021-09-03; validation 2021-09-04 through 2022-10-24; test after 2022-10-24. Strict 30-bar sequence-window isolation is enabled. Per-stock decision date ranges exclude boundary windows and feature warmup.

{md_table(split_rows, ['Split','First decision','Last decision','Complete bars','Proxy samples','% assigned bars','% all panel bars'])}

Percentages are based on complete, assigned decision bars; the all-panel column separately includes warmup and boundary-isolated bars. Exact panel totals and counts are in `reports/chronological_split.json`. Per-stock, position-state, action, and split label counts are in `reports/sample_counts_per_stock_state_action.csv`. Training weights were fitted on all training labels; fitting itself used the documented deterministic subset. Validation and test classification use the complete generated samples.

## Model, objective, and safeguards

- GRU(43,64), two layers, dropout 0.25; static Linear(12,16)+GELU; 80-wide combined representation; Linear(80,64)+GELU+Dropout(0.25)+Linear(64,4). 51,540 parameters. Sequence length 30; action map HOLD/BUY/SELL/EXIT.
- Labels are **strategy-derived decision-time proxy labels**, not human demonstrations and not future-outcome-confirmed labels. Confidence measures agreement with this proxy objective, not probability of avoiding a loss.
- The network proposes actions only. Deterministic `RiskController` has execution authority. Configured limits include 10% max position, 60% max gross, 5% position stop, 120-bar maximum hold, 8% drawdown halt, 15% flatten, 6% recent-loss halt, and 90th percentile volatility ceiling.
- Adversarial checks passed: a high-confidence BUY during drawdown halt was blocked; a hard stop forced EXIT at zero confidence. EXIT remains independent of confidence. These checks are in `reports/risk_controller_adversarial_checks.json`.
- Decision at close *t*, fill at open *t+1*, 5 bps fee plus 5 bps slippage; short borrowing cost follows the configured backtest convention. A deterministic stop does not prevent gap losses beyond the configured stop.

## Classification and calibration

Validation-only fitted temperature: **{test['temperature']:.4f}**. Test rows are {num(test['n_samples'])}; uncalibrated and calibrated values below come from separate saved test metrics. Per-action counts and scores:

{md_table(test_actions, ['Action','Support','Precision','Recall','F1'])}

| Test metric | Calibrated | Uncalibrated |
|---|---:|---:|
| Accuracy | {pct(test['accuracy'])} | {pct(uncal['accuracy'])} |
| Balanced accuracy | {pct(test['balanced_accuracy'])} | {pct(uncal['balanced_accuracy'])} |
| Macro-F1 | {test['macro_f1']:.4f} | {uncal['macro_f1']:.4f} |
| Weighted-F1 | {test['weighted_f1']:.4f} | {uncal['weighted_f1']:.4f} |
| NLL | {test['nll']:.5f} | {uncal['nll']:.5f} |
| ECE | {test['ece']:.5f} | {uncal['ece']:.5f} |

Confusion matrix (rows actual, columns predicted):

{confusion_table}

Validation metrics, per-position-state metrics, full per-action statistics, reliability bins, and probability confidence distributions are in `evaluation_metrics.json`. Reliability diagram: `figures/reliability_test.png`; confusion matrix plot: `figures/confusion_matrix_test.png`.

## Generalization and regime experiments

- Unseen-stock cohort (100 stocks; held out of scaler, class weights, fitting, calibration, and model selection): {num(unseen['n_samples'])} test samples; accuracy {pct(unseen['accuracy'])}, balanced accuracy {pct(unseen['balanced_accuracy'])}, macro-F1 {unseen['macro_f1']:.3f}, ECE {unseen['ece']:.4f}. Performance is materially weaker than main test and SELL support is sparse.
- Top-20 bluechip cohort: {num(blue['n_samples'])} test samples across 20 symbols; accuracy {pct(blue['accuracy'])}, balanced accuracy {pct(blue['balanced_accuracy'])}, macro-F1 {blue['macro_f1']:.3f}, ECE {blue['ece']:.4f}.
- Profile sensitivity, regimes, and risk-condition classification are fully recorded under `generalization_experiments`, `classification_by_risk_condition`, and `entry_proposals_in_forbidden_states` in `evaluation_metrics.json`; plotted in `figures/profile_sensitivity.png`, `figures/regime_comparison.png`, and `figures/gate_and_stress.png`.

## Portfolio backtest results

Backtesting is on the common 20-stock liquid bluechip basket with available aligned bars, not on all 1,795 equities. The final test backtest spans 495 sessions (2024-10-18 to 2026-10-23); the earlier full test classification window begins 2022-12-07. Therefore the portfolio results and classifier metrics have different test coverage. `Equal-weight universe buy & hold` in the implementation is the equal-weight common bluechip basket used by this backtest, not all 1,795 stocks. Long/short results are theoretical; Indian cash-equity shorting is restricted and these figures are not a SEBI-compliant cash-equity execution claim.

{md_table(backtest_rows, ['Strategy','Return','CAGR','Max DD','Sharpe','Sortino','Ann. vol.','Avg gross','Turnover','Entries','Closed','Overrides'])}

*The common-basket equal-weight reference is labelled accurately above. Detailed reports include VaR/CVaR, worst day, ulcer index, underwater share, maximum exposure, trade win rate/mean return, and validation-period values in `evaluation_metrics.json`; plotted equity comparison is `figures/backtest_test.png`. Fees, slippage, and short borrowing costs are applied to fills, but this pipeline does not separately itemize cumulative charges in the final metrics JSON.*

## Risk-controller, stress, and failure analysis

The balanced long/short test strategy had {int(balanced['gate_overrides'])} gate overrides; total overrides by strategy and reason are in the table and `reports/gate_overrides_test.json`.

{md_table(override_rows, ['Strategy','Stop loss','Max hold','DD halt','Recent loss','Volatility','Abnormal move/gap','Market stress','Gross exposure'])}

Hindsight-only blocked-entry forward-return analysis distinguishes losses avoided from profitable entries blocked. For the balanced test policy, gross-exposure blocks averaged +0.87% direction-adjusted 10-bar return (43.36% would have lost), while volatility blocks averaged +2.49% (29.63% would have lost); those observed cases show that gate blocks were not automatically beneficial. Full breakdowns, stress periods, severe drawdowns, worst days, forced exits, and gap-loss budgets are in `blocked_entries_hindsight.json`, `stress_tests.json`, `failure_analysis.json`, and `evaluation_metrics.json`.

## Training record and artifacts

- Seed 42; fresh initialization; AdamW, initial LR 0.001, weight decay 0.0001, batch 512, gradient clip 1.0, input noise 0.02, capped inverse-frequency weights, ReduceLROnPlateau, patience 7, max 40 epochs. Early stopping at epoch 32; best checkpoint epoch 25.
- All 32 epoch losses, validation macro-F1, LR and durations are in `training_history.csv` and `reports/training_history_full_precision.csv`. The figure is `figures/learning_curves.png`.
- Main deliverables: `conservative_policy.pt`, `best_checkpoint.pt`, `final_checkpoint.pt`, `conservative_config.json`, `feature_schema.json`, `preprocessing.pkl`, `evaluation_metrics.json`, `training_history.csv`, `equity_curve.csv`, copied `sp05_features.py`, `sp05_risk.py`, `sp05_model.py`, and `execution.log`.
- Nine figures are in `figures/`. Machine-readable backtest summary, confusion matrix, gate override, stress, failure, blocked-entry and split reports are in `reports/`.
- The final deterministic reload and risk-limit round-trip assertions passed.

## Limitations

Results are historical, proxy-label-driven and dependent on the supplied OHLCV coverage, modeled execution costs, and the common bluechip portfolio backtest. The fit subset was 360,000 of 10.37 million generated training proxy samples for CPU/memory feasibility; all training labels still informed class weights. SELL examples are rare. The unseen-stock generalization result is weak. The evaluated SP-05 balanced configuration lost money over the 495-session portfolio test. These findings do not establish profitability, durable risk control, or capital preservation.
'''
(OUT / "final_training_report.md").write_text(report, encoding="utf-8")
print(f"Wrote {OUT / 'final_training_report.md'}")

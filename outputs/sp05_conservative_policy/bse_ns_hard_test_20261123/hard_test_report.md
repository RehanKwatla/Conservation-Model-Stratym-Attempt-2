# BSE.NS SP-05 hard-test results

## Test scope

Input: `C:\Users\ssama\Downloads\BSE.NS_stock_data.csv` (1,785 clean rows, 2017-02-03 to 2024-04-29). SP-05 used the supplied CSV with the saved scaler/model and locally cached NIFTY 50 benchmark. It ran the balanced profile, with the saved deterministic RiskController and close-*t*/next-open execution. Fees and slippage are each 5 bps per fill; short borrow is 50 bps annualized.

The strict post-validation test window is **2022-12-08 to 2024-04-29** (341 sessions). Earlier dates are excluded from the reported strategy result. BSE is in the 1,795-stock model training universe, so this is not an unseen-stock test. The attached series overlaps the local BSE series on 1,513 dates; daily-return correlation is 0.9974. It is a temporal test on the policy's original held-out period, not an independent data-source or unseen-asset test.

## Returns

| Strategy | Total return | CAGR | Max drawdown | Annual volatility | Sharpe | Avg. gross exposure |
|---|---:|---:|---:|---:|---:|---:|
| SP-05 + RiskController, BSE only | 1.80% | 1.33% | -1.90% | 1.78% | -2.62 | 1.61% |
| BSE.NS buy and hold | 384.26% | 220.83% | -31.02% | 47.27% | 2.58 | 100% |
| Local NIFTY 50 buy and hold | 21.56% | 15.52% | -9.19% | 10.43% | 0.86 | 100% |
| BSE buy and hold scaled to SP-05 average exposure | 2.82% | 2.07% | -0.58% | 0.76% | -5.19 | 1.61% |

SP-05 made 8 entries, closed 7 trades, won 28.6% of closed trades, and had 0 gate overrides. One position remained open at the final mark. The BSE buy-and-hold result was exceptionally strong in this window; these results are single-stock historical outcomes and do not predict future returns.

## Interpretation and caveats

SP-05 returned **1.80%** while averaging only **1.61% gross exposure**; the low exposure reduced both participation and risk. It lagged both full-exposure benchmarks and the BSE buy-and-hold return scaled to its average exposure. The maximum drawdown was lower, but this alone does not prove capital protection. BSE.NS ends 2024-04-29, so this covers only part of the model's full chronological test horizon. One asset cannot establish portfolio performance. Long/short capability is theoretical and does not imply SEBI-compliant cash-equity short execution.

Detailed daily actions, equity, closed trades, gate reasons, input metrics, and limitations are in `daily_policy_results.csv`, `equity_curves.csv`, `closed_trades.csv`, and `hard_test_metrics.json`.

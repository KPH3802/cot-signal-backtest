# COT Signal Backtest

Tests whether extreme commercial hedger positioning (CFTC Commitments of Traders)
predicts forward returns in commodity ETFs.

## Hypothesis
When commercial hedgers (producers/consumers who actually use the commodity) reach
extreme net long positioning vs. their trailing history, the commodity tends to rally.
The inverse holds for extreme net short positioning.

## Commodities Covered
| Commodity    | ETF  | CFTC Code(s)       |
|--------------|------|--------------------|
| WTI Crude    | USO  | 067651, 06765A     |
| Natural Gas  | UNG  | 023651             |
| Gold         | GLD  | 088691             |
| Corn         | CORN | 002602             |
| Wheat (SRW)  | WEAT | 001602             |

## Signal Logic
- **BULL**: Commercial net position >= BULL_THRESHOLD percentile of trailing window
- **BEAR**: Commercial net position <= BEAR_THRESHOLD percentile of trailing window
- Default thresholds: 80th / 20th percentile over a 3-year (156-week) rolling window

## Setup

```bash
pip install pandas numpy yfinance scipy
cp config_example.py config.py
```

Requires `cot_data.db` in `../macro_data/` (populated by `cot_collector.py --backfill`)

## Usage

```bash
# Run all commodities
python3 cot_backtest.py

# Single commodity
python3 cot_backtest.py --commodity "Gold"

# Override thresholds
python3 cot_backtest.py --bull 75 --bear 25

# Override lookback window
python3 cot_backtest.py --lookback 104  # 2 years

# Export trades to CSV
python3 cot_backtest.py --export
```

## Output
- Per-commodity alpha, win rate, t-stat, p-value at 4, 8, 13, 26-week horizons
- Cross-commodity summary table
- Optional CSV export to `results/`

## Disclaimer
For research purposes only. Not financial advice.

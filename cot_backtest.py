#!/usr/bin/env python3
"""
COT Signal Backtest
===================
Tests the hypothesis that extreme commercial hedger positioning (per CFTC COT data)
predicts forward returns in commodity ETFs.

Signal logic:
  - BULL: commercial net position >= BULL_THRESHOLD percentile of trailing LOOKBACK_WEEKS window
  - BEAR: commercial net position <= BEAR_THRESHOLD percentile of trailing LOOKBACK_WEEKS window

Usage:
  python3 cot_backtest.py                      # Run all commodities, all horizons
  python3 cot_backtest.py --commodity "Gold"   # Single commodity
  python3 cot_backtest.py --bull 75 --bear 25  # Override thresholds
  python3 cot_backtest.py --lookback 104       # Override lookback (2 years)
  python3 cot_backtest.py --export             # Save results CSV to results/
"""

import sys
import os
import argparse
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / cfg.RESULTS_DIR
DB_PATH = (SCRIPT_DIR / cfg.COT_DB_PATH).resolve()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_commercial_positions(codes: list) -> pd.DataFrame:
    """Load commercial net positions for a list of contract codes."""
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ','.join('?' * len(codes))
    query = f"""
        SELECT report_date, cftc_contract_market_code, net_positions
        FROM commercial_positions
        WHERE cftc_contract_market_code IN ({placeholders})
        ORDER BY report_date
    """
    df = pd.read_sql_query(query, conn, params=codes)
    conn.close()

    df['report_date'] = pd.to_datetime(df['report_date'])

    # If multiple codes (contract rename), keep one row per date preferring newer code
    if len(codes) > 1:
        df = df.sort_values(['report_date', 'cftc_contract_market_code'], ascending=[True, False])
        df = df.drop_duplicates(subset=['report_date'], keep='first')

    df = df.set_index('report_date').sort_index()
    df = df[['net_positions']].rename(columns={'net_positions': 'comm_net'})
    return df


def load_etf_prices(ticker: str, start: str, end: str = None) -> pd.Series:
    """Download adjusted close prices via yfinance."""
    end = end or datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")
    prices = data['Close']
    if hasattr(prices, 'squeeze'):
        prices = prices.squeeze()
    prices.index = pd.to_datetime(prices.index)
    return prices


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def compute_signals(comm_net: pd.Series, bull_threshold: int, bear_threshold: int,
                    lookback_weeks: int) -> pd.DataFrame:
    """
    Compute rolling percentile and generate BULL/BEAR signals.
    Returns DataFrame with columns: comm_net, percentile, signal
    signal: +1 = bull, -1 = bear, 0 = neutral
    """
    df = pd.DataFrame({'comm_net': comm_net})

    # Rolling percentile rank (0-100)
    def pct_rank(series):
        vals = series.values
        if len(vals) < 2:
            return np.nan
        current = vals[-1]
        history = vals[:-1]
        if np.isnan(current) or len(history) == 0:
            return np.nan
        return float(np.sum(history <= current) / len(history) * 100)

    df['percentile'] = (
        df['comm_net']
        .rolling(window=lookback_weeks + 1, min_periods=max(52, lookback_weeks // 3))
        .apply(pct_rank, raw=False)
    )

    df['signal'] = 0
    df.loc[df['percentile'] >= bull_threshold, 'signal'] = 1
    df.loc[df['percentile'] <= bear_threshold, 'signal'] = -1

    return df


# ---------------------------------------------------------------------------
# Forward return calculation
# ---------------------------------------------------------------------------

def compute_forward_returns(signal_df: pd.DataFrame, prices: pd.Series,
                             hold_periods: list, direction: str) -> pd.DataFrame:
    """
    For each signal date, look up the ETF price at signal date and N weeks forward.
    Returns DataFrame with one row per trade.
    """
    trades = []

    for sig_date, row in signal_df[signal_df['signal'] != 0].iterrows():
        signal = row['signal']

        # Find entry price: first available ETF price on or after signal date
        future_prices = prices[prices.index >= sig_date]
        if future_prices.empty:
            continue
        entry_date = future_prices.index[0]
        entry_price = future_prices.iloc[0]

        trade = {
            'signal_date': sig_date,
            'entry_date': entry_date,
            'signal': 'BULL' if signal == 1 else 'BEAR',
            'percentile': row['percentile'],
            'comm_net': row['comm_net'],
            'entry_price': entry_price,
        }

        for weeks in hold_periods:
            target_date = entry_date + timedelta(weeks=weeks)
            future = prices[prices.index >= target_date]
            if future.empty:
                trade[f'ret_{weeks}w'] = np.nan
                continue
            exit_price = future.iloc[0]
            raw_ret = (exit_price - entry_price) / entry_price

            # Direction adjustment: BULL signal = long, BEAR signal = short
            # For 'long' commodities, bull=long (+), bear=short (-)
            if signal == 1:
                ret = raw_ret  # long trade
            else:
                ret = -raw_ret  # short trade (bear signal)

            if direction == 'inverse':
                ret = -ret

            trade[f'ret_{weeks}w'] = ret

        trades.append(trade)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades: pd.DataFrame, hold_periods: list) -> pd.DataFrame:
    """Compute alpha, win rate, t-stat, and trade count per horizon."""
    rows = []
    for weeks in hold_periods:
        col = f'ret_{weeks}w'
        if col not in trades.columns:
            continue
        rets = trades[col].dropna()
        if len(rets) < 5:
            rows.append({
                'horizon': f'{weeks}w',
                'n_trades': len(rets),
                'mean_ret': np.nan,
                'win_rate': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
            })
            continue

        mean_ret = rets.mean()
        win_rate = (rets > 0).mean()
        t_stat, p_value = stats.ttest_1samp(rets, 0)

        rows.append({
            'horizon': f'{weeks}w',
            'n_trades': len(rets),
            'mean_ret': mean_ret,
            'win_rate': win_rate,
            't_stat': t_stat,
            'p_value': p_value,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_separator(char='=', width=70):
    print(char * width)


def print_results(commodity_name: str, etf: str, stats_df: pd.DataFrame,
                  trades: pd.DataFrame, bull_threshold: int, bear_threshold: int,
                  lookback_weeks: int):
    print()
    print_separator()
    print(f"  {commodity_name.upper()}  |  ETF: {etf}")
    print(f"  Thresholds: Bull >={bull_threshold}th pct  Bear <={bear_threshold}th pct  |  Lookback: {lookback_weeks}w")
    print_separator()

    bull_trades = trades[trades['signal'] == 'BULL']
    bear_trades = trades[trades['signal'] == 'BEAR']
    print(f"  Total signals: {len(trades)}  (Bull: {len(bull_trades)}  Bear: {len(bear_trades)})")
    if not trades.empty:
        print(f"  Date range:    {trades['signal_date'].min().date()} to {trades['signal_date'].max().date()}")
    print()

    if stats_df.empty:
        print("  Not enough data to compute statistics.")
        return

    print(f"  {'Horizon':<10} {'N':>5} {'Alpha':>9} {'Win%':>8} {'t-stat':>8} {'p-val':>8}  Sig")
    print('  ' + '-' * 58)

    for _, r in stats_df.iterrows():
        if pd.isna(r['mean_ret']):
            print(f"  {r['horizon']:<10} {int(r['n_trades']):>5} {'--':>9} {'--':>8} {'--':>8} {'--':>8}")
            continue
        sig = ''
        if r['p_value'] < 0.01:
            sig = '***'
        elif r['p_value'] < 0.05:
            sig = '**'
        elif r['p_value'] < 0.10:
            sig = '*'

        print(
            f"  {r['horizon']:<10} {int(r['n_trades']):>5} "
            f"{r['mean_ret']:>+8.2%} {r['win_rate']:>8.1%} "
            f"{r['t_stat']:>8.2f} {r['p_value']:>8.4f}  {sig}"
        )

    print()
    print("  * p<0.10  ** p<0.05  *** p<0.01")


def print_summary_table(all_stats: dict):
    """Print a single cross-commodity summary for the best horizon."""
    print()
    print_separator('=')
    print("  CROSS-COMMODITY SUMMARY (best horizon per commodity)")
    print_separator('=')
    print(f"  {'Commodity':<20} {'ETF':<6} {'Best Horizon':<13} {'Alpha':>9} {'Win%':>8} {'t-stat':>8} {'N':>5}")
    print('  ' + '-' * 72)

    for name, (stats_df, etf) in all_stats.items():
        if stats_df.empty:
            print(f"  {name:<20} {etf:<6} {'--':<13} {'--':>9} {'--':>8} {'--':>8} {'--':>5}")
            continue
        valid = stats_df.dropna(subset=['t_stat'])
        if valid.empty:
            continue
        best = valid.loc[valid['t_stat'].abs().idxmax()]
        print(
            f"  {name:<20} {etf:<6} {best['horizon']:<13} "
            f"{best['mean_ret']:>+8.2%} {best['win_rate']:>8.1%} "
            f"{best['t_stat']:>8.2f} {int(best['n_trades']):>5}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='COT Signal Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cot_backtest.py
  python3 cot_backtest.py --commodity "Gold"
  python3 cot_backtest.py --bull 75 --bear 25
  python3 cot_backtest.py --lookback 104
  python3 cot_backtest.py --export
        """
    )
    parser.add_argument('--commodity', type=str, default=None,
                        help='Run a single commodity by name (e.g. "Gold")')
    parser.add_argument('--bull', type=int, default=cfg.BULL_THRESHOLD,
                        help=f'Bull threshold percentile (default: {cfg.BULL_THRESHOLD})')
    parser.add_argument('--bear', type=int, default=cfg.BEAR_THRESHOLD,
                        help=f'Bear threshold percentile (default: {cfg.BEAR_THRESHOLD})')
    parser.add_argument('--lookback', type=int, default=cfg.LOOKBACK_WEEKS,
                        help=f'Lookback window in weeks (default: {cfg.LOOKBACK_WEEKS})')
    parser.add_argument('--export', action='store_true',
                        help='Export results CSV to results/ directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # Select commodities
    if args.commodity:
        if args.commodity not in cfg.COMMODITIES:
            print(f"ERROR: '{args.commodity}' not found in config. Available:")
            for name in cfg.COMMODITIES:
                print(f"  {name}")
            sys.exit(1)
        universe = {args.commodity: cfg.COMMODITIES[args.commodity]}
    else:
        universe = cfg.COMMODITIES

    print()
    print_separator()
    print(f"  COT SIGNAL BACKTEST")
    print(f"  Run date:   {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  DB:         {DB_PATH}")
    print(f"  Bull:       >= {args.bull}th percentile")
    print(f"  Bear:       <= {args.bear}th percentile")
    print(f"  Lookback:   {args.lookback} weeks ({args.lookback/52:.1f} years)")
    print(f"  Horizons:   {cfg.HOLD_PERIODS}")
    print_separator()

    all_stats = {}
    all_trades = []

    for name, params in universe.items():
        codes = params['codes']
        etf = params['etf']
        direction = params.get('direction', 'long')

        print(f"\n  Loading {name} ({etf})...", end=' ', flush=True)

        try:
            comm_df = load_commercial_positions(codes)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        if comm_df.empty or comm_df['comm_net'].isna().all():
            print("No data found for contract codes:", codes)
            continue

        # Filter date range
        start = cfg.BACKTEST_START
        end = cfg.BACKTEST_END
        if start:
            comm_df = comm_df[comm_df.index >= start]
        if end:
            comm_df = comm_df[comm_df.index <= end]

        # ETF price range: go back far enough for entry prices
        price_start = (pd.to_datetime(start) - timedelta(weeks=4)).strftime('%Y-%m-%d') if start else '2010-01-01'
        price_end = end or datetime.today().strftime('%Y-%m-%d')

        try:
            prices = load_etf_prices(etf, price_start, price_end)
        except Exception as e:
            print(f"ETF price fetch failed: {e}")
            continue

        print(f"{len(comm_df)} weeks COT, {len(prices)} days ETF prices")

        signal_df = compute_signals(comm_df['comm_net'], args.bull, args.bear, args.lookback)
        trades = compute_forward_returns(signal_df, prices, cfg.HOLD_PERIODS, direction)

        if trades.empty:
            print(f"  No signals generated for {name}.")
            all_stats[name] = (pd.DataFrame(), etf)
            continue

        trades['commodity'] = name
        trades['etf'] = etf
        all_trades.append(trades)

        stats_df = compute_stats(trades, cfg.HOLD_PERIODS)
        all_stats[name] = (stats_df, etf)
        print_results(name, etf, stats_df, trades, args.bull, args.bear, args.lookback)

    # Cross-commodity summary
    if len(all_stats) > 1:
        print_summary_table(all_stats)

    # Export
    if args.export and all_trades:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_trades, ignore_index=True)
        ts = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_path = RESULTS_DIR / f'cot_backtest_{ts}.csv'
        combined.to_csv(out_path, index=False)
        print(f"  Results exported to: {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
COT Vehicle Analysis -- USO vs XLE vs XOP for WTI Crude Signal
==============================================================
Tests the same COT commercial hedger signal across three vehicles:
  - USO: Direct crude oil ETF (futures-linked, subject to contango decay)
  - XLE: Energy Select Sector SPDR (large integrated oil companies)
  - XOP: SPDR S&P Oil & Gas E&P ETF (pure-play upstream, higher crude beta)

Also applies the validated OVX regime filter (OVX >= CRUDE_OVX_MIN) from config.py
so we're comparing vehicles on the same filtered signal, not the raw one.

Usage:
  python3 cot_vehicle_analysis.py                  # All vehicles, OVX filter on
  python3 cot_vehicle_analysis.py --no-filter      # Skip OVX regime filter
  python3 cot_vehicle_analysis.py --export         # Save CSV to results/
"""

import sys
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

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / cfg.RESULTS_DIR
DB_PATH = (SCRIPT_DIR / cfg.COT_DB_PATH).resolve()

# Vehicles to compare for WTI Crude COT signal
VEHICLES = {
    'USO': 'Crude Oil ETF (futures-linked, contango exposure)',
    'XLE': 'Energy Select Sector SPDR (large integrated oil/gas)',
    'XOP': 'S&P Oil & Gas E&P ETF (pure-play upstream, high crude beta)',
}

# WTI contract codes (same as cot_backtest config)
CRUDE_CODES = ['067651', '06765A']
OVX_TICKER = '^OVX'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_commercial_positions(codes):
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ','.join('?' * len(codes))
    df = pd.read_sql_query(f"""
        SELECT report_date, cftc_contract_market_code, net_positions
        FROM commercial_positions
        WHERE cftc_contract_market_code IN ({placeholders})
        ORDER BY report_date
    """, conn, params=codes)
    conn.close()
    df['report_date'] = pd.to_datetime(df['report_date'])
    if len(codes) > 1:
        df = df.sort_values(['report_date', 'cftc_contract_market_code'], ascending=[True, False])
        df = df.drop_duplicates(subset=['report_date'], keep='first')
    df = df.set_index('report_date').sort_index()
    return df[['net_positions']].rename(columns={'net_positions': 'comm_net'})


def load_prices(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f'No price data for {ticker}')
    prices = data['Close'].squeeze()
    prices.index = pd.to_datetime(prices.index)
    return prices


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def compute_signals(comm_net, bull_threshold, bear_threshold, lookback_weeks):
    df = pd.DataFrame({'comm_net': comm_net})

    def pct_rank(series):
        vals = series.values
        if len(vals) < 2:
            return np.nan
        current = vals[-1]
        history = vals[:-1]
        if np.isnan(current):
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
# OVX regime filter
# ---------------------------------------------------------------------------

def apply_ovx_filter(signal_df, ovx_prices, ovx_min):
    """Zero out signals where OVX < ovx_min at signal date."""
    filtered = signal_df.copy()
    for sig_date in filtered[filtered['signal'] != 0].index:
        ovx_at_date = ovx_prices[ovx_prices.index <= sig_date]
        if ovx_at_date.empty:
            filtered.loc[sig_date, 'signal'] = 0
            continue
        ovx_val = ovx_at_date.iloc[-1]
        if ovx_val < ovx_min:
            filtered.loc[sig_date, 'signal'] = 0
    return filtered


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def compute_forward_returns(signal_df, prices, hold_periods):
    """
    Compute forward returns for each signal.
    Crude signal direction: inverse (commercial structural short hedgers)
    BULL signal = long (commercial extreme long means less hedging -> bullish)
    BEAR signal = short
    Direction is already baked in via the 'inverse' config -- here we just
    compute raw ETF returns and flip for BEAR signals.
    """
    trades = []
    for sig_date, row in signal_df[signal_df['signal'] != 0].iterrows():
        signal = int(row['signal'])
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
            # Crude is 'inverse': BULL signal = buy ETF (+), BEAR signal = short ETF (-)
            ret = raw_ret if signal == 1 else -raw_ret
            trade[f'ret_{weeks}w'] = ret

        trades.append(trade)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------

def compute_stats(rets):
    if len(rets) < 5:
        return dict(n=len(rets), alpha=np.nan, win_rate=np.nan, t_stat=np.nan, p_value=np.nan)
    t, p = stats.ttest_1samp(rets, 0)
    return dict(n=len(rets), alpha=rets.mean(), win_rate=(rets > 0).mean(), t_stat=t, p_value=p)


def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def print_sep(char='=', w=72):
    print(char * w)


def print_vehicle_comparison(all_trades, hold_periods, ovx_filtered):
    """Print side-by-side comparison table across all vehicles."""
    filter_label = f'OVX >= {cfg.CRUDE_OVX_MIN} (validated regime filter)' if ovx_filtered else 'No regime filter (all signals)'

    for weeks in hold_periods:
        col = f'ret_{weeks}w'
        print()
        print_sep()
        print(f'  {weeks}-WEEK FORWARD RETURN  |  {filter_label}')
        print_sep()
        print(f'  {"Vehicle":<6}  {"Description":<44}  {"N":>5}  {"Alpha":>9}  {"Win%":>8}  {"t-stat":>8}  {"p-val":>8}  Sig')
        print('  ' + '-' * 98)

        for ticker, desc in VEHICLES.items():
            if ticker not in all_trades or all_trades[ticker].empty:
                print(f'  {ticker:<6}  {desc:<44}  {"--":>5}')
                continue
            rets = all_trades[ticker][col].dropna()
            s = compute_stats(rets)
            if pd.isna(s['alpha']):
                print(f'  {ticker:<6}  {desc:<44}  {int(s["n"]):>5}  {"--":>9}')
            else:
                stars = sig_stars(s['p_value'])
                print(
                    f'  {ticker:<6}  {desc:<44}  {int(s["n"]):>5}  '
                    f'{s["alpha"]:>+8.2%}  {s["win_rate"]:>8.1%}  '
                    f'{s["t_stat"]:>8.2f}  {s["p_value"]:>8.4f}  {stars}'
                )

    print()
    print('  * p<0.10  ** p<0.05  *** p<0.01')


def print_signal_block(label, subset, hold_periods):
    """Print stats for a subset of trades (ALL, BULL only, or BEAR only)."""
    print(f'  {label} ({len(subset)} trades)')
    print(f'  {"Horizon":<10} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8} {"p-val":>8}  Sig')
    print('  ' + '-' * 58)
    for weeks in hold_periods:
        col = f'ret_{weeks}w'
        if col not in subset.columns:
            continue
        rets = subset[col].dropna()
        s = compute_stats(rets)
        if pd.isna(s['alpha']):
            print(f'  {weeks}w{"":<8} {int(s["n"]):>5} {"--":>9}')
        else:
            stars = sig_stars(s['p_value'])
            print(
                f'  {weeks}w{"":<8} {int(s["n"]):>5} '
                f'{s["alpha"]:>+8.2%} {s["win_rate"]:>8.1%} '
                f'{s["t_stat"]:>8.2f} {s["p_value"]:>8.4f}  {stars}'
            )
    print()


def print_detail_table(ticker, desc, trades, hold_periods):
    """Per-vehicle breakdown split by ALL / BULL / BEAR."""
    print()
    print_sep('-')
    bull = trades[trades['signal'] == 'BULL']
    bear = trades[trades['signal'] == 'BEAR']
    print(f'  {ticker}  ({desc})')
    print(f'  Signals: {len(trades)} total  (Bull: {len(bull)}  Bear: {len(bear)})')
    if not trades.empty:
        print(f'  Date range: {trades["signal_date"].min().date()} to {trades["signal_date"].max().date()}')
    print()
    print_signal_block('ALL signals', trades, hold_periods)
    print_signal_block('BULL signals only (long ETF)', bull, hold_periods)
    print_signal_block('BEAR signals only (short ETF)', bear, hold_periods)
    print('  * p<0.10  ** p<0.05  *** p<0.01')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='COT Vehicle Analysis -- USO vs XLE vs XOP')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip OVX regime filter (run all signals regardless of vol regime)')
    parser.add_argument('--export', action='store_true', help='Export trades CSV to results/')
    return parser.parse_args()


def main():
    args = parse_args()
    ovx_filtered = not args.no_filter

    print()
    print_sep()
    print('  COT VEHICLE ANALYSIS -- WTI Crude: USO vs XLE vs XOP')
    print(f'  Run date:     {datetime.today().strftime("%Y-%m-%d")}')
    print(f'  Signal:       Commercial net >= {cfg.BULL_THRESHOLD}th / <= {cfg.BEAR_THRESHOLD}th pct  |  Lookback: {cfg.LOOKBACK_WEEKS}w')
    print(f'  OVX filter:   {"ON -- only signals when OVX >= " + str(cfg.CRUDE_OVX_MIN) if ovx_filtered else "OFF -- all signals included"}')
    print(f'  Vehicles:     {list(VEHICLES.keys())}')
    print_sep()

    # Load COT data
    start = cfg.BACKTEST_START
    price_start = (pd.to_datetime(start) - timedelta(weeks=8)).strftime('%Y-%m-%d') if start else '2010-01-01'
    price_end = cfg.BACKTEST_END or datetime.today().strftime('%Y-%m-%d')

    print('\n  Loading COT commercial positions...', end=' ', flush=True)
    comm_df = load_commercial_positions(CRUDE_CODES)
    if start:
        comm_df = comm_df[comm_df.index >= start]
    print(f'{len(comm_df)} weeks')

    # Compute signals
    signal_df = compute_signals(comm_df['comm_net'], cfg.BULL_THRESHOLD, cfg.BEAR_THRESHOLD, cfg.LOOKBACK_WEEKS)
    raw_count = (signal_df['signal'] != 0).sum()

    # Load OVX and apply regime filter
    if ovx_filtered:
        print(f'  Loading OVX ({OVX_TICKER})...', end=' ', flush=True)
        try:
            ovx_prices = load_prices(OVX_TICKER, price_start, price_end)
            print(f'{len(ovx_prices)} days')
            signal_df = apply_ovx_filter(signal_df, ovx_prices, cfg.CRUDE_OVX_MIN)
            filtered_count = (signal_df['signal'] != 0).sum()
            print(f'  Signals after OVX filter: {filtered_count} of {raw_count} ({raw_count - filtered_count} removed, OVX < {cfg.CRUDE_OVX_MIN})')
        except Exception as e:
            print(f'FAILED ({e}) -- running without OVX filter')
            ovx_filtered = False
    else:
        print(f'  Total signals (unfiltered): {raw_count}')

    # Load vehicle prices and compute returns
    all_trades = {}
    all_export = []

    for ticker, desc in VEHICLES.items():
        print(f'  Loading {ticker}...', end=' ', flush=True)
        try:
            prices = load_prices(ticker, price_start, price_end)
            print(f'{len(prices)} days')
        except Exception as e:
            print(f'FAILED: {e}')
            continue

        trades = compute_forward_returns(signal_df, prices, cfg.HOLD_PERIODS)
        if trades.empty:
            print(f'  No trades for {ticker}')
            continue

        trades['vehicle'] = ticker
        all_trades[ticker] = trades
        all_export.append(trades)

        print_detail_table(ticker, desc, trades, cfg.HOLD_PERIODS)

    # Side-by-side comparison
    if all_trades:
        print()
        print_sep('=')
        print('  CROSS-VEHICLE COMPARISON')
        print_sep('=')
        print_vehicle_comparison(all_trades, cfg.HOLD_PERIODS, ovx_filtered)

    # Best vehicle recommendation
    print()
    print_sep('=')
    print('  RECOMMENDATION SUMMARY')
    print_sep('=')
    for weeks in [8, 13]:
        col = f'ret_{weeks}w'
        best_ticker = None
        best_t = -999
        for ticker in VEHICLES:
            if ticker not in all_trades:
                continue
            rets = all_trades[ticker][col].dropna()
            s = compute_stats(rets)
            if not pd.isna(s['t_stat']) and s['t_stat'] > best_t:
                best_t = s['t_stat']
                best_ticker = ticker
                best_s = s
        if best_ticker:
            stars = sig_stars(best_s['p_value'])
            print(f'  Best at {weeks}w: {best_ticker}  alpha={best_s["alpha"]:+.2%}  win={best_s["win_rate"]:.1%}  t={best_s["t_stat"]:.2f}  {stars}')

    print()

    # Export
    if args.export and all_export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_export, ignore_index=True)
        ts = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_path = RESULTS_DIR / f'cot_vehicle_{ts}.csv'
        combined.to_csv(out_path, index=False)
        print(f'  Exported to: {out_path}')


if __name__ == '__main__':
    main()

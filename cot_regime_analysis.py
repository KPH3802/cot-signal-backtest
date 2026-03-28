#!/usr/bin/env python3
"""
COT Regime Analysis (L2) -- Commodity Volatility Regime Segmentation
=====================================================================
Tests whether COT signals perform differently in low vs high volatility
regimes, using the commodity's OWN realized volatility (not VIX).

Regime definition:
  - Realized vol = trailing 26-week annualized std dev of ETF weekly returns
  - Low Vol  = bottom tercile of realized vol at signal date
  - Mid Vol  = middle tercile
  - High Vol = top tercile

Usage:
  python3 cot_regime_analysis.py                     # All commodities
  python3 cot_regime_analysis.py --commodity "Gold"  # Single commodity
  python3 cot_regime_analysis.py --vol-window 13     # 13-week vol window
  python3 cot_regime_analysis.py --export            # Save CSV to results/
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

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / cfg.RESULTS_DIR
DB_PATH = (SCRIPT_DIR / cfg.COT_DB_PATH).resolve()

DEFAULT_VOL_WINDOW = 26   # weeks for realized vol fallback

# CBOE implied vol index tickers (free via yfinance)
# Where available, use implied vol (forward-looking) instead of realized vol (backward-looking)
# ^OVX = CBOE Crude Oil Volatility Index (same methodology as VIX, based on USO options)
# ^GVZ = CBOE Gold Volatility Index (based on GLD options)
# No free CBOE implied vol index exists for natgas, corn, or wheat -- use realized vol for those
IMPLIED_VOL_TICKERS = {
    "WTI Crude Oil": "^OVX",
    "Gold": "^GVZ",
}


# ---------------------------------------------------------------------------
# Data loading (reuse cot_backtest logic)
# ---------------------------------------------------------------------------

def load_commercial_positions(codes):
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
    if len(codes) > 1:
        df = df.sort_values(['report_date', 'cftc_contract_market_code'], ascending=[True, False])
        df = df.drop_duplicates(subset=['report_date'], keep='first')
    df = df.set_index('report_date').sort_index()
    return df[['net_positions']].rename(columns={'net_positions': 'comm_net'})


def load_etf_prices(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f'No price data for {ticker}')
    prices = data['Close']
    if hasattr(prices, 'squeeze'):
        prices = prices.squeeze()
    prices.index = pd.to_datetime(prices.index)
    return prices


# ---------------------------------------------------------------------------
# Signal generation (same as cot_backtest)
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
# Realized volatility computation
# ---------------------------------------------------------------------------

def compute_realized_vol(prices, vol_window_weeks):
    """
    Compute trailing realized vol from daily prices, resampled to weekly.
    Returns a weekly Series of annualized realized vol.
    vol_window_weeks: number of weekly returns to use
    """
    # Resample to weekly (Friday close)
    weekly = prices.resample('W-FRI').last().dropna()
    weekly_ret = weekly.pct_change().dropna()

    # Rolling std, annualized (52 weeks)
    roll_std = weekly_ret.rolling(window=vol_window_weeks, min_periods=vol_window_weeks // 2).std()
    realized_vol = roll_std * np.sqrt(52)
    realized_vol.name = 'realized_vol'
    return realized_vol


# ---------------------------------------------------------------------------
# Build trade log with vol regime
# ---------------------------------------------------------------------------

def build_trades_with_regime(signal_df, prices, vol_series, hold_periods, direction, vol_label='vol'):
    """
    Same as cot_backtest forward return calc, but also attaches vol at entry.
    vol_series: either implied vol (^OVX/^GVZ) or realized vol -- handled the same way.
    vol_label: column name to use in output ('implied_vol' or 'realized_vol')
    """
    trades = []

    for sig_date, row in signal_df[signal_df['signal'] != 0].iterrows():
        signal = row['signal']

        future_prices = prices[prices.index >= sig_date]
        if future_prices.empty:
            continue
        entry_date = future_prices.index[0]
        entry_price = future_prices.iloc[0]

        # Vol at signal date: find most recent reading on or before sig_date
        vol_at_signal = vol_series[vol_series.index <= sig_date]
        rv = vol_at_signal.iloc[-1] if not vol_at_signal.empty else np.nan

        trade = {
            'signal_date': sig_date,
            'entry_date': entry_date,
            'signal': 'BULL' if signal == 1 else 'BEAR',
            'percentile': row['percentile'],
            'comm_net': row['comm_net'],
            'entry_price': entry_price,
            vol_label: rv,
        }

        for weeks in hold_periods:
            target_date = entry_date + timedelta(weeks=weeks)
            future = prices[prices.index >= target_date]
            if future.empty:
                trade[f'ret_{weeks}w'] = np.nan
                continue
            exit_price = future.iloc[0]
            raw_ret = (exit_price - entry_price) / entry_price
            if signal == 1:
                ret = raw_ret
            else:
                ret = -raw_ret
            if direction == 'inverse':
                ret = -ret
            trade[f'ret_{weeks}w'] = ret

        trades.append(trade)

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Regime segmentation
# ---------------------------------------------------------------------------

def assign_vol_regime(trades_df):
    """Assign Low/Mid/High vol regime based on terciles of vol column (implied or realized)."""
    vol_col = 'implied_vol' if 'implied_vol' in trades_df.columns else 'realized_vol'
    vol = trades_df[vol_col].dropna()
    if len(vol) < 9:
        trades_df['vol_regime'] = 'Unknown'
        return trades_df

    low_cut = np.percentile(vol, 33.3)
    high_cut = np.percentile(vol, 66.7)

    def regime(v):
        if pd.isna(v):
            return 'Unknown'
        if v <= low_cut:
            return 'Low Vol'
        elif v <= high_cut:
            return 'Mid Vol'
        else:
            return 'High Vol'

    trades_df['vol_regime'] = trades_df[vol_col].apply(regime)
    return trades_df, low_cut, high_cut


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(rets):
    if len(rets) < 5:
        return {'n': len(rets), 'alpha': np.nan, 'win_rate': np.nan, 't_stat': np.nan, 'p_value': np.nan}
    t, p = stats.ttest_1samp(rets, 0)
    return {
        'n': len(rets),
        'alpha': rets.mean(),
        'win_rate': (rets > 0).mean(),
        't_stat': t,
        'p_value': p,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sep(char='=', w=72):
    print(char * w)


def sig_stars(p):
    if pd.isna(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def print_regime_table(commodity, etf, trades, hold_periods, low_cut, high_cut, vol_type):
    print()
    print_sep()
    print(f'  {commodity.upper()}  |  ETF: {etf}  |  Vol measure: {vol_type}')
    # Implied vol (OVX/GVZ) arrives as index points (e.g. 33.2) -- display raw
    # Realized vol arrives as decimal (e.g. 0.332) -- display as %
    if 'Implied' in str(vol_type):
        print(f'  Regime cutoffs:  Low Vol <= {low_cut:.1f}  |  High Vol >= {high_cut:.1f}  (index points)')
    else:
        print(f'  Regime cutoffs:  Low Vol <= {low_cut:.1%}  |  High Vol >= {high_cut:.1%}  (annualized realized vol)')
    print_sep()

    regimes = ['Low Vol', 'Mid Vol', 'High Vol', 'ALL']
    for regime in regimes:
        if regime == 'ALL':
            subset = trades
            label = f'  ALL ({len(subset)} trades)'
        else:
            subset = trades[trades['vol_regime'] == regime]
            label = f'  {regime} ({len(subset)} trades)'

        if subset.empty:
            print(f'{label}: no trades')
            continue

        print(f'{label}')
        print(f'  {"Horizon":<10} {"N":>5} {"Alpha":>9} {"Win%":>8} {"t-stat":>8} {"p-val":>8}  Sig')
        print('  ' + '-' * 58)

        for weeks in hold_periods:
            col = f'ret_{weeks}w'
            if col not in subset.columns:
                continue
            rets = subset[col].dropna()
            s = compute_stats(rets)
            if pd.isna(s['alpha']):
                print(f'  {weeks}w{"":<8} {s["n"]:>5} {"--":>9} {"--":>8} {"--":>8} {"--":>8}')
            else:
                stars = sig_stars(s['p_value'])
                print(
                    f'  {weeks}w{"":<8} {s["n"]:>5} '
                    f'{s["alpha"]:>+8.2%} {s["win_rate"]:>8.1%} '
                    f'{s["t_stat"]:>8.2f} {s["p_value"]:>8.4f}  {stars}'
                )
        print()

    print('  * p<0.10  ** p<0.05  *** p<0.01')


def print_best_regime_summary(all_results):
    """Cross-commodity summary: best regime per commodity at 8w horizon."""
    print()
    print_sep('=')
    print('  REGIME SUMMARY -- 8-WEEK HORIZON (best regime per commodity)')
    print_sep('=')
    print(f'  {"Commodity":<20} {"ETF":<6} {"Regime":<10} {"Alpha":>9} {"Win%":>8} {"t-stat":>8} {"N":>5}')
    print('  ' + '-' * 68)

    for name, (etf, trades) in all_results.items():
        col = 'ret_8w'
        if col not in trades.columns or trades.empty:
            print(f'  {name:<20} {etf:<6} {"--":<10} {"--":>9}')
            continue

        best_alpha = -999
        best_row = None
        best_regime = '--'

        for regime in ['Low Vol', 'Mid Vol', 'High Vol', 'ALL']:
            if regime == 'ALL':
                subset = trades
            else:
                subset = trades[trades['vol_regime'] == regime]
            rets = subset[col].dropna()
            if len(rets) < 5:
                continue
            s = compute_stats(rets)
            if not pd.isna(s['alpha']) and s['alpha'] > best_alpha:
                best_alpha = s['alpha']
                best_row = s
                best_regime = regime

        if best_row:
            stars = sig_stars(best_row['p_value'])
            print(
                f'  {name:<20} {etf:<6} {best_regime:<10} '
                f'{best_row["alpha"]:>+8.2%} {best_row["win_rate"]:>8.1%} '
                f'{best_row["t_stat"]:>8.2f} {int(best_row["n"]):>5}  {stars}'
            )
        else:
            print(f'  {name:<20} {etf:<6} {"--":<10}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='COT Regime Analysis -- Commodity Vol Regimes')
    parser.add_argument('--commodity', type=str, default=None)
    parser.add_argument('--bull', type=int, default=cfg.BULL_THRESHOLD)
    parser.add_argument('--bear', type=int, default=cfg.BEAR_THRESHOLD)
    parser.add_argument('--lookback', type=int, default=cfg.LOOKBACK_WEEKS)
    parser.add_argument('--vol-window', type=int, default=DEFAULT_VOL_WINDOW,
                        help=f'Trailing weeks for realized vol (default: {DEFAULT_VOL_WINDOW})')
    parser.add_argument('--export', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    universe = {args.commodity: cfg.COMMODITIES[args.commodity]} if args.commodity else cfg.COMMODITIES
    if args.commodity and args.commodity not in cfg.COMMODITIES:
        print(f'ERROR: {args.commodity} not in config')
        sys.exit(1)

    print()
    print_sep()
    print(f'  COT REGIME ANALYSIS (L2) -- Commodity Realized Volatility')
    print(f'  Run date:    {datetime.today().strftime("%Y-%m-%d")}')
    print(f'  Vol window:  {args.vol_window} weeks trailing realized vol (annualized)')
    print(f'  Regime cuts: Bottom 33rd / Middle / Top 33rd percentile of realized vol')
    print(f'  Signal:      Bull >={args.bull}th  Bear <={args.bear}th  Lookback: {args.lookback}w')
    print_sep()

    all_results = {}
    all_export = []

    for name, params in universe.items():
        codes = params['codes']
        etf = params['etf']
        direction = params.get('direction', 'long')

        print(f'\n  Loading {name} ({etf})...', end=' ', flush=True)

        try:
            comm_df = load_commercial_positions(codes)
        except Exception as e:
            print(f'FAILED: {e}')
            continue

        start = cfg.BACKTEST_START
        end = cfg.BACKTEST_END
        if start:
            comm_df = comm_df[comm_df.index >= start]
        if end:
            comm_df = comm_df[comm_df.index <= end]

        price_start = (pd.to_datetime(start) - timedelta(weeks=args.vol_window + 4)).strftime('%Y-%m-%d') if start else '2010-01-01'
        price_end = end or datetime.today().strftime('%Y-%m-%d')

        try:
            prices = load_etf_prices(etf, price_start, price_end)
        except Exception as e:
            print(f'ETF price fetch failed: {e}')
            continue

        print(f'{len(comm_df)} weeks COT, {len(prices)} days ETF prices')

        # Compute signals
        signal_df = compute_signals(comm_df['comm_net'], args.bull, args.bear, args.lookback)

        # Use implied vol (^OVX/^GVZ) if available for this commodity, else realized vol
        if name in IMPLIED_VOL_TICKERS:
            iv_ticker = IMPLIED_VOL_TICKERS[name]
            try:
                iv_raw = yf.download(iv_ticker, start=price_start, end=price_end, progress=False, auto_adjust=True)
                iv_series = iv_raw['Close'].squeeze()
                iv_series.index = pd.to_datetime(iv_series.index)
                vol_series = iv_series
                vol_label = 'implied_vol'
                vol_type = f'Implied Vol ({iv_ticker})'
            except Exception as e:
                print(f'  WARNING: Could not fetch {iv_ticker} ({e}), falling back to realized vol')
                vol_series = compute_realized_vol(prices, args.vol_window)
                vol_label = 'realized_vol'
                vol_type = f'Realized Vol ({args.vol_window}w trailing)'
        else:
            vol_series = compute_realized_vol(prices, args.vol_window)
            vol_label = 'realized_vol'
            vol_type = f'Realized Vol ({args.vol_window}w trailing)'

        print(f'  Vol measure: {vol_type}')

        # Build trade log with vol at entry
        trades = build_trades_with_regime(signal_df, prices, vol_series, cfg.HOLD_PERIODS, direction, vol_label)

        if trades.empty:
            print(f'  No signals for {name}.')
            continue

        # Assign regime
        result = assign_vol_regime(trades)
        if isinstance(result, tuple):
            trades, low_cut, high_cut = result
        else:
            trades = result
            low_cut = high_cut = np.nan

        trades['commodity'] = name
        trades['etf'] = etf
        all_results[name] = (etf, trades)
        all_export.append(trades)

        print_regime_table(name, etf, trades, cfg.HOLD_PERIODS, low_cut, high_cut, vol_type)

    # Cross-commodity summary
    if len(all_results) > 1:
        print_best_regime_summary(all_results)

    # Export
    if args.export and all_export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_export, ignore_index=True)
        ts = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_path = RESULTS_DIR / f'cot_regime_{ts}.csv'
        combined.to_csv(out_path, index=False)
        print(f'  Exported to: {out_path}')


if __name__ == '__main__':
    main()

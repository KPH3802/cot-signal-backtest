#!/usr/bin/env python3
"""
Path B Scoring Analysis — COT and CEL Signals
==============================================
PART A: Does COT percentile rank predict return magnitude?
PART B: Does USO move magnitude predict equity return magnitude?

Runs both analyses sequentially, prints verdicts.

Usage:
    python3 path_b_cot_cel.py
"""

import sys
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
COT_DATA_DB = (SCRIPT_DIR / "../macro_data/cot_data.db").resolve()
COT_TRADES_DB = SCRIPT_DIR / "cot_backtest.db"
CEL_TRADES_DB = (SCRIPT_DIR / "../commodity_lag_backtest/commodity_lag_backtest.db").resolve()

# Import COT config
sys.path.insert(0, str(SCRIPT_DIR))
import config as cot_cfg

# Import CEL config
sys.path.insert(0, str(SCRIPT_DIR / "../commodity_lag_backtest"))
import importlib
cel_cfg_spec = importlib.util.spec_from_file_location(
    "cel_config", SCRIPT_DIR / "../commodity_lag_backtest/config.py")
cel_cfg = importlib.util.module_from_spec(cel_cfg_spec)
cel_cfg_spec.loader.exec_module(cel_cfg)

# ===== Validated deployment parameters =====
# COT
COT_DEPLOYMENTS = {
    "WTI Crude Oil BULL": {
        "commodity": "WTI Crude Oil",
        "signal": "BULL",
        "codes": ["067651", "06765A"],
        "direction": "inverse",
        "etf": cot_cfg.CRUDE_BULL_VEHICLE,     # XOP
        "hold_weeks": cot_cfg.CRUDE_BULL_HOLD_WEEKS,  # 13
        "regime_filter": "OVX",
        "regime_min": cot_cfg.CRUDE_OVX_MIN,   # 40
    },
    "WTI Crude Oil BEAR": {
        "commodity": "WTI Crude Oil",
        "signal": "BEAR",
        "codes": ["067651", "06765A"],
        "direction": "inverse",
        "etf": cot_cfg.CRUDE_BEAR_VEHICLE,     # USO
        "hold_weeks": cot_cfg.CRUDE_BEAR_HOLD_WEEKS,  # 8
        "regime_filter": "OVX",
        "regime_min": cot_cfg.CRUDE_OVX_MIN,   # 40
    },
    "Gold BULL": {
        "commodity": "Gold",
        "signal": "BULL",
        "codes": ["088691"],
        "direction": "inverse",
        "etf": "GLD",
        "hold_weeks": 13,
        "regime_filter": "GVZ",
        "regime_min": cot_cfg.GOLD_GVZ_MIN,    # 15
    },
    "Gold BEAR": {
        "commodity": "Gold",
        "signal": "BEAR",
        "codes": ["088691"],
        "direction": "inverse",
        "etf": "GLD",
        "hold_weeks": 13,
        "regime_filter": "GVZ",
        "regime_min": cot_cfg.GOLD_GVZ_MIN,
    },
    "Wheat BULL": {
        "commodity": "Wheat (SRW)",
        "signal": "BULL",
        "codes": ["001602"],
        "direction": "long",
        "etf": "WEAT",
        "hold_weeks": 13,
        "regime_filter": "RVOL",
        "regime_max": cot_cfg.WHEAT_RVOL_MAX,  # 0.23
    },
    "Wheat BEAR": {
        "commodity": "Wheat (SRW)",
        "signal": "BEAR",
        "codes": ["001602"],
        "direction": "long",
        "etf": "WEAT",
        "hold_weeks": 13,
        "regime_filter": "RVOL",
        "regime_max": cot_cfg.WHEAT_RVOL_MAX,
    },
    "Corn BULL": {
        "commodity": "Corn",
        "signal": "BULL",
        "codes": ["002602"],
        "direction": "long",
        "etf": "CORN",
        "hold_weeks": 13,
        "regime_filter": "RVOL",
        "regime_max": cot_cfg.CORN_RVOL_MAX,   # 0.196
    },
    "Corn BEAR": {
        "commodity": "Corn",
        "signal": "BEAR",
        "codes": ["002602"],
        "direction": "long",
        "etf": "CORN",
        "hold_weeks": 13,
        "regime_filter": "RVOL",
        "regime_max": cot_cfg.CORN_RVOL_MAX,
    },
}


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def load_prices_yf(ticker, start, end=None):
    """Load adjusted close prices via yfinance. Returns Series or None."""
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        s = data['Close'].squeeze().dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None


def load_prices_batch(tickers, start, end=None):
    """Batch download prices. Returns dict {ticker: Series}."""
    end = end or datetime.today().strftime('%Y-%m-%d')
    result = {}
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return result
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    s = data[('Close', t)].dropna()
                    if not s.empty:
                        s.index = pd.to_datetime(s.index)
                        result[t] = s
                except (KeyError, TypeError):
                    pass
        else:
            if len(tickers) == 1 and 'Close' in data.columns:
                s = data['Close'].squeeze().dropna()
                s.index = pd.to_datetime(s.index)
                result[tickers[0]] = s
    except Exception as e:
        print(f"  Batch download failed: {e}")
    return result


def bucket_table(df, bucket_col, ret_col, alpha_col, label_order, baseline_label):
    """Print a bucket analysis table."""
    header = f"{'Bucket':>10s} | {'Trades':>6s} | {'Win%':>5s} | {'Avg Return':>10s} | {'Avg Alpha':>10s} | {'vs baseline':>18s}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    baseline_ret = None
    for bucket in label_order:
        bk = df[df[bucket_col] == bucket]
        n = len(bk)
        if n == 0:
            continue
        rets = bk[ret_col].dropna()
        if len(rets) == 0:
            continue
        wins = (rets > 0).sum()
        wr = wins / len(rets) * 100
        avg_ret = rets.mean()
        alphas = bk[alpha_col].dropna()
        avg_alpha = alphas.mean() if len(alphas) > 0 else float('nan')

        if baseline_ret is None:
            baseline_ret = avg_ret
            delta_str = "baseline"
        else:
            delta = avg_ret - baseline_ret
            delta_str = f"{delta:+.2f}% delta"

        alpha_str = f"{avg_alpha:+.2f}%" if not np.isnan(avg_alpha) else "N/A"
        print(f"  {bucket:>10s} | {n:>6d} | {wr:>4.1f}% | {avg_ret:>+9.2f}% | {alpha_str:>10s} | {delta_str:>18s}")

    return baseline_ret


def ttest_buckets(df, bucket_col, ret_col, b1_label, b2_label):
    """Run t-test between two buckets. Returns (t, p) or (nan, nan)."""
    g1 = df[df[bucket_col] == b1_label][ret_col].dropna()
    g2 = df[df[bucket_col] == b2_label][ret_col].dropna()
    if len(g1) < 5 or len(g2) < 5:
        return np.nan, np.nan
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return t, p


# ===========================================================================
# PART A: COT PATH B
# ===========================================================================

def load_commercial_positions(codes):
    """Load COT commercial net positions from SQLite."""
    conn = sqlite3.connect(str(COT_DATA_DB))
    placeholders = ','.join('?' * len(codes))
    df = pd.read_sql_query(
        f"SELECT report_date, cftc_contract_market_code, net_positions "
        f"FROM commercial_positions WHERE cftc_contract_market_code IN ({placeholders}) "
        f"ORDER BY report_date",
        conn, params=codes)
    conn.close()
    df['report_date'] = pd.to_datetime(df['report_date'])
    if len(codes) > 1:
        df = df.sort_values(['report_date', 'cftc_contract_market_code'], ascending=[True, False])
        df = df.drop_duplicates(subset=['report_date'], keep='first')
    df = df.set_index('report_date').sort_index()
    return df[['net_positions']].rename(columns={'net_positions': 'comm_net'})


def compute_percentile_signals(comm_net, bull_th, bear_th, lookback):
    """Compute rolling percentile rank and generate signals."""
    df = pd.DataFrame({'comm_net': comm_net})

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
        .rolling(window=lookback + 1, min_periods=max(52, lookback // 3))
        .apply(pct_rank, raw=False)
    )
    df['signal'] = 0
    df.loc[df['percentile'] >= bull_th, 'signal'] = 1
    df.loc[df['percentile'] <= bear_th, 'signal'] = -1
    return df


def get_regime_value(sig_date, regime_type, regime_prices):
    """Get the regime filter value on or before signal date."""
    if regime_prices is None:
        return None
    before = regime_prices[regime_prices.index <= sig_date]
    if before.empty:
        return None
    return float(before.iloc[-1])


def pct_bucket_bull(pct):
    """Bucket for BULL signals (high percentile)."""
    if pct >= 95: return '95+'
    if pct >= 90: return '90-95'
    if pct >= 80: return '80-90'
    return None


def pct_bucket_bear(pct):
    """Bucket for BEAR signals (low percentile = strong bear)."""
    if pct <= 5:  return '95+'    # Below 5th = extreme (mirror of 95+)
    if pct <= 10: return '90-95'  # Below 10th = strong (mirror of 90-95)
    if pct <= 20: return '80-90'  # Below 20th = moderate (mirror of 80-90)
    return None


def run_cot_backtest():
    """Run Part A: COT scoring analysis."""
    print("=" * 78)
    print("  PART A: COT PATH B SCORING ANALYSIS")
    print("  Question: Does percentile rank predict return magnitude?")
    print("=" * 78)

    start = cot_cfg.BACKTEST_START
    price_start = (pd.to_datetime(start) - timedelta(weeks=4)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')
    bull_th = cot_cfg.BULL_THRESHOLD
    bear_th = cot_cfg.BEAR_THRESHOLD
    lookback = cot_cfg.LOOKBACK_WEEKS

    # Load regime data
    print("\n  Loading regime filters...")
    regime_data = {}
    for ticker, label in [("^OVX", "OVX"), ("^GVZ", "GVZ")]:
        p = load_prices_yf(ticker, '2014-01-01', end)
        if p is not None:
            regime_data[label] = p
            print(f"    {label}: {len(p)} days")
        else:
            print(f"    {label}: FAILED")

    # Load SPY for alpha calc
    spy = load_prices_yf("SPY", price_start, end)
    if spy is not None:
        print(f"    SPY: {len(spy)} days")

    # Collect all ETF tickers needed
    etf_tickers = list(set(d['etf'] for d in COT_DEPLOYMENTS.values()))
    print(f"\n  Loading ETF prices: {etf_tickers}")
    etf_prices = load_prices_batch(etf_tickers, price_start, end)
    for t in etf_tickers:
        if t in etf_prices:
            print(f"    {t}: {len(etf_prices[t])} days")
        else:
            print(f"    {t}: FAILED")

    # Process each deployment
    all_cot_trades = []
    verdicts = {}

    for deploy_name, params in COT_DEPLOYMENTS.items():
        commodity = params['commodity']
        sig_type = params['signal']
        codes = params['codes']
        direction = params['direction']
        etf = params['etf']
        hold_weeks = params['hold_weeks']

        print(f"\n  Processing {deploy_name} ({etf} {hold_weeks}w)...")

        # Load COT data
        comm_df = load_commercial_positions(codes)
        if comm_df.empty:
            print(f"    No COT data")
            continue
        comm_df = comm_df[comm_df.index >= start]

        # Compute signals
        sig_df = compute_percentile_signals(comm_df['comm_net'], bull_th, bear_th, lookback)

        # Filter to target signal type
        if sig_type == "BULL":
            signal_rows = sig_df[sig_df['signal'] == 1].copy()
        else:
            signal_rows = sig_df[sig_df['signal'] == -1].copy()

        if signal_rows.empty:
            print(f"    No {sig_type} signals")
            continue

        # Get ETF prices
        prices = etf_prices.get(etf)
        if prices is None:
            print(f"    No {etf} prices")
            continue

        # Regime filter data
        regime_prices = None
        if params['regime_filter'] in ('OVX', 'GVZ'):
            regime_prices = regime_data.get(params['regime_filter'])
        elif params['regime_filter'] == 'RVOL':
            # Realized vol from ETF prices
            pass  # computed below per signal

        # Compute trades
        trades = []
        for sig_date, row in signal_rows.iterrows():
            pct = row['percentile']

            # Regime filter check
            if params['regime_filter'] in ('OVX', 'GVZ'):
                rv = get_regime_value(sig_date, params['regime_filter'], regime_prices)
                if rv is None or rv < params['regime_min']:
                    continue
            elif params['regime_filter'] == 'RVOL':
                # Realized vol: 52-week rolling std of weekly returns
                etf_before = prices[prices.index <= sig_date]
                if len(etf_before) < 260:
                    continue
                weekly = etf_before.resample('W').last().dropna()
                if len(weekly) < 52:
                    continue
                rvol = weekly.pct_change().dropna().tail(52).std() * np.sqrt(52)
                if rvol > params['regime_max']:
                    continue

            # Bucket
            if sig_type == "BULL":
                bucket = pct_bucket_bull(pct)
            else:
                bucket = pct_bucket_bear(pct)
            if bucket is None:
                continue

            # Entry: first trading day on or after signal date
            future_prices = prices[prices.index >= sig_date]
            if future_prices.empty:
                continue
            entry_date = future_prices.index[0]
            entry_price = float(future_prices.iloc[0])

            # Exit: hold_weeks forward
            target_exit = entry_date + timedelta(weeks=hold_weeks)
            exit_candidates = prices[prices.index >= target_exit]
            if exit_candidates.empty:
                continue
            exit_date = exit_candidates.index[0]
            exit_price = float(exit_candidates.iloc[0])

            # Return (direction-aware)
            raw_ret = (exit_price - entry_price) / entry_price
            if sig_type == "BULL":
                if direction == 'inverse':
                    ret = -raw_ret  # inverse commodity: commercial long = bearish underlying, but we buy equity ETF
                else:
                    ret = raw_ret
            else:  # BEAR
                # Short trade
                raw_ret_short = -raw_ret
                if direction == 'inverse':
                    ret = -raw_ret_short
                else:
                    ret = raw_ret_short

            # Actually, let me re-derive this carefully from the original backtest logic:
            # cot_backtest compute_forward_returns:
            #   signal == 1 (BULL): ret = raw_ret; if direction=='inverse': ret = -ret
            #   signal == -1 (BEAR): ret = -raw_ret; if direction=='inverse': ret = -ret
            # So: BULL+inverse = -(exit-entry)/entry  -- expects commodity ETF to drop
            #     BEAR+inverse = (exit-entry)/entry   -- expects commodity ETF to rise (short the short = long)
            # But for deployed vehicles:
            #   Crude BULL -> BUY XOP (long equity): ret = (exit-entry)/entry
            #   Crude BEAR -> SHORT USO: ret = -(exit-entry)/entry = (entry-exit)/entry
            # The original backtest uses the same ETF as commodity proxy. Our deployed vehicles differ.
            # For scoring analysis, use deployed vehicle with correct trade direction:

            # Simpler: deployed direction
            if sig_type == "BULL":
                ret_pct = (exit_price - entry_price) / entry_price * 100  # long trade
            else:
                ret_pct = (entry_price - exit_price) / entry_price * 100  # short trade

            # SPY return for alpha
            spy_ret_pct = None
            alpha_pct = None
            if spy is not None:
                spy_entry_cands = spy[spy.index >= entry_date]
                spy_exit_cands = spy[spy.index >= exit_date]
                if not spy_entry_cands.empty and not spy_exit_cands.empty:
                    spy_e = float(spy_entry_cands.iloc[0])
                    spy_x = float(spy_exit_cands.iloc[0])
                    if sig_type == "BULL":
                        spy_ret_pct = (spy_x - spy_e) / spy_e * 100
                    else:
                        spy_ret_pct = (spy_e - spy_x) / spy_e * 100  # short benchmark
                    alpha_pct = ret_pct - spy_ret_pct

            trades.append({
                'commodity': commodity,
                'signal': sig_type,
                'signal_date': sig_date.strftime('%Y-%m-%d'),
                'etf': etf,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 4),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'exit_price': round(exit_price, 4),
                'hold_weeks': hold_weeks,
                'ret_pct': round(ret_pct, 4),
                'spy_ret_pct': round(spy_ret_pct, 4) if spy_ret_pct is not None else None,
                'alpha_pct': round(alpha_pct, 4) if alpha_pct is not None else None,
                'percentile_rank': round(pct, 2),
                'percentile_bucket': bucket,
            })

        print(f"    {len(trades)} trades after regime filter")
        all_cot_trades.extend(trades)

        if len(trades) < 10:
            verdicts[deploy_name] = "INSUFFICIENT DATA"
            continue

        # Analysis for this deployment
        tdf = pd.DataFrame(trades)
        print(f"\n  === {deploy_name} ({etf} {hold_weeks}w) ===")
        bucket_table(tdf, 'percentile_bucket', 'ret_pct', 'alpha_pct',
                     ['80-90', '90-95', '95+'], '80-90')

        # T-tests
        t1, p1 = ttest_buckets(tdf, 'percentile_bucket', 'ret_pct', '90-95', '80-90')
        t2, p2 = ttest_buckets(tdf, 'percentile_bucket', 'ret_pct', '95+', '80-90')
        sig1 = "SIGNIFICANT" if not np.isnan(p1) and p1 < 0.10 else "not significant"
        sig2 = "SIGNIFICANT" if not np.isnan(p2) and p2 < 0.10 else "not significant"
        print(f"  T-tests: 90-95 vs 80-90: t={t1:.2f} p={p1:.3f} [{sig1}]")
        print(f"           95+ vs 80-90:   t={t2:.2f} p={p2:.3f} [{sig2}]")

        if (not np.isnan(p1) and p1 < 0.10) or (not np.isnan(p2) and p2 < 0.10):
            verdicts[deploy_name] = "GRADIENT CONFIRMED"
        else:
            verdicts[deploy_name] = "NO GRADIENT"

    # Save trades to DB
    if all_cot_trades:
        conn = sqlite3.connect(str(COT_TRADES_DB))
        conn.execute("DROP TABLE IF EXISTS cot_trades")
        conn.execute("""
            CREATE TABLE cot_trades (
                commodity TEXT, signal TEXT, signal_date TEXT,
                etf TEXT, entry_date TEXT, entry_price REAL,
                exit_date TEXT, exit_price REAL,
                hold_weeks INTEGER, ret_pct REAL, spy_ret_pct REAL, alpha_pct REAL,
                percentile_rank REAL, percentile_bucket TEXT
            )
        """)
        for t in all_cot_trades:
            conn.execute(
                "INSERT INTO cot_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t['commodity'], t['signal'], t['signal_date'], t['etf'],
                 t['entry_date'], t['entry_price'], t['exit_date'], t['exit_price'],
                 t['hold_weeks'], t['ret_pct'], t['spy_ret_pct'], t['alpha_pct'],
                 t['percentile_rank'], t['percentile_bucket']))
        conn.commit()
        conn.close()
        print(f"\n  Saved {len(all_cot_trades)} trades to {COT_TRADES_DB}")

    # Print verdicts
    print("\n" + "=" * 78)
    print("  COT VERDICTS:")
    for name, v in verdicts.items():
        print(f"    {name:<30s}: {v}")
    print("=" * 78)

    return verdicts


# ===========================================================================
# PART B: CEL PATH B
# ===========================================================================

def run_cel_backtest():
    """Run Part B: CEL scoring analysis."""
    print("\n" + "=" * 78)
    print("  PART B: CEL PATH B SCORING ANALYSIS")
    print("  Question: Does USO move magnitude predict equity return magnitude?")
    print("=" * 78)

    start = cel_cfg.BACKTEST_START
    end = datetime.today().strftime('%Y-%m-%d')
    price_start = (pd.to_datetime(start) - timedelta(days=10)).strftime('%Y-%m-%d')
    threshold = cel_cfg.MOVE_THRESHOLD  # 2.0%

    # Only Crude Oil is deployed (BEAR side)
    commodity = 'Crude Oil'
    comm_ticker = cel_cfg.COMMODITY_TICKERS[commodity]  # USO
    equity_tickers = cel_cfg.EQUITY_TICKERS[commodity]  # XOP, XLE, CVX, XOM, COP

    # Load all prices in batch
    all_tickers = [comm_ticker, 'SPY'] + equity_tickers
    print(f"\n  Loading prices: {all_tickers}")
    price_cache = load_prices_batch(all_tickers, price_start, end)
    for t in all_tickers:
        if t in price_cache:
            print(f"    {t}: {len(price_cache[t])} days")
        else:
            print(f"    {t}: FAILED")

    comm_prices = price_cache.get(comm_ticker)
    if comm_prices is None:
        print(f"  FATAL: {comm_ticker} prices not available")
        return {}

    spy = price_cache.get('SPY')

    # Filter to backtest range
    comm_prices = comm_prices[comm_prices.index >= start]

    # Find large moves
    daily_ret = comm_prices.pct_change().dropna() * 100
    bear_dates_rets = daily_ret[daily_ret <= -threshold]  # USO drops
    bull_dates_rets = daily_ret[daily_ret >= threshold]    # USO spikes

    print(f"\n  {comm_ticker} signals (>= {threshold}% move):")
    print(f"    BULL (USO up):   {len(bull_dates_rets)}")
    print(f"    BEAR (USO down): {len(bear_dates_rets)}")

    # Process each equity ticker — BEAR side only (deployed signal)
    all_cel_trades = []
    verdicts = {}

    for eq_ticker in equity_tickers:
        eq_prices = price_cache.get(eq_ticker)
        if eq_prices is None:
            print(f"    {eq_ticker}: no prices, skipping")
            continue
        eq_prices = eq_prices[eq_prices.index >= start]
        eq_dates = eq_prices.index

        trades = []

        # BEAR signals: USO drops → SHORT equity (expect equity to follow down)
        for sig_date, uso_move in bear_dates_rets.items():
            # Move bucket
            abs_move = abs(float(uso_move))
            if abs_move >= 5:
                bucket = '5+'
            elif abs_move >= 3:
                bucket = '3-5'
            else:
                bucket = '2-3'

            # Entry: next trading day
            future = eq_dates[eq_dates > sig_date]
            if len(future) < 6:
                continue
            entry_date = future[0]
            entry_price = float(eq_prices[entry_date])

            # Exit: 5 trading days after entry
            exit_idx = 4  # 0-indexed, so future[4] = 5th trading day
            if exit_idx >= len(future):
                continue
            exit_date = future[exit_idx]
            exit_price = float(eq_prices[exit_date])

            # Short trade return
            ret_5d = (entry_price - exit_price) / entry_price * 100

            # SPY return
            spy_ret_5d = None
            alpha_5d = None
            if spy is not None:
                spy_e_cands = spy[spy.index >= entry_date]
                spy_x_cands = spy[spy.index >= exit_date]
                if not spy_e_cands.empty and not spy_x_cands.empty:
                    spy_e = float(spy_e_cands.iloc[0])
                    spy_x = float(spy_x_cands.iloc[0])
                    spy_ret_5d = (spy_e - spy_x) / spy_e * 100  # short benchmark
                    alpha_5d = ret_5d - spy_ret_5d

            trades.append({
                'signal_date': sig_date.strftime('%Y-%m-%d'),
                'commodity': commodity,
                'signal': 'BEAR',
                'etf': eq_ticker,
                'uso_move_pct': round(float(uso_move), 4),
                'move_bucket': bucket,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 4),
                'exit_price': round(exit_price, 4),
                'ret_5d': round(ret_5d, 4),
                'spy_ret_5d': round(spy_ret_5d, 4) if spy_ret_5d is not None else None,
                'alpha_5d': round(alpha_5d, 4) if alpha_5d is not None else None,
            })

        all_cel_trades.extend(trades)
        tdf = pd.DataFrame(trades)

        print(f"\n  === CRUDE BEAR → SHORT {eq_ticker} (5-day hold) ===")
        print(f"  {len(trades)} trades")

        if len(trades) < 20:
            verdicts[f"CEL BEAR {eq_ticker}"] = "INSUFFICIENT DATA"
            continue

        bucket_table(tdf, 'move_bucket', 'ret_5d', 'alpha_5d',
                     ['2-3', '3-5', '5+'], '2-3')

        t1, p1 = ttest_buckets(tdf, 'move_bucket', 'ret_5d', '3-5', '2-3')
        t2, p2 = ttest_buckets(tdf, 'move_bucket', 'ret_5d', '5+', '2-3')
        sig1 = "SIGNIFICANT" if not np.isnan(p1) and p1 < 0.10 else "not significant"
        sig2 = "SIGNIFICANT" if not np.isnan(p2) and p2 < 0.10 else "not significant"
        print(f"  T-tests: 3-5% vs 2-3%: t={t1:.2f} p={p1:.3f} [{sig1}]")
        print(f"           5%+ vs 2-3%:  t={t2:.2f} p={p2:.3f} [{sig2}]")

        if (not np.isnan(p1) and p1 < 0.10) or (not np.isnan(p2) and p2 < 0.10):
            verdicts[f"CEL BEAR {eq_ticker}"] = "GRADIENT CONFIRMED"
        else:
            verdicts[f"CEL BEAR {eq_ticker}"] = "NO GRADIENT"

    # Save to DB
    if all_cel_trades:
        conn = sqlite3.connect(str(CEL_TRADES_DB))
        conn.execute("DROP TABLE IF EXISTS cel_trades")
        conn.execute("""
            CREATE TABLE cel_trades (
                signal_date TEXT, commodity TEXT, signal TEXT, etf TEXT,
                uso_move_pct REAL, move_bucket TEXT,
                entry_date TEXT, exit_date TEXT,
                entry_price REAL, exit_price REAL,
                ret_5d REAL, spy_ret_5d REAL, alpha_5d REAL
            )
        """)
        for t in all_cel_trades:
            conn.execute(
                "INSERT INTO cel_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t['signal_date'], t['commodity'], t['signal'], t['etf'],
                 t['uso_move_pct'], t['move_bucket'], t['entry_date'], t['exit_date'],
                 t['entry_price'], t['exit_price'], t['ret_5d'],
                 t['spy_ret_5d'], t['alpha_5d']))
        conn.commit()
        conn.close()
        print(f"\n  Saved {len(all_cel_trades)} trades to {CEL_TRADES_DB}")

    # Print verdicts
    print("\n" + "=" * 78)
    print("  CEL VERDICTS:")
    for name, v in verdicts.items():
        print(f"    {name:<30s}: {v}")
    print("=" * 78)

    return verdicts


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 78)
    print("  PATH B SCORING ANALYSIS: COT + CEL SIGNALS")
    print(f"  Run date: {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 78)

    cot_verdicts = run_cot_backtest()
    cel_verdicts = run_cel_backtest()

    # Final summary
    print("\n" + "=" * 78)
    print("  FINAL SCORING VERDICTS")
    print("=" * 78)

    all_verdicts = {}
    all_verdicts.update(cot_verdicts)
    all_verdicts.update(cel_verdicts)

    any_gradient = False
    for name, v in all_verdicts.items():
        marker = " <<<" if v == "GRADIENT CONFIRMED" else ""
        print(f"    {name:<35s}: {v}{marker}")
        if v == "GRADIENT CONFIRMED":
            any_gradient = True

    if any_gradient:
        print("\n  ACTION: Implement tiered scoring for signals marked above.")
    else:
        print("\n  ACTION: Keep flat scoring. No magnitude gradients detected.")
        print("  COT stays at Score 3 for all percentile levels above threshold.")
        print("  CEL stays at Score 3 for all USO move sizes above 2%.")

    # Verification
    print("\n  VERIFICATION:")
    try:
        conn = sqlite3.connect(str(COT_TRADES_DB))
        c = conn.cursor()
        c.execute("SELECT commodity, signal, COUNT(*) FROM cot_trades GROUP BY commodity, signal ORDER BY commodity, signal")
        for row in c.fetchall():
            status = "PASS" if row[2] >= 50 else f"BELOW TARGET ({row[2]}<50)"
            print(f"    COT {row[0]} {row[1]}: {row[2]} trades [{status}]")
        conn.close()
    except Exception:
        print("    COT trades DB: not populated")

    try:
        conn = sqlite3.connect(str(CEL_TRADES_DB))
        c = conn.cursor()
        c.execute("SELECT etf, COUNT(*) FROM cel_trades GROUP BY etf ORDER BY etf")
        for row in c.fetchall():
            status = "PASS" if row[1] >= 100 else f"BELOW TARGET ({row[1]}<100)"
            print(f"    CEL BEAR {row[0]}: {row[1]} trades [{status}]")
        conn.close()
    except Exception:
        print("    CEL trades DB: not populated")

    print("=" * 78)


if __name__ == '__main__':
    main()

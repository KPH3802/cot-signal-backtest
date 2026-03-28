# COT Backtest Configuration -- EXAMPLE FILE (safe for GitHub)
# Copy this to config.py and adjust as needed

# --- Database path ---
COT_DB_PATH = "../macro_data/cot_data.db"

# --- Signal thresholds (percentile of trailing window) ---
BULL_THRESHOLD = 80   # commercials extreme net long -> bullish
BEAR_THRESHOLD = 20   # commercials extreme net short -> bearish

# --- Rolling lookback window for percentile calculation ---
LOOKBACK_WEEKS = 156  # 3 years of weekly data

# --- Forward return horizons (in weeks) ---
HOLD_PERIODS = [4, 8, 13, 26]

# --- Commodity universe ---
COMMODITIES = {
    "WTI Crude Oil": {
        "codes": ["067651", "06765A"],
        "etf": "USO",
        "direction": "long",
    },
    "Natural Gas": {
        "codes": ["023651"],
        "etf": "UNG",
        "direction": "long",
    },
    "Gold": {
        "codes": ["088691"],
        "etf": "GLD",
        "direction": "long",
    },
    "Corn": {
        "codes": ["002602"],
        "etf": "CORN",
        "direction": "long",
    },
    "Wheat (SRW)": {
        "codes": ["001602"],
        "etf": "WEAT",
        "direction": "long",
    },
}

# --- Backtest date range ---
BACKTEST_START = "2015-01-01"
BACKTEST_END   = None

# --- Output ---
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# VALIDATED REGIME FILTERS (from L2 regime backtest, Mar 27 2026)
# ---------------------------------------------------------------------------
CRUDE_OVX_MIN = 40        # Only trade crude COT signals when OVX >= this level
GOLD_GVZ_MIN = 15         # Only trade gold COT signals when GVZ >= this level
WHEAT_RVOL_MAX = 0.23     # Only trade wheat COT signals when realized vol <= this level
CORN_RVOL_MAX = 0.196     # Only trade corn COT signals when realized vol <= this level
# Natural Gas: NO EDGE -- do not deploy
